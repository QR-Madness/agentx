# pyright: reportAttributeAccessIssue=false
"""
Golden-set retrieval eval (Memory-Roadmap §2.7 — the read-path sibling of
``eval_consolidation``).

Seeds a known corpus of facts/entities/turns under an isolated eval user,
then runs `remember()` for every golden query under each technique **arm**
(base-only, single techniques, fused, optional cross-encoder/HyDE/self-query),
scoring recall@k + MRR against the expected ids. This is the harness that makes
recall changes measurable — §2.10/§2.11 items are eval-gated on it (e.g. the
cross-encoder stage keeps only on ≥ +5pp MRR here).

Unlike eval_consolidation there is NO sterility gate: recall is user/channel
scoped, so the default run is safe on a live dev cluster (a very full cluster
can crowd the global vector-index over-fetch — the run records the cluster
size, and --snapshot offers a clean room). Arm switching never touches
data/memory_settings.json: settings are pinned process-locally per arm and
restored in ``finally``.

Usage:
    python manage.py eval_recall                       # default arms, save run
    python manage.py eval_recall --list-arms
    python manage.py eval_recall --arms fused_default,fused_cross_encoder
    python manage.py eval_recall --category temporal --no-save
    python manage.py eval_recall --snapshot            # clean-room (wipe+restore)
    python manage.py eval_recall --keep                # leave seeded data in place
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError

EVAL_USER = "_eval_recall_user"
CHANNEL = "_evalr_main"
CHANNEL_PREFIX = "_evalr_"
PIPELINE_VERSION = 1  # manual constant until §2.1 lands; keys run comparability
DEFAULT_KS = (1, 5, 10)
NEGATIVE_K = 3  # top-k window a forbidden distractor must stay out of


@dataclass(frozen=True)
class Arm:
    """One recall configuration to score. ``overrides`` are Settings fields."""
    name: str
    overrides: dict = field(default_factory=dict)
    use_recall_layer: bool = True
    requires_llm: bool = False
    default: bool = True
    note: str = ""


_ALL_TECHNIQUES_OFF = {
    "recall_enable_hybrid": False,
    "recall_enable_entity_centric": False,
    "recall_enable_query_expansion": False,
    "recall_enable_hyde": False,
    "recall_enable_self_query": False,
}
_FUSED_DEFAULT = {
    "recall_enable_hybrid": True,
    "recall_enable_entity_centric": True,
    "recall_enable_query_expansion": True,
    "recall_enable_hyde": False,
    "recall_enable_self_query": False,
}

ARMS: list[Arm] = [
    Arm("base_only", use_recall_layer=False,
        note="base MemoryRetriever only (no RecallLayer)"),
    Arm("vector_only", overrides=_ALL_TECHNIQUES_OFF,
        note="RecallLayer with all 5 techniques off (base path inside the layer)"),
    Arm("hybrid_only", overrides={**_ALL_TECHNIQUES_OFF, "recall_enable_hybrid": True},
        note="BM25+vector RRF fusion only"),
    Arm("entity_only", overrides={**_ALL_TECHNIQUES_OFF, "recall_enable_entity_centric": True},
        note="entity-centric graph traversal only"),
    Arm("expansion_only", overrides={**_ALL_TECHNIQUES_OFF, "recall_enable_query_expansion": True},
        note="rule-based query expansion only"),
    Arm("fused_default", overrides=dict(_FUSED_DEFAULT),
        note="production default set (incl. the §2.11 cross-encoder stage since v0.21.155)"),
    Arm("fused_no_ce",
        overrides={**_FUSED_DEFAULT, "cross_encoder_enabled": False},
        default=False,
        note="fused without the cross-encoder stage — isolates the §2.11 stage-2 gain"),
    Arm("fused_ce_pure",
        overrides={**_FUSED_DEFAULT, "cross_encoder_enabled": True, "reranking_enabled": True,
                   "recall_ce_max_demotion": 0},
        default=False,
        note="pure cross-encoder order (no demotion cap) — the cap-vs-pure ablation"),
    Arm("fused_guard",
        overrides={**_FUSED_DEFAULT, "recall_first_person_guard": True},
        default=False,
        note="fused + first-person guard (default-OFF: failed its §2.11 abstention gate)"),
    Arm("hyde", overrides={**_FUSED_DEFAULT, "recall_enable_hyde": True},
        requires_llm=True, default=False,
        note="fused + HyDE (needs recall_hyde_model provider)"),
    Arm("self_query", overrides={**_FUSED_DEFAULT, "recall_enable_self_query": True},
        requires_llm=True, default=False,
        note="fused + self-query filters (needs recall_self_query_model provider)"),
]


class Command(BaseCommand):
    help = "Golden-set retrieval eval: recall@k/MRR per technique arm (Memory-Roadmap §2.7)."

    def add_arguments(self, parser):
        parser.add_argument("--arms", default=None,
                            help="Comma-separated arm names (default: all default arms). See --list-arms.")
        parser.add_argument("--list-arms", action="store_true",
                            help="Print available arms and exit.")
        parser.add_argument("--corpus", default="builtin",
                            help="Corpus name (default: builtin).")
        parser.add_argument("--only", default=None,
                            help="Comma-separated golden-query names to run.")
        parser.add_argument("--category", default=None,
                            help="Comma-separated categories to run (see corpus CATEGORIES).")
        parser.add_argument("--k", default=",".join(str(k) for k in DEFAULT_KS),
                            help="Comma-separated k values for recall@k (default: 1,5,10).")
        parser.add_argument("--top-k", type=int, default=10,
                            help="top_k passed to remember() (default: 10; scoring uses max(k, top-k)).")
        parser.add_argument("--llm-model", default=None, metavar="PROVIDER:MODEL",
                            help="Pin recall_hyde_model/recall_self_query_model for LLM arms "
                                 "this run (process-local, never written to disk).")
        parser.add_argument("--keep", action="store_true",
                            help="Leave seeded eval data in place after the run (for inspection).")
        parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                            help="Persist a JSON run file + index.jsonl line (default: on).")
        parser.add_argument("--output-dir", default=None,
                            help="Where to write run files (default: <repo>/data/eval_runs).")
        parser.add_argument("--snapshot", action="store_true",
                            help="Clean-room mode: snapshot all memory, wipe, run, restore afterward.")
        parser.add_argument("--wipe", action="store_true",
                            help="DESTROY all memory data first (dev only; prefer --snapshot).")
        parser.add_argument("--restore", default=None, metavar="FILE",
                            help="Recovery mode: wipe and restore a snapshot file, then exit.")

    # -- environment (mirrors eval_consolidation) -----------------------------
    @staticmethod
    def _load_dotenv():
        import os
        env_path = Path(__file__).resolve().parents[4] / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # -- seeding ---------------------------------------------------------------
    def _seed(self, corpus):
        """Seed the corpus once; return {'facts': {key: id}, 'turns': {...}, 'entities': {...}}."""
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
        from agentx_ai.kit.agent_memory.models import Entity, Turn

        conv_id = str(uuid4())
        mem = AgentMemory(user_id=EVAL_USER, conversation_id=conv_id, channel=CHANNEL)

        entity_ids: dict[str, str] = {}
        for e in corpus.entities:
            stored = mem.upsert_entity(Entity(name=e.name, type=e.type, description=e.description or None))
            entity_ids[e.key] = stored.id

        fact_ids: dict[str, str] = {}
        for f in corpus.facts:
            stored = mem.learn_fact(
                f.claim,
                source="eval_seed",
                confidence=f.confidence,
                entity_ids=[entity_ids[k] for k in f.entity_keys] or None,
                temporal_context=f.temporal_context,
            )
            fact_ids[f.key] = stored.id

        turn_ids: dict[str, str] = {}
        turn_content_to_id: dict[str, str] = {}
        for i, t in enumerate(corpus.turns):
            turn = Turn(conversation_id=conv_id, index=i, role=t.role,
                        content=t.content, channel=CHANNEL)
            mem.store_turn(turn)
            turn_ids[t.key] = turn.id
            turn_content_to_id[t.content] = turn.id

        self.stdout.write(
            f"  seeded {len(fact_ids)} facts, {len(entity_ids)} entities, "
            f"{len(turn_ids)} turns under {EVAL_USER}/{CHANNEL}")
        return {"facts": fact_ids, "entities": entity_ids, "turns": turn_ids,
                "turn_content_to_id": turn_content_to_id}

    def _reset_access_stats(self):
        """Zero the access reinforcement recall writes as a side effect (D4)."""
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            s.run(
                "MATCH (n) WHERE n.user_id = $uid "
                "SET n.access_count = 0, n.last_accessed = null",
                uid=EVAL_USER,
            ).consume()

    def _cleanup(self):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection, get_postgres_session
        from sqlalchemy import text
        with Neo4jConnection.session() as s:
            s.run("MATCH (n) WHERE n.user_id = $uid DETACH DELETE n", uid=EVAL_USER).consume()
            s.run("MATCH (n) WHERE n.channel STARTS WITH $p DETACH DELETE n",
                  p=CHANNEL_PREFIX).consume()
        try:
            with get_postgres_session() as ses:
                for tbl in ("conversation_logs", "memory_audit_log"):
                    ses.execute(text(f"DELETE FROM {tbl} WHERE channel LIKE :p"),
                                {"p": CHANNEL_PREFIX + "%"})
        except Exception:
            pass

    # -- arm resolution / preflight ---------------------------------------------
    def _resolve_arms(self, opts) -> list[tuple[Arm, str | None]]:
        """Return [(arm, skip_reason|None)]. LLM arms preflight their provider."""
        by_name = {a.name: a for a in ARMS}
        if opts["arms"]:
            wanted = [n.strip() for n in opts["arms"].split(",") if n.strip()]
            unknown = [n for n in wanted if n not in by_name]
            if unknown:
                raise CommandError(f"Unknown arm(s) {unknown}. See --list-arms.")
            chosen = [by_name[n] for n in wanted]
        else:
            chosen = [a for a in ARMS if a.default]

        from agentx_ai.kit.agent_memory.config import get_settings
        settings = get_settings()
        resolved: list[tuple[Arm, str | None]] = []
        for arm in chosen:
            skip = None
            if arm.requires_llm:
                model = opts.get("llm_model") or (
                    settings.recall_hyde_model if arm.name == "hyde"
                    else settings.recall_self_query_model)
                try:
                    from agentx_ai.providers.registry import get_registry
                    get_registry().get_provider_for_model(model)
                except Exception as e:
                    skip = f"model {model!r} unusable: {e}"
            resolved.append((arm, skip))
        return resolved

    # -- per-arm run ---------------------------------------------------------------
    def _run_arm(self, arm, seeded, queries, ks, top_k, llm_model=None):
        """Score one arm; returns the arm result dict. Settings are pinned
        process-locally via ``pin_memory_settings`` — never written to disk."""
        from agentx_ai.kit.agent_memory.config import get_settings, pin_memory_settings
        from agentx_ai.kit.agent_memory.evals import recall_scoring as sc
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory

        max_k = max([*ks, top_k])
        updates = {**arm.overrides, "retrieval_cache_enabled": False}
        if llm_model and arm.requires_llm:
            updates["recall_hyde_model"] = llm_model
            updates["recall_self_query_model"] = llm_model
        override = get_settings().model_copy(update=updates)

        self._reset_access_stats()
        mem = AgentMemory(user_id=EVAL_USER, conversation_id=str(uuid4()), channel=CHANNEL)

        cross_encoder_loaded = None
        rows = []
        with pin_memory_settings(override):
            # Un-timed warmup: embed cache, BM25 prep, lazy cross-encoder load.
            mem.remember("warmup probe", top_k=1, channels=[CHANNEL],
                         use_recall_layer=arm.use_recall_layer)
            if override.cross_encoder_enabled:
                cross_encoder_loaded = mem.retriever._get_cross_encoder() is not None

            for q in queries:
                t0 = perf_counter()
                bundle = mem.remember(q.query, top_k=max_k, channels=[CHANNEL],
                                      use_recall_layer=arm.use_recall_layer)
                latency_ms = (perf_counter() - t0) * 1000
                fact_rank = [f["id"] for f in bundle.facts]
                # Base-path turn dicts from get_recent_turns carry no id —
                # resolve those to seeded ids by verbatim content.
                turn_rank = []
                for i, t in enumerate(bundle.relevant_turns):
                    tid = t.get("id") or seeded["turn_content_to_id"].get(t.get("content", ""))
                    turn_rank.append(tid or f"_unseeded_{i}")

                if q.category == "negative":
                    forbidden = [seeded["facts"][k] for k in q.forbid_fact_keys]
                    rows.append({
                        "name": q.name, "category": q.category, "negative": True,
                        "abstention_pass": sc.score_negative(fact_rank, forbidden, NEGATIVE_K),
                        "latency_ms": latency_ms,
                    })
                    continue

                if q.category == "callback":
                    ranked = turn_rank
                    relevant = [seeded["turns"][k] for k in q.expected_turn_keys]
                else:
                    ranked = fact_rank
                    relevant = [seeded["facts"][k] for k in q.expected_fact_keys]
                best = relevant[0] if relevant else None
                row = {
                    "name": q.name, "category": q.category, "negative": False,
                    "mrr": sc.mrr(ranked, best),
                    "rank_of_best": sc.rank_of(ranked, best),
                    "latency_ms": latency_ms,
                }
                for k in ks:
                    row[f"recall@{k}"] = sc.recall_at_k(ranked, relevant, k)
                rows.append(row)

        result = {
            "name": arm.name, "status": "OK", "skip_reason": None,
            "use_recall_layer": arm.use_recall_layer, "overrides": arm.overrides,
            "metrics": sc.aggregate(rows, list(ks)), "queries": rows,
        }
        if cross_encoder_loaded is not None:
            result["cross_encoder_loaded"] = cross_encoder_loaded
        return result

    # -- main --------------------------------------------------------------------
    def handle(self, *args, **opts):
        self._load_dotenv()
        from agentx_ai.kit.agent_memory.portability import cluster

        if opts["list_arms"]:
            for a in ARMS:
                tag = "default" if a.default else ("llm" if a.requires_llm else "extra")
                self.stdout.write(f"  {a.name:<20} [{tag}] {a.note}")
            return

        if opts["restore"]:
            if not Path(opts["restore"]).exists():
                raise CommandError(f"Snapshot file not found: {opts['restore']}")
            self.stdout.write(self.style.WARNING("Restoring snapshot (wipes current memory first)..."))
            cluster.restore_snapshot(opts["restore"], self.stdout.write, self.stderr.write)
            return

        from agentx_ai.kit.agent_memory.evals.recall_corpus import CATEGORIES, load_corpus
        try:
            corpus = load_corpus(opts["corpus"])
        except ValueError as e:
            raise CommandError(str(e)) from None

        queries = list(corpus.queries)
        if opts["only"]:
            wanted = {n.strip() for n in opts["only"].split(",")}
            queries = [q for q in queries if q.name in wanted]
        if opts["category"]:
            cats = {c.strip() for c in opts["category"].split(",")}
            unknown = cats - set(CATEGORIES)
            if unknown:
                raise CommandError(f"Unknown categories {sorted(unknown)}; valid: {CATEGORIES}")
            queries = [q for q in queries if q.category in cats]
        if not queries:
            raise CommandError("No golden queries selected.")

        ks = tuple(sorted({int(k) for k in str(opts["k"]).split(",")}))
        arms = self._resolve_arms(opts)

        # Preflight the embedder BEFORE touching the graph.
        from agentx_ai.kit.agent_memory.embeddings import get_embedder
        try:
            get_embedder().embed_single("preflight probe")
        except Exception as e:
            raise CommandError(f"Embedder unavailable — cannot seed or recall: {e}") from e

        # Optional clean room; default runs live but records cluster size.
        snapshot_path = None
        existing = cluster.count_conversations()
        if opts["snapshot"]:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            snap_dir = Path(__file__).resolve().parents[4] / "data" / "eval_snapshots"
            snapshot_path = cluster.make_snapshot(snap_dir / f"{ts}.json", self.stdout.write)
        if snapshot_path or opts["wipe"]:
            self.stdout.write(self.style.WARNING("Wiping all memory data..."))
            cluster.wipe_cluster(self.stdout.write, self.stderr.write)
        elif existing:
            self.stdout.write(self.style.WARNING(
                f"  note: cluster holds {existing} conversation(s) — global vector-index "
                "crowding can depress scores; use --snapshot for a clean room."))

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nRecall eval — corpus={corpus.name}  queries={len(queries)}  "
            f"arms={[a.name for a, _ in arms]}  k={list(ks)}\n"))

        arm_results = []
        try:
            seeded = self._seed(corpus)
            for arm, skip in arms:
                if skip:
                    self.stdout.write(f"  {arm.name:<20} {self.style.WARNING('SKIPPED')}  {skip}")
                    arm_results.append({
                        "name": arm.name, "status": "SKIPPED", "skip_reason": skip,
                        "use_recall_layer": arm.use_recall_layer,
                        "overrides": arm.overrides, "metrics": None, "queries": [],
                    })
                    continue
                result = self._run_arm(arm, seeded, queries, ks, opts["top_k"],
                                       llm_model=opts.get("llm_model"))
                arm_results.append(result)
                self._render_arm(result, ks)
        finally:
            if snapshot_path:
                cluster.restore_snapshot(
                    snapshot_path,
                    lambda m: self.stdout.write(self.style.SUCCESS(m)),
                    self.stderr.write,
                )
                self.stdout.write(self.style.WARNING(
                    f"\nsnapshot kept for recovery: {snapshot_path}"))
            elif opts["keep"]:
                self.stdout.write(self.style.WARNING(
                    f"\n--keep set: eval data left under {EVAL_USER}/{CHANNEL_PREFIX}*"))
            else:
                self._cleanup()
                self.stdout.write("\ncleaned up seeded eval data")

        self._render_summary(arm_results, ks)
        if opts["save"]:
            self._persist_run(opts, corpus, queries, arm_results, ks, existing)

    # -- rendering -------------------------------------------------------------
    def _render_arm(self, result, ks):
        m = result["metrics"]
        parts = [f"mrr={m['mrr']}"] + [f"r@{k}={m[f'recall@{k}']}" for k in ks]
        if "abstention_pass_rate" in m:
            parts.append(f"abstain={m['abstention_pass_rate']}")
        parts.append(f"p95={m['latency_ms']['p95']}ms")
        if result.get("cross_encoder_loaded") is False:
            parts.append(self.style.ERROR("cross-encoder FAILED to load"))
        self.stdout.write(f"  {result['name']:<20} {'  '.join(str(p) for p in parts)}")

    def _render_summary(self, arm_results, ks):
        self.stdout.write("\n" + "=" * 78)
        header = f"{'arm':<20} {'mrr':>6} " + " ".join(f"{'r@' + str(k):>6}" for k in ks) + \
                 f" {'abstain':>8} {'p95 ms':>8}"
        self.stdout.write(header)
        for r in arm_results:
            if r["status"] != "OK":
                self.stdout.write(f"{r['name']:<20} SKIPPED — {r['skip_reason']}")
                continue
            m = r["metrics"]
            row = f"{r['name']:<20} {m['mrr'] if m['mrr'] is not None else '-':>6} "
            row += " ".join(f"{m[f'recall@{k}'] if m[f'recall@{k}'] is not None else '-':>6}" for k in ks)
            row += f" {m.get('abstention_pass_rate', '-'):>8} {m['latency_ms']['p95']:>8}"
            self.stdout.write(row)
        self.stdout.write("=" * 78)

    # -- persistence -------------------------------------------------------------
    def _persist_run(self, opts, corpus, queries, arm_results, ks, cluster_conversations):
        from agentx_ai.kit.agent_memory.config import get_recall_settings, get_settings

        out_dir = Path(opts["output_dir"]) if opts["output_dir"] else (
            Path(__file__).resolve().parents[4] / "data" / "eval_runs")
        out_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4]}"
        path = out_dir / f"{run_id}_recall_{corpus.name}.json"

        settings = get_settings()
        categories: dict[str, int] = {}
        for q in queries:
            categories[q.category] = categories.get(q.category, 0) + 1

        payload = {
            "harness": "recall",
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "corpus": {"name": corpus.name, "facts": len(corpus.facts),
                       "entities": len(corpus.entities), "turns": len(corpus.turns),
                       "queries": len(queries), "categories": categories},
            "flags": {"arms": [r["name"] for r in arm_results], "only": opts["only"],
                      "category": opts["category"], "top_k": opts["top_k"],
                      "ks": list(ks), "negative_k": NEGATIVE_K,
                      "llm_model": opts.get("llm_model"),
                      "snapshot": bool(opts["snapshot"])},
            "cluster_conversations_at_start": cluster_conversations,
            "base_settings": {
                **get_recall_settings(),
                "cross_encoder_enabled": settings.cross_encoder_enabled,
                "cross_encoder_model": settings.cross_encoder_model,
                "reranking_enabled": settings.reranking_enabled,
                "embedding_provider": settings.embedding_provider,
            },
            "arms": arm_results,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

        index_line = {
            "run_id": run_id, "timestamp": payload["timestamp"], "harness": "recall",
            "corpus": corpus.name, "queries": len(queries),
            "arms": {
                r["name"]: {"mrr": r["metrics"]["mrr"],
                            **{f"recall@{k}": r["metrics"][f"recall@{k}"] for k in ks}}
                for r in arm_results if r["status"] == "OK"
            },
        }
        with (out_dir / "index.jsonl").open("a") as fh:
            fh.write(json.dumps(index_line) + "\n")

        self.stdout.write(f"\nsaved run → {path}")

# pyright: reportAttributeAccessIssue=false
"""
End-to-end consolidation quality eval (prompt tuning + regression checking).

Seeds graded conversation cases (ranging from a single fact to multi-turn,
multi-entity narratives with temporal/negation/refinement) into isolated memory
channels, runs the REAL consolidation pipeline against live Neo4j + a configured
LLM provider, and scores each case's extracted entities/facts against
expectations. Use it to compare models and tune extraction prompts.

⚠️  consolidate_episodic_to_semantic() is GLOBAL — it processes every
unconsolidated conversation in the connected memory cluster. This command must
run against a STERILE dev instance. On a cluster that holds data you care about,
pass --snapshot: it exports ALL memory, wipes, runs, then restores it afterward
(even if the eval errors). --wipe is the destructive alternative (no restore).

Usage:
    python manage.py eval_consolidation --snapshot   # back up + restore around the run
    python manage.py eval_consolidation --wipe        # destructive; sterile/dev only
    python manage.py eval_consolidation --model openrouter:minimax/minimax-m2.7 --snapshot
    python manage.py eval_consolidation --snapshot --full   # also correction + contradiction stages
    python manage.py eval_consolidation --snapshot --only occupation_location,temporal_job_change
    python manage.py eval_consolidation --restore data/eval_snapshots/<ts>.json  # recovery
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError

EVAL_USER = "_eval_user"
CHANNEL_PREFIX = "_eval_"


@dataclass
class EvalCase:
    """A graded consolidation eval case.

    Two kinds of case share this shape:

    **Extraction** (default) — graded conversation turns scored against the
    entities/facts that consolidation should store. `expect_facts` /
    `expect_entities` are case-insensitive substrings expected to appear in some
    stored claim / entity name. `forbid_facts` substrings must NOT appear in any
    claim (e.g. the positive form of a negated statement). `relevant=False`
    asserts the turns are pure noise and nothing should store.

    **Procedural** — any case with a non-empty `tool_sequence`. Instead of turns,
    it seeds a tool-usage trajectory plus a success/failure `Outcome` and is
    scored against the `Strategy` that `detect_patterns()` learns. `task_type`
    only flavors the learned strategy description (not a scoring key — learned
    strategies carry no channel). `expect_strategy_tools` defaults to
    `tool_sequence`; set it to assert a subset. `outcome_success=False` is a
    negative case — the detector's `WHERE success:true` gate must learn nothing.
    """
    name: str
    level: int  # 1 (simple) .. 5 (complex)
    turns: list[str]
    note: str = ""
    relevant: bool = True
    expect_facts: list[str] = field(default_factory=list)
    expect_entities: list[str] = field(default_factory=list)
    forbid_facts: list[str] = field(default_factory=list)
    # -- §2.10 Slice 1 graph-honesty assertions -------------------------------
    # Each inner list = phrasings of ONE concept; PASS iff exactly one stored
    # entity matches (by name or alias) — catches duplicate-entity minting.
    expect_single_entity: list[list[str]] = field(default_factory=list)
    # Cardinality ceiling for the case's channel (None = unchecked).
    expect_entity_count_max: int | None = None
    # [[A, B], ...]: >=1 RELATES_TO edge between A and B (either direction) —
    # catches cross-turn relationship blindness + silent endpoint drops.
    expect_relationship_pairs: list[list[str]] = field(default_factory=list)
    # Groups of phrasings that must resolve to DIFFERENT entities (each group
    # nonempty, pairwise-disjoint ids) — the inverse of expect_single_entity:
    # similar contacts must NOT be over-merged ("Nadia Osei" ≠ "Dr. Osei").
    expect_distinct_entities: list[list[str]] = field(default_factory=list)
    # Every stored fact must carry a non-null source_turn_id.
    expect_fact_provenance: bool = False
    # -- procedural-only fields ---------------------------------------------
    tool_sequence: list[str] = field(default_factory=list)
    task_type: str = "general"
    outcome_success: bool = True
    expect_strategy_tools: list[str] = field(default_factory=list)

    @property
    def is_procedural(self) -> bool:
        return bool(self.tool_sequence)

    @property
    def expected_tools(self) -> list[str]:
        """Tools the learned strategy must cover (defaults to the seeded set)."""
        return self.expect_strategy_tools or self.tool_sequence

    @property
    def kind(self) -> str:
        return "procedural" if self.is_procedural else "extraction"


# Cases range in complexity so prompt/model changes can be evaluated across
# difficulty. Add new cases here — they're plain data.
CASES: list[EvalCase] = [
    # ---- Level 1: single clear signal --------------------------------------
    EvalCase(
        name="single_name", level=1,
        turns=["My name is Sofia Almeida."],
        note="single identity fact + person entity",
        expect_facts=["Sofia"], expect_entities=["Sofia"],
    ),
    EvalCase(
        name="irrelevant_pleasantry", level=1,
        turns=["ok thanks, sounds good!"],
        note="pure pleasantry — nothing should be stored",
        relevant=False,
    ),
    # ---- Level 2: multiple facts/entities in one turn ----------------------
    EvalCase(
        name="occupation_location", level=2,
        turns=["I work as a data scientist at Spotify in Stockholm."],
        note="role + employer + city packed into one turn",
        expect_facts=["data scientist", "Spotify", "Stockholm"],
        expect_entities=["Spotify", "Stockholm"],
    ),
    # ---- Level 3: relationships, negation -----------------------------------
    EvalCase(
        name="colleague_relationship", level=3,
        turns=["My colleague Marcus leads the recommendations team at Spotify."],
        note="person↔organization relationship",
        expect_facts=["Marcus"], expect_entities=["Marcus", "Spotify"],
    ),
    EvalCase(
        name="negation_preference", level=3,
        turns=["I don't drink coffee, but I really love green tea."],
        note="explicit negation must not flip to a positive claim",
        expect_facts=["green tea"],
        forbid_facts=["user drinks coffee"],
    ),
    # ---- Level 4: multi-turn refinement, temporal, mixed relevance ----------
    EvalCase(
        name="multiturn_refinement", level=4,
        turns=[
            "I'm learning Rust for systems programming.",
            "Mostly I use it for a CLI tool I'm building called Ferro.",
        ],
        note="accumulate + refine across turns; surface the project entity",
        expect_facts=["Rust"], expect_entities=["Rust", "Ferro"],
    ),
    EvalCase(
        name="temporal_job_change", level=4,
        turns=["I used to work at Google, but I moved to Anthropic last month."],
        note="past vs current employer (temporal disambiguation)",
        expect_facts=["Anthropic"], expect_entities=["Google", "Anthropic"],
    ),
    EvalCase(
        name="mixed_relevance", level=4,
        turns=["Thanks so much for the help! By the way, I'm based in Berlin and "
               "my role is backend engineer."],
        note="pleasantry + real substance in the same turn",
        expect_facts=["Berlin", "backend engineer"], expect_entities=["Berlin"],
    ),
    # ---- Level 5: complex multi-entity narratives ---------------------------
    EvalCase(
        name="complex_event", level=5,
        turns=["I'm organizing a conference called PyNordic in Oslo this coming "
               "June. We expect around 800 attendees and Guido van Rossum is keynoting."],
        note="event + location + person + future temporal + counts",
        expect_facts=["PyNordic", "Oslo"],
        expect_entities=["PyNordic", "Oslo", "Guido"],
    ),
    EvalCase(
        name="complex_narrative", level=5,
        turns=[
            "I'm a PhD student at ETH Zurich studying reinforcement learning.",
            "My advisor is Professor Chen and I'm collaborating with DeepMind on "
            "a sample-efficiency project.",
            "We're targeting a NeurIPS submission in May.",
        ],
        note="multi-turn: multiple orgs/people/topics + future temporal",
        expect_facts=["reinforcement learning"],
        expect_entities=["ETH Zurich", "DeepMind"],
    ),
    # ---- §2.10 Slice 1: graph honesty (windowed extraction + resolution) ----
    EvalCase(
        name="dedup_three_phrasings", level=4,
        turns=[
            "I've been reading about RAG for my research notes.",
            "Retrieval-augmented generation seems like the right approach for grounding them.",
            "So retrieval augmented generation is what I'll build the notes system on.",
        ],
        note="one concept, three phrasings across turns → exactly ONE entity",
        expect_facts=["retrieval"],
        expect_single_entity=[["RAG", "retrieval-augmented generation",
                               "retrieval augmented generation"]],
        expect_entity_count_max=3,
    ),
    EvalCase(
        name="cross_turn_relationship", level=4,
        turns=[
            "My colleague Amara has been job hunting for months.",
            "Separately — I finally fixed my bike's gear cable today.",
            "She accepted the offer! She now works at Northwind Observatory.",
        ],
        note="edge endpoints introduced in DIFFERENT turns (T1 person, T3 employer)",
        expect_entities=["Amara", "Northwind"],
        expect_relationship_pairs=[["Amara", "Northwind"]],
    ),
    EvalCase(
        name="user_fact_provenance", level=1,
        turns=["My favorite tea is oolong."],
        note="user facts must carry source_turn_id (was None pre-Slice-1)",
        expect_facts=["oolong"],
        expect_fact_provenance=True,
    ),
    # ---- Level 5+: juicy multi-contact business narratives -------------------
    EvalCase(
        name="startup_org_map", level=5,
        turns=[
            "Quick update on Brightline Analytics, the supply-chain startup I founded with Priya Raman.",
            "We just closed a seed round led by Harbor Peak Capital — their partner Tomás Vega is joining our board.",
            "Priya is moving from CTO to Chief Product Officer; our new CTO is Deng Wei, who we poached from Kestrel Robotics.",
            "We're also piloting with two customers: Meridian Foods and the Aster Hotel Group.",
        ],
        note="org map: founders/investors/role change/customers across 4 turns",
        expect_facts=["Chief Product Officer"],
        expect_entities=["Brightline", "Priya", "Harbor Peak", "Tomás", "Deng Wei",
                         "Kestrel", "Meridian", "Aster"],
        expect_single_entity=[["Brightline Analytics", "Brightline"]],
        expect_relationship_pairs=[["Priya", "Brightline"],
                                   ["Tomás", "Harbor Peak"],
                                   ["Deng Wei", "Brightline"]],
    ),
    EvalCase(
        name="contact_web_two_oseis", level=5,
        turns=[
            "My accountant Nadia Osei just moved her practice to Geneva.",
            "Not to be confused with my mentor Dr. Osei — he's still in Accra. Nadia is his niece, funnily enough.",
            "Nadia's firm, Ledgerline Partners, now handles both my business and personal filings.",
        ],
        note="two similar contacts must stay DISTINCT nodes + kinship edge",
        expect_entities=["Nadia", "Ledgerline"],
        expect_distinct_entities=[["Nadia Osei"], ["Dr. Osei"]],
        expect_relationship_pairs=[["Nadia", "Ledgerline"]],
    ),
    EvalCase(
        name="vendor_project_pipeline", level=5,
        turns=[
            "The Foxglove rollout is the data-platform project I'm running this quarter.",
            "Our vendor contact is Sofia Marchetti from Delta Harbor Systems.",
            "Foxglove's steering committee meets Thursdays; Sofia joins remotely.",
            "If the rollout succeeds, Delta Harbor becomes our primary vendor next year.",
        ],
        note="project + vendor contact + future commitment, endpoints across turns",
        expect_facts=["primary vendor"],
        expect_entities=["Foxglove", "Sofia", "Delta Harbor"],
        expect_relationship_pairs=[["Sofia", "Delta Harbor"]],
        expect_fact_provenance=True,
    ),
    # ---- Procedural: tool-usage → strategy learning -------------------------
    # These seed a tool trajectory + success/failure Outcome and are scored
    # against the Strategy that detect_patterns() learns. They're embedding-only
    # (no extraction LLM) — a plumbing/regression check, constant across --model.
    EvalCase(
        name="proc_web_research", level=2, turns=[],
        note="successful research run → strategy over both tools",
        tool_sequence=["web_search", "read_stored_output"], task_type="research",
    ),
    EvalCase(
        name="proc_code_edit", level=2, turns=[],
        note="successful coding run → strategy over both tools",
        tool_sequence=["read_file", "edit_file"], task_type="coding",
    ),
    EvalCase(
        name="proc_failed_run", level=2, turns=[],
        note="failed outcome → detector's success-gate must learn nothing",
        tool_sequence=["web_search"], task_type="research", outcome_success=False,
    ),
]


class Command(BaseCommand):
    help = "Run an end-to-end consolidation quality eval against a sterile dev instance."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model", default=None,
            help="Extraction model as provider:model_id (default: configured combined_extraction_model).",
        )
        parser.add_argument(
            "--wipe", action="store_true",
            help="DESTROY all memory data first (required unless the instance is already sterile). "
                 "Prefer --snapshot on a dev cluster you care about.",
        )
        parser.add_argument(
            "--snapshot", action="store_true",
            help="Non-destructive alternative to --wipe: export all memory to "
                 "data/eval_snapshots/<ts>.json, wipe, run, then restore it afterward "
                 "(even if the eval errors). Bypasses the sterility gate. Snapshots are "
                 "text-only; embeddings are regenerated on restore.",
        )
        parser.add_argument(
            "--restore", default=None, metavar="FILE",
            help="Recovery mode: wipe and restore a prior --snapshot file, then exit "
                 "(use if a snapshot run crashed between wipe and restore).",
        )
        parser.add_argument(
            "--keep", action="store_true",
            help="Leave seeded eval data in place after the run (for inspection).",
        )
        parser.add_argument(
            "--full", action="store_true",
            help="Also exercise the correction + contradiction stages (more LLM calls).",
        )
        parser.add_argument(
            "--only", default=None,
            help="Comma-separated case names to run (default: all).",
        )
        parser.add_argument(
            "--save", action=argparse.BooleanOptionalAction, default=True,
            help="Persist a JSON run file + index.jsonl line (default: on; --no-save to skip).",
        )
        parser.add_argument(
            "--output-dir", default=None,
            help="Where to write run files (default: <repo>/data/eval_runs).",
        )

    # -- environment ---------------------------------------------------------
    @staticmethod
    def _load_dotenv():
        """Best-effort: load repo .env into os.environ so provider keys resolve."""
        env_path = Path(__file__).resolve().parents[4] / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # -- sterility / wipe / snapshot (shared impl: portability.cluster) -------
    def _count_conversations(self):
        from agentx_ai.kit.agent_memory.portability import cluster
        return cluster.count_conversations()

    def _wipe(self):
        from agentx_ai.kit.agent_memory.portability import cluster
        cluster.wipe_cluster(self.stdout.write, self.stderr.write)

    def _make_snapshot(self, path):
        from agentx_ai.kit.agent_memory.portability import cluster
        return cluster.make_snapshot(path, self.stdout.write)

    def _restore_snapshot(self, path):
        from agentx_ai.kit.agent_memory.portability import cluster
        cluster.restore_snapshot(
            path, lambda m: self.stdout.write(self.style.SUCCESS(m)), self.stderr.write
        )

    # -- model override ------------------------------------------------------
    def _build_override(self, model, full):
        """Build the run's pinned Settings — applied via ``pin_memory_settings``
        (covers every live read: jobs, extraction service, recall) in handle()."""
        from agentx_ai.kit.agent_memory.config import get_settings

        update = {"combined_extraction_model": model}
        if full:
            update.update(correction_model=model, contradiction_model=model)
        else:
            update.update(correction_detection_enabled=False,
                          contradiction_detection_enabled=False)
        return get_settings().model_copy(update=update)

    # -- seeding -------------------------------------------------------------
    def _seed(self, cases):
        for case in cases:
            if case.is_procedural:
                self._seed_procedural(case)
            else:
                self._seed_extraction(case)

    def _seed_extraction(self, case):
        from agentx_ai.kit.agent_memory.models import Turn
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
        channel = CHANNEL_PREFIX + case.name
        conv_id = str(uuid4())
        mem = AgentMemory(user_id=EVAL_USER, conversation_id=conv_id, channel=channel)
        for i, content in enumerate(case.turns):
            mem.store_turn(Turn(conversation_id=conv_id, index=i, role="user",
                                content=content, channel=channel))

    def _seed_procedural(self, case):
        """Seed a tool trajectory + Outcome directly (no extraction pipeline).

        Procedural cases don't need consolidation — routing them through
        store_turn would burn LLM calls and risk polluting extraction scores.
        record_invocation only needs the Conversation node to exist, so we
        MERGE it directly, record the tool invocations, then seed the Outcome
        that detect_patterns() gates on.
        """
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        from agentx_ai.kit.agent_memory.memory.procedural import ProceduralMemory
        channel = CHANNEL_PREFIX + case.name
        conv_id = str(uuid4())
        with Neo4jConnection.session() as s:
            s.run("MERGE (c:Conversation {id:$id}) SET c.channel = $ch",
                  id=conv_id, ch=channel).consume()
        proc = ProceduralMemory()
        for i, tool in enumerate(case.tool_sequence):
            proc.record_invocation(
                conversation_id=conv_id, turn_id=None, tool_name=tool,
                tool_input={}, tool_output="ok", success=True, latency_ms=10,
                channel=channel, turn_index=i,
            )
        with Neo4jConnection.session() as s:
            s.run(
                """
                MATCH (c:Conversation {id:$id})
                CREATE (o:Outcome {success:$success, task_type:$task_type, channel:$ch})
                MERGE (c)-[:RESULTED_IN]->(o)
                """,
                id=conv_id, success=case.outcome_success,
                task_type=case.task_type, ch=channel,
            ).consume()

    # -- scoring -------------------------------------------------------------
    @staticmethod
    def _stored(channel):
        """Structured channel contents: entities carry aliases (dedup-quality
        assertions need them), facts carry provenance, plus RELATES_TO pairs."""
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            facts = [{"claim": r["c"], "source_turn_id": r["st"]} for r in s.run(
                "MATCH (f:Fact {channel:$ch}) RETURN f.claim AS c, "
                "f.source_turn_id AS st", ch=channel)]
            ents = [{"id": r["id"], "name": r["n"], "aliases": r["a"] or []}
                    for r in s.run(
                        "MATCH (e:Entity {channel:$ch}) RETURN e.id AS id, "
                        "e.name AS n, coalesce(e.aliases, []) AS a", ch=channel)]
            rels = [(r["src"], r["t"], r["tgt"]) for r in s.run(
                "MATCH (a:Entity {channel:$ch})-[r:RELATES_TO]->(b:Entity {channel:$ch}) "
                "RETURN a.name AS src, r.type AS t, b.name AS tgt", ch=channel)]
        return {"facts": facts, "entities": ents, "relationships": rels}

    @staticmethod
    def _entity_ids_matching(entities, phrasings):
        """Distinct entity ids whose name OR any alias contains any phrasing."""
        matched = set()
        needles = [p.lower() for p in phrasings]
        for ent in entities:
            haystacks = [ent["name"], *ent["aliases"]]
            if any(n in h.lower() for h in haystacks for n in needles):
                matched.add(ent["id"])
        return matched

    def _score_case(self, case):
        stored = self._stored(CHANNEL_PREFIX + case.name)
        facts = [f["claim"] for f in stored["facts"]]
        ents = [e["name"] for e in stored["entities"]]
        fact_blob = " || ".join(facts).lower()
        # Aliases count toward entity expectations (a dedup merge may demote an
        # expected surface form to an alias — still a correct store).
        ent_blob = " || ".join(
            n for e in stored["entities"] for n in [e["name"], *e["aliases"]]
        ).lower()

        if not case.relevant:
            ok = not facts and not ents
            return {"status": "PASS" if ok else "FAIL", "facts": facts, "entities": ents,
                    "detail": "nothing stored" if ok else "expected nothing, got data"}

        fact_hits = sum(1 for sub in case.expect_facts if sub.lower() in fact_blob)
        ent_hits = sum(1 for sub in case.expect_entities if sub.lower() in ent_blob)
        forbidden = [sub for sub in case.forbid_facts if sub.lower() in fact_blob]
        total = len(case.expect_facts) + len(case.expect_entities)
        hits = fact_hits + ent_hits
        extra = []

        # -- §2.10 Slice 1 assertions: dedup, cardinality, edges, provenance --
        for phrasings in case.expect_single_entity:
            total += 1
            matched = self._entity_ids_matching(stored["entities"], phrasings)
            if len(matched) == 1:
                hits += 1
            else:
                extra.append(f"single-entity[{phrasings[0]}]: {len(matched)} nodes (want 1)")
        if case.expect_entity_count_max is not None:
            total += 1
            if len(stored["entities"]) <= case.expect_entity_count_max:
                hits += 1
            else:
                extra.append(f"entity-count {len(stored['entities'])} > "
                             f"max {case.expect_entity_count_max}")
        for pair in case.expect_relationship_pairs:
            total += 1
            a_ids = self._entity_ids_matching(stored["entities"], [pair[0]])
            b_ids = self._entity_ids_matching(stored["entities"], [pair[1]])
            a_names = {e["name"] for e in stored["entities"] if e["id"] in a_ids}
            b_names = {e["name"] for e in stored["entities"] if e["id"] in b_ids}
            found = any(
                (src in a_names and tgt in b_names) or (src in b_names and tgt in a_names)
                for src, _t, tgt in stored["relationships"]
            )
            if found:
                hits += 1
            else:
                extra.append(f"relationship {pair[0]}<->{pair[1]}: missing")
        if case.expect_distinct_entities:
            total += 1
            groups = [self._entity_ids_matching(stored["entities"], g)
                      for g in case.expect_distinct_entities]
            all_nonempty = all(groups)
            disjoint = all(
                groups[i].isdisjoint(groups[j])
                for i in range(len(groups)) for j in range(i + 1, len(groups))
            )
            if all_nonempty and disjoint:
                hits += 1
            elif not all_nonempty:
                missing = [case.expect_distinct_entities[i][0]
                           for i, g in enumerate(groups) if not g]
                extra.append(f"distinct-entities: missing {missing}")
            else:
                extra.append("distinct-entities: over-merged (groups share a node)")
        if case.expect_fact_provenance:
            total += 1
            if stored["facts"] and all(f["source_turn_id"] for f in stored["facts"]):
                hits += 1
            else:
                lacking = sum(1 for f in stored["facts"] if not f["source_turn_id"])
                extra.append(f"provenance: {lacking}/{len(stored['facts'])} "
                             "facts lack source_turn_id")

        if forbidden:
            status = "FAIL"
        elif hits == total and (facts or ents):
            status = "PASS"
        elif hits > 0:
            status = "PARTIAL"
        else:
            status = "FAIL"
        detail = f"facts {fact_hits}/{len(case.expect_facts)}, entities {ent_hits}/{len(case.expect_entities)}"
        if extra:
            detail += " — " + "; ".join(extra)
        if forbidden:
            detail += f" — FORBIDDEN matched: {forbidden}"
        return {"status": status, "facts": facts, "entities": ents, "detail": detail}

    @staticmethod
    def _learned_strategy(channel):
        """Strategy learned for the case's conversation, via SUCCEEDED_IN.

        detect_patterns() doesn't tag strategies with a channel, so match
        through the edge back to the eval Conversation (which IS channel-tagged)
        rather than by channel/context_pattern.
        """
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            rec = s.run(
                """
                MATCH (c:Conversation {channel:$ch})<-[:SUCCEEDED_IN]-(s:Strategy)
                RETURN s.tool_sequence AS tools, s.description AS d LIMIT 1
                """,
                ch=channel,
            ).single()
        if not rec:
            return None
        return {"tools": rec["tools"] or [], "description": rec["d"] or ""}

    def _score_procedural_case(self, case):
        strat = self._learned_strategy(CHANNEL_PREFIX + case.name)

        # Negative case: the success-gate must have learned nothing.
        if not case.outcome_success:
            ok = strat is None
            return {"status": "PASS" if ok else "FAIL", "facts": [], "entities": [],
                    "detail": "no strategy learned (correct)" if ok
                              else f"expected none, learned tools {strat['tools']}"}

        if strat is None:
            return {"status": "FAIL", "facts": [], "entities": [],
                    "detail": "no strategy learned"}

        learned = {t.lower() for t in strat["tools"]}
        expected = case.expected_tools
        hits = [t for t in expected if t.lower() in learned]
        if len(hits) == len(expected):
            status = "PASS"
        elif hits:
            status = "PARTIAL"
        else:
            status = "FAIL"
        # Reuse the facts/entities slots so the existing table renderer is unchanged.
        return {"status": status, "facts": [strat["description"]], "entities": strat["tools"],
                "detail": f"strategy tools {len(hits)}/{len(expected)}"}

    def _score(self, case):
        return self._score_procedural_case(case) if case.is_procedural else self._score_case(case)

    # -- cleanup -------------------------------------------------------------
    def _cleanup(self, cases):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection, get_postgres_session
        from sqlalchemy import text
        # Tools created by procedural cases are global (no channel) — collect
        # their names so we can delete them explicitly.
        eval_tools = sorted({t for c in cases if c.is_procedural for t in c.tool_sequence})
        with Neo4jConnection.session() as s:
            # Strategies aren't channel-tagged — delete via the SUCCEEDED_IN edge
            # back to the eval conversations BEFORE those conversations are gone.
            s.run("""
                MATCH (c:Conversation) WHERE c.channel STARTS WITH $p
                MATCH (c)<-[:SUCCEEDED_IN]-(s:Strategy) DETACH DELETE s
            """, p=CHANNEL_PREFIX).consume()
            s.run("MATCH (n) WHERE n.channel STARTS WITH $p DETACH DELETE n", p=CHANNEL_PREFIX).consume()
            s.run("MATCH (c:Conversation) WHERE c.channel STARTS WITH $p DETACH DELETE c", p=CHANNEL_PREFIX).consume()
            s.run("MATCH (u:User {id:$uid}) DETACH DELETE u", uid=EVAL_USER).consume()
            if eval_tools:
                s.run("MATCH (t:Tool) WHERE t.name IN $names DETACH DELETE t", names=eval_tools).consume()
        try:
            with get_postgres_session() as ses:
                for tbl in ("conversation_logs", "tool_invocations"):
                    ses.execute(text(f"DELETE FROM {tbl} WHERE channel LIKE :p"),
                                {"p": CHANNEL_PREFIX + "%"})
        except Exception:
            pass

    # -- main ----------------------------------------------------------------
    def handle(self, *args, **opts):
        self._load_dotenv()

        # Recovery mode: just restore a prior snapshot and exit (no model/cases needed).
        if opts["restore"]:
            if not Path(opts["restore"]).exists():
                raise CommandError(f"Snapshot file not found: {opts['restore']}")
            self.stdout.write(self.style.WARNING("Restoring snapshot (wipes current memory first)..."))
            self._restore_snapshot(opts["restore"])
            return

        cases = CASES
        if opts["only"]:
            wanted = {n.strip() for n in opts["only"].split(",")}
            cases = [c for c in CASES if c.name in wanted]
            if not cases:
                raise CommandError(f"No cases match --only={opts['only']!r}")

        # Resolve the extraction model.
        from agentx_ai.kit.agent_memory.config import get_settings, pin_memory_settings
        model = opts["model"] or get_settings().combined_extraction_model

        # Validate the provider BEFORE touching data — a snapshot run must not
        # wipe the cluster and then bail because the model is unusable.
        override = self._build_override(model, opts["full"])
        from agentx_ai.providers.registry import reset_registry, get_registry
        reset_registry()
        try:
            get_registry().get_provider_for_model(model)
        except Exception as e:
            raise CommandError(
                f"Extraction model {model!r} is not usable: {e}. Configure the "
                "provider (e.g. set OPENROUTER_API_KEY) or pass --model provider:model_id."
            ) from e

        # Sterility gate — consolidation is global. --snapshot satisfies it
        # non-destructively (snapshot → wipe → run → restore).
        existing = self._count_conversations()
        if existing and not (opts["wipe"] or opts["snapshot"]):
            raise CommandError(
                f"Memory cluster is not sterile ({existing} conversations present). "
                "Consolidation runs GLOBALLY, so this eval must start from an empty "
                "cluster. Re-run with --snapshot to back up + restore your data around "
                "the run (recommended), or --wipe to DESTROY all memory data first — "
                "dev instances only, never against real data."
            )

        # Snapshot the whole cluster before wiping so we can restore it afterward.
        snapshot_path = None
        if opts["snapshot"]:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            snapshot_dir = Path(__file__).resolve().parents[4] / "data" / "eval_snapshots"
            snapshot_path = self._make_snapshot(snapshot_dir / f"{ts}.json")
        if snapshot_path or opts["wipe"]:
            self.stdout.write(self.style.WARNING("Wiping all memory data..."))
            self._wipe()

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nConsolidation eval — model={model}  cases={len(cases)}  "
            f"full={'yes' if opts['full'] else 'no'}\n"))

        # Run the eval, then ALWAYS restore the snapshot (or clean up) — even if
        # seeding/consolidation/scoring raises — so a snapshot run can't leave the
        # cluster wiped. The pin unwinds with the block (it used to leak — an
        # in-process call_command left the override applied forever).
        try:
            with pin_memory_settings(override):
                self._run_eval(cases, model, opts)
        finally:
            if snapshot_path:
                self._restore_snapshot(snapshot_path)
                self.stdout.write(self.style.WARNING(
                    f"\nsnapshot kept for recovery: {snapshot_path}"))
            elif opts["keep"]:
                self.stdout.write(self.style.WARNING(
                    f"\n--keep set: eval data left under channels '{CHANNEL_PREFIX}*'"))
            else:
                self._cleanup(cases)
                self.stdout.write("\ncleaned up seeded eval data")

    def _run_eval(self, cases, model, opts):
        """Seed → consolidate (extraction) → detect patterns (procedural) → score → persist."""
        self._seed(cases)
        from agentx_ai.kit.agent_memory.consolidation import jobs
        # The sweep's discovery caps at 10 conversations per run (production
        # cadence); the suite seeds one conversation per case, so DRAIN the
        # queue — otherwise cases beyond the first 10 score as empty stores.
        result = None
        for _ in range(10):
            sweep = asyncio.run(jobs.consolidate_episodic_to_semantic())
            if result is None:
                result = sweep
            else:
                for key in ("items_processed", "entities", "facts", "relationships"):
                    result[key] += sweep[key]
                result["errors"].extend(sweep["errors"])
                for key in ("extraction_calls", "total_tokens_used"):
                    result["metrics"][key] += sweep["metrics"][key]
            if sweep["items_processed"] == 0:
                break
        assert result is not None  # loop body runs at least once
        patterns_extracted = 0
        if any(c.is_procedural for c in cases):
            patterns_extracted = jobs.detect_patterns().get("items_processed", 0)

        passed = partial = failed = 0
        by_kind = {"extraction": {"pass": 0, "partial": 0, "fail": 0},
                   "procedural": {"pass": 0, "partial": 0, "fail": 0}}
        case_results = []
        self.stdout.write("=" * 78)
        for case in sorted(cases, key=lambda c: (c.is_procedural, c.level, c.name)):
            r = self._score(case)
            passed += r["status"] == "PASS"
            partial += r["status"] == "PARTIAL"
            failed += r["status"] == "FAIL"
            by_kind[case.kind][r["status"].lower()] += 1
            case_results.append({"name": case.name, "level": case.level, "kind": case.kind,
                                 "status": r["status"], "detail": r["detail"]})
            style = {"PASS": self.style.SUCCESS, "PARTIAL": self.style.WARNING,
                     "FAIL": self.style.ERROR}[r["status"]]
            tag = "P" if case.is_procedural else "L"
            self.stdout.write(
                f"{tag}{case.level} {case.name:<24} {style(r['status']):<8} {r['detail']}")
            self.stdout.write(f"     note: {case.note}")
            if r["facts"]:
                for c in r["facts"]:
                    self.stdout.write(f"       {'strategy' if case.is_procedural else 'fact':<8}: {c}")
            if r["entities"]:
                label = "tools" if case.is_procedural else "entities"
                self.stdout.write(f"       {label}: {', '.join(r['entities'])}")
            if not r["facts"] and not r["entities"]:
                self.stdout.write("       (nothing stored)")
        self.stdout.write("=" * 78)

        m = result["metrics"]
        self.stdout.write(
            f"\nResult: {self.style.SUCCESS(str(passed) + ' PASS')}, "
            f"{self.style.WARNING(str(partial) + ' PARTIAL')}, "
            f"{self.style.ERROR(str(failed) + ' FAIL')}  "
            f"of {len(cases)} cases")
        self.stdout.write(
            f"LLM calls={m['extraction_calls']}  tokens={m['total_tokens_used']}  "
            f"stored=[{result['entities']}e, {result['facts']}f]  "
            f"strategies={patterns_extracted}  errors={len(result['errors'])}")
        if result["errors"]:
            for err in result["errors"][:5]:
                self.stderr.write(f"  error: {err}")

        if opts["save"]:
            self._persist_run(opts, model, cases, case_results,
                              {"pass": passed, "partial": partial, "fail": failed,
                               "total": len(cases), "by_kind": by_kind},
                              m, patterns_extracted)

    # -- persistence ---------------------------------------------------------
    def _persist_run(self, opts, model, cases, case_results, summary, metrics, patterns_extracted):
        """Write a JSON run file + append a one-line index entry for comparison."""
        out_dir = Path(opts["output_dir"]) if opts["output_dir"] else (
            Path(__file__).resolve().parents[4] / "data" / "eval_runs")
        out_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:4]}"
        slug = model.replace("/", "-").replace(":", "-")
        path = out_dir / f"{run_id}_{slug}_{'full' if opts['full'] else 'quick'}.json"

        payload = {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": model,
            "flags": {"full": opts["full"], "only": opts["only"]},
            "cases": case_results,
            "summary": summary,
            "metrics": metrics,
            "procedural": {"patterns_extracted": patterns_extracted},
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

        ext, proc = summary["by_kind"]["extraction"], summary["by_kind"]["procedural"]
        ext_total = sum(ext.values())
        proc_total = sum(proc.values())
        index_line = {
            "run_id": run_id, "timestamp": payload["timestamp"], "model": model,
            "extraction_pass": ext["pass"], "extraction_total": ext_total,
            "procedural_pass": proc["pass"], "procedural_total": proc_total,
            "tokens": metrics.get("total_tokens_used", 0),
        }
        with (out_dir / "index.jsonl").open("a") as fh:
            fh.write(json.dumps(index_line) + "\n")

        self.stdout.write(f"\nsaved run → {path}")

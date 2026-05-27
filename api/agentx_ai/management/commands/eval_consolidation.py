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
run against a STERILE dev instance. Pass --wipe to clear ALL memory data first.
NEVER run against a cluster holding real data. (A future memory export/import —
Todo.md backlog — would let us snapshot/restore instead of wiping.)

Usage:
    python manage.py eval_consolidation --wipe
    python manage.py eval_consolidation --model openrouter:minimax/minimax-m2.7 --wipe
    python manage.py eval_consolidation --wipe --full     # also exercise correction + contradiction stages
    python manage.py eval_consolidation --wipe --keep     # leave seeded data for inspection
    python manage.py eval_consolidation --wipe --only occupation_location,temporal_job_change
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError

EVAL_USER = "_eval_user"
CHANNEL_PREFIX = "_eval_"


@dataclass
class EvalCase:
    """A graded consolidation eval case.

    `expect_facts` / `expect_entities` are case-insensitive substrings expected
    to appear in some stored claim / entity name. `forbid_facts` substrings must
    NOT appear in any claim (e.g. the positive form of a negated statement).
    `relevant=False` asserts the turns are pure noise and nothing should store.
    """
    name: str
    level: int  # 1 (simple) .. 5 (complex)
    turns: list[str]
    note: str = ""
    relevant: bool = True
    expect_facts: list[str] = field(default_factory=list)
    expect_entities: list[str] = field(default_factory=list)
    forbid_facts: list[str] = field(default_factory=list)


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
            help="DESTROY all memory data first (required unless the instance is already sterile).",
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

    # -- sterility / wipe ----------------------------------------------------
    def _count_conversations(self):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            rec = s.run("MATCH (c:Conversation) RETURN count(c) AS n").single()
        return rec["n"] if rec else 0

    def _wipe(self):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection, get_postgres_session
        from sqlalchemy import text
        with Neo4jConnection.session() as s:
            rec = s.run("MATCH (n) DETACH DELETE n RETURN count(n) AS n").single()
            neo = rec["n"] if rec else 0
        pg = 0
        try:
            with get_postgres_session() as ses:
                for tbl in ("conversation_logs", "memory_audit_log"):
                    pg += ses.execute(text(f"DELETE FROM {tbl}")).rowcount
        except Exception as e:
            self.stderr.write(f"  postgres wipe note: {e}")
        self.stdout.write(f"  wiped Neo4j nodes={neo}, postgres rows={pg}")

    # -- model override ------------------------------------------------------
    def _pin_model(self, model, full):
        from agentx_ai.kit.agent_memory.config import get_settings
        from agentx_ai.kit.agent_memory.extraction.service import get_extraction_service
        from agentx_ai.kit.agent_memory.consolidation import jobs

        update = {"combined_extraction_model": model}
        if full:
            update.update(correction_model=model, contradiction_model=model)
        else:
            update.update(correction_detection_enabled=False,
                          contradiction_detection_enabled=False)
        override = get_settings().model_copy(update=update)
        # Pin everywhere consolidation reads settings so the get_settings() TTL
        # cache cannot revert the model mid-run.
        jobs.settings = override
        get_extraction_service()._settings = override

    # -- seeding -------------------------------------------------------------
    def _seed(self, cases):
        from agentx_ai.kit.agent_memory.models import Turn
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
        for case in cases:
            channel = CHANNEL_PREFIX + case.name
            conv_id = str(uuid4())
            mem = AgentMemory(user_id=EVAL_USER, conversation_id=conv_id, channel=channel)
            for i, content in enumerate(case.turns):
                mem.store_turn(Turn(conversation_id=conv_id, index=i, role="user",
                                    content=content, channel=channel))

    # -- scoring -------------------------------------------------------------
    @staticmethod
    def _stored(channel):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            facts = [r["c"] for r in s.run(
                "MATCH (f:Fact {channel:$ch}) RETURN f.claim AS c", ch=channel)]
            ents = [r["n"] for r in s.run(
                "MATCH (e:Entity {channel:$ch}) RETURN e.name AS n", ch=channel)]
        return facts, ents

    def _score_case(self, case):
        facts, ents = self._stored(CHANNEL_PREFIX + case.name)
        fact_blob = " || ".join(facts).lower()
        ent_blob = " || ".join(ents).lower()

        if not case.relevant:
            ok = not facts and not ents
            return {"status": "PASS" if ok else "FAIL", "facts": facts, "entities": ents,
                    "detail": "nothing stored" if ok else "expected nothing, got data"}

        fact_hits = sum(1 for sub in case.expect_facts if sub.lower() in fact_blob)
        ent_hits = sum(1 for sub in case.expect_entities if sub.lower() in ent_blob)
        forbidden = [sub for sub in case.forbid_facts if sub.lower() in fact_blob]
        total = len(case.expect_facts) + len(case.expect_entities)
        hits = fact_hits + ent_hits

        if forbidden:
            status = "FAIL"
        elif hits == total and (facts or ents):
            status = "PASS"
        elif hits > 0:
            status = "PARTIAL"
        else:
            status = "FAIL"
        detail = f"facts {fact_hits}/{len(case.expect_facts)}, entities {ent_hits}/{len(case.expect_entities)}"
        if forbidden:
            detail += f" — FORBIDDEN matched: {forbidden}"
        return {"status": status, "facts": facts, "entities": ents, "detail": detail}

    # -- cleanup -------------------------------------------------------------
    def _cleanup(self):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection, get_postgres_session
        from sqlalchemy import text
        with Neo4jConnection.session() as s:
            s.run("MATCH (n) WHERE n.channel STARTS WITH $p DETACH DELETE n", p=CHANNEL_PREFIX).consume()
            s.run("MATCH (c:Conversation) WHERE c.channel STARTS WITH $p DETACH DELETE c", p=CHANNEL_PREFIX).consume()
            s.run("MATCH (u:User {id:$uid}) DETACH DELETE u", uid=EVAL_USER).consume()
        try:
            with get_postgres_session() as ses:
                ses.execute(text("DELETE FROM conversation_logs WHERE channel LIKE :p"),
                            {"p": CHANNEL_PREFIX + "%"})
        except Exception:
            pass

    # -- main ----------------------------------------------------------------
    def handle(self, *args, **opts):
        self._load_dotenv()

        cases = CASES
        if opts["only"]:
            wanted = {n.strip() for n in opts["only"].split(",")}
            cases = [c for c in CASES if c.name in wanted]
            if not cases:
                raise CommandError(f"No cases match --only={opts['only']!r}")

        # Resolve the extraction model.
        from agentx_ai.kit.agent_memory.config import get_settings
        model = opts["model"] or get_settings().combined_extraction_model

        # Sterility gate — consolidation is global.
        existing = self._count_conversations()
        if existing and not opts["wipe"]:
            raise CommandError(
                f"Memory cluster is not sterile ({existing} conversations present). "
                "Consolidation runs GLOBALLY, so this eval must start from an empty "
                "cluster. Re-run with --wipe to DESTROY all memory data first — dev "
                "instances only, never against real data."
            )
        if opts["wipe"]:
            self.stdout.write(self.style.WARNING("Wiping all memory data..."))
            self._wipe()

        # Pin the model and (re)build the registry so provider keys resolve.
        self._pin_model(model, opts["full"])
        from agentx_ai.providers.registry import reset_registry, get_registry
        reset_registry()
        try:
            get_registry().get_provider_for_model(model)
        except Exception as e:
            raise CommandError(
                f"Extraction model {model!r} is not usable: {e}. Configure the "
                "provider (e.g. set OPENROUTER_API_KEY) or pass --model provider:model_id."
            )

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nConsolidation eval — model={model}  cases={len(cases)}  "
            f"full={'yes' if opts['full'] else 'no'}\n"))

        # Seed → consolidate → score.
        self._seed(cases)
        from agentx_ai.kit.agent_memory.consolidation import jobs
        result = asyncio.run(jobs.consolidate_episodic_to_semantic())

        passed = partial = failed = 0
        self.stdout.write("=" * 78)
        for case in sorted(cases, key=lambda c: (c.level, c.name)):
            r = self._score_case(case)
            passed += r["status"] == "PASS"
            partial += r["status"] == "PARTIAL"
            failed += r["status"] == "FAIL"
            style = {"PASS": self.style.SUCCESS, "PARTIAL": self.style.WARNING,
                     "FAIL": self.style.ERROR}[r["status"]]
            self.stdout.write(
                f"L{case.level} {case.name:<24} {style(r['status']):<8} {r['detail']}")
            self.stdout.write(f"     note: {case.note}")
            if r["facts"]:
                for c in r["facts"]:
                    self.stdout.write(f"       fact:   {c}")
            if r["entities"]:
                self.stdout.write(f"       entities: {', '.join(r['entities'])}")
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
            f"errors={len(result['errors'])}")
        if result["errors"]:
            for err in result["errors"][:5]:
                self.stderr.write(f"  error: {err}")

        if opts["keep"]:
            self.stdout.write(self.style.WARNING(
                f"\n--keep set: eval data left under channels '{CHANNEL_PREFIX}*'"))
        else:
            self._cleanup()
            self.stdout.write("\ncleaned up seeded eval data")

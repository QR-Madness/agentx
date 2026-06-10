"""
Django management command: drive a scripted multi-agent conversation through the
REAL consolidation pipeline and report where every fact landed.

This is the end-to-end exercise that unit tests can't give: it runs the actual LLM
extraction (the ``subject_agent`` → ``agent_id`` resolution, per-agent self-channel
routing, and ``[:ABOUT]`` agent-entity linking) on a scripted conversation, then prints
the resulting facts per channel so you can see, e.g., that "Mobius, think step-by-step"
homed to ``_self_{mobius}`` and nowhere else.

NON-DESTRUCTIVE: it seeds the scenario under a throwaway ``user_id`` and consolidates
**only that conversation** (`consolidate_episodic_to_semantic(only_conversation_id=…)`),
so the rest of the cluster is never touched. Cleanup deletes just that user (scoped
``WHERE n.user_id = $u``). All three extraction-stage models (combined / correction /
contradiction) are pinned to ``--model`` so a single provider serves the whole run.

Usage:
    python manage.py debug_attribution --scenario directive --agents "Mobius,Jeff"
    python manage.py debug_attribution --scenario mixed --model openrouter:openai/gpt-4o-mini
    python manage.py debug_attribution --scenario cross-agent --keep      # leave data to inspect
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError


# ── Scenarios ───────────────────────────────────────────────────────────────
# Turns reference agents by slot index (0, 1, …); --agents fills those slots with
# real profiles. ``{a0}``/``{a1}`` in text is substituted with the agent's display
# name. ``expect_self`` lists, per agent slot, substrings expected in that agent's
# _self_ channel; ``expect_active`` substrings expected in the active channel — a
# cheap ✅/❌ regression signal. Assistant content is padded past the 100-char
# self-extraction floor.
SCENARIOS: dict[str, dict] = {
    "directive": {
        "agents": 1,
        "turns": [
            {"role": "user",
             "content": "{a0}, from now on always think step-by-step and lay out your "
                        "reasoning in stages before giving me a final answer."},
            {"role": "assistant", "agent": 0,
             "content": "Understood. From now on I'll decompose each problem into explicit "
                        "stages and show my reasoning before the final answer, rather than "
                        "jumping straight to a conclusion."},
        ],
        "expect_self": {0: ["step-by-step"]},
        "expect_active": [],
    },
    "cross-agent": {
        "agents": 2,
        "turns": [
            {"role": "user", "content": "{a1}, how did the SQL tuning task go?"},
            {"role": "assistant", "agent": 1,
             "content": "I handed the SQL query-plan tuning over to {a0}, because {a0} is "
                        "noticeably faster and more accurate at optimizing complex query "
                        "plans than I am. I focused on the schema review instead."},
        ],
        "expect_self": {1: ["SQL"]},
        "expect_active": [],
    },
    "mixed": {
        "agents": 2,
        "turns": [
            {"role": "user",
             "content": "I prefer all measurements in metric units. Also {a0}, please "
                        "always cite your sources when you make a factual claim."},
            {"role": "assistant", "agent": 0,
             "content": "Got it — I'll cite sources for every factual claim from now on, "
                        "and I'll keep measurements in metric units as you prefer."},
            {"role": "user",
             "content": "{a1}, you tend to be far too verbose — keep your answers tight."},
            {"role": "assistant", "agent": 1,
             "content": "Understood, I'll be much more concise going forward and cut the "
                        "preamble, giving you tighter answers without the extra padding."},
        ],
        "expect_self": {0: ["cite"], 1: ["concise"]},
        "expect_active": ["metric"],
    },
}


class Command(BaseCommand):
    help = (
        "Run a scripted multi-agent conversation through real consolidation (scoped to "
        "just that conversation) and report per-channel attribution. Non-destructive."
    )

    def add_arguments(self, parser):
        parser.add_argument("--scenario", default="directive",
                            choices=sorted(SCENARIOS.keys()),
                            help="Which built-in scenario to run.")
        parser.add_argument("--agents", default="",
                            help="Comma-separated profile names to fill agent slots "
                                 "(e.g. 'Mobius,Jeff'). Defaults to the first profiles.")
        parser.add_argument("--model", default=None,
                            help="Extraction model as provider:model_id (pins all stages; "
                                 "default: configured combined_extraction_model).")
        parser.add_argument("--user-id", default=None,
                            help="User id to seed under (default: throwaway debug-<uuid>).")
        parser.add_argument("--keep", action="store_true",
                            help="Skip cleanup; leave seeded data to inspect.")
        parser.add_argument("--json", action="store_true",
                            help="Emit the report as JSON (for tooling).")

    # ── env / provider ──────────────────────────────────────────────────────
    @staticmethod
    def _load_dotenv():
        """Best-effort: load repo .env so provider keys resolve (mirrors eval)."""
        env_path = Path(__file__).resolve().parents[4] / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    def _pin_and_validate_model(self, model: str) -> None:
        """Pin the model for ALL extraction stages (combined + correction +
        contradiction) so one provider serves the run, and fail fast if it's
        unusable — before any data is written."""
        from agentx_ai.kit.agent_memory.config import get_settings
        from agentx_ai.kit.agent_memory.extraction.service import get_extraction_service
        from agentx_ai.kit.agent_memory.consolidation import jobs
        from agentx_ai.providers.registry import reset_registry, get_registry

        override = get_settings().model_copy(update={
            "combined_extraction_model": model,
            "correction_model": model,
            "contradiction_model": model,
        })
        jobs.settings = override
        get_extraction_service()._settings = override

        reset_registry()
        try:
            get_registry().get_provider_for_model(model)
        except Exception as e:
            raise CommandError(
                f"Extraction model {model!r} is not usable: {e}. Configure the provider "
                "(e.g. set OPENROUTER_API_KEY / run LM Studio) or pass --model provider:model_id."
            ) from e

    # ── agents / seeding ──────────────────────────────────────────────────────
    def _resolve_agents(self, names: str, needed: int):
        """Return ``needed`` profiles, by the given names or the first available."""
        from agentx_ai.agent.profiles import get_profile_manager
        pm = get_profile_manager()
        chosen = []
        for nm in [n.strip() for n in names.split(",") if n.strip()]:
            p = pm.get_profile_by_name(nm) or pm.get_profile_by_agent_id(nm)
            if p is None:
                raise CommandError(f"No profile named {nm!r}")
            chosen.append(p)
        if len(chosen) < needed:
            for p in pm.list_profiles():
                if p not in chosen:
                    chosen.append(p)
                if len(chosen) >= needed:
                    break
        if len(chosen) < needed:
            raise CommandError(f"Scenario needs {needed} agents; only {len(chosen)} available.")
        return chosen[:needed]

    def _seed(self, scenario: dict, agents, user_id: str, channel: str) -> str:
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
        from agentx_ai.kit.agent_memory.models import Turn
        conv_id = str(uuid4())
        mem = AgentMemory(user_id=user_id, conversation_id=conv_id, channel=channel)
        names = {f"a{i}": a.name for i, a in enumerate(agents)}
        for i, turn in enumerate(scenario["turns"]):
            content = turn["content"].format(**names)
            kwargs = {}
            if turn["role"] == "assistant":
                a = agents[turn["agent"]]
                kwargs = {"agent_id": a.agent_id, "metadata": {"agent_name": a.name}}
            mem.store_turn(Turn(conversation_id=conv_id, index=i, role=turn["role"],
                                content=content, channel=channel, **kwargs))
        return conv_id

    def _wipe_user(self, user_id: str) -> None:
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            s.run("MATCH (n) WHERE n.user_id = $u DETACH DELETE n", u=user_id)

    # ── reporting ─────────────────────────────────────────────────────────────
    def _facts_in(self, user_id: str, channel: str):
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as s:
            rows = s.run(
                """
                MATCH (f:Fact {user_id: $u, channel: $c})
                WHERE f.superseded_at IS NULL
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                RETURN f.claim AS claim, collect(e.name) AS about
                ORDER BY claim
                """, u=user_id, c=channel).data()
        return [{"claim": r["claim"], "about": [a for a in r["about"] if a]} for r in rows]

    def handle(self, *args, **opts):
        self._load_dotenv()
        scenario = SCENARIOS[opts["scenario"]]
        agents = self._resolve_agents(opts["agents"], scenario["agents"])
        user_id = opts["user_id"] or f"debug-{uuid4().hex[:8]}"
        channel = "_default"

        from agentx_ai.kit.agent_memory.config import get_settings
        model = opts["model"] or get_settings().combined_extraction_model
        self._pin_and_validate_model(model)  # fail fast before touching data

        self.stdout.write(self.style.MIGRATE_HEADING(  # type: ignore[attr-defined]
            f"\ndebug_attribution — scenario={opts['scenario']} model={model} "
            f"agents={[a.name for a in agents]} user={user_id}\n"))

        try:
            conv_id = self._seed(scenario, agents, user_id, channel)
            from agentx_ai.kit.agent_memory.consolidation import jobs
            # Scoped to just this conversation — the rest of the cluster is untouched.
            asyncio.run(jobs.consolidate_episodic_to_semantic(only_conversation_id=conv_id))

            report = self._build_report(scenario, agents, user_id, channel)
            if opts["json"]:
                self.stdout.write(json.dumps(report, indent=2))
            else:
                self._print_report(report)
        finally:
            if opts["keep"]:
                self.stdout.write(self.style.WARNING(f"\n--keep: data left under user {user_id}"))  # type: ignore[attr-defined]
            else:
                self._wipe_user(user_id)
                self.stdout.write(f"\ncleaned up debug user {user_id}")

    def _build_report(self, scenario: dict, agents, user_id: str, channel: str) -> dict:
        labels = {f"_self_{a.agent_id}": a.name for a in agents}
        labels[channel] = "(active)"
        labels["_global"] = "(global)"
        per_channel = {ch: self._facts_in(user_id, ch) for ch in labels}

        checks = []
        for slot, subs in scenario.get("expect_self", {}).items():
            ch = f"_self_{agents[slot].agent_id}"
            blob = " ".join(f["claim"].lower() for f in per_channel.get(ch, []))
            for sub in subs:
                checks.append({"where": f"{agents[slot].name} self", "expect": sub,
                               "ok": sub.lower() in blob})
        active_blob = " ".join(f["claim"].lower() for f in per_channel.get(channel, []))
        for sub in scenario.get("expect_active", []):
            checks.append({"where": "active", "expect": sub,
                           "ok": sub.lower() in active_blob})
        return {"labels": labels, "facts": per_channel, "checks": checks}

    def _print_report(self, report: dict) -> None:
        self.stdout.write("=" * 78)
        for ch, label in report["labels"].items():
            facts = report["facts"][ch]
            self.stdout.write(f"\n{ch}  {label}")
            if not facts:
                self.stdout.write("    (none)")
            for f in facts:
                about = f"  →ABOUT {f['about']}" if f["about"] else ""
                self.stdout.write(f"    • {f['claim']}{about}")
        checks = report["checks"]
        if checks:
            self.stdout.write("\n" + "-" * 78)
            for c in checks:
                mark = "✅" if c["ok"] else "❌"
                self.stdout.write(f"  {mark} [{c['where']}] expected substring: {c['expect']!r}")
            n_ok = sum(1 for c in checks if c["ok"])
            self.stdout.write(f"\n  {n_ok}/{len(checks)} expectations met")

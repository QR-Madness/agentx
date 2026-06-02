"""
Django management command to backfill multi-agent attribution onto legacy
self-knowledge facts (Phase 16 multi-agent attribution).

Before agents became first-class, self-extraction stored generic claims like
"Agent forgets the user's name" in each agent's ``_self_{agent_id}`` channel —
the *channel* encoded which agent, but the prose did not, and there was no
``(Fact)-[:ABOUT]->(Entity agent)`` link. With several agents this reads as an
unusable wall of "Agent ...". This command brings the existing graph in line
with the write-path semantics, deterministically (no LLM, no embeddings beyond
the one Agent entity per channel):

  For each agent profile, in its ``_self_{agent_id}`` channel:
    1. Ensure the first-class Agent entity exists (id ``agent:{agent_id}``,
       name = the profile's display name, ``properties.agent_id`` = the key).
    2. Rewrite facts whose claim starts with a generic "Agent"/"The agent" so
       they lead with the agent's name ("Mobius forgets the user's name"),
       recomputing ``claim_hash``.
    3. Link each such fact to the Agent entity via ``[:ABOUT]`` (MERGE, so
       re-runs are no-ops).

The channel is the source of truth for *which* agent, so the rename is fully
deterministic. Idempotent: a claim already leading with the name is skipped,
and the ABOUT MERGE never duplicates.

Usage:
    python manage.py backfill_agent_attribution --dry-run   # default
    python manage.py backfill_agent_attribution --apply
    python manage.py backfill_agent_attribution --agent-id bold-cosmic-falcon --apply
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from django.core.management.base import BaseCommand, CommandError


logger = logging.getLogger(__name__)

# Matches a leading generic agent reference: "Agent", "The agent", "the Agent's".
# Captures the trailing possessive so "Agent's" → "Mobius's".
_GENERIC_AGENT_RE = re.compile(r"^\s*(?:the\s+)?agent('s)?\b", re.IGNORECASE)

# Candidate facts per channel (cheap server-side prefilter; the Python _rename
# below is the authoritative check). Claims are stored trimmed.
_FETCH_CANDIDATES = """
MATCH (f:Fact {channel: $channel})
WHERE f.claim IS NOT NULL
  AND (toLower(f.claim) STARTS WITH 'agent'
       OR toLower(f.claim) STARTS WITH 'the agent')
RETURN f.id AS id, f.claim AS claim, f.user_id AS user_id
"""


class Command(BaseCommand):
    help = (
        "Backfill agent name + ABOUT links onto legacy generic 'Agent ...' "
        "self-knowledge facts. Default is dry-run; pass --apply to commit."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Commit changes. Without this flag the command only reports.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Explicit dry-run (the default). Overridden by --apply.",
        )
        parser.add_argument(
            "--agent-id",
            type=str,
            default=None,
            help="Restrict to a single agent_id. Default: all profiles.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print one line per rewritten fact.",
        )

    def handle(self, *args, **opts):
        apply = bool(opts["apply"])
        only_agent = opts.get("agent_id")
        verbose = bool(opts["verbose"])

        self.stdout.write(
            f"backfill_agent_attribution: {'APPLY (committing)' if apply else 'DRY-RUN'}"
        )

        try:
            from agentx_ai.kit.agent_memory.connections import Neo4jConnection
            from agentx_ai.kit.agent_memory.models import Entity, compute_claim_hash
            from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
            from agentx_ai.agent.profiles import get_profile_manager
        except ImportError as e:
            raise CommandError(f"Import failed: {e}") from e

        profiles = get_profile_manager().list_profiles()
        if only_agent:
            profiles = [p for p in profiles if p.agent_id == only_agent]
            if not profiles:
                raise CommandError(f"No profile with agent_id {only_agent!r}")

        driver = Neo4jConnection.get_driver()
        totals = {"facts_renamed": 0, "links_added": 0, "entities_ensured": 0}
        # Cache of (user_id, agent_id) → ensured, so we embed each Agent entity once.
        ensured: set[tuple[str, str]] = set()

        with driver.session() as session:
            for profile in profiles:
                agent_id = profile.agent_id
                name = profile.name
                channel = f"_self_{agent_id}"

                candidates = list(session.run(_FETCH_CANDIDATES, channel=channel))
                if not candidates:
                    continue

                for rec in candidates:
                    fid = rec["id"]
                    claim = rec["claim"]
                    user_id = rec["user_id"] or "default"
                    # Agent entities live in _global, one per (user, agent).
                    ent_id = f"agent:{user_id}:{agent_id}"

                    new_claim = self._rename(claim, name)
                    if new_claim is None:
                        continue  # already named / not a generic-agent claim

                    if verbose:
                        self.stdout.write(f"  [{agent_id}] {claim!r} → {new_claim!r}")

                    # Ensure the Agent entity exists in _global (once per user, embedded).
                    if (user_id, agent_id) not in ensured:
                        ensured.add((user_id, agent_id))
                        if apply:
                            try:
                                AgentMemory(user_id=user_id, channel="_global").upsert_entity(
                                    Entity(
                                        id=ent_id,
                                        name=name,
                                        type="Agent",
                                        description=f"AI agent (id {agent_id})",
                                        properties={"agent_id": agent_id},
                                    )
                                )
                            except Exception as e:  # noqa: BLE001
                                logger.warning(f"ensure agent entity failed ({agent_id}): {e}")
                        totals["entities_ensured"] += 1

                    totals["facts_renamed"] += 1
                    totals["links_added"] += 1
                    if apply:
                        session.run(
                            """
                            MATCH (f:Fact {id: $id})
                            SET f.claim = $claim, f.claim_hash = $hash
                            WITH f
                            MATCH (e:Entity {id: $eid})
                            MERGE (f)-[:ABOUT]->(e)
                            """,
                            id=fid,
                            claim=new_claim,
                            hash=compute_claim_hash(new_claim),
                            eid=ent_id,
                        )

        self.stdout.write(
            "  facts_renamed={facts_renamed} links_added={links_added} "
            "entities_ensured={entities_ensured}".format(**totals)
        )
        if not apply:
            self.stdout.write(self.style.WARNING("Dry-run — nothing written. Re-run with --apply."))  # type: ignore[attr-defined]
        else:
            self.stdout.write(self.style.SUCCESS("Done."))  # type: ignore[attr-defined]

    @staticmethod
    def _rename(claim: str, name: str) -> Optional[str]:
        """Rewrite a leading generic "Agent"/"The agent" to ``name``.

        Returns the new claim, or None if the claim doesn't start with a generic
        agent reference or already leads with the agent's name (idempotent).
        """
        if not claim:
            return None
        if claim.lstrip().lower().startswith(name.lower()):
            return None  # already named
        m = _GENERIC_AGENT_RE.match(claim)
        if not m:
            return None
        possessive = m.group(1) or ""
        replacement = f"{name}{possessive}"
        new_claim = _GENERIC_AGENT_RE.sub(replacement, claim, count=1).strip()
        return new_claim if new_claim != claim.strip() else None

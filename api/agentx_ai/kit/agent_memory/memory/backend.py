"""Which store the memory classes run on (Decisions.md ADR-21, Slice 3).

The memory classes are the only door to the memory store. While the store moves
from Neo4j to PostgreSQL each class has a twin in ``memory/postgres/`` with the
same public methods, and callers build a class through this module instead of
naming one, so the choice is made in one place:

    from agentx_ai.kit.agent_memory.memory.backend import episodic_memory
    episodic = episodic_memory(audit_logger=logger)

``MEMORY_STORE`` picks the store and is read once per process. It is a
development seam, not a supported setting (ADR-22): ``postgres`` is refused
until every class has its twin, because a half-moved store would keep turns in
one database and the facts drawn from them in another. The twins are exercised
on their own until then, by the contract tests that run each class's
behaviour against both stores.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from ..config import get_settings
from .episodic import EpisodicMemory

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger

NEO4J = "neo4j"
POSTGRES = "postgres"

# The memory classes that still have no PostgreSQL twin. Each twin's pull request
# removes its line; the store can be selected once this is empty. A test holds
# the list to the factories below, so it cannot be emptied by hand.
TWINS_MISSING: tuple[str, ...] = (
    "SemanticMemory",
    "ProceduralMemory",
    "GoalMemory",
    "ConsolidationStore",
    "CrossUserStore",
    "PortabilityStore",
    "MemoryOverview",
)


@cache
def memory_store() -> str:
    """The store this process runs on: ``neo4j`` or ``postgres``. Boot-frozen: the
    first read stands until the process ends, whatever the settings do later."""
    value = (get_settings().memory_store or NEO4J).strip().lower()
    if value not in (NEO4J, POSTGRES):
        raise ValueError(f"MEMORY_STORE must be '{NEO4J}' or '{POSTGRES}', not {value!r}")
    if value == POSTGRES and TWINS_MISSING:
        raise RuntimeError(
            "MEMORY_STORE=postgres cannot be used yet: these memory classes have no "
            f"PostgreSQL twin: {', '.join(TWINS_MISSING)} (ADR-21, Slice 3)."
        )
    return value


def episodic_memory(audit_logger: MemoryAuditLogger | None = None) -> EpisodicMemory:
    """The episodic memory class for this process's store."""
    if memory_store() == POSTGRES:
        from .postgres.episodic import PgEpisodicMemory

        return PgEpisodicMemory(audit_logger=audit_logger)
    return EpisodicMemory(audit_logger=audit_logger)

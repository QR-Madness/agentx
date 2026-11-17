"""Cross-user memory operations: the one memory class whose methods span every user.

Every other memory class works within a user (and usually a channel). Admin, eval
and maintenance work needs the whole store: cluster snapshots and wipes, and agent
renames that reach every user's copy of an agent. That work goes through here and
nowhere else. When the store moves to PostgreSQL (Decisions.md ADR-21), this class's
twin is the only code that runs under the separate system role that row-level
security lets past.
"""

import logging
from typing import NamedTuple

from ..connections import Neo4jConnection, get_postgres_session

logger = logging.getLogger(__name__)


class WipeResult(NamedTuple):
    """What `CrossUserStore.wipe_all` removed."""

    graph_nodes: int
    pg_rows: int
    # Set when the PostgreSQL half failed; the graph half has already run.
    pg_error: str | None


class CrossUserStore:
    """Operations across every user. Callers own any safety gating (see `wipe_all`)."""

    def count_conversations(self) -> int:
        """Conversations stored, across all users."""
        with Neo4jConnection.session() as session:
            record = session.run("MATCH (c:Conversation) RETURN count(c) AS n").single()
        return record["n"] if record else 0

    def list_user_ids(self) -> list[str]:
        """Every user id that has memory in the store."""
        with Neo4jConnection.session() as session:
            return [r["uid"] for r in session.run("MATCH (u:User) RETURN u.id AS uid") if r["uid"]]

    def wipe_all(self) -> WipeResult:
        """DESTROY all memory: every graph node, then the PostgreSQL mirror tables
        (`conversation_logs`, `memory_audit_log`). A PostgreSQL failure doesn't undo
        or stop the graph wipe; it comes back as `pg_error` with the rows removed so far.
        """
        from sqlalchemy import text

        with Neo4jConnection.session() as session:
            record = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS n").single()
            graph_nodes = record["n"] if record else 0
        pg_rows = 0
        try:
            with get_postgres_session() as pg:
                for statement in ("DELETE FROM conversation_logs", "DELETE FROM memory_audit_log"):
                    pg_rows += pg.execute(text(statement)).rowcount  # pyright: ignore[reportAttributeAccessIssue]
        except Exception as e:  # noqa: BLE001 — reported to the caller, which decides what to say
            return WipeResult(graph_nodes, pg_rows, str(e))
        return WipeResult(graph_nodes, pg_rows, None)

    def rename_agent_entities(self, agent_id: str, old_name: str | None, new_name: str) -> None:
        """Rename the agent's memory entity in every user's graph, keeping the old name
        as an alias. `agent_id` is the stable key (`agent:{agent_id}` entity ids), so the
        rename is unambiguous; an old name already present as an alias, or equal to the
        new name ignoring case, isn't added again.
        """
        with Neo4jConnection.session() as session:
            session.run(
                """
                MATCH (e:Entity)
                WHERE coalesce(e.type, '') = 'Agent'
                  AND e.id ENDS WITH (':' + $agent_id)
                SET e.aliases = CASE
                        WHEN $old_name IS NULL
                             OR toLower($old_name) = toLower($new_name)
                             OR toLower($old_name) IN [a IN coalesce(e.aliases, []) | toLower(a)]
                        THEN coalesce(e.aliases, [])
                        ELSE coalesce(e.aliases, []) + [$old_name]
                    END,
                    e.name = $new_name
                """,
                agent_id=agent_id,
                old_name=old_name,
                new_name=new_name,
            )

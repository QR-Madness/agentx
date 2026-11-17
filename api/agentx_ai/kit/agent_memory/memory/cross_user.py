"""Cross-user memory operations: the one memory class whose methods span every user.

Every other memory class works within a user (and usually a channel). Admin, eval
and maintenance work needs the whole store: cluster snapshots and wipes, and agent
renames that reach every user's copy of an agent. That work goes through here and
nowhere else. When the store moves to PostgreSQL (Decisions.md ADR-21), this class's
twin is the only code that runs under the separate system role that row-level
security lets past.
"""

import logging
from typing import Any, NamedTuple

from ..connections import Neo4jConnection, get_postgres_session

logger = logging.getLogger(__name__)


class WipeResult(NamedTuple):
    """What `CrossUserStore.wipe_all` removed."""

    graph_nodes: int
    pg_rows: int
    # Set when the PostgreSQL half failed; the graph half has already run.
    pg_error: str | None


class CrossUserStore:
    """Operations across every user. Callers own any safety gating (see `wipe_all`).

    Some of these back API endpoints that ought to be scoped to the requesting user
    (the channel list, channel existence, channel deletion). They span users today and
    moved here as they are, so the gap sits in one visible place until row-level
    security scopes them (ADR-21).
    """

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

    def graph_size_mb(self) -> float | None:
        """Approximate graph store size in MB: Neo4j's JMX store-size counter, or, if
        that query fails, an estimate of about 500 bytes a node. None if neither works."""
        try:
            with Neo4jConnection.session() as session:
                record = session.run("""
                    CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store sizes')
                    YIELD attributes
                    RETURN attributes.TotalStoreSize.value AS total_bytes
                """).single()
                if record and record["total_bytes"]:
                    return round(record["total_bytes"] / 1024 / 1024, 2)
            return None
        except Exception:  # noqa: BLE001 — JMX is often unavailable; estimate instead
            try:
                with Neo4jConnection.session() as session:
                    record = session.run("MATCH (n) RETURN count(n) as nodes").single()
                    if record:
                        return round(record["nodes"] * 500 / 1024 / 1024, 2)
                return None
            except Exception:  # noqa: BLE001
                return None

    def list_channels(self) -> list[dict[str, Any]]:
        """Every channel any node carries, with its turn, entity, fact, strategy and goal
        counts, ordered by name."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                    // Get all channels from different node types
                    MATCH (n)
                    WHERE n.channel IS NOT NULL
                    WITH DISTINCT n.channel AS channel

                    // Count items per channel per type
                    OPTIONAL MATCH (t:Turn {channel: channel})
                    WITH channel, count(DISTINCT t) AS turn_count

                    OPTIONAL MATCH (e:Entity {channel: channel})
                    WITH channel, turn_count, count(DISTINCT e) AS entity_count

                    OPTIONAL MATCH (f:Fact {channel: channel})
                    WITH channel, turn_count, entity_count, count(DISTINCT f) AS fact_count

                    OPTIONAL MATCH (s:Strategy {channel: channel})
                    WITH channel, turn_count, entity_count, fact_count, count(DISTINCT s) AS strategy_count

                    OPTIONAL MATCH (g:Goal {channel: channel})

                    RETURN channel,
                           turn_count AS turns,
                           entity_count AS entities,
                           fact_count AS facts,
                           strategy_count AS strategies,
                           count(DISTINCT g) AS goals
                    ORDER BY channel
                """)
            return [
                {
                    "channel": record["channel"],
                    "turns": record["turns"],
                    "entities": record["entities"],
                    "facts": record["facts"],
                    "strategies": record["strategies"],
                    "goals": record["goals"],
                }
                for record in result
            ]

    def channel_exists(self, channel: str) -> bool:
        """Whether any node carries `channel`."""
        with Neo4jConnection.session() as session:
            record = session.run("""
                    MATCH (n {channel: $channel})
                    RETURN count(n) AS count
                """, channel=channel).single()
        return bool(record and record["count"] > 0)

    def delete_channel(self, channel: str) -> dict[str, int]:
        """Delete every turn, entity, fact, strategy, goal and conversation in `channel`.
        Returns how many of each went. Procedures in the channel are not touched."""
        deleted = {"turns": 0, "entities": 0, "facts": 0, "strategies": 0, "goals": 0, "conversations": 0}
        with Neo4jConnection.session() as session:
            # Delete turns
            result = session.run("""
                MATCH (t:Turn {channel: $channel})
                WITH t, count(t) AS cnt
                DETACH DELETE t
                RETURN cnt
            """, channel=channel)
            record = result.single()
            if record:
                deleted["turns"] = record["cnt"] or 0

            # Delete entities
            result = session.run("""
                MATCH (e:Entity {channel: $channel})
                WITH e, count(e) AS cnt
                DETACH DELETE e
                RETURN cnt
            """, channel=channel)
            record = result.single()
            if record:
                deleted["entities"] = record["cnt"] or 0

            # Delete facts
            result = session.run("""
                MATCH (f:Fact {channel: $channel})
                WITH f, count(f) AS cnt
                DETACH DELETE f
                RETURN cnt
            """, channel=channel)
            record = result.single()
            if record:
                deleted["facts"] = record["cnt"] or 0

            # Delete strategies
            result = session.run("""
                MATCH (s:Strategy {channel: $channel})
                WITH s, count(s) AS cnt
                DETACH DELETE s
                RETURN cnt
            """, channel=channel)
            record = result.single()
            if record:
                deleted["strategies"] = record["cnt"] or 0

            # Delete goals
            result = session.run("""
                MATCH (g:Goal {channel: $channel})
                WITH g, count(g) AS cnt
                DETACH DELETE g
                RETURN cnt
            """, channel=channel)
            record = result.single()
            if record:
                deleted["goals"] = record["cnt"] or 0

            # Delete conversations
            result = session.run("""
                MATCH (c:Conversation {channel: $channel})
                WITH c, count(c) AS cnt
                DETACH DELETE c
                RETURN cnt
            """, channel=channel)
            record = result.single()
            if record:
                deleted["conversations"] = record["cnt"] or 0
        return deleted

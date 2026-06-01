"""Serialize a user's memory graph + PostgreSQL audit mirror into a MemoryExport.

Nodes are dumped as opaque property dicts (datetimes → ISO) sorted by ``id`` for
deterministic, diffable output. Relationships that aren't already encoded as
foreign-key-like props are captured as helper keys on the owning node (e.g. a
fact's ``entity_ids`` from its ABOUT edges, a goal's ``parent_goal_id`` from
SUBGOAL_OF).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from sqlalchemy import bindparam, text
from typing_extensions import LiteralString

from ..connections import Neo4jConnection, get_postgres_session
from ..query_utils import (
    COMMON_DATETIME_FIELDS,
    CypherFilterBuilder,
    convert_record_datetimes,
)
from .schema import MemoryExport, current_embedder_info

logger = logging.getLogger(__name__)

# Datetime fields beyond the common set that appear on memory nodes/rows.
# (`consolidated` lives on Conversation; without it the raw Neo4j DateTime would
# break JSON serialization of the node dict.)
_DATETIME_FIELDS = COMMON_DATETIME_FIELDS + ["superseded_at", "deadline", "consolidated"]


class MemoryExporter:
    """Reads the memory graph for one user (optionally one channel) into a MemoryExport."""

    def __init__(
        self,
        user_id: str,
        channel: Optional[str] = None,
    ):
        self.user_id = user_id
        # Normalize "_all"/None → no channel filter.
        self.channel = None if channel in (None, "_all") else channel

    # -- helpers ---------------------------------------------------------

    def _channel_inline(self, alias: str) -> str:
        """Inline channel-scope clause for `alias` (empty when exporting all channels)."""
        return CypherFilterBuilder(alias).add_channel_filter(self.channel).build_inline()

    def _node(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Neo4j node property dict for JSON. Exports are text-only:
        the embedding is always dropped (regenerated on import)."""
        d = convert_record_datetimes(dict(raw), fields=_DATETIME_FIELDS)
        d.pop("embedding", None)
        return d

    # -- node collections ------------------------------------------------

    def export(self) -> MemoryExport:
        """Run every collection query and assemble the envelope."""
        export = MemoryExport(
            user_id=self.user_id,
            channel=self.channel,
            embedder=current_embedder_info(),
        )
        with Neo4jConnection.session() as session:
            params = {"user_id": self.user_id, "channel": self.channel}

            export.conversations = [
                self._node(r["c"])
                for r in session.run(cast(LiteralString, f"""
                    MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                    WHERE true {self._channel_inline('c')}
                    RETURN c ORDER BY c.id
                """), **params)
            ]

            for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                      -[:HAS_TURN]->(t:Turn)
                WHERE true {self._channel_inline('t')}
                RETURN t, c.id AS conversation_id ORDER BY t.id
            """), **params):
                node = self._node(r["t"])
                node["conversation_id"] = r["conversation_id"]
                export.turns.append(node)

            export.entities = [
                self._node(r["e"])
                for r in session.run(cast(LiteralString, f"""
                    MATCH (u:User {{id: $user_id}})-[:HAS_ENTITY]->(e:Entity)
                    WHERE true {self._channel_inline('e')}
                    RETURN e ORDER BY e.id
                """), **params)
            ]

            for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_FACT]->(f:Fact)
                WHERE true {self._channel_inline('f')}
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                WITH f, collect(DISTINCT e.id) AS entity_ids
                RETURN f, entity_ids ORDER BY f.id
            """), **params):
                node = self._node(r["f"])
                node["entity_ids"] = [eid for eid in r["entity_ids"] if eid]
                export.facts.append(node)

            for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal)
                WHERE true {self._channel_inline('g')}
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent.id AS parent_goal_id ORDER BY g.id
            """), **params):
                node = self._node(r["g"])
                node["parent_goal_id"] = r["parent_goal_id"]
                export.goals.append(node)

            for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_STRATEGY]->(s:Strategy)
                WHERE true {self._channel_inline('s')}
                OPTIONAL MATCH (s)-[:USES_TOOL]->(tool:Tool)
                OPTIONAL MATCH (s)-[:SUCCEEDED_IN]->(sc:Conversation)
                OPTIONAL MATCH (s)-[:FAILED_IN]->(fc:Conversation)
                WITH s, collect(DISTINCT tool.name) AS tool_names,
                     collect(DISTINCT sc.id) AS succeeded_in,
                     collect(DISTINCT fc.id) AS failed_in
                RETURN s, tool_names, succeeded_in, failed_in ORDER BY s.id
            """), **params):
                node = self._node(r["s"])
                # USES_TOOL edges are authoritative; fall back to the stored prop.
                node["tool_sequence"] = (
                    [t for t in r["tool_names"] if t] or node.get("tool_sequence") or []
                )
                node["succeeded_in"] = [c for c in r["succeeded_in"] if c]
                node["failed_in"] = [c for c in r["failed_in"] if c]
                export.strategies.append(node)

            for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                      -[:USED_TOOL]->(inv:ToolInvocation)
                WHERE true {self._channel_inline('inv')}
                RETURN inv, c.id AS conversation_id
                ORDER BY c.id, inv.turn_index, inv.tool_name
            """), **params):
                node = self._node(r["inv"])
                node["conversation_id"] = r["conversation_id"]
                export.tool_invocations.append(node)

        # -- PostgreSQL audit mirror, scoped to the exported conversations ---
        conv_ids = [c["id"] for c in export.conversations]
        if conv_ids:
            export.pg_conversation_logs = self._export_pg_logs(conv_ids)
            export.pg_tool_invocations = self._export_pg_tools(conv_ids)

        logger.info("Memory export for user=%s channel=%s: %s",
                    self.user_id, self.channel, export.counts())
        return export

    # -- PostgreSQL ------------------------------------------------------

    def _pg_rows(self, sql: str, conv_ids: List[str]) -> List[Dict[str, Any]]:
        stmt = text(sql).bindparams(bindparam("ids", expanding=True))
        with get_postgres_session() as session:
            result = session.execute(stmt, {"ids": conv_ids})
            rows = [dict(row._mapping) for row in result]
        out: List[Dict[str, Any]] = []
        for row in rows:
            row["conversation_id"] = str(row["conversation_id"])
            if row.get("timestamp") is not None and hasattr(row["timestamp"], "isoformat"):
                row["timestamp"] = row["timestamp"].isoformat()
            out.append(row)
        return out

    def _export_pg_logs(self, conv_ids: List[str]) -> List[Dict[str, Any]]:
        # Text-only: the embedding column is omitted (regenerated on import).
        return self._pg_rows("""
            SELECT conversation_id::text AS conversation_id, turn_index, timestamp,
                   role, content, content_hash, token_count, model, channel, agent_id,
                   metadata
            FROM conversation_logs
            WHERE conversation_id::text IN :ids
            ORDER BY conversation_id, turn_index
        """, conv_ids)

    def _export_pg_tools(self, conv_ids: List[str]) -> List[Dict[str, Any]]:
        return self._pg_rows("""
            SELECT conversation_id::text AS conversation_id, turn_index, timestamp,
                   tool_name, tool_input, tool_output, success, latency_ms,
                   error_message, channel
            FROM tool_invocations
            WHERE conversation_id::text IN :ids
            ORDER BY conversation_id, turn_index, tool_name
        """, conv_ids)

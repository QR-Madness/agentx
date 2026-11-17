"""Serialize a user's memory graph + PostgreSQL audit mirror into a MemoryExport.

Nodes are dumped as opaque property dicts (datetimes → ISO) sorted by ``id`` for
deterministic, diffable output. Relationships that aren't already encoded as
foreign-key-like props are captured as helper keys on the owning node (e.g. a
fact's ``entity_ids`` from its ABOUT edges, a goal's ``parent_goal_id`` from
SUBGOAL_OF). The graph reads are ``memory.portability_store.PortabilityStore``'s;
this module shapes the envelope and reads the PostgreSQL mirror.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from sqlalchemy import bindparam, text

from ..connections import get_postgres_session
from ..memory.portability_store import PortabilityStore
from ..query_utils import convert_all_datetimes
from .schema import MemoryExport, current_embedder_info

logger = logging.getLogger(__name__)


class MemoryExporter:
    """Reads the memory graph for one user (optionally a channel set) into a MemoryExport.

    ``channels`` (a list) wins over the legacy single ``channel`` when both are
    given. ``include_global=False`` makes the scope exact-match — no ``_global``
    nodes ride along — which is what Extract needs so the artifact reconciles
    1:1 against the wiped set; plain exports keep the inclusive default.
    """

    def __init__(
        self,
        user_id: str,
        channel: str | None = None,
        channels: list[str] | None = None,
        include_global: bool = True,
    ):
        self.user_id = user_id
        self.include_global = include_global
        # Normalize both spellings into one internal list (None = no filter).
        if channels is not None:
            chans = [c for c in channels if c not in (None, "")]
            self.channels: list[str] | None = (
                None if (not chans or "_all" in chans) else list(dict.fromkeys(chans))
            )
        elif channel in (None, "_all"):
            self.channels = None
        else:
            self.channels = [cast(str, channel)]
        # Back-compat single-channel view (used by the envelope + callers).
        self.channel = self.channels[0] if self.channels and len(self.channels) == 1 else None

    # -- helpers ---------------------------------------------------------

    def _node(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Neo4j node property dict for JSON. Exports are text-only:
        the embedding is always dropped (regenerated on import). Every temporal
        property is converted to ISO so no raw Neo4j DateTime (e.g. a Conversation's
        ``self_consolidated``) can break serialization."""
        d = convert_all_datetimes(dict(raw))
        d.pop("embedding", None)
        return d

    # -- node collections ------------------------------------------------

    def export(self) -> MemoryExport:
        """Run every collection query and assemble the envelope."""
        export = MemoryExport(
            user_id=self.user_id,
            channel=self.channel,
            channels=self.channels if self.channels and len(self.channels) > 1 else None,
            embedder=current_embedder_info(),
        )
        collections = PortabilityStore().read_collections(
            self.user_id, self.channels, include_global=self.include_global
        )
        export.conversations = [self._node(n) for n in collections["conversations"]]
        export.turns = [self._node(n) for n in collections["turns"]]
        export.entities = [self._node(n) for n in collections["entities"]]
        export.facts = [self._node(n) for n in collections["facts"]]
        export.goals = [self._node(n) for n in collections["goals"]]
        export.strategies = [self._node(n) for n in collections["strategies"]]
        export.procedures = [self._node(n) for n in collections["procedures"]]
        export.tool_invocations = [self._node(n) for n in collections["tool_invocations"]]

        # -- PostgreSQL audit mirror, scoped to the exported conversations ---
        conv_ids = [c["id"] for c in export.conversations]
        if conv_ids:
            export.pg_conversation_logs = self._export_pg_logs(conv_ids)
            export.pg_tool_invocations = self._export_pg_tools(conv_ids)

        logger.info("Memory export for user=%s channels=%s: %s",
                    self.user_id, self.channels or "_all", export.counts())
        return export

    # -- PostgreSQL ------------------------------------------------------

    def _pg_rows(self, sql: str, conv_ids: list[str]) -> list[dict[str, Any]]:
        stmt = text(sql).bindparams(bindparam("ids", expanding=True))
        with get_postgres_session() as session:
            result = session.execute(stmt, {"ids": conv_ids})
            rows = [dict(row._mapping) for row in result]
        out: list[dict[str, Any]] = []
        for row in rows:
            row["conversation_id"] = str(row["conversation_id"])
            if row.get("timestamp") is not None and hasattr(row["timestamp"], "isoformat"):
                row["timestamp"] = row["timestamp"].isoformat()
            out.append(row)
        return out

    def _export_pg_logs(self, conv_ids: list[str]) -> list[dict[str, Any]]:
        # Text-only: the embedding column is omitted (regenerated on import).
        return self._pg_rows("""
            SELECT conversation_id::text AS conversation_id, turn_index, timestamp,
                   role, content, content_hash, token_count, model, channel, agent_id,
                   metadata
            FROM conversation_logs
            WHERE conversation_id::text IN :ids
            ORDER BY conversation_id, turn_index
        """, conv_ids)

    def _export_pg_tools(self, conv_ids: list[str]) -> list[dict[str, Any]]:
        return self._pg_rows("""
            SELECT conversation_id::text AS conversation_id, turn_index, timestamp,
                   tool_name, tool_input, tool_output, success, latency_ms,
                   error_message, channel
            FROM tool_invocations
            WHERE conversation_id::text IN :ids
            ORDER BY conversation_id, turn_index, tool_name
        """, conv_ids)

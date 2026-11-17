"""Idempotently restore a MemoryExport into Neo4j + PostgreSQL.

Every node is written with ``MERGE`` on its stable ``id`` (or, for the id-less
``ToolInvocation``, on a natural key), so re-importing the same envelope is a
no-op. Exports are text-only, so every embedding is recomputed from the node's
canonical text with this instance's model on import (which also makes exports
portable across embedding models).

Modes:
    - ``merge``   — upsert; leaves nodes outside the envelope untouched.
    - ``replace`` — deletes the user's nodes in the target channel(s) first, so
      the channel ends up matching the envelope exactly.

The graph writes are ``memory.portability_store.PortabilityStore``'s; this module
regenerates embeddings, restores the PostgreSQL mirror and reports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import bindparam, text
from typing import Literal

from ..connections import get_postgres_session
from ..embeddings import get_embedder
from ..memory.portability_store import PortabilityStore
from .schema import SCHEMA_VERSION, MemoryExport

logger = logging.getLogger(__name__)

ImportMode = Literal["merge", "replace"]

# Import re-embeds every node; batch in bounded chunks (queue interleaving +
# remote input caps) with a generous background wait (ingestion precedent).
_IMPORT_EMBED_CHUNK = 128
_IMPORT_EMBED_TIMEOUT_S = 600.0


def wipe_pg_mirror(channel: str | list[str] | None) -> dict[str, int]:
    """Delete the PostgreSQL audit-mirror rows in `channel` (exact match).

    The mirror tables carry no user column (single-user until auth lands), so
    None / "_all" clears every row — mirroring the Neo4j wipe-all semantics.
    Returns per-table deleted counts. Shared by replace-mode import (which
    previously left the mirror stale — the observed asymmetry) and Extract.
    """
    from ..query_utils import SQLFilterBuilder

    chan: str | list[str] | None = None if channel in (None, "_all") else channel
    counts: dict[str, int] = {}
    with get_postgres_session() as session:
        for table in ("conversation_logs", "tool_invocations"):
            builder = SQLFilterBuilder().add_channel_filter(chan, include_global=False)
            where = builder.build()
            result = session.execute(
                text(f"DELETE FROM {table} {where}"), builder.params  # noqa: S608 — fixed table names, bound params
            )
            counts[f"pg_{table}"] = result.rowcount or 0  # pyright: ignore[reportAttributeAccessIssue]
    return counts


class MemoryImporter:
    """Writes a MemoryExport back into the live stores under one user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._embedder = None
        self._recomputed = 0

    # -- public ----------------------------------------------------------

    def import_export(
        self,
        payload: MemoryExport | dict[str, Any],
        mode: ImportMode = "merge",
        channel: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Restore `payload`. `channel` overrides the wipe scope for replace mode
        (a list wipes exactly that channel set).

        Returns a summary dict with per-type counts and `recomputed_embeddings`.
        """
        export = self._coerce(payload)
        self._recomputed = 0

        # Exports are text-only: regenerate every embedding from the node's text
        # with THIS instance's model, in Python, before touching the DB (keeps the
        # write transaction short and makes imports portable across embedders).
        self._prepare_embeddings(export)

        wipe_channel: str | list[str] | None = (
            channel if channel is not None else (export.channels or export.channel)
        )
        # One graph transaction: the replace-mode wipe, every family, the turn chains.
        results = PortabilityStore().write_collections(
            self.user_id, export, replace=(mode == "replace"), channel=wipe_channel
        )

        # Replace mode also resets the PG mirror for the wiped channel(s) —
        # leaving it stale was the observed Neo4j/PG asymmetry.
        pg_wiped = wipe_pg_mirror(wipe_channel) if mode == "replace" else None

        pg_logs = self._import_pg_logs(export.pg_conversation_logs)
        pg_tools = self._import_pg_tools(export.pg_tool_invocations)

        summary = {
            "mode": mode,
            "channel": wipe_channel,
            "recomputed_embeddings": self._recomputed,
            "imported": {
                k: {"created": c, "total": t} for k, (c, t) in results.items()
            },
            "pg_conversation_logs": pg_logs,
            "pg_tool_invocations": pg_tools,
        }
        if pg_wiped is not None:
            summary["pg_wiped"] = pg_wiped
        logger.info("Memory import (user=%s, mode=%s): %s", self.user_id, mode, summary)
        return summary

    # -- embedding handling ---------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed in bounded chunks. Chunking keeps each queue request small
        enough that live chat embeds interleave between chunks, and under remote
        per-request input caps."""
        if self._embedder is None:
            self._embedder = get_embedder()
        out: list[list[float]] = []
        for start in range(0, len(texts), _IMPORT_EMBED_CHUNK):
            out.extend(self._embedder.embed(
                texts[start:start + _IMPORT_EMBED_CHUNK],
                timeout=_IMPORT_EMBED_TIMEOUT_S,
            ))
        self._recomputed += len(texts)
        return out

    def _prepare_embeddings(self, export: MemoryExport) -> None:
        """Regenerate every node embedding from its canonical text — batched per
        record type (re-embedding is the expensive import step)."""
        groups: list[tuple[list[dict], Callable[[dict], str]]] = [
            (export.turns, lambda t: t.get("content") or ""),
            (export.entities,
             lambda e: f"{e.get('name', '')}: {e.get('description') or e.get('type', '')}"),
            (export.facts, lambda f: f.get("claim") or ""),
            (export.goals, lambda g: g.get("description") or ""),
            (export.strategies, lambda s: s.get("description") or ""),
            # Matches ProceduralMemory.learn_procedure / find_procedures text.
            (export.procedures,
             lambda p: f"{p.get('trigger') or ''}\n{p.get('body') or ''}"),
        ]
        for rows, text_of in groups:
            if not rows:
                continue
            vectors = self._embed_batch([text_of(r) for r in rows])
            for row, vec in zip(rows, vectors, strict=True):
                row["embedding"] = vec



    # -- PostgreSQL ------------------------------------------------------

    def _import_pg_logs(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        # The audit-mirror embedding is left NULL on import — recall uses the
        # recomputed Neo4j `turn_embeddings` vectors, not this column.
        with get_postgres_session() as session:
            for row in rows:
                session.execute(text("""
                    INSERT INTO conversation_logs
                    (conversation_id, turn_index, timestamp, role, content, content_hash,
                     token_count, model, channel, agent_id, metadata)
                    VALUES (CAST(:conversation_id AS UUID), :turn_index, :timestamp, :role,
                            :content, :content_hash, :token_count, :model, :channel, :agent_id,
                            CAST(:metadata AS JSONB))
                    ON CONFLICT (conversation_id, turn_index) DO UPDATE
                    SET timestamp = EXCLUDED.timestamp,
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        token_count = EXCLUDED.token_count,
                        model = EXCLUDED.model,
                        channel = EXCLUDED.channel,
                        agent_id = EXCLUDED.agent_id,
                        metadata = EXCLUDED.metadata
                """), self._pg_log_params(row))
        return len(rows)

    def _import_pg_tools(self, rows: list[dict[str, Any]]) -> int:
        # No natural key on tool_invocations → delete the exported conversations'
        # rows, then re-insert. Idempotent across re-imports.
        if not rows:
            return 0
        conv_ids = sorted({r["conversation_id"] for r in rows})
        with get_postgres_session() as session:
            session.execute(
                text("DELETE FROM tool_invocations WHERE conversation_id::text IN :ids")
                .bindparams(bindparam("ids", expanding=True)),
                {"ids": conv_ids},
            )
            for row in rows:
                session.execute(text("""
                    INSERT INTO tool_invocations
                    (conversation_id, turn_index, timestamp, tool_name, tool_input,
                     tool_output, success, latency_ms, error_message, channel)
                    VALUES (CAST(:conversation_id AS UUID), :turn_index, :timestamp, :tool_name,
                            CAST(:tool_input AS JSONB), CAST(:tool_output AS JSONB), :success,
                            :latency_ms, :error_message, :channel)
                """), self._pg_tool_params(row))
        return len(rows)

    @staticmethod
    def _json_param(value: Any) -> Any:
        import json
        if value is None or isinstance(value, str):
            return value
        return json.dumps(value)

    def _pg_log_params(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "conversation_id": row["conversation_id"],
            "turn_index": row["turn_index"],
            "timestamp": row.get("timestamp"),
            "role": row.get("role"),
            "content": row.get("content"),
            "content_hash": row.get("content_hash"),
            "token_count": row.get("token_count"),
            "model": row.get("model"),
            "channel": row.get("channel") or "_global",
            "agent_id": row.get("agent_id"),
            "metadata": self._json_param(row.get("metadata")) or "{}",
        }

    def _pg_tool_params(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "conversation_id": row["conversation_id"],
            "turn_index": row.get("turn_index"),
            "timestamp": row.get("timestamp"),
            "tool_name": row.get("tool_name"),
            "tool_input": self._json_param(row.get("tool_input")) or "{}",
            "tool_output": self._json_param(row.get("tool_output")),
            "success": row.get("success"),
            "latency_ms": row.get("latency_ms"),
            "error_message": row.get("error_message"),
            "channel": row.get("channel") or "_global",
        }

    # -- validation ------------------------------------------------------

    @staticmethod
    def _coerce(payload: MemoryExport | dict[str, Any]) -> MemoryExport:
        export = payload if isinstance(payload, MemoryExport) else MemoryExport.model_validate(payload)
        if export.schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"Export schema_version {export.schema_version} is newer than this "
                f"build supports ({SCHEMA_VERSION}). Upgrade AgentX to import it."
            )
        return export

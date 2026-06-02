"""Idempotently restore a MemoryExport into Neo4j + PostgreSQL.

Every node is written with ``MERGE`` on its stable ``id`` (or, for the id-less
``ToolInvocation``, on a natural key), so re-importing the same envelope is a
no-op. Exports are text-only, so every embedding is recomputed from the node's
canonical text with this instance's model on import (which also makes exports
portable across embedding models).

Modes:
    - ``merge``   — upsert; leaves nodes outside the envelope untouched.
    - ``replace`` — scoped ``DETACH DELETE`` of the target channel(s) first, so
      the channel ends up matching the envelope exactly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from sqlalchemy import bindparam, text
from typing_extensions import Literal, LiteralString

from ..connections import Neo4jConnection, get_postgres_session
from ..embeddings import get_embedder
from ..query_utils import CypherFilterBuilder
from .schema import SCHEMA_VERSION, MemoryExport

logger = logging.getLogger(__name__)

ImportMode = Literal["merge", "replace"]


class MemoryImporter:
    """Writes a MemoryExport back into the live stores under one user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._embedder = None
        self._recomputed = 0

    # -- public ----------------------------------------------------------

    def import_export(
        self,
        payload: Union[MemoryExport, Dict[str, Any]],
        mode: ImportMode = "merge",
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore `payload`. `channel` overrides the wipe scope for replace mode.

        Returns a summary dict with per-type counts and `recomputed_embeddings`.
        """
        export = self._coerce(payload)
        self._recomputed = 0

        # Exports are text-only: regenerate every embedding from the node's text
        # with THIS instance's model, in Python, before touching the DB (keeps the
        # write transaction short and makes imports portable across embedders).
        self._prepare_embeddings(export)

        wipe_channel = channel if channel is not None else export.channel
        results: Dict[str, Tuple[int, int]] = {}

        with Neo4jConnection.session() as session:
            tx = session.begin_transaction()
            try:
                if mode == "replace":
                    self._wipe(tx, wipe_channel)
                results["entities"] = self._import_entities(tx, export.entities)
                results["conversations"] = self._import_conversations(tx, export.conversations)
                results["turns"] = self._import_turns(tx, export.turns)
                results["facts"] = self._import_facts(tx, export.facts)
                results["goals"] = self._import_goals(tx, export.goals)
                results["strategies"] = self._import_strategies(tx, export.strategies)
                results["tool_invocations"] = self._import_tool_invocations(tx, export.tool_invocations)
                self._rebuild_turn_chains(tx, [c["id"] for c in export.conversations])
                tx.commit()
            except Exception:
                tx.rollback()
                raise

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
        logger.info("Memory import (user=%s, mode=%s): %s", self.user_id, mode, summary)
        return summary

    # -- embedding handling ---------------------------------------------

    def _embed(self, text_value: str) -> List[float]:
        if self._embedder is None:
            self._embedder = get_embedder()
        self._recomputed += 1
        return self._embedder.embed_single(text_value)

    def _prepare_embeddings(self, export: MemoryExport) -> None:
        """Regenerate every node embedding from its canonical text."""
        for t in export.turns:
            t["embedding"] = self._embed(t.get("content") or "")
        for e in export.entities:
            text_value = f"{e.get('name', '')}: {e.get('description') or e.get('type', '')}"
            e["embedding"] = self._embed(text_value)
        for f in export.facts:
            f["embedding"] = self._embed(f.get("claim") or "")
        for g in export.goals:
            g["embedding"] = self._embed(g.get("description") or "")
        for s in export.strategies:
            s["embedding"] = self._embed(s.get("description") or "")

    # -- replace-mode wipe ----------------------------------------------

    def _wipe(self, tx, channel: Optional[str]) -> None:
        """Scoped DETACH DELETE of the user's nodes in `channel` (exact match, no _global spill)."""
        chan = None if channel in (None, "_all") else channel

        def f(alias: str) -> str:
            # include_global=False → exact channel match (don't nuke _global).
            return CypherFilterBuilder(alias).add_channel_filter(
                chan, include_global=False
            ).build_inline()

        params = {"user_id": self.user_id, "channel": chan}
        for label, rel, alias in (
            ("Fact", "HAS_FACT", "n"),
            ("Entity", "HAS_ENTITY", "n"),
            ("Strategy", "HAS_STRATEGY", "n"),
            ("Goal", "HAS_GOAL", "n"),
        ):
            tx.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:{rel}]->(n:{label})
                WHERE true {f(alias)}
                DETACH DELETE n
            """), **params)

        # Conversations + their turns / tool-invocations / participants.
        tx.run(cast(LiteralString, f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
            WHERE true {f('c')}
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            OPTIONAL MATCH (c)-[:USED_TOOL]->(inv:ToolInvocation)
            OPTIONAL MATCH (ap:AgentParticipant)-[:PARTICIPATED_IN]->(c)
            DETACH DELETE t, inv, ap, c
        """), **params)

    # -- node writers ----------------------------------------------------

    @staticmethod
    def _counts(result) -> Tuple[int, int]:
        rec = result.single()
        if rec is None:
            return (0, 0)
        return (rec["created"] or 0, rec["total"] or 0)

    def _import_entities(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        return self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Entity {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (e:Entity {id: row.id})
            SET e.name = row.name,
                e.type = row.type,
                e.aliases = row.aliases,
                e.description = row.description,
                e.embedding = row.embedding,
                e.salience = row.salience,
                e.user_id = $user_id,
                e.channel = row.channel,
                e.access_count = row.access_count,
                e.properties = row.properties,
                e.first_seen = CASE WHEN row.first_seen IS NULL
                    THEN coalesce(e.first_seen, datetime()) ELSE datetime(row.first_seen) END,
                e.last_accessed = CASE WHEN row.last_accessed IS NULL
                    THEN coalesce(e.last_accessed, datetime()) ELSE datetime(row.last_accessed) END
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_ENTITY]->(e)
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=self.user_id))

    def _import_conversations(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        return self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Conversation {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (c:Conversation {id: row.id})
            SET c.user_id = $user_id,
                c.channel = row.channel,
                c.agent_id = row.agent_id,
                c.started_at = CASE WHEN row.started_at IS NULL
                    THEN coalesce(c.started_at, datetime()) ELSE datetime(row.started_at) END,
                c.consolidated = CASE WHEN row.consolidated IS NULL
                    THEN c.consolidated ELSE datetime(row.consolidated) END,
                c.self_consolidated = CASE WHEN row.self_consolidated IS NULL
                    THEN c.self_consolidated ELSE datetime(row.self_consolidated) END
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c)
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=self.user_id))

    def _import_turns(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        return self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Turn {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (c:Conversation {id: row.conversation_id})
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c)
            MERGE (t:Turn {id: row.id})
            SET t.index = row.index,
                t.role = row.role,
                t.content = row.content,
                t.embedding = row.embedding,
                t.token_count = row.token_count,
                t.channel = row.channel,
                t.user_id = $user_id,
                t.agent_id = row.agent_id,
                t.timestamp = CASE WHEN row.timestamp IS NULL
                    THEN coalesce(t.timestamp, datetime()) ELSE datetime(row.timestamp) END
            MERGE (c)-[:HAS_TURN]->(t)
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=self.user_id))

    def _import_facts(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        counts = self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Fact {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (f:Fact {id: row.id})
            SET f.claim = row.claim,
                f.claim_hash = row.claim_hash,
                f.confidence = row.confidence,
                f.source = row.source,
                f.source_turn_id = row.source_turn_id,
                f.embedding = row.embedding,
                f.user_id = $user_id,
                f.channel = row.channel,
                f.access_count = row.access_count,
                f.salience = row.salience,
                f.temporal_context = row.temporal_context,
                f.superseded_by_id = row.superseded_by_id,
                f.supersedes_id = row.supersedes_id,
                f.flagged_for_review = row.flagged_for_review,
                f.created_at = CASE WHEN row.created_at IS NULL
                    THEN coalesce(f.created_at, datetime()) ELSE datetime(row.created_at) END,
                f.last_accessed = CASE WHEN row.last_accessed IS NULL
                    THEN coalesce(f.last_accessed, datetime()) ELSE datetime(row.last_accessed) END,
                f.superseded_at = CASE WHEN row.superseded_at IS NULL
                    THEN null ELSE datetime(row.superseded_at) END
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_FACT]->(f)
            FOREACH (eid IN row.entity_ids |
                MERGE (e:Entity {id: eid})
                MERGE (f)-[:ABOUT]->(e)
            )
            FOREACH (_ IN CASE WHEN row.source_turn_id IS NOT NULL THEN [1] ELSE [] END |
                MERGE (st:Turn {id: row.source_turn_id})
                MERGE (f)-[:DERIVED_FROM]->(st)
            )
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=self.user_id))

        # Supersession edges (second pass — both facts must exist first).
        supersedes = [r for r in rows if r.get("supersedes_id")]
        if supersedes:
            tx.run("""
                UNWIND $rows AS row
                MATCH (new:Fact {id: row.id}), (old:Fact {id: row.supersedes_id})
                MERGE (new)-[:SUPERSEDES]->(old)
            """, rows=supersedes)
        return counts

    def _import_goals(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        counts = self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Goal {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (g:Goal {id: row.id})
            SET g.description = row.description,
                g.status = row.status,
                g.priority = row.priority,
                g.channel = row.channel,
                g.embedding = row.embedding,
                g.created_at = CASE WHEN row.created_at IS NULL
                    THEN coalesce(g.created_at, datetime()) ELSE datetime(row.created_at) END,
                g.deadline = CASE WHEN row.deadline IS NULL
                    THEN null ELSE datetime(row.deadline) END
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_GOAL]->(g)
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=self.user_id))

        children = [r for r in rows if r.get("parent_goal_id")]
        if children:
            tx.run("""
                UNWIND $rows AS row
                MATCH (child:Goal {id: row.id}), (parent:Goal {id: row.parent_goal_id})
                MERGE (child)-[:SUBGOAL_OF]->(parent)
            """, rows=children)
        return counts

    def _import_strategies(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        return self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Strategy {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (s:Strategy {id: row.id})
            SET s.description = row.description,
                s.context_pattern = row.context_pattern,
                s.tool_sequence = row.tool_sequence,
                s.embedding = row.embedding,
                s.success_count = row.success_count,
                s.failure_count = row.failure_count,
                s.user_id = $user_id,
                s.channel = row.channel,
                s.last_used = CASE WHEN row.last_used IS NULL
                    THEN null ELSE datetime(row.last_used) END
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_STRATEGY]->(s)
            FOREACH (tn IN row.tool_sequence |
                MERGE (tool:Tool {name: tn})
                MERGE (s)-[:USES_TOOL]->(tool)
            )
            FOREACH (cid IN row.succeeded_in |
                MERGE (c:Conversation {id: cid})
                MERGE (s)-[:SUCCEEDED_IN]->(c)
            )
            FOREACH (cid IN row.failed_in |
                MERGE (c:Conversation {id: cid})
                MERGE (s)-[:FAILED_IN]->(c)
            )
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=self.user_id))

    def _import_tool_invocations(self, tx, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        # id-less node → MERGE on a natural key (conversation + tool + turn + ts).
        rows = [r for r in rows if r.get("timestamp") and r.get("conversation_id")]
        if not rows:
            return (0, 0)
        tx.run("""
            UNWIND $rows AS row
            MATCH (c:Conversation {id: row.conversation_id})
            MERGE (c)-[:USED_TOOL]->(inv:ToolInvocation {
                tool_name: row.tool_name,
                turn_index: row.turn_index,
                timestamp: datetime(row.timestamp)
            })
            ON CREATE SET inv.success = row.success,
                          inv.latency_ms = row.latency_ms,
                          inv.channel = row.channel
            MERGE (tool:Tool {name: row.tool_name})
            MERGE (inv)-[:INVOKED]->(tool)
        """, rows=rows)
        return (0, len(rows))

    def _rebuild_turn_chains(self, tx, conv_ids: List[str]) -> None:
        """Re-derive FOLLOWED_BY (from turn index) and AgentParticipant for imported convos."""
        if not conv_ids:
            return
        tx.run("""
            UNWIND $conv_ids AS cid
            MATCH (c:Conversation {id: cid})-[:HAS_TURN]->(a:Turn),
                  (c)-[:HAS_TURN]->(b:Turn)
            WHERE b.index = a.index + 1
            MERGE (a)-[:FOLLOWED_BY]->(b)
        """, conv_ids=conv_ids)
        tx.run("""
            UNWIND $conv_ids AS cid
            MATCH (c:Conversation {id: cid})-[:HAS_TURN]->(t:Turn)
            WHERE t.agent_id IS NOT NULL
            MERGE (ap:AgentParticipant {id: cid + ':' + t.agent_id})
            ON CREATE SET ap.conversation_id = cid,
                          ap.agent_id = t.agent_id,
                          ap.user_id = $user_id,
                          ap.channel = c.channel,
                          ap.first_seen = datetime()
            MERGE (ap)-[:PARTICIPATED_IN]->(c)
        """, conv_ids=conv_ids, user_id=self.user_id)

    # -- PostgreSQL ------------------------------------------------------

    def _import_pg_logs(self, rows: List[Dict[str, Any]]) -> int:
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

    def _import_pg_tools(self, rows: List[Dict[str, Any]]) -> int:
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

    def _pg_log_params(self, row: Dict[str, Any]) -> Dict[str, Any]:
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

    def _pg_tool_params(self, row: Dict[str, Any]) -> Dict[str, Any]:
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
    def _coerce(payload: Union[MemoryExport, Dict[str, Any]]) -> MemoryExport:
        export = payload if isinstance(payload, MemoryExport) else MemoryExport.model_validate(payload)
        if export.schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"Export schema_version {export.schema_version} is newer than this "
                f"build supports ({SCHEMA_VERSION}). Upgrade AgentX to import it."
            )
        return export

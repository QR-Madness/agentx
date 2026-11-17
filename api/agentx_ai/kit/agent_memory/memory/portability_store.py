"""Whole-memory bulk reads and writes for one user: export, import and Extract.

Recall, consolidation and the API reach memory one record or one search at a time,
through the per-type classes. Export, import and Extract move everything a user holds
in a channel scope at once, against one envelope (``portability/schema.py``), so their
store queries live together here. The portability modules keep everything that isn't
a store query: envelope assembly, embedding regeneration, the PostgreSQL audit mirror,
artifacts and receipts. The PostgreSQL twin (Decisions.md ADR-21, Slice 3) reads and
writes the same envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from typing import LiteralString

from ..connections import Neo4jConnection
from ..query_utils import CypherFilterBuilder

if TYPE_CHECKING:
    from ..portability.schema import MemoryExport


def _exact_channel_clause(alias: str) -> str:
    """Exact-channel inline clause bound to $channels (never includes _global).

    The builder shapes the clause from the list form only — actual channel
    values bind at query time through the ``$channels`` parameter.
    """
    return CypherFilterBuilder(alias).add_channel_filter(
        ["_shape_only_"], include_global=False
    ).build_inline()


# Envelope-agnostic count table: (family, live-count Cypher). Every query takes
# $user_id + $channels and counts with the exact-channel scope that export and wipe
# use (include_global=False) — Extract's verify step reconciles against these.
def _count_queries() -> list[tuple[str, str]]:
    return [
        ("conversations", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
            WHERE true {_exact_channel_clause('c')}
            RETURN count(c) AS c"""),
        ("turns", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                  -[:HAS_TURN]->(t:Turn)
            WHERE true {_exact_channel_clause('t')}
            RETURN count(t) AS c"""),
        ("entities", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_ENTITY]->(e:Entity)
            WHERE true {_exact_channel_clause('e')}
            RETURN count(e) AS c"""),
        ("facts", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_FACT]->(f:Fact)
            WHERE true {_exact_channel_clause('f')}
            RETURN count(f) AS c"""),
        ("goals", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal)
            WHERE true {_exact_channel_clause('g')}
            RETURN count(g) AS c"""),
        ("strategies", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_STRATEGY]->(s:Strategy)
            WHERE true {_exact_channel_clause('s')}
            RETURN count(s) AS c"""),
        ("procedures", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_PROCEDURE]->(p:Procedure)
            WHERE true {_exact_channel_clause('p')}
            RETURN count(p) AS c"""),
        ("tool_invocations", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                  -[:USED_TOOL]->(inv:ToolInvocation)
            WHERE true {_exact_channel_clause('inv')}
            RETURN count(inv) AS c"""),
    ]


class PortabilityStore:
    """One user's memory in bulk, scoped to a channel set."""

    # -- export ----------------------------------------------------------

    def read_collections(
        self,
        user_id: str,
        channels: list[str] | None,
        include_global: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """Every node `user_id` holds in `channels` (None = every channel), as raw
        property dicts sorted by id, plus the relationship keys the envelope carries:
        a turn's or tool invocation's ``conversation_id``, a fact's ``entity_ids``, a
        goal's ``parent_goal_id``, a strategy's ``tool_sequence`` (its USES_TOOL edges,
        else the stored property), ``succeeded_in`` and ``failed_in``, and a
        procedure's ``conversation_ids``. ``include_global=False`` makes the scope
        exact: no ``_global`` nodes ride along.
        """
        def chan(alias: str) -> str:
            return CypherFilterBuilder(alias).add_channel_filter(
                channels, include_global=include_global
            ).build_inline()

        params = {
            "user_id": user_id,
            "channel": channels[0] if channels and len(channels) == 1 else None,
            "channels": channels,
        }
        out: dict[str, list[dict[str, Any]]] = {}
        with Neo4jConnection.session() as session:
            out["conversations"] = [
                dict(r["c"])
                for r in session.run(cast(LiteralString, f"""
                    MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                    WHERE true {chan('c')}
                    RETURN c ORDER BY c.id
                """), **params)
            ]

            out["turns"] = [
                {**dict(r["t"]), "conversation_id": r["conversation_id"]}
                for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                      -[:HAS_TURN]->(t:Turn)
                WHERE true {chan('t')}
                RETURN t, c.id AS conversation_id ORDER BY t.id
            """), **params)
            ]

            out["entities"] = [
                dict(r["e"])
                for r in session.run(cast(LiteralString, f"""
                    MATCH (u:User {{id: $user_id}})-[:HAS_ENTITY]->(e:Entity)
                    WHERE true {chan('e')}
                    RETURN e ORDER BY e.id
                """), **params)
            ]

            out["facts"] = [
                {**dict(r["f"]), "entity_ids": [eid for eid in r["entity_ids"] if eid]}
                for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_FACT]->(f:Fact)
                WHERE true {chan('f')}
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                WITH f, collect(DISTINCT e.id) AS entity_ids
                RETURN f, entity_ids ORDER BY f.id
            """), **params)
            ]

            out["goals"] = [
                {**dict(r["g"]), "parent_goal_id": r["parent_goal_id"]}
                for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal)
                WHERE true {chan('g')}
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent.id AS parent_goal_id ORDER BY g.id
            """), **params)
            ]

            out["strategies"] = []
            for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_STRATEGY]->(s:Strategy)
                WHERE true {chan('s')}
                OPTIONAL MATCH (s)-[:USES_TOOL]->(tool:Tool)
                OPTIONAL MATCH (s)-[:SUCCEEDED_IN]->(sc:Conversation)
                OPTIONAL MATCH (s)-[:FAILED_IN]->(fc:Conversation)
                WITH s, collect(DISTINCT tool.name) AS tool_names,
                     collect(DISTINCT sc.id) AS succeeded_in,
                     collect(DISTINCT fc.id) AS failed_in
                RETURN s, tool_names, succeeded_in, failed_in ORDER BY s.id
            """), **params):
                node = dict(r["s"])
                # USES_TOOL edges are authoritative; fall back to the stored prop.
                node["tool_sequence"] = (
                    [t for t in r["tool_names"] if t] or node.get("tool_sequence") or []
                )
                node["succeeded_in"] = [c for c in r["succeeded_in"] if c]
                node["failed_in"] = [c for c in r["failed_in"] if c]
                out["strategies"].append(node)

            out["procedures"] = [
                {**dict(r["p"]), "conversation_ids": [cid for cid in r["conversation_ids"] if cid]}
                for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_PROCEDURE]->(p:Procedure)
                WHERE true {chan('p')}
                OPTIONAL MATCH (p)-[:DISTILLED_FROM]->(c:Conversation)
                WITH p, collect(DISTINCT c.id) AS conversation_ids
                RETURN p, conversation_ids ORDER BY p.id
            """), **params)
            ]

            out["tool_invocations"] = [
                {**dict(r["inv"]), "conversation_id": r["conversation_id"]}
                for r in session.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                      -[:USED_TOOL]->(inv:ToolInvocation)
                WHERE true {chan('inv')}
                RETURN inv, c.id AS conversation_id
                ORDER BY c.id, inv.turn_index, inv.tool_name
            """), **params)
            ]
        return out

    # -- import ----------------------------------------------------------

    def write_collections(
        self,
        user_id: str,
        export: MemoryExport,
        *,
        replace: bool = False,
        channel: str | list[str] | None = None,
    ) -> dict[str, tuple[int, int]]:
        """Write an envelope's nodes under `user_id` in one transaction: all of it or,
        on any error, none of it. With `replace`, first deletes the user's nodes in
        `channel` (None or "_all" = every channel) inside the same transaction.
        Embeddings must already be on the rows. Returns per-family (created, total).
        """
        results: dict[str, tuple[int, int]] = {}
        with Neo4jConnection.session() as session:
            tx = session.begin_transaction()
            try:
                if replace:
                    self._wipe(tx, user_id, channel)
                results["entities"] = self._write_entities(tx, user_id, export.entities)
                results["conversations"] = self._write_conversations(tx, user_id, export.conversations)
                results["turns"] = self._write_turns(tx, user_id, export.turns)
                results["facts"] = self._write_facts(tx, user_id, export.facts)
                results["goals"] = self._write_goals(tx, user_id, export.goals)
                results["strategies"] = self._write_strategies(tx, user_id, export.strategies)
                results["procedures"] = self._write_procedures(tx, user_id, export.procedures)
                results["tool_invocations"] = self._write_tool_invocations(
                    tx, user_id, export.tool_invocations
                )
                self._rebuild_turn_chains(tx, user_id, [c["id"] for c in export.conversations])
                tx.commit()
            except Exception:
                tx.rollback()
                raise
        return results

    # -- Extract ---------------------------------------------------------

    def count_channels(self, user_id: str, channels: list[str]) -> dict[str, int]:
        """Per-family node counts under the exact channel scope Extract exports and wipes."""
        counts: dict[str, int] = {}
        params = {"user_id": user_id, "channels": channels}
        with Neo4jConnection.session() as session:
            for family, query in _count_queries():
                rec = session.run(cast(LiteralString, query), **params).single()
                counts[family] = (rec["c"] if rec else 0) or 0
        return counts

    def wipe_channels(self, user_id: str, channel: str | list[str] | None) -> dict[str, int]:
        """Delete `user_id`'s nodes in `channel` in their own transaction (Extract's
        wipe step). Returns per-family deleted counts — see `_wipe`."""
        with Neo4jConnection.session() as session:
            tx = session.begin_transaction()
            try:
                wiped = self._wipe(tx, user_id, channel)
                tx.commit()
            except Exception:
                tx.rollback()
                raise
        return wiped

    # -- internals -------------------------------------------------------

    def _wipe(self, tx, user_id: str, channel: str | list[str] | None) -> dict[str, int]:
        """Scoped DETACH DELETE of `user_id`'s nodes in `channel` (exact match, no
        _global spill). Accepts a single channel or a channel set; None / "_all"
        wipes every channel for the user. Returns per-family deleted counts.

        Shared by replace-mode import and Extract (which reconciles these counts
        against the artifact it just verified).
        """
        chan: str | list[str] | None = None if channel in (None, "_all") else channel

        def f(alias: str) -> str:
            # include_global=False → exact channel match (don't nuke _global).
            return CypherFilterBuilder(alias).add_channel_filter(
                chan, include_global=False
            ).build_inline()

        params = {
            "user_id": user_id,
            "channel": chan if isinstance(chan, str) else None,
            "channels": list(chan) if isinstance(chan, (list, tuple)) else None,
        }
        wiped: dict[str, int] = {}
        for family, label, rel in (
            ("facts", "Fact", "HAS_FACT"),
            ("entities", "Entity", "HAS_ENTITY"),
            ("strategies", "Strategy", "HAS_STRATEGY"),
            ("goals", "Goal", "HAS_GOAL"),
            ("procedures", "Procedure", "HAS_PROCEDURE"),
        ):
            rec = tx.run(cast(LiteralString, f"""
                MATCH (u:User {{id: $user_id}})-[:{rel}]->(n:{label})
                WHERE true {f('n')}
                DETACH DELETE n
                RETURN count(n) AS c
            """), **params).single()
            wiped[family] = (rec["c"] if rec else 0) or 0

        # Conversations + their turns / tool-invocations / participants.
        # Count first (same tx — nothing writes in between), then reuse the
        # proven cascade delete shape.
        rec = tx.run(cast(LiteralString, f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
            WHERE true {f('c')}
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            OPTIONAL MATCH (c)-[:USED_TOOL]->(inv:ToolInvocation)
            RETURN count(DISTINCT c) AS convs, count(DISTINCT t) AS turns,
                   count(DISTINCT inv) AS invs
        """), **params).single()
        wiped["conversations"] = (rec["convs"] if rec else 0) or 0
        wiped["turns"] = (rec["turns"] if rec else 0) or 0
        wiped["tool_invocations"] = (rec["invs"] if rec else 0) or 0

        tx.run(cast(LiteralString, f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
            WHERE true {f('c')}
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            OPTIONAL MATCH (c)-[:USED_TOOL]->(inv:ToolInvocation)
            OPTIONAL MATCH (ap:AgentParticipant)-[:PARTICIPATED_IN]->(c)
            DETACH DELETE t, inv, ap, c
        """), **params)
        return wiped

    @staticmethod
    def _counts(result) -> tuple[int, int]:
        rec = result.single()
        if rec is None:
            return (0, 0)
        return (rec["created"] or 0, rec["total"] or 0)

    def _write_entities(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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
        """, rows=rows, user_id=user_id))

    def _write_conversations(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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
        """, rows=rows, user_id=user_id))

    def _write_turns(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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
                    THEN coalesce(t.timestamp, datetime()) ELSE datetime(row.timestamp) END,
                // Preserve turn-level consolidation state so a re-imported bundle
                // resumes where it left off instead of re-consolidating everything.
                // Absent in pre-0.21 exports → keep any existing value.
                t.consolidated = CASE WHEN row.consolidated IS NULL
                    THEN t.consolidated ELSE datetime(row.consolidated) END,
                t.self_consolidated = CASE WHEN row.self_consolidated IS NULL
                    THEN t.self_consolidated ELSE datetime(row.self_consolidated) END
            MERGE (c)-[:HAS_TURN]->(t)
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=user_id))

    def _write_facts(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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
        """, rows=rows, user_id=user_id))

        # Supersession edges (second pass — both facts must exist first).
        supersedes = [r for r in rows if r.get("supersedes_id")]
        if supersedes:
            tx.run("""
                UNWIND $rows AS row
                MATCH (new:Fact {id: row.id}), (old:Fact {id: row.supersedes_id})
                MERGE (new)-[:SUPERSEDES]->(old)
            """, rows=supersedes)
        return counts

    def _write_goals(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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
        """, rows=rows, user_id=user_id))

        children = [r for r in rows if r.get("parent_goal_id")]
        if children:
            tx.run("""
                UNWIND $rows AS row
                MATCH (child:Goal {id: row.id}), (parent:Goal {id: row.parent_goal_id})
                MERGE (child)-[:SUBGOAL_OF]->(parent)
            """, rows=children)
        return counts

    def _write_strategies(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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
        """, rows=rows, user_id=user_id))

    def _write_procedures(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
        if not rows:
            return (0, 0)
        counts = self._counts(tx.run("""
            UNWIND $rows AS row
            OPTIONAL MATCH (ex:Procedure {id: row.id})
            WITH row, ex IS NOT NULL AS existed
            MERGE (p:Procedure {id: row.id})
            SET p.trigger = row.trigger,
                p.trigger_features = row.trigger_features,
                p.body = row.body,
                p.rationale = row.rationale,
                // Written as twins by learn_procedure; tolerate either in old data.
                p.channel = coalesce(row.channel, row.scope),
                p.scope = coalesce(row.scope, row.channel),
                p.agent_id = row.agent_id,
                p.strength = row.strength,
                p.evidence_refs = row.evidence_refs,
                p.signal_kinds = row.signal_kinds,
                p.embedding = row.embedding,
                p.user_id = $user_id,
                p.created_at = CASE WHEN row.created_at IS NULL
                    THEN coalesce(p.created_at, datetime()) ELSE datetime(row.created_at) END,
                p.last_reinforced = CASE WHEN row.last_reinforced IS NULL
                    THEN coalesce(p.last_reinforced, datetime()) ELSE datetime(row.last_reinforced) END
            MERGE (u:User {id: $user_id})
            MERGE (u)-[:HAS_PROCEDURE]->(p)
            RETURN sum(CASE WHEN existed THEN 0 ELSE 1 END) AS created, count(*) AS total
        """, rows=rows, user_id=user_id))

        # DISTILLED_FROM edges (second pass; MATCH not MERGE — mirrors the write
        # path, which never mints a Conversation for an absent evidence id).
        distilled = [r for r in rows if r.get("conversation_ids")]
        if distilled:
            tx.run("""
                UNWIND $rows AS row
                UNWIND row.conversation_ids AS cid
                MATCH (p:Procedure {id: row.id}), (c:Conversation {id: cid})
                MERGE (p)-[:DISTILLED_FROM]->(c)
            """, rows=distilled)
        return counts

    def _write_tool_invocations(self, tx, user_id: str, rows: list[dict[str, Any]]) -> tuple[int, int]:
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

    def _rebuild_turn_chains(self, tx, user_id: str, conv_ids: list[str]) -> None:
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
        """, conv_ids=conv_ids, user_id=user_id)

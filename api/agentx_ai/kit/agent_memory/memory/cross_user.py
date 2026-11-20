"""Cross-user memory operations: the one memory class whose methods span every user.

Every other memory class works within a user (and usually a channel). Admin, eval
and maintenance work needs the whole store: cluster snapshots and wipes, agent
renames that reach every user's copy of an agent, and the scheduled consolidation
maintenance (decay, archiving, promotion to ``_global``, pattern detection, the
entity backfills). That work goes through here and nowhere else. When the store
moves to PostgreSQL (Decisions.md ADR-21), this class's twin is the only code that
runs under the separate system role that row-level security lets past.

Writes consume their results before the session closes. An error that arrives after
``run()`` returns, such as a failed commit of the auto-commit transaction, is still
pending at close, and the driver's ``Session.close()`` swallows it there.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

from ..connections import Neo4jConnection, get_postgres_session

if TYPE_CHECKING:
    from .entity_maintenance import EntityMaintenance

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
            ).consume()

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

    # -- consolidation maintenance (the scheduled background jobs) -------------

    def decay_salience(self, entity_rate: float, fact_rate: float) -> tuple[int, int]:
        """Multiply the salience of entities untouched for a day by `entity_rate`, and
        the confidence of week-old inferred facts by `fact_rate`. Returns
        ``(entities_decayed, facts_decayed)``."""
        with Neo4jConnection.session() as session:
            # Decay entity salience
            result = session.run("""
            MATCH (e:Entity)
            WHERE e.last_accessed < datetime() - duration('P1D')
            SET e.salience = e.salience * $decay_rate
            RETURN count(e) AS decayed_count
        """, decay_rate=entity_rate)
            record = result.single()
            entities_decayed = record["decayed_count"] if record else 0

            # Decay fact confidence (slower decay)
            result = session.run("""
            MATCH (f:Fact)
            WHERE f.created_at < datetime() - duration('P7D')
              AND f.source = 'inferred'
            SET f.confidence = f.confidence * $decay_rate
            RETURN count(f) AS decayed_count
        """, decay_rate=fact_rate)
            record = result.single()
            facts_decayed = record["decayed_count"] if record else 0
        return entities_decayed, facts_decayed

    def archive_and_prune(self, retention_days: int) -> tuple[int, int]:
        """Archive conversations (and their turns) older than `retention_days`, then
        delete entities of very low salience, untouched for 30 days, that no fact is
        about. Returns ``(archived_conversations, deleted_entities)``."""
        with Neo4jConnection.session() as session:
            # Archive old episodic memories
            result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE c.started_at < datetime() - duration('P' + $days + 'D')
              AND c.archived IS NULL
            SET c.archived = true, t.archived = true
            RETURN count(DISTINCT c) AS archived_count
        """, days=str(retention_days))
            record = result.single()
            archived_count = record["archived_count"] if record else 0

            # Delete very low salience entities
            result = session.run("""
            MATCH (e:Entity)
            WHERE e.salience < 0.1
              AND e.last_accessed < datetime() - duration('P30D')
              AND NOT EXISTS { (e)<-[:ABOUT]-(:Fact) }
            DETACH DELETE e
            RETURN count(e) AS deleted_count
        """)
            record = result.single()
            deleted_count = record["deleted_count"] if record else 0
        return archived_count, deleted_count

    def reset_consolidation_markers(self) -> int:
        """Clear the consolidation markers on every conversation and turn, so the next
        sweep consolidates everything again. Returns how many conversations were reset."""
        with Neo4jConnection.session() as session:
            # Reset consolidation timestamps (conversation-level coarse markers)
            result = session.run("""
            MATCH (c:Conversation)
            WHERE c.consolidated IS NOT NULL OR c.self_consolidated IS NOT NULL
            REMOVE c.consolidated, c.self_consolidated
            RETURN count(c) AS reset_count
        """)
            record = result.single()
            reset_count = record["reset_count"] if record else 0

            # Turn-level markers are the real gate now — clear them (batched) so the
            # next sweep actually re-consolidates every turn. Without this, reset is a
            # no-op (turns stay marked → nothing reprocesses).
            session.run("""
            MATCH (t:Turn)
            WHERE t.consolidated IS NOT NULL OR t.self_consolidated IS NOT NULL
            CALL { WITH t REMOVE t.consolidated, t.self_consolidated }
            IN TRANSACTIONS OF 10000 ROWS
        """).consume()
        return reset_count

    def delete_semantic_memories(self) -> tuple[int, int, int]:
        """DESTROY every fact, entity and strategy (turns and conversations stay).
        Returns ``(facts, entities, strategies)`` deleted."""
        with Neo4jConnection.session() as session:
            # Delete all Facts
            result = session.run("""
                MATCH (f:Fact)
                DETACH DELETE f
                RETURN count(f) AS deleted
            """)
            record = result.single()
            facts_deleted = record["deleted"] if record else 0

            # Delete all Entities
            result = session.run("""
                MATCH (e:Entity)
                DETACH DELETE e
                RETURN count(e) AS deleted
            """)
            record = result.single()
            entities_deleted = record["deleted"] if record else 0

            # Delete all Strategies
            result = session.run("""
                MATCH (s:Strategy)
                DETACH DELETE s
                RETURN count(s) AS deleted
            """)
            record = result.single()
            strategies_deleted = record["deleted"] if record else 0
        return facts_deleted, entities_deleted, strategies_deleted

    # Promotion to _global: candidates come from any user's channel.

    def promotable_entities(
        self, min_confidence: float, min_access: int, min_conversations: int,
    ) -> list[Any]:
        """Entities outside ``_global`` that meet the promotion bar, with the number of
        conversations that mention them (``conv_count``)."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
            MATCH (e:Entity)
            WHERE e.channel IS NOT NULL
              AND e.channel <> '_global'
              AND coalesce(e.salience, 0) >= $min_confidence
              AND coalesce(e.access_count, 0) >= $min_access
            WITH e
            OPTIONAL MATCH (c:Conversation)-[:MENTIONS]->(e)
            WITH e, count(DISTINCT c) AS conv_count
            WHERE conv_count >= $min_conversations
            RETURN e.id AS id,
                   e.name AS name,
                   e.type AS type,
                   e.description AS description,
                   e.salience AS salience,
                   e.access_count AS access_count,
                   e.channel AS source_channel,
                   conv_count
        """, min_confidence=min_confidence, min_access=min_access, min_conversations=min_conversations))

    def global_entity_named(self, name: str) -> Any | None:
        """The ``_global`` entity called `name` (its ``id`` and ``salience``), if any."""
        with Neo4jConnection.session() as session:
            return session.run("""
                    MATCH (e:Entity {name: $name, channel: '_global'})
                    RETURN e.id AS id, e.salience AS salience
                """, name=name).single()

    def raise_promoted_entity(self, entity_id: str, salience: float, source_channel: str) -> None:
        """Raise a ``_global`` entity's salience and record where the promotion came from."""
        with Neo4jConnection.session() as session:
            session.run("""
                            MATCH (e:Entity {id: $id})
                            SET e.salience = $salience,
                                e.last_promoted = datetime(),
                                e.promoted_from = coalesce(e.promoted_from, []) + $source
                        """, id=entity_id, salience=salience, source=source_channel).consume()

    def create_global_entity(
        self, entity_id: str, name: str, entity_type: str | None, description: str | None,
        salience: float, source_channel: str,
    ) -> None:
        """Create a ``_global`` copy of an entity promoted from `source_channel`."""
        with Neo4jConnection.session() as session:
            session.run("""
                        CREATE (e:Entity {
                            id: $id,
                            name: $name,
                            type: $type,
                            description: $description,
                            salience: $salience,
                            access_count: 0,
                            channel: '_global',
                            promoted_from: [$source],
                            promoted_at: datetime(),
                            created_at: datetime()
                        })
                    """,
                id=entity_id,
                name=name,
                type=entity_type,
                description=description,
                salience=salience,
                source=source_channel
            ).consume()

    def promotable_facts(self, min_confidence: float, min_access: int) -> list[Any]:
        """Facts outside ``_global`` that meet the promotion bar."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
            MATCH (f:Fact)
            WHERE f.channel IS NOT NULL
              AND f.channel <> '_global'
              AND coalesce(f.confidence, 0) >= $min_confidence
              AND coalesce(f.access_count, 0) >= $min_access
            RETURN f.id AS id,
                   f.claim AS claim,
                   f.confidence AS confidence,
                   f.access_count AS access_count,
                   f.source AS source,
                   f.channel AS source_channel
        """, min_confidence=min_confidence, min_access=min_access))

    def global_fact_with_claim(self, claim: str) -> Any | None:
        """The ``_global`` fact with exactly this claim (its ``id`` and ``confidence``), if any."""
        with Neo4jConnection.session() as session:
            return session.run("""
                    MATCH (f:Fact {claim: $claim, channel: '_global'})
                    RETURN f.id AS id, f.confidence AS confidence
                """, claim=claim).single()

    def raise_promoted_fact(self, fact_id: str, confidence: float, source_channel: str) -> None:
        """Raise a ``_global`` fact's confidence and record where the promotion came from."""
        with Neo4jConnection.session() as session:
            session.run("""
                            MATCH (f:Fact {id: $id})
                            SET f.confidence = $confidence,
                                f.last_promoted = datetime(),
                                f.promoted_from = coalesce(f.promoted_from, []) + $source
                        """, id=fact_id, confidence=confidence, source=source_channel).consume()

    def create_global_fact(
        self, fact_id: str, claim: str, confidence: float, fact_source: str | None,
        source_channel: str,
    ) -> None:
        """Create a ``_global`` copy of a fact promoted from `source_channel`."""
        with Neo4jConnection.session() as session:
            session.run("""
                        CREATE (f:Fact {
                            id: $id,
                            claim: $claim,
                            confidence: $confidence,
                            access_count: 0,
                            source: $fact_source,
                            channel: '_global',
                            promoted_from: [$source_channel],
                            promoted_at: datetime(),
                            created_at: datetime()
                        })
                    """,
                id=fact_id,
                claim=claim,
                confidence=confidence,
                fact_source=fact_source,
                source_channel=source_channel
            ).consume()

    # Procedural patterns, conversation ownership, backfills.

    def successful_conversations_without_patterns(self) -> list[Any]:
        """Up to 20 conversations with a successful outcome whose tool pattern hasn't
        been extracted yet: ``conversation_id``, ``tools``, ``task_type``."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
            MATCH (c:Conversation)-[:RESULTED_IN]->(o:Outcome {success: true})
            WHERE c.patterns_extracted IS NULL
            WITH c, o
            MATCH (c)-[:USED_TOOL]->(inv:ToolInvocation)-[:INVOKED]->(t:Tool)
            WITH c, o, collect(DISTINCT t.name) AS tools
            WHERE size(tools) > 0
            RETURN c.id AS conversation_id,
                   tools,
                   o.task_type AS task_type
            LIMIT 20
        """))

    def record_pattern_success(
        self, task_type: str, tools: list[str], description: str, embedding: list[float],
        conversation_id: str,
    ) -> None:
        """Count a success for the strategy (task type + tool sequence), creating it on
        first sight, link it to the conversation and mark the conversation done."""
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (s:Strategy {context_pattern: $task_type, tool_sequence: $tools})
                ON CREATE SET
                    s.id = randomUUID(),
                    s.description = $description,
                    s.embedding = $embedding,
                    s.success_count = 1,
                    s.failure_count = 0,
                    s.created_at = datetime()
                ON MATCH SET
                    s.success_count = s.success_count + 1,
                    s.last_used = datetime()

                WITH s
                MATCH (c:Conversation {id: $conv_id})
                MERGE (s)-[:SUCCEEDED_IN]->(c)
                SET c.patterns_extracted = true
            """,
                task_type=task_type,
                tools=tools,
                description=description,
                embedding=embedding,
                conv_id=conversation_id
            ).consume()

    def owner_of_conversation(self, conversation_id: str) -> str | None:
        """The id of the user who owns the conversation, or None if it has no owner."""
        with Neo4jConnection.session() as session:
            rec = session.run(
                """
                MATCH (u:User)-[:HAS_CONVERSATION]->(c:Conversation {id: $id})
                RETURN u.id AS uid LIMIT 1
            """,
                id=conversation_id,
            ).single()
        return rec["uid"] if rec and rec["uid"] else None

    def entities_missing_embeddings(self, limit: int) -> list[Any]:
        """Up to `limit` named entities stored without an embedding: ``id``, ``name``,
        ``description``, ``type``."""
        with Neo4jConnection.session() as session:
            return list(session.run(
                """
            MATCH (e:Entity)
            WHERE e.embedding IS NULL AND e.name IS NOT NULL
            RETURN e.id AS id, e.name AS name, e.description AS description,
                   coalesce(e.type, 'Concept') AS type
            LIMIT $limit
            """, limit=limit,
            ))

    def set_entity_embeddings(self, items: list[dict[str, Any]]) -> None:
        """Write ``{id, embedding}`` pairs onto their entities."""
        with Neo4jConnection.session() as session:
            session.run(
                """
            UNWIND $items AS item
            MATCH (e:Entity {id: item.id})
            SET e.embedding = item.embedding
            """,
                items=items,
            ).consume()

    def orphan_facts(self, cap: int) -> list[Any]:
        """Up to `cap` facts that are about no entity, across all users: ``fact_id``,
        ``claim``, ``user_id``, ``channel``."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
            MATCH (f:Fact)
            WHERE NOT (f)-[:ABOUT]->(:Entity)
            RETURN f.id AS fact_id,
                   f.claim AS claim,
                   coalesce(f.user_id, 'default') AS user_id,
                   coalesce(f.channel, '_default') AS channel
            ORDER BY f.user_id, f.channel, f.id
            LIMIT $cap
        """, cap=cap))

    def link_fact_to_entities(self, fact_id: str, entity_ids: list[str]) -> None:
        """Add ``ABOUT`` edges from a fact to existing entities (backfill name match)."""
        with Neo4jConnection.session() as session:
            session.run("""
                    MATCH (f:Fact {id: $fact_id})
                    UNWIND $entity_ids AS eid
                    MATCH (e:Entity {id: eid})
                    MERGE (f)-[r:ABOUT]->(e)
                    ON CREATE SET r.confidence = 0.85,
                                  r.linked_at = datetime(),
                                  r.method = 'backfill_namematch'
                """, fact_id=fact_id, entity_ids=entity_ids).consume()


    # The consolidation sweep's discovery: which conversations have turns to consolidate.
    # The per-conversation work then runs through memory/consolidation_store.py.

    def conversation_census(self) -> list[Any]:
        """Up to 20 conversations with their ``consolidated`` marker and
        ``turn_count``, for the sweep's debug log."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
        MATCH (c:Conversation)
        OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
        RETURN c.id AS id, c.consolidated AS consolidated, count(t) AS turn_count
        LIMIT 20
    """))

    def pending_user_conversations(self, only_conversation_id: str | None) -> list[Any]:
        """The 10 most recently started conversations with user turns not yet
        consolidated: ``conversation_id``, ``user_id``, ``channel``, ``agent_id`` and
        ``turns`` (``id``, ``content`` and ``responder_agent_id``, the agent of the
        turn that follows, in turn order). Only the unmarked user turns are collected,
        so a fully consolidated conversation yields nothing. `only_conversation_id`
        restricts the search to one conversation."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
        MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
        WHERE t.role = 'user'
          AND t.consolidated IS NULL
          AND ($only IS NULL OR c.id = $only)
        OPTIONAL MATCH (u:User)-[:HAS_CONVERSATION]->(c)
        OPTIONAL MATCH (t)-[:FOLLOWED_BY]->(resp:Turn)
        WITH c, u, t, resp
        ORDER BY t.index
        WITH c, u, collect({id: t.id, content: t.content, responder_agent_id: resp.agent_id}) AS turns
        ORDER BY c.started_at DESC
        LIMIT 10
        RETURN c.id AS conversation_id,
               coalesce(u.id, 'default') AS user_id,
               coalesce(c.channel, '_default') AS channel,
               c.agent_id AS agent_id,
               turns
    """, only=only_conversation_id))

    def pending_assistant_conversations(self, only_conversation_id: str | None) -> list[Any]:
        """The 10 most recently started conversations, with an agent, whose
        assistant turns aren't yet self-consolidated: ``conversation_id``,
        ``user_id``, ``channel``, ``agent_id`` and ``turns`` (``content``, ``id``,
        ``agent_id``). Only the unmarked assistant turns are collected.
        `only_conversation_id` restricts the search to one conversation."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE t.role = 'assistant'
              AND t.self_consolidated IS NULL
              AND c.agent_id IS NOT NULL
              AND ($only IS NULL OR c.id = $only)
            OPTIONAL MATCH (u:User)-[:HAS_CONVERSATION]->(c)
            WITH c, u, collect(t) AS turns
            ORDER BY c.started_at DESC
            LIMIT 10
            RETURN c.id AS conversation_id,
                   coalesce(u.id, 'default') AS user_id,
                   coalesce(c.channel, '_default') AS channel,
                   c.agent_id AS agent_id,
                   [t IN turns | {content: t.content, id: t.id, agent_id: t.agent_id}] AS turns
        """, only=only_conversation_id))

    # Entity maintenance: deduplication, and the agent-attribution backfill.

    @contextmanager
    def entity_maintenance(self) -> Iterator[EntityMaintenance]:
        """One transaction for entity dedupe and merge (`EntityMaintenance`). Nothing
        persists unless the caller calls ``commit()``; leaving the block any other
        way, including by an exception, rolls the transaction back."""
        from .entity_maintenance import EntityMaintenance

        with Neo4jConnection.session() as session:
            tx = session.begin_transaction()
            try:
                yield EntityMaintenance(tx)
            finally:
                if not tx.closed():
                    tx.rollback()

    def generic_agent_fact_candidates(self, channel: str) -> list[Any]:
        """Facts in `channel`, from every user, whose claim starts with a generic
        "agent" or "the agent" (a cheap prefilter; the caller decides): ``id``,
        ``claim``, ``user_id``. Claims are stored trimmed."""
        with Neo4jConnection.session() as session:
            return list(session.run("""
MATCH (f:Fact {channel: $channel})
WHERE f.claim IS NOT NULL
  AND (toLower(f.claim) STARTS WITH 'agent'
       OR toLower(f.claim) STARTS WITH 'the agent')
RETURN f.id AS id, f.claim AS claim, f.user_id AS user_id
""", channel=channel))

    def attribute_fact_to_agent(
        self, fact_id: str, claim: str, claim_hash: str, agent_entity_id: str,
    ) -> None:
        """Rewrite a fact's claim (and its hash) to name the agent, and link the fact
        to the agent's entity. Re-running adds no second link."""
        with Neo4jConnection.session() as session:
            session.run(
                """
                            MATCH (f:Fact {id: $id})
                            SET f.claim = $claim, f.claim_hash = $hash
                            WITH f
                            MATCH (e:Entity {id: $eid})
                            MERGE (f)-[:ABOUT]->(e)
                            """,
                id=fact_id,
                claim=claim,
                hash=claim_hash,
                eid=agent_entity_id,
            ).consume()

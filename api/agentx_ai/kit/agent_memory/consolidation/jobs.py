"""Background consolidation jobs for memory processing."""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, TYPE_CHECKING
from uuid import uuid4
import logging
import json
import re

from ..connections import Neo4jConnection, get_postgres_session, RedisConnection
from ..embeddings import get_embedder
from ..config import get_settings
from ..extraction.service import get_extraction_service
from ..models import Entity

if TYPE_CHECKING:
    from ..memory.interface import AgentMemory

logger = logging.getLogger(__name__)
settings = get_settings()


def _is_duplicate_fact(session, claim: str, user_id: str, channel: str) -> bool:
    """
    Check if a fact with the same or very similar claim already exists.

    Args:
        session: Neo4j session
        claim: The fact claim text
        user_id: User ID
        channel: Memory channel

    Returns:
        True if a duplicate exists
    """
    # Normalize the claim for comparison
    normalized = claim.strip().lower()

    # Check for exact or near-exact matches
    result = session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND toLower(trim(f.claim)) = $normalized_claim
        RETURN f.id AS id
        LIMIT 1
    """, user_id=user_id, channel=channel, normalized_claim=normalized)

    return result.single() is not None


def _get_memory_for_user(user_id: str, channel: str = "_default") -> "AgentMemory":
    """
    Get or create an AgentMemory instance for a user.

    Args:
        user_id: User ID
        channel: Memory channel (default: "_default")

    Returns:
        AgentMemory instance
    """
    from ..memory.interface import AgentMemory
    return AgentMemory(user_id=user_id, channel=channel)


def consolidate_episodic_to_semantic() -> Dict[str, Any]:
    """
    Extract entities, facts, and relationships from recent episodic memory
    and store in semantic memory using the AgentMemory interface.

    Returns:
        Dictionary with consolidation metrics
    """
    # Cache memory instances per user to avoid repeated initialization
    memory_cache: Dict[str, "AgentMemory"] = {}

    # Metrics for logging
    total_entities = 0
    total_facts = 0
    total_relationships = 0
    total_conversations = 0
    errors: List[str] = []

    with Neo4jConnection.session() as session:
        # First, check what conversations exist at all (for debugging)
        debug_result = session.run("""
            MATCH (c:Conversation)
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            RETURN c.id AS id, c.consolidated AS consolidated, count(t) AS turn_count
            LIMIT 20
        """)
        conversations_found = list(debug_result)
        logger.info(f"Consolidation: Found {len(conversations_found)} conversations in Neo4j")
        for conv in conversations_found:
            logger.debug(f"  - {conv['id']}: {conv['turn_count']} turns, consolidated={conv['consolidated']}")

        # Get recent conversations not yet processed, including user info
        # Only extract from user turns (not assistant/system/tool responses)
        result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE (c.consolidated IS NULL OR c.consolidated < datetime() - duration('PT15M'))
              AND t.role = 'user'
            OPTIONAL MATCH (u:User)-[:HAS_CONVERSATION]->(c)
            WITH c, u, collect(t) AS turns
            ORDER BY c.started_at DESC
            LIMIT 10
            RETURN c.id AS conversation_id,
                   coalesce(u.id, 'default') AS user_id,
                   coalesce(c.channel, '_default') AS channel,
                   [t IN turns | {content: t.content}] AS turns
        """)

        records = list(result)
        logger.info(f"Consolidation: {len(records)} conversations need processing")

        for record in records:
            conv_id = record["conversation_id"]
            user_id = record["user_id"]
            channel = record["channel"]
            turns = record["turns"]

            total_conversations += 1

            # Get or create memory instance for this user/channel
            cache_key = f"{user_id}:{channel}"
            if cache_key not in memory_cache:
                memory_cache[cache_key] = _get_memory_for_user(user_id, channel)
            memory = memory_cache[cache_key]

            # Filter turns by relevance (skip "thanks", "ok", etc.)
            extraction_service = get_extraction_service()
            relevant_turns = []
            for turn in turns:
                content = turn['content']
                relevance = extraction_service.check_relevance(content)
                if relevance.is_relevant:
                    relevant_turns.append(turn)
                else:
                    logger.debug(f"Skipping irrelevant turn: {content[:50]}... ({relevance.reason})")

            if not relevant_turns:
                logger.debug(f"No relevant turns in conversation {conv_id}, skipping extraction")
                # Still mark as consolidated
                session.run("""
                    MATCH (c:Conversation {id: $conv_id})
                    SET c.consolidated = datetime()
                """, conv_id=conv_id)
                continue

            # Combine relevant user turn content for extraction
            full_text = "\n".join(t['content'] for t in relevant_turns)

            # Single extraction call for entities, facts, and relationships
            try:
                extraction_result = extraction_service.extract_all(full_text)
                extracted_entities = extraction_result.entities
                extracted_facts = extraction_result.facts
                extracted_relationships = extraction_result.relationships
                logger.debug(
                    f"Extraction result: {len(extracted_entities)} entities, "
                    f"{len(extracted_facts)} facts, {len(extracted_relationships)} relationships"
                )
            except Exception as e:
                logger.warning(f"Extraction failed for {conv_id}: {e}")
                errors.append(f"extraction:{conv_id}:{e}")
                extracted_entities = []
                extracted_facts = []
                extracted_relationships = []

            entity_count = 0
            # Use lowercase keys for case-insensitive matching with relationships
            entity_map: Dict[str, str] = {}  # lowercase_name -> entity_id for relationship linking

            # Store entities via AgentMemory interface
            for entity_dict in extracted_entities:
                try:
                    entity = Entity(
                        id=str(uuid4()),
                        name=entity_dict["name"],
                        type=entity_dict["type"],
                        description=entity_dict.get("description"),
                        salience=entity_dict.get("confidence", 0.5),
                    )
                    memory.upsert_entity(entity)
                    # Store with lowercase key for case-insensitive lookup
                    entity_map[entity_dict["name"].lower()] = entity.id
                    entity_count += 1

                    # Create MENTIONS relationship (this is still direct Neo4j for the relationship)
                    session.run("""
                        MATCH (c:Conversation {id: $conv_id}), (e:Entity {id: $entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, conv_id=conv_id, entity_id=entity.id)
                except Exception as e:
                    logger.warning(f"Failed to store entity {entity_dict.get('name')}: {e}")

            total_entities += entity_count

            fact_count = 0
            skipped_duplicates = 0

            # Store facts via AgentMemory interface
            for fact_dict in extracted_facts:
                try:
                    claim = fact_dict["claim"]

                    # Check for duplicate facts before storing
                    if _is_duplicate_fact(session, claim, user_id, channel):
                        skipped_duplicates += 1
                        logger.debug(f"Skipping duplicate fact: {claim[:50]}...")
                        continue

                    # Link fact to mentioned entities (case-insensitive lookup)
                    entity_ids = [
                        entity_map[name.lower()]
                        for name in fact_dict.get("entity_names", [])
                        if name.lower() in entity_map
                    ]

                    fact = memory.learn_fact(
                        claim=claim,
                        source="extraction",
                        confidence=fact_dict.get("confidence", 0.7),
                        source_turn_id=fact_dict.get("source_turn_id"),
                        entity_ids=entity_ids if entity_ids else None,
                    )
                    fact_count += 1

                    # Create DERIVED_FROM relationship
                    session.run("""
                        MATCH (c:Conversation {id: $conv_id}), (f:Fact {id: $fact_id})
                        MERGE (f)-[:DERIVED_FROM]->(c)
                    """, conv_id=conv_id, fact_id=fact.id)
                except Exception as e:
                    logger.warning(f"Failed to store fact: {e}")

            total_facts += fact_count

            # Relationships already extracted from extract_all() above
            rel_count = 0

            # Store relationships in Neo4j (case-insensitive entity lookup)
            for rel in extracted_relationships:
                try:
                    source_id = entity_map.get(rel["source"].lower())
                    target_id = entity_map.get(rel["target"].lower())

                    if source_id and target_id:
                        session.run("""
                            MATCH (source:Entity {id: $source_id}),
                                  (target:Entity {id: $target_id})
                            MERGE (source)-[r:RELATES_TO {type: $rel_type}]->(target)
                            ON CREATE SET r.confidence = $confidence,
                                          r.created_at = datetime()
                            ON MATCH SET r.confidence =
                                CASE WHEN r.confidence < $confidence
                                     THEN $confidence
                                     ELSE r.confidence END
                        """,
                            source_id=source_id,
                            target_id=target_id,
                            rel_type=rel["type"],
                            confidence=rel.get("confidence", 0.7)
                        )
                        rel_count += 1
                except Exception as e:
                    logger.warning(f"Failed to store relationship: {e}")

            total_relationships += rel_count

            # Mark conversation as consolidated in finally block to ensure it's set
            # even if some extraction steps fail (prevents reprocessing)
            try:
                dup_msg = f", {skipped_duplicates} duplicates skipped" if skipped_duplicates > 0 else ""
                logger.info(
                    f"Consolidated conversation {conv_id}: "
                    f"{entity_count} entities, {fact_count} facts, {rel_count} relationships{dup_msg}"
                )
            finally:
                session.run("""
                    MATCH (c:Conversation {id: $conv_id})
                    SET c.consolidated = datetime()
                """, conv_id=conv_id)

    # Clean up memory cache to release resources
    memory_cache.clear()

    # Log summary metrics
    if total_conversations > 0:
        logger.info(
            f"Consolidation complete: {total_conversations} conversations, "
            f"{total_entities} entities, {total_facts} facts, "
            f"{total_relationships} relationships"
        )

    if errors:
        logger.warning(f"Consolidation had {len(errors)} errors: {errors[:5]}")

    return {
        "items_processed": total_conversations,
        "entities": total_entities,
        "facts": total_facts,
        "relationships": total_relationships,
        "errors": errors
    }


def detect_patterns() -> Dict[str, Any]:
    """
    Analyze successful conversations to extract procedural patterns.

    Returns:
        Dictionary with pattern detection metrics
    """
    patterns_extracted = 0

    with Neo4jConnection.session() as session:
        # Find conversations with successful outcomes
        result = session.run("""
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
        """)

        embedder = get_embedder()

        for record in result:
            conv_id = record["conversation_id"]
            tools = record["tools"]
            task_type = record.get("task_type", "general")

            # Create or update strategy
            strategy_desc = f"Use {', '.join(tools)} for {task_type} tasks"
            embedding = embedder.embed_single(strategy_desc)

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
                description=strategy_desc,
                embedding=embedding,
                conv_id=conv_id
            )

            patterns_extracted += 1
            logger.info(f"Extracted pattern from {conv_id}: {tools}")

    return {"items_processed": patterns_extracted}


def apply_memory_decay() -> Dict[str, Any]:
    """
    Apply time-based decay to memory salience scores.

    Returns:
        Dictionary with decay metrics
    """
    decay_rate = settings.salience_decay_rate
    entities_decayed = 0
    facts_decayed = 0

    with Neo4jConnection.session() as session:
        # Decay entity salience
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.last_accessed < datetime() - duration('P1D')
            SET e.salience = e.salience * $decay_rate
            RETURN count(e) AS decayed_count
        """, decay_rate=decay_rate)

        record = result.single()
        entities_decayed = record["decayed_count"] if record else 0
        logger.info(f"Decayed {entities_decayed} entities")

        # Decay fact confidence (slower decay)
        result = session.run("""
            MATCH (f:Fact)
            WHERE f.created_at < datetime() - duration('P7D')
              AND f.source = 'inferred'
            SET f.confidence = f.confidence * $decay_rate
            RETURN count(f) AS decayed_count
        """, decay_rate=decay_rate ** 0.5)  # Slower decay for facts

        record = result.single()
        facts_decayed = record["decayed_count"] if record else 0
        logger.info(f"Decayed {facts_decayed} facts")

    return {
        "items_processed": entities_decayed + facts_decayed,
        "entities_decayed": entities_decayed,
        "facts_decayed": facts_decayed
    }


def promote_to_global() -> Dict[str, Any]:
    """
    Promote high-quality facts/entities from project channels to _global.

    Criteria (all three must be met for entities):
    - Salience >= promotion_min_confidence
    - Access count >= promotion_min_access_count
    - Referenced in >= promotion_min_conversations conversations

    For facts:
    - Confidence >= promotion_min_confidence
    - Access count >= promotion_min_access_count

    Returns:
        Dictionary with promotion results
    """
    from ..audit import MemoryAuditLogger

    min_confidence = settings.promotion_min_confidence
    min_access = settings.promotion_min_access_count
    min_conversations = settings.promotion_min_conversations

    entities_promoted = 0
    facts_promoted = 0
    entities_updated = 0
    facts_updated = 0
    errors: List[str] = []

    audit_logger = MemoryAuditLogger(settings)

    with Neo4jConnection.session() as session:
        # Find entities meeting promotion criteria
        result = session.run("""
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
        """, min_confidence=min_confidence, min_access=min_access, min_conversations=min_conversations)

        for record in result:
            entity_name = record["name"]
            source_channel = record["source_channel"]

            try:
                # Check if entity already exists in _global
                existing = session.run("""
                    MATCH (e:Entity {name: $name, channel: '_global'})
                    RETURN e.id AS id, e.salience AS salience
                """, name=entity_name).single()

                if existing:
                    # Update salience if higher
                    if record["salience"] > existing["salience"]:
                        session.run("""
                            MATCH (e:Entity {id: $id})
                            SET e.salience = $salience,
                                e.last_promoted = datetime(),
                                e.promoted_from = coalesce(e.promoted_from, []) + $source
                        """, id=existing["id"], salience=record["salience"], source=source_channel)
                        entities_updated += 1
                else:
                    # Create new entity in _global
                    new_id = str(uuid4())
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
                        id=new_id,
                        name=entity_name,
                        type=record["type"],
                        description=record["description"],
                        salience=record["salience"],
                        source=source_channel
                    )
                    entities_promoted += 1

                    # Log promotion
                    audit_logger.log_promotion(
                        source_channel=source_channel,
                        promoted_ids=[new_id],
                        promoted_type="entity",
                        confidence=float(record["salience"]),
                        access_count=int(record["access_count"]),
                        conversation_count=int(record["conv_count"]),
                    )

            except Exception as e:
                logger.warning(f"Failed to promote entity {entity_name}: {e}")
                errors.append(f"entity:{entity_name}:{e}")

        # Find facts meeting promotion criteria
        result = session.run("""
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
        """, min_confidence=min_confidence, min_access=min_access)

        for record in result:
            claim = record["claim"]
            source_channel = record["source_channel"]

            try:
                # Check if fact already exists in _global (by claim text)
                existing = session.run("""
                    MATCH (f:Fact {claim: $claim, channel: '_global'})
                    RETURN f.id AS id, f.confidence AS confidence
                """, claim=claim).single()

                if existing:
                    # Update confidence if higher
                    if record["confidence"] > existing["confidence"]:
                        session.run("""
                            MATCH (f:Fact {id: $id})
                            SET f.confidence = $confidence,
                                f.last_promoted = datetime(),
                                f.promoted_from = coalesce(f.promoted_from, []) + $source
                        """, id=existing["id"], confidence=record["confidence"], source=source_channel)
                        facts_updated += 1
                else:
                    # Create new fact in _global
                    new_id = str(uuid4())
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
                        id=new_id,
                        claim=claim,
                        confidence=record["confidence"],
                        fact_source=record["source"],
                        source_channel=source_channel
                    )
                    facts_promoted += 1

                    # Log promotion
                    audit_logger.log_promotion(
                        source_channel=source_channel,
                        promoted_ids=[new_id],
                        promoted_type="fact",
                        confidence=float(record["confidence"]),
                        access_count=int(record["access_count"]),
                        conversation_count=0,  # Facts don't track conversation count
                    )

            except Exception as e:
                logger.warning(f"Failed to promote fact: {e}")
                errors.append(f"fact:{claim[:50]}:{e}")

    total_promoted = entities_promoted + facts_promoted
    total_updated = entities_updated + facts_updated

    if total_promoted > 0 or total_updated > 0:
        logger.info(
            f"Promotion complete: {entities_promoted} entities promoted, "
            f"{facts_promoted} facts promoted, {entities_updated} entities updated, "
            f"{facts_updated} facts updated"
        )

    if errors:
        logger.warning(f"Promotion had {len(errors)} errors: {errors[:5]}")

    return {
        "items_processed": total_promoted + total_updated,
        "entities_promoted": entities_promoted,
        "facts_promoted": facts_promoted,
        "entities_updated": entities_updated,
        "facts_updated": facts_updated,
        "errors": errors
    }


def cleanup_old_memories() -> Dict[str, Any]:
    """Archive or delete old, low-salience memories."""
    retention_days = settings.episodic_retention_days
    archived_count = 0
    deleted_count = 0

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
        logger.info(f"Archived {archived_count} old conversations")

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
        logger.info(f"Cleaned up {deleted_count} low-salience orphan entities")

    return {
        "archived_conversations": archived_count,
        "deleted_entities": deleted_count,
    }


def reset_consolidation(delete_memories: bool = False) -> Dict[str, Any]:
    """
    Reset consolidation timestamps for all conversations.

    This allows all conversations to be reprocessed by the consolidation job.
    Useful when extraction logic has changed or to rebuild semantic memory.

    Args:
        delete_memories: If True, also delete all entities, facts, and relationships
                        (but keep turns/conversations for their embeddings)

    Returns:
        Dictionary with reset metrics
    """
    memories_deleted = 0

    with Neo4jConnection.session() as session:
        # Reset consolidation timestamps
        result = session.run("""
            MATCH (c:Conversation)
            WHERE c.consolidated IS NOT NULL
            REMOVE c.consolidated
            RETURN count(c) AS reset_count
        """)

        record = result.single()
        reset_count = record["reset_count"] if record else 0

        if delete_memories:
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

            memories_deleted = facts_deleted + entities_deleted + strategies_deleted
            logger.info(
                f"Deleted memories: {facts_deleted} facts, {entities_deleted} entities, "
                f"{strategies_deleted} strategies"
            )

        logger.info(f"Reset consolidation for {reset_count} conversations")

        return {
            "conversations_reset": reset_count,
            "memories_deleted": memories_deleted if delete_memories else None,
            "success": True
        }


def trigger_reflection(
    conversation_id: str,
    user_id: str,
    outcome: Dict[str, Any]
) -> None:
    """
    Trigger reflection job for a conversation.
    Adds to Redis queue for async processing.

    Args:
        conversation_id: Conversation ID
        user_id: User ID
        outcome: Outcome dictionary
    """
    redis = RedisConnection.get_client()

    job_data = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "outcome": outcome,
        "triggered_at": datetime.now(timezone.utc).isoformat()
    }

    redis.xadd("reflection_jobs", {"data": json.dumps(job_data)})
    logger.info(f"Queued reflection for conversation {conversation_id}")


def manage_audit_partitions() -> Dict[str, Any]:
    """
    Manage audit log partitions.
    Creates future partitions and drops old ones based on retention settings.

    Returns:
        Dictionary with partition management results
    """
    from sqlalchemy import text

    retention_days = settings.audit_retention_days
    ahead_days = settings.audit_partition_ahead_days

    partitions_created = 0
    partitions_dropped = 0
    errors: List[str] = []

    with get_postgres_session() as session:
        # Create future partitions
        for day_offset in range(ahead_days + 1):
            partition_date = datetime.utcnow().date() + timedelta(days=day_offset)
            next_date = partition_date + timedelta(days=1)

            partition_name = f"memory_audit_log_{partition_date.strftime('%Y%m%d')}"

            try:
                # Check if partition already exists
                result = session.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_class
                        WHERE relname = :partition_name
                        AND relkind = 'r'
                    )
                """), {"partition_name": partition_name})

                exists = result.scalar()

                if not exists:
                    # Validate partition name to prevent SQL injection
                    # (should only contain alphanumeric and underscore)
                    if not re.match(r'^[a-zA-Z0-9_]+$', partition_name):
                        logger.error(f"Invalid partition name format: {partition_name}")
                        errors.append(f"invalid_name:{partition_name}")
                        continue

                    # Create partition - use identifier quoting for safety
                    session.execute(text(f"""
                        CREATE TABLE IF NOT EXISTS "{partition_name}"
                        PARTITION OF memory_audit_log
                        FOR VALUES FROM ('{partition_date.isoformat()}')
                        TO ('{next_date.isoformat()}')
                    """))
                    partitions_created += 1
                    logger.info(f"Created audit partition: {partition_name}")
            except Exception as e:
                logger.warning(f"Failed to create partition {partition_name}: {e}")
                errors.append(f"create:{partition_name}:{e}")

        # Drop old partitions beyond retention
        cutoff_date = datetime.utcnow().date() - timedelta(days=retention_days)

        try:
            # Get list of existing partitions
            result = session.execute(text("""
                SELECT relname FROM pg_class c
                JOIN pg_inherits i ON c.oid = i.inhrelid
                JOIN pg_class p ON i.inhparent = p.oid
                WHERE p.relname = 'memory_audit_log'
                AND c.relkind = 'r'
                ORDER BY relname
            """))

            for row in result:
                partition_name = row[0]
                # Extract date from partition name (format: memory_audit_log_YYYYMMDD)
                try:
                    date_str = partition_name.replace("memory_audit_log_", "")
                    partition_date = datetime.strptime(date_str, "%Y%m%d").date()

                    if partition_date < cutoff_date:
                        # Validate partition name before dropping
                        if re.match(r'^[a-zA-Z0-9_]+$', partition_name):
                            session.execute(text(f'DROP TABLE IF EXISTS "{partition_name}"'))
                            partitions_dropped += 1
                            logger.info(f"Dropped old audit partition: {partition_name}")
                except ValueError:
                    # Skip partitions with unexpected naming
                    continue
        except Exception as e:
            logger.warning(f"Failed to drop old partitions: {e}")
            errors.append(f"drop:{e}")

    logger.info(
        f"Audit partition management: created={partitions_created}, "
        f"dropped={partitions_dropped}, errors={len(errors)}"
    )

    return {
        "items_processed": partitions_created + partitions_dropped,
        "partitions_created": partitions_created,
        "partitions_dropped": partitions_dropped,
        "errors": errors
    }


def link_facts_to_entities() -> Dict[str, Any]:
    """
    Link unlinked facts to existing entities using embedding similarity.

    Finds facts without entity connections, extracts potential entity mentions
    from the claim text, and creates ABOUT relationships to matching entities.

    Returns:
        Dictionary with linking metrics
    """
    if not settings.entity_linking_enabled:
        logger.debug("Entity linking is disabled")
        return {"items_processed": 0, "links_created": 0, "skipped": "disabled"}

    embedder = get_embedder()
    threshold = settings.entity_linking_similarity_threshold

    links_created = 0
    facts_processed = 0
    errors = []

    with Neo4jConnection.session() as session:
        # Find facts that have no ABOUT relationships to entities
        result = session.run("""
            MATCH (f:Fact)
            WHERE NOT (f)-[:ABOUT]->(:Entity)
              AND f.created_at > datetime() - duration('P7D')
            RETURN f.id AS fact_id,
                   f.claim AS claim,
                   f.user_id AS user_id,
                   f.channel AS channel
            LIMIT 100
        """)

        facts_to_link = list(result)
        logger.info(f"Entity linking: found {len(facts_to_link)} unlinked facts")

        for record in facts_to_link:
            fact_id = record["fact_id"]
            claim = record["claim"]
            user_id = record["user_id"]
            facts_processed += 1

            try:
                # Embed the fact claim
                claim_embedding = embedder.embed_single(claim)

                # Find similar entities by vector search
                # Note: This requires Neo4j vector index on entity embeddings
                entity_result = session.run("""
                    CALL db.index.vector.queryNodes('entity_embeddings', 5, $embedding)
                    YIELD node, score
                    WHERE node.user_id = $user_id
                      AND score >= $threshold
                    RETURN node.id AS entity_id,
                           node.name AS entity_name,
                           score
                    ORDER BY score DESC
                    LIMIT 3
                """, embedding=claim_embedding, user_id=user_id, threshold=threshold)

                matching_entities: list[dict] = [dict(r) for r in entity_result]

                # Also try text-based matching for entity names mentioned in claim
                claim_lower = claim.lower()
                text_result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.user_id = $user_id
                      AND toLower(e.name) IN $words
                    RETURN e.id AS entity_id, e.name AS entity_name
                """, user_id=user_id, words=claim_lower.split())

                for text_match in text_result:
                    if text_match["entity_id"] not in [m["entity_id"] for m in matching_entities]:
                        matching_entities.append({
                            "entity_id": text_match["entity_id"],
                            "entity_name": text_match["entity_name"],
                            "score": 0.8  # Text match gets decent confidence
                        })

                # Create ABOUT relationships for matches
                for match in matching_entities:
                    entity_id = match["entity_id"]
                    entity_name = match["entity_name"]
                    score = match.get("score", 0.8)

                    session.run("""
                        MATCH (f:Fact {id: $fact_id}), (e:Entity {id: $entity_id})
                        MERGE (f)-[r:ABOUT]->(e)
                        SET r.confidence = $score,
                            r.linked_at = datetime(),
                            r.method = 'auto_embedding'
                    """, fact_id=fact_id, entity_id=entity_id, score=score)

                    links_created += 1
                    logger.debug(f"Linked fact '{claim[:50]}...' to entity '{entity_name}'")

            except Exception as e:
                logger.warning(f"Failed to link fact {fact_id}: {e}")
                errors.append(f"{fact_id}:{e}")

    logger.info(f"Entity linking complete: {facts_processed} facts processed, {links_created} links created")

    return {
        "items_processed": facts_processed,
        "links_created": links_created,
        "errors": errors
    }

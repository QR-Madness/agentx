"""Background consolidation jobs for memory processing."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from uuid import uuid4
import logging
import json

from ..connections import Neo4jConnection, get_postgres_session, RedisConnection
from ..embeddings import get_embedder
from ..config import get_settings
from ..extraction.entities import extract_entities
from ..extraction.facts import extract_facts
from ..models import Entity

if TYPE_CHECKING:
    from ..memory.interface import AgentMemory

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_memory_for_user(user_id: str, channel: str = "_global") -> "AgentMemory":
    """
    Get or create an AgentMemory instance for a user.

    Args:
        user_id: User ID
        channel: Memory channel (default: "_global")

    Returns:
        AgentMemory instance
    """
    from ..memory.interface import AgentMemory
    return AgentMemory(user_id=user_id, channel=channel)


def consolidate_episodic_to_semantic():
    """
    Extract entities and facts from recent episodic memory
    and store in semantic memory using the AgentMemory interface.
    """
    # Cache memory instances per user to avoid repeated initialization
    memory_cache: Dict[str, "AgentMemory"] = {}

    with Neo4jConnection.session() as session:
        # Get recent conversations not yet processed, including user info
        result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE NOT EXISTS(c.consolidated)
              OR c.consolidated < datetime() - duration('PT15M')
            OPTIONAL MATCH (u:User)-[:HAS_CONVERSATION]->(c)
            WITH c, u, collect(t) AS turns
            ORDER BY c.started_at DESC
            LIMIT 10
            RETURN c.id AS conversation_id,
                   coalesce(u.id, 'system') AS user_id,
                   coalesce(c.channel, '_global') AS channel,
                   [t IN turns | {role: t.role, content: t.content}] AS turns
        """)

        for record in result:
            conv_id = record["conversation_id"]
            user_id = record["user_id"]
            channel = record["channel"]
            turns = record["turns"]

            # Get or create memory instance for this user/channel
            cache_key = f"{user_id}:{channel}"
            if cache_key not in memory_cache:
                memory_cache[cache_key] = _get_memory_for_user(user_id, channel)
            memory = memory_cache[cache_key]

            # Combine turn content for extraction
            full_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in turns
            )

            # Entity extraction
            extracted_entities = extract_entities(full_text)
            entity_count = 0

            # Store entities via AgentMemory interface
            for entity_dict in extracted_entities:
                try:
                    entity = Entity(
                        id=str(uuid4()),
                        name=entity_dict["name"],
                        type=entity_dict["type"],
                        description=entity_dict.get("description"),
                        salience=0.5,
                    )
                    memory.upsert_entity(entity)
                    entity_count += 1

                    # Create MENTIONS relationship (this is still direct Neo4j for the relationship)
                    session.run("""
                        MATCH (c:Conversation {id: $conv_id}), (e:Entity {id: $entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, conv_id=conv_id, entity_id=entity.id)
                except Exception as e:
                    logger.warning(f"Failed to store entity {entity_dict.get('name')}: {e}")

            # Fact extraction
            extracted_facts = extract_facts(full_text)
            fact_count = 0

            # Store facts via AgentMemory interface
            for fact_dict in extracted_facts:
                try:
                    fact = memory.learn_fact(
                        claim=fact_dict["claim"],
                        source="extraction",
                        confidence=fact_dict.get("confidence", 0.7),
                        source_turn_id=None,  # Could link to specific turns if we tracked them
                    )
                    fact_count += 1

                    # Create DERIVED_FROM relationship
                    session.run("""
                        MATCH (c:Conversation {id: $conv_id}), (f:Fact {id: $fact_id})
                        MERGE (f)-[:DERIVED_FROM]->(c)
                    """, conv_id=conv_id, fact_id=fact.id)
                except Exception as e:
                    logger.warning(f"Failed to store fact: {e}")

            # Mark conversation as consolidated
            session.run("""
                MATCH (c:Conversation {id: $conv_id})
                SET c.consolidated = datetime()
            """, conv_id=conv_id)

            logger.info(f"Consolidated conversation {conv_id}: {entity_count} entities, {fact_count} facts")


def detect_patterns():
    """Analyze successful conversations to extract procedural patterns."""
    with Neo4jConnection.session() as session:
        # Find conversations with successful outcomes
        result = session.run("""
            MATCH (c:Conversation)-[:RESULTED_IN]->(o:Outcome {success: true})
            WHERE NOT EXISTS(c.patterns_extracted)
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

            logger.info(f"Extracted pattern from {conv_id}: {tools}")


def apply_memory_decay():
    """Apply time-based decay to memory salience scores."""
    decay_rate = settings.salience_decay_rate

    with Neo4jConnection.session() as session:
        # Decay entity salience
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.last_accessed < datetime() - duration('P1D')
            SET e.salience = e.salience * $decay_rate
            RETURN count(e) AS decayed_count
        """, decay_rate=decay_rate)

        count = result.single()["decayed_count"]
        logger.info(f"Decayed {count} entities")

        # Decay fact confidence (slower decay)
        result = session.run("""
            MATCH (f:Fact)
            WHERE f.created_at < datetime() - duration('P7D')
              AND f.source = 'inferred'
            SET f.confidence = f.confidence * $decay_rate
            RETURN count(f) AS decayed_count
        """, decay_rate=decay_rate ** 0.5)  # Slower decay for facts

        count = result.single()["decayed_count"]
        logger.info(f"Decayed {count} facts")


def cleanup_old_memories():
    """Archive or delete old, low-salience memories."""
    retention_days = settings.episodic_retention_days

    with Neo4jConnection.session() as session:
        # Archive old episodic memories
        result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE c.started_at < datetime() - duration('P' + $days + 'D')
              AND NOT EXISTS(c.archived)
            SET c.archived = true, t.archived = true
            RETURN count(DISTINCT c) AS archived_count
        """, days=str(retention_days))

        count = result.single()["archived_count"]
        logger.info(f"Archived {count} old conversations")

        # Delete very low salience entities
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.salience < 0.1
              AND e.last_accessed < datetime() - duration('P30D')
              AND NOT EXISTS((e)<-[:ABOUT]-(:Fact))
            DETACH DELETE e
            RETURN count(e) AS deleted_count
        """)

        logger.info("Cleaned up low-salience orphan entities")


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
        "triggered_at": datetime.utcnow().isoformat()
    }

    redis.xadd("reflection_jobs", {"data": json.dumps(job_data)})
    logger.info(f"Queued reflection for conversation {conversation_id}")

"""Background consolidation jobs for memory processing."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json

from ..connections import Neo4jConnection, get_postgres_session, RedisConnection
from ..embeddings import get_embedder
from ..config import get_settings
from ..extraction.entities import extract_entities
from ..extraction.facts import extract_facts

logger = logging.getLogger(__name__)
settings = get_settings()


def consolidate_episodic_to_semantic():
    """
    Extract entities and facts from recent episodic memory
    and store in semantic memory.
    """
    embedder = get_embedder()

    with Neo4jConnection.session() as session:
        # Get recent conversations not yet processed
        result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE NOT EXISTS(c.consolidated)
              OR c.consolidated < datetime() - duration('PT15M')
            WITH c, collect(t) AS turns
            ORDER BY c.started_at DESC
            LIMIT 10
            RETURN c.id AS conversation_id,
                   [t IN turns | {role: t.role, content: t.content}] AS turns
        """)

        for record in result:
            conv_id = record["conversation_id"]
            turns = record["turns"]

            # Combine turn content for extraction
            full_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in turns
            )

            # Entity extraction (simplified - use NER in production)
            entities = extract_entities(full_text)

            # Store entities
            for entity in entities:
                entity_embedding = embedder.embed_single(
                    f"{entity['name']}: {entity.get('description', entity['type'])}"
                )

                session.run("""
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET
                        e.id = randomUUID(),
                        e.type = $type,
                        e.embedding = $embedding,
                        e.first_seen = datetime(),
                        e.salience = 0.5
                    ON MATCH SET
                        e.last_accessed = datetime(),
                        e.access_count = coalesce(e.access_count, 0) + 1

                    WITH e
                    MATCH (c:Conversation {id: $conv_id})
                    MERGE (c)-[:MENTIONS]->(e)
                """,
                    name=entity["name"],
                    type=entity["type"],
                    embedding=entity_embedding,
                    conv_id=conv_id
                )

            # Fact extraction (simplified)
            facts = extract_facts(full_text)

            for fact in facts:
                fact_embedding = embedder.embed_single(fact["claim"])

                session.run("""
                    CREATE (f:Fact {
                        id: randomUUID(),
                        claim: $claim,
                        confidence: $confidence,
                        source: 'extraction',
                        embedding: $embedding,
                        created_at: datetime()
                    })

                    WITH f
                    MATCH (c:Conversation {id: $conv_id})
                    CREATE (f)-[:DERIVED_FROM]->(c)
                """,
                    claim=fact["claim"],
                    confidence=fact.get("confidence", 0.7),
                    embedding=fact_embedding,
                    conv_id=conv_id
                )

            # Mark conversation as consolidated
            session.run("""
                MATCH (c:Conversation {id: $conv_id})
                SET c.consolidated = datetime()
            """, conv_id=conv_id)

            logger.info(f"Consolidated conversation {conv_id}: {len(entities)} entities, {len(facts)} facts")


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

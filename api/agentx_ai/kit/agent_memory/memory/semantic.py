"""Semantic memory - entities, facts, and conceptual knowledge."""

import json
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..models import Entity, Fact
from ..connections import Neo4jConnection
from ..config import get_settings
from ..query_utils import CypherFilterBuilder, convert_record_datetimes

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger

logger = logging.getLogger(__name__)
settings = get_settings()


class SemanticMemory:
    """Handles entities, facts, and conceptual knowledge."""

    def __init__(self, audit_logger: Optional["MemoryAuditLogger"] = None):
        """Initialize semantic memory.

        Args:
            audit_logger: Optional audit logger for operation tracking.
        """
        self._audit_logger = audit_logger

    def _validate_entity_type(self, entity_type: str) -> str:
        """
        Validate and sanitize entity type against allowed types.

        Args:
            entity_type: The entity type to validate

        Returns:
            Validated entity type (defaults to 'Entity' if invalid)
        """
        # Get allowed types from settings
        allowed_types = set(settings.entity_types)

        # Normalize: uppercase first letter, lowercase rest
        normalized = entity_type.strip().title().replace(" ", "").replace("_", "")

        if normalized in allowed_types:
            return normalized

        # Log warning but don't fail - use generic type
        logger.warning(
            f"Entity type '{entity_type}' not in allowed types {allowed_types}, "
            f"using 'Entity' instead"
        )
        return "Entity"

    def upsert_entity(self, entity: Entity, user_id: Optional[str] = None, channel: str = "_global") -> Entity:
        """
        Create or update an entity.

        Args:
            entity: Entity object to upsert
            user_id: User ID for linking
            channel: Memory channel

        Returns:
            Updated entity object
        """
        # Validate entity type against whitelist to prevent arbitrary label creation
        validated_type = self._validate_entity_type(entity.type)

        with Neo4jConnection.session() as session:
            # Use MERGE for idempotent upsert
            # Store type as property (not dynamic label) for safety
            session.run("""
                MERGE (e:Entity {id: $id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.aliases = $aliases,
                    e.description = $description,
                    e.embedding = $embedding,
                    e.salience = $salience,
                    e.user_id = $user_id,
                    e.channel = $channel,
                    e.first_seen = datetime(),
                    e.last_accessed = datetime(),
                    e.access_count = 1,
                    e.properties = $properties
                ON MATCH SET
                    e.name = $name,
                    e.type = $type,
                    e.description = coalesce($description, e.description),
                    e.embedding = coalesce($embedding, e.embedding),
                    e.aliases = $aliases + [x IN e.aliases WHERE NOT x IN $aliases],
                    e.last_accessed = datetime(),
                    e.access_count = e.access_count + 1

                // Link to user
                WITH e
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:HAS_ENTITY]->(e)

                RETURN e
            """,
                id=entity.id,
                name=entity.name,
                type=validated_type,
                aliases=entity.aliases,
                description=entity.description,
                embedding=entity.embedding,
                salience=entity.salience,
                properties=json.dumps(entity.properties) if entity.properties else None,
                user_id=user_id,
                channel=channel
            ).consume()  # Ensure transaction commits
        return entity

    def store_fact(self, fact: Fact, user_id: Optional[str] = None, channel: str = "_global") -> None:
        """
        Store a fact and link to entities.

        Args:
            fact: Fact object to store
            user_id: User ID for linking
            channel: Memory channel
        """
        with Neo4jConnection.session() as session:
            session.run("""
                CREATE (f:Fact {
                    id: $id,
                    claim: $claim,
                    confidence: $confidence,
                    source: $source,
                    source_turn_id: $source_turn_id,
                    embedding: $embedding,
                    user_id: $user_id,
                    channel: $channel,
                    created_at: datetime()
                })

                // Link to user
                WITH f
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:HAS_FACT]->(f)

                // Link to source turn if provided
                WITH f
                OPTIONAL MATCH (t:Turn {id: $source_turn_id})
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:DERIVED_FROM]->(t)
                )

                // Link to entities
                WITH f
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {id: entity_id})
                MERGE (f)-[:ABOUT]->(e)
            """,
                id=fact.id,
                claim=fact.claim,
                confidence=fact.confidence,
                source=fact.source,
                source_turn_id=fact.source_turn_id,
                embedding=fact.embedding,
                entity_ids=fact.entity_ids,
                user_id=user_id,
                channel=channel
            ).consume()  # Ensure transaction commits

    def vector_search_facts(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_confidence: float = 0.5,
        user_id: Optional[str] = None,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search facts by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of matching facts
        """
        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("f")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            result = session.run(f"""
                CALL db.index.vector.queryNodes('fact_embeddings', $k, $embedding)
                YIELD node AS f, score
                WHERE f.confidence >= $min_confidence {filters.build_inline()}
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                RETURN f.id AS id,
                       f.claim AS claim,
                       f.confidence AS confidence,
                       f.source AS source,
                       f.channel AS channel,
                       collect(e.name) AS entities,
                       score
                ORDER BY score DESC
            """,
                k=top_k * 2,  # Over-fetch for filtering
                embedding=query_embedding,
                min_confidence=min_confidence,
                user_id=user_id,
                channel=channel
            )

            return [dict(record) for record in result][:top_k]

    def vector_search_entities(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search entities by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of matching entities
        """
        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("e")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            result = session.run(f"""
                CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
                YIELD node AS e, score
                WHERE true {filters.build_inline()}

                // Update access stats
                SET e.last_accessed = datetime(),
                    e.access_count = e.access_count + 1

                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       e.salience AS salience,
                       e.channel AS channel,
                       score
                ORDER BY score DESC
            """,
                k=top_k * 2,  # Over-fetch for filtering
                embedding=query_embedding,
                user_id=user_id,
                channel=channel
            )

            return [dict(record) for record in result][:top_k]

    def get_entity_graph(
        self,
        entity_ids: List[str],
        depth: int = 2,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
        max_related: int = 50
    ) -> Dict[str, Any]:
        """
        Traverse entity relationships to get context.

        Args:
            entity_ids: List of entity IDs to start from
            depth: Relationship traversal depth (capped at 3)
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)
            max_related: Maximum number of related entities to return (default 50)

        Returns:
            Dictionary with entities, related entities, and facts
        """
        # Validate depth to prevent expensive traversals
        validated_depth = max(1, min(int(depth), 3))
        validated_max_related = max(1, min(int(max_related), 100))

        with Neo4jConnection.session() as session:
            # Build filters using utilities for each node alias
            entity_filters = CypherFilterBuilder("e")
            entity_filters.add_user_filter(user_id).add_channel_filter(channel)

            related_filters = CypherFilterBuilder("related")
            related_filters.add_channel_filter(channel)

            fact_filters = CypherFilterBuilder("f")
            fact_filters.add_channel_filter(channel)

            result = session.run(f"""
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {{id: eid}})
                WHERE true {entity_filters.build_inline()}

                // Get related entities within depth, filtered by channel
                // Use validated_depth directly in query (it's an integer, safe)
                OPTIONAL MATCH path = (e)-[r*1..{validated_depth}]-(related:Entity)
                WHERE true {related_filters.build_inline()}

                // Get facts about these entities, filtered by channel
                OPTIONAL MATCH (f:Fact)-[:ABOUT]->(e)
                WHERE true {fact_filters.build_inline()}

                // Collect results with limits to prevent memory issues
                WITH e,
                     collect(DISTINCT {{
                         entity: related,
                         relationship: type(last(relationships(path))),
                         path_length: length(path)
                     }})[0..$max_related] AS related_limited,
                     collect(DISTINCT f)[0..50] AS facts_limited

                RETURN e AS entity,
                       related_limited AS related,
                       facts_limited AS facts
            """,
                entity_ids=entity_ids,
                max_related=validated_max_related,
                user_id=user_id,
                channel=channel
            )

            entities = []
            all_related = []
            all_facts = []

            for record in result:
                if record["entity"]:
                    entities.append(dict(record["entity"]))
                all_related.extend([r for r in record["related"] if r.get("entity")])
                all_facts.extend([dict(f) for f in record["facts"] if f])

            return {
                "entities": entities,
                "related": all_related,
                "facts": all_facts
            }

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        channel: str = "_global"
    ) -> None:
        """
        Create a relationship between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            properties: Optional relationship properties
            user_id: User ID (for validation)
            channel: Channel for the relationship
        """
        with Neo4jConnection.session() as session:
            props = properties or {}

            # Build user filter for security
            user_filter = ""
            if user_id:
                user_filter = "AND s.user_id = $user_id AND t.user_id = $user_id"

            session.run(f"""
                MATCH (s:Entity {{id: $source_id}})
                MATCH (t:Entity {{id: $target_id}})
                WHERE true {user_filter}
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $properties,
                    r.channel = $channel,
                    r.created_at = datetime()
            """,
                source_id=source_id,
                target_id=target_id,
                properties=props,
                user_id=user_id,
                channel=channel
            )

    def list_entities(
        self,
        user_id: str,
        channel: str = "_global",
        offset: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
        entity_type: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List entities with pagination and optional filtering.

        Args:
            user_id: User ID to filter by
            channel: Filter by channel (searches channel + _global)
            offset: Number of records to skip
            limit: Maximum number of records to return
            search: Optional text search on entity name
            entity_type: Optional filter by entity type

        Returns:
            Tuple of (entities list, total count)
        """
        with Neo4jConnection.session() as session:
            # Build WHERE conditions
            conditions = ["e.user_id = $user_id"]

            # Channel filter - _all means no filter, otherwise search both specified channel and _global
            if channel == "_all":
                pass  # No channel filter - show all channels
            elif channel and channel != "_global":
                conditions.append("(e.channel = $channel OR e.channel = '_global')")
            else:
                conditions.append("e.channel = '_global'")

            # Text search on name
            if search:
                conditions.append("toLower(e.name) CONTAINS toLower($search)")

            # Filter by entity type
            if entity_type:
                conditions.append("e.type = $entity_type")

            where_clause = " AND ".join(conditions)

            # Get total count
            count_result = session.run(f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                RETURN count(e) AS total
            """,
                user_id=user_id,
                channel=channel,
                search=search,
                entity_type=entity_type
            )
            total = count_result.single()["total"]

            # Get paginated results
            result = session.run(f"""
                MATCH (e:Entity)
                WHERE {where_clause}
                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.channel AS channel,
                       e.salience AS salience,
                       e.description AS description,
                       e.last_accessed AS last_accessed,
                       e.access_count AS access_count,
                       e.first_seen AS first_seen
                ORDER BY e.salience DESC, e.last_accessed DESC
                SKIP $offset
                LIMIT $limit
            """,
                user_id=user_id,
                channel=channel,
                search=search,
                entity_type=entity_type,
                offset=offset,
                limit=limit
            )

            entities = []
            for record in result:
                entity = convert_record_datetimes(dict(record))
                entities.append(entity)
            return entities, total

    def list_facts(
        self,
        user_id: str,
        channel: str = "_global",
        offset: int = 0,
        limit: int = 20,
        min_confidence: float = 0.0,
        search: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List facts with pagination and optional filtering.

        Args:
            user_id: User ID to filter by
            channel: Filter by channel (searches channel + _global)
            offset: Number of records to skip
            limit: Maximum number of records to return
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            search: Optional text search on fact claim

        Returns:
            Tuple of (facts list, total count)
        """
        with Neo4jConnection.session() as session:
            # Build WHERE conditions
            conditions = ["f.user_id = $user_id", "f.confidence >= $min_confidence"]

            # Channel filter - _all means no filter, otherwise search both specified channel and _global
            if channel == "_all":
                pass  # No channel filter - show all channels
            elif channel and channel != "_global":
                conditions.append("(f.channel = $channel OR f.channel = '_global')")
            else:
                conditions.append("f.channel = '_global'")

            # Text search on claim
            if search:
                conditions.append("toLower(f.claim) CONTAINS toLower($search)")

            where_clause = " AND ".join(conditions)

            # Get total count
            count_result = session.run(f"""
                MATCH (f:Fact)
                WHERE {where_clause}
                RETURN count(f) AS total
            """,
                user_id=user_id,
                channel=channel,
                min_confidence=min_confidence,
                search=search
            )
            total = count_result.single()["total"]

            # Get paginated results with entity names
            result = session.run(f"""
                MATCH (f:Fact)
                WHERE {where_clause}
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                WITH f, collect(e.id) AS entity_ids
                RETURN f.id AS id,
                       f.claim AS claim,
                       f.confidence AS confidence,
                       f.source AS source,
                       f.channel AS channel,
                       f.source_turn_id AS source_turn_id,
                       f.created_at AS created_at,
                       f.promoted_from AS promoted_from,
                       entity_ids
                ORDER BY f.confidence DESC, f.created_at DESC
                SKIP $offset
                LIMIT $limit
            """,
                user_id=user_id,
                channel=channel,
                min_confidence=min_confidence,
                search=search,
                offset=offset,
                limit=limit
            )

            facts = []
            for record in result:
                fact = convert_record_datetimes(dict(record))
                facts.append(fact)
            return facts, total

    def get_entity_by_id(
        self,
        entity_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single entity by ID.

        Args:
            entity_id: Entity ID to retrieve
            user_id: User ID for access control

        Returns:
            Entity dict or None if not found
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (e:Entity {id: $entity_id})
                WHERE e.user_id = $user_id
                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.channel AS channel,
                       e.salience AS salience,
                       e.description AS description,
                       e.aliases AS aliases,
                       e.last_accessed AS last_accessed,
                       e.access_count AS access_count,
                       e.first_seen AS first_seen
            """,
                entity_id=entity_id,
                user_id=user_id
            )

            record = result.single()
            if not record:
                return None

            return convert_record_datetimes(dict(record))

    def get_entity_facts_and_relationships(
        self,
        entity_id: str,
        user_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get an entity with its connected facts and relationships.

        Args:
            entity_id: Entity ID to retrieve
            user_id: User ID for access control
            depth: Relationship traversal depth (capped at 3)

        Returns:
            Dictionary with entity, facts, and relationships
        """
        validated_depth = max(1, min(int(depth), 3))

        with Neo4jConnection.session() as session:
            # Get entity
            entity = self.get_entity_by_id(entity_id, user_id)
            if not entity:
                return {"entity": None, "facts": [], "relationships": []}

            # Get facts about this entity
            facts_result = session.run("""
                MATCH (f:Fact)-[:ABOUT]->(e:Entity {id: $entity_id})
                WHERE f.user_id = $user_id
                RETURN f.id AS id,
                       f.claim AS claim,
                       f.confidence AS confidence,
                       f.source AS source,
                       f.channel AS channel,
                       f.promoted_from AS promoted_from
                ORDER BY f.confidence DESC
                LIMIT 50
            """,
                entity_id=entity_id,
                user_id=user_id
            )
            facts = [dict(record) for record in facts_result]

            # Get relationships to other entities
            rel_result = session.run(f"""
                MATCH (e:Entity {{id: $entity_id}})-[r]->(target:Entity)
                WHERE e.user_id = $user_id AND target.user_id = $user_id
                RETURN type(r) AS type,
                       target.id AS target_id,
                       target.name AS target_name,
                       target.type AS target_type
                LIMIT 50
            """,
                entity_id=entity_id,
                user_id=user_id
            )
            relationships = [
                {
                    "type": record["type"],
                    "target": {
                        "id": record["target_id"],
                        "name": record["target_name"],
                        "type": record["target_type"]
                    }
                }
                for record in rel_result
            ]

            return {
                "entity": entity,
                "facts": facts,
                "relationships": relationships
            }

"""Semantic memory - entities, facts, and conceptual knowledge."""

from typing import List, Dict, Any, Optional

from ..models import Entity, Fact
from ..connections import Neo4jConnection


class SemanticMemory:
    """Handles entities, facts, and conceptual knowledge."""

    def upsert_entity(self, entity: Entity) -> Entity:
        """
        Create or update an entity.

        Args:
            entity: Entity object to upsert

        Returns:
            Updated entity object
        """
        with Neo4jConnection.session() as session:
            # Use MERGE for idempotent upsert
            session.run("""
                MERGE (e:Entity {id: $id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.aliases = $aliases,
                    e.description = $description,
                    e.embedding = $embedding,
                    e.salience = $salience,
                    e.first_seen = datetime(),
                    e.last_accessed = datetime(),
                    e.access_count = 1,
                    e.properties = $properties
                ON MATCH SET
                    e.name = $name,
                    e.description = coalesce($description, e.description),
                    e.embedding = coalesce($embedding, e.embedding),
                    e.aliases = $aliases + [x IN e.aliases WHERE NOT x IN $aliases],
                    e.last_accessed = datetime(),
                    e.access_count = e.access_count + 1

                // Add type-specific label
                WITH e
                CALL apoc.create.addLabels(e, [$type]) YIELD node
                RETURN node
            """,
                id=entity.id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                description=entity.description,
                embedding=entity.embedding,
                salience=entity.salience,
                properties=entity.properties
            )
        return entity

    def store_fact(self, fact: Fact) -> None:
        """
        Store a fact and link to entities.

        Args:
            fact: Fact object to store
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
                    created_at: datetime()
                })

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
                entity_ids=fact.entity_ids
            )

    def vector_search_facts(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search facts by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching facts
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('fact_embeddings', $k, $embedding)
                YIELD node AS f, score
                WHERE f.confidence >= $min_confidence
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                RETURN f.id AS id,
                       f.claim AS claim,
                       f.confidence AS confidence,
                       f.source AS source,
                       collect(e.name) AS entities,
                       score
                ORDER BY score DESC
            """,
                k=top_k,
                embedding=query_embedding,
                min_confidence=min_confidence
            )

            return [dict(record) for record in result]

    def vector_search_entities(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search entities by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of matching entities
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
                YIELD node AS e, score

                // Update access stats
                SET e.last_accessed = datetime(),
                    e.access_count = e.access_count + 1

                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       e.salience AS salience,
                       score
                ORDER BY score DESC
            """,
                k=top_k,
                embedding=query_embedding
            )

            return [dict(record) for record in result]

    def get_entity_graph(
        self,
        entity_ids: List[str],
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Traverse entity relationships to get context.

        Args:
            entity_ids: List of entity IDs to start from
            depth: Relationship traversal depth

        Returns:
            Dictionary with entities, related entities, and facts
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {id: eid})

                // Get related entities within depth
                OPTIONAL MATCH path = (e)-[r*1..$depth]-(related:Entity)

                // Get facts about these entities
                OPTIONAL MATCH (f:Fact)-[:ABOUT]->(e)

                RETURN e AS entity,
                       collect(DISTINCT {
                           entity: related,
                           relationship: type(last(relationships(path))),
                           path_length: length(path)
                       }) AS related,
                       collect(DISTINCT f) AS facts
            """,
                entity_ids=entity_ids,
                depth=depth
            )

            entities = []
            all_related = []
            all_facts = []

            for record in result:
                entities.append(dict(record["entity"]))
                all_related.extend(record["related"])
                all_facts.extend([dict(f) for f in record["facts"]])

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
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a relationship between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            properties: Optional relationship properties
        """
        with Neo4jConnection.session() as session:
            props = properties or {}
            session.run(f"""
                MATCH (s:Entity {{id: $source_id}})
                MATCH (t:Entity {{id: $target_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $properties
            """,
                source_id=source_id,
                target_id=target_id,
                properties=props
            )

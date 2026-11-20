"""The consolidation sweep's queries, over one store session per sweep phase.

The sweep (``consolidation/jobs.py``: ``consolidate_episodic_to_semantic`` and its
helpers) opens a ``ConsolidationStore`` for each phase and threads it through its
helpers, which call these methods instead of running queries themselves. Every
query here works on the conversation being consolidated: its user's facts in one
channel plus ``_global``, or nodes the sweep already holds ids for. Finding the
conversations to consolidate spans every user, so that lives in ``CrossUserStore``.
When the store moves to PostgreSQL (Decisions.md ADR-21), the twin opens its own
session per phase and offers the same methods.

Writes consume their results before returning. An error that arrives after
``run()`` returns, such as a failed commit of the auto-commit transaction, then
surfaces at the write that caused it, not at whichever query runs next.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..connections import Neo4jConnection
from ..models import Entity

# The turn markers the sweep stamps: the only names `mark_turns` writes into a query.
TURN_MARKERS = frozenset({"consolidated", "self_consolidated"})


class ConsolidationStore:
    """The sweep's queries over one open store session. See the module docstring."""

    def __init__(self, session: Any) -> None:
        self._session = session

    @classmethod
    @contextmanager
    def open(cls) -> Iterator[ConsolidationStore]:
        """A store over a fresh session, held for one phase of the sweep."""
        with Neo4jConnection.session() as session:
            yield cls(session)

    # -- fact checks: one user's facts in a channel plus _global -----------------

    def fact_with_hash_exists(self, user_id: str, channel: str, claim_hash: str) -> bool:
        """Whether the user has a fact with this claim hash (the indexed lookup)."""
        result = self._session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.claim_hash = $claim_hash
        RETURN f.id AS id
        LIMIT 1
    """, user_id=user_id, channel=channel, claim_hash=claim_hash)
        return result.single() is not None

    def legacy_fact_with_claim_exists(
        self, user_id: str, channel: str, normalized_claim: str,
    ) -> bool:
        """Whether a fact stored without a claim hash has this claim, compared
        lowercased and trimmed (`normalized_claim` arrives normalized)."""
        result = self._session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.claim_hash IS NULL
          AND toLower(trim(f.claim)) = $normalized_claim
        RETURN f.id AS id
        LIMIT 1
    """, user_id=user_id, channel=channel, normalized_claim=normalized_claim)
        return result.single() is not None

    def similar_fact_exists(
        self, embedding: list[float], user_id: str, channel: str, threshold: float,
    ) -> bool:
        """Whether a live fact in the user's scope scores above `threshold` against
        `embedding`. The vector index ranks the three nearest facts across every
        user before the scope filter applies, so a near neighbour of another user's
        can crowd out this user's match (a known recall limit of this backend)."""
        result = self._session.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', 3, $embedding)
            YIELD node, score
            WHERE score > $threshold
              AND node.user_id = $user_id
              AND (node.channel = $channel OR node.channel = '_global')
              AND node.superseded_at IS NULL
            RETURN node.id AS id, score
            LIMIT 1
        """, embedding=embedding, threshold=threshold, user_id=user_id, channel=channel)
        return result.single() is not None

    def facts_about_entities(
        self, entity_names: list[str], user_id: str, channel: str, limit: int,
    ) -> list[dict[str, Any]]:
        """Live facts in the user's scope about any entity with one of these names
        (lowercase): ``id``, ``claim``, ``confidence``, ``temporal_context`` and a
        ``similarity_score`` of 1.0."""
        entity_result = self._session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) IN $entity_names
                MATCH (f:Fact)-[:ABOUT]->(e)
                WHERE f.user_id = $user_id
                  AND (f.channel = $channel OR f.channel = '_global')
                  AND f.superseded_at IS NULL
                RETURN DISTINCT f.id AS id, f.claim AS claim,
                       f.confidence AS confidence,
                       f.temporal_context AS temporal_context,
                       1.0 AS similarity_score
                LIMIT $limit
            """, entity_names=entity_names, user_id=user_id,
            channel=channel, limit=limit)
        return [dict(record) for record in entity_result]

    def similar_facts(
        self, embedding: list[float], user_id: str, channel: str, threshold: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Live facts in the user's scope scoring above `threshold` against
        `embedding`, from the `limit` nearest across every user: ``id``, ``claim``,
        ``confidence``, ``temporal_context``, ``similarity_score``."""
        vector_result = self._session.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', $k, $embedding)
            YIELD node, score
            WHERE score > $threshold
              AND node.user_id = $user_id
              AND (node.channel = $channel OR node.channel = '_global')
              AND node.superseded_at IS NULL
            RETURN node.id AS id, node.claim AS claim,
                   node.confidence AS confidence,
                   node.temporal_context AS temporal_context,
                   score AS similarity_score
            LIMIT $limit
        """, k=limit, embedding=embedding,
            threshold=threshold, user_id=user_id,
            channel=channel, limit=limit)
        return [dict(record) for record in vector_result]

    def recent_facts(self, user_id: str, channel: str, limit: int) -> list[dict[str, Any]]:
        """The user's newest live facts in scope: ``id``, ``claim``, ``confidence``,
        ``created_at``."""
        result = self._session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.superseded_at IS NULL
        RETURN f.id AS id,
               f.claim AS claim,
               f.confidence AS confidence,
               f.created_at AS created_at
        ORDER BY f.created_at DESC
        LIMIT $limit
    """, user_id=user_id, channel=channel, limit=limit)
        return [dict(record) for record in result]

    def facts_containing(self, user_id: str, channel: str, needle: str) -> list[dict[str, Any]]:
        """Up to five live facts in scope whose claim contains `needle` (lowercase),
        newest first: ``id``, ``claim``. Finds the fact a user correction targets."""
        result = self._session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.superseded_at IS NULL
          AND toLower(f.claim) CONTAINS $original_lower
        RETURN f.id AS id, f.claim AS claim
        ORDER BY f.created_at DESC
        LIMIT 5
    """, user_id=user_id, channel=channel, original_lower=needle)
        return [dict(record) for record in result]

    # -- writes --------------------------------------------------------------------

    def store_entities(
        self, entities: list[Entity], conv_id: str, user_id: str, channel: str,
    ) -> None:
        """Upsert entities for the user in `channel`, each linked to its user and
        mentioned by the conversation. A known entity keeps its first-seen fields,
        takes the new name, type and salience, and counts one more access."""
        entity_data = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "salience": e.salience,
                "aliases": e.aliases,
                "embedding": e.embedding,
                "user_id": user_id,
                "channel": channel,
            }
            for e in entities
        ]
        self._session.run("""
        UNWIND $entities AS e
        MERGE (entity:Entity {id: e.id})
        ON CREATE SET
            entity.name = e.name,
            entity.type = e.type,
            entity.description = e.description,
            entity.salience = e.salience,
            entity.aliases = e.aliases,
            entity.embedding = e.embedding,
            entity.user_id = e.user_id,
            entity.channel = e.channel,
            entity.first_seen = datetime(),
            entity.last_accessed = datetime(),
            entity.access_count = 1
        ON MATCH SET
            entity.name = e.name,
            entity.type = e.type,
            entity.description = coalesce(e.description, entity.description),
            entity.salience = e.salience,
            entity.last_accessed = datetime(),
            entity.access_count = entity.access_count + 1
        WITH entity, e
        MERGE (u:User {id: e.user_id})
        MERGE (u)-[:HAS_ENTITY]->(entity)
        WITH entity
        MATCH (c:Conversation {id: $conv_id})
        MERGE (c)-[:MENTIONS]->(entity)
    """, entities=entity_data, conv_id=conv_id).consume()

    def store_relationships(self, rels: list[dict[str, Any]]) -> None:
        """Upsert typed relationships between entities that exist, from
        ``{source_id, target_id, rel_type, confidence}`` rows. A repeated relationship
        keeps the higher confidence."""
        self._session.run("""
        UNWIND $rels AS r
        MATCH (source:Entity {id: r.source_id}),
              (target:Entity {id: r.target_id})
        MERGE (source)-[rel:RELATES_TO {type: r.rel_type}]->(target)
        ON CREATE SET rel.confidence = r.confidence,
                      rel.created_at = datetime()
        ON MATCH SET rel.confidence =
            CASE WHEN rel.confidence < r.confidence
                 THEN r.confidence
                 ELSE rel.confidence END
    """, rels=rels).consume()

    def entity_exists(self, entity_id: str) -> bool:
        """Whether an entity with this id exists."""
        record = self._session.run(
            "MATCH (e:Entity {id: $id}) RETURN e.id LIMIT 1", id=entity_id,
        ).single()
        return record is not None

    def link_facts_to_conversation(self, conv_id: str, fact_ids: list[str]) -> None:
        """Record that each fact was derived from the conversation."""
        self._session.run("""
                UNWIND $fact_ids AS fid
                MATCH (c:Conversation {id: $conv_id}), (f:Fact {id: fid})
                MERGE (f)-[:DERIVED_FROM]->(c)
            """, conv_id=conv_id, fact_ids=fact_ids).consume()

    def mark_turns(self, turn_ids: list[str], prop: str) -> None:
        """Stamp ``prop`` (one of `TURN_MARKERS`) with the current time on each turn."""
        if prop not in TURN_MARKERS:
            raise ValueError(f"unknown turn marker: {prop!r}")
        self._session.run(
            f"UNWIND $ids AS tid MATCH (t:Turn {{id: tid}}) SET t.{prop} = datetime()",
            ids=turn_ids,
        ).consume()

    def mark_conversation_consolidated(self, conv_id: str) -> None:
        """Stamp the conversation's coarse last-consolidated time (user turns)."""
        self._session.run(
            "MATCH (c:Conversation {id: $conv_id}) SET c.consolidated = datetime()",
            conv_id=conv_id,
        ).consume()

    def mark_conversation_self_consolidated(self, conv_id: str) -> None:
        """Stamp the conversation's coarse last-self-consolidated time (assistant turns)."""
        self._session.run(
            "MATCH (c:Conversation {id: $conv_id}) SET c.self_consolidated = datetime()",
            conv_id=conv_id,
        ).consume()

"""Background consolidation jobs for memory processing."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any, TYPE_CHECKING
from collections.abc import Callable
from uuid import uuid4
import logging
import json
import re
import time

from ..connections import Neo4jConnection, get_postgres_session, RedisConnection
from ..embeddings import get_embedder
from ..config import get_settings
from ..extraction.service import get_extraction_service
from ..models import Entity, compute_claim_hash
from .metrics import ConsolidationMetrics

if TYPE_CHECKING:
    from ..memory.interface import AgentMemory

logger = logging.getLogger(__name__)


def _is_duplicate_fact(session, claim: str, user_id: str, channel: str) -> bool:
    """
    Check if a fact with the same or very similar claim already exists.

    Uses claim_hash for indexed lookup when available, falls back to normalized
    string comparison for legacy facts without hash.

    Args:
        session: Neo4j session
        claim: The fact claim text
        user_id: User ID
        channel: Memory channel

    Returns:
        True if a duplicate exists
    """
    claim_hash = compute_claim_hash(claim)

    # Check for duplicate using indexed claim_hash (fast path)
    result = session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.claim_hash = $claim_hash
        RETURN f.id AS id
        LIMIT 1
    """, user_id=user_id, channel=channel, claim_hash=claim_hash)

    if result.single() is not None:
        return True

    # Fallback check for legacy facts without claim_hash
    normalized = " ".join(claim.lower().split())
    result = session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.claim_hash IS NULL
          AND toLower(trim(f.claim)) = $normalized_claim
        RETURN f.id AS id
        LIMIT 1
    """, user_id=user_id, channel=channel, normalized_claim=normalized)

    return result.single() is not None


def _is_semantic_duplicate(
    session,
    claim_embedding: list[float],
    user_id: str,
    channel: str,
    threshold: float = 0.92,
) -> bool:
    """
    Check if a semantically identical fact already exists via vector search.

    Uses the Neo4j fact_embeddings index for fast cosine similarity lookup.
    A threshold of 0.92 catches paraphrases like "User likes Python" vs
    "The user enjoys Python" while avoiding false positives.

    Args:
        session: Neo4j session
        claim_embedding: Embedding vector for the new claim
        user_id: User ID
        channel: Memory channel
        threshold: Minimum cosine similarity to consider duplicate

    Returns:
        True if a semantically identical fact exists
    """
    try:
        result = session.run("""
            CALL db.index.vector.queryNodes('fact_embeddings', 3, $embedding)
            YIELD node, score
            WHERE score > $threshold
              AND node.user_id = $user_id
              AND (node.channel = $channel OR node.channel = '_global')
              AND node.superseded_at IS NULL
            RETURN node.id AS id, score
            LIMIT 1
        """, embedding=claim_embedding, threshold=threshold, user_id=user_id, channel=channel)
        return result.single() is not None
    except Exception as e:
        logger.warning(
            f"Semantic duplicate check failed [{type(e).__name__}] "
            f"(fact_embeddings index missing/offline or param shape mismatch): {e}. "
            f"If the index is missing, run: python manage.py init_memory_schema"
        )
        return False


def _get_contradiction_candidates(
    session,
    claim: str,
    claim_embedding: list[float],
    entity_names: list[str],
    user_id: str,
    channel: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 10,
) -> list[dict[str, Any]]:
    """
    Find facts that might contradict a new claim using entity-scoped
    retrieval + embedding similarity (Layer 2 of the verification pipeline).

    Strategy:
    1. Entity-scoped: Find all active facts about the same entities
    2. Embedding similarity: Vector search for semantically similar facts
    3. Merge + deduplicate, order by similarity descending

    Args:
        session: Neo4j session
        claim: The new fact claim text
        claim_embedding: Embedding vector for the new claim
        entity_names: Entity names mentioned in the new fact
        user_id: User ID
        channel: Memory channel
        similarity_threshold: Min cosine similarity for vector search
        max_candidates: Max candidates to return

    Returns:
        List of candidate facts with id, claim, confidence, temporal_context, similarity_score
    """
    seen_ids: set = set()
    candidates: list[dict[str, Any]] = []

    # Strategy 1: Entity-scoped retrieval
    if entity_names:
        entity_names_lower = [n.lower() for n in entity_names]
        try:
            entity_result = session.run("""
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
            """, entity_names=entity_names_lower, user_id=user_id,
                channel=channel, limit=max_candidates)
            for record in entity_result:
                fid = record["id"]
                if fid not in seen_ids:
                    seen_ids.add(fid)
                    candidates.append(dict(record))
        except Exception as e:
            logger.debug(f"Entity-scoped contradiction search failed: {e}")

    # Strategy 2: Embedding similarity search
    try:
        vector_result = session.run("""
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
        """, k=max_candidates, embedding=claim_embedding,
            threshold=similarity_threshold, user_id=user_id,
            channel=channel, limit=max_candidates)
        for record in vector_result:
            fid = record["id"]
            if fid not in seen_ids:
                seen_ids.add(fid)
                candidates.append(dict(record))
    except Exception as e:
        logger.warning(
            f"Vector contradiction search failed [{type(e).__name__}] "
            f"(index missing or param shape mismatch): {e}"
        )

    # Sort by similarity descending, cap at max_candidates
    candidates.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
    return candidates[:max_candidates]


def _is_temporal_progression(new_fact: dict[str, Any], old_fact: dict[str, Any]) -> bool:
    """
    Detect natural temporal progressions that don't need LLM adjudication.

    A temporal progression is when a new 'current' fact naturally supersedes
    an old 'current' or unspecified fact about the same entity (e.g., job changes,
    tool switches, location moves).

    Args:
        new_fact: The newly extracted fact dict
        old_fact: The existing fact dict from contradiction candidates

    Returns:
        True if this is a natural temporal progression
    """
    new_temporal = new_fact.get("temporal_context")
    old_temporal = old_fact.get("temporal_context")
    # New "current" supersedes old "current" or unspecified about same entity
    if new_temporal == "current" and old_temporal in ("current", None):
        return True
    return False


def _get_memory_for_user(user_id: str, channel: str = "_default") -> AgentMemory:
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


def _get_recent_facts(
    session,
    user_id: str,
    channel: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """
    Get recent facts for a user/channel for contradiction detection.

    Args:
        session: Neo4j session
        user_id: User ID
        channel: Memory channel
        limit: Maximum facts to return

    Returns:
        List of facts with id, claim, confidence, created_at
    """
    result = session.run("""
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


def _handle_user_correction(
    memory: AgentMemory,
    session,
    correction,
    user_id: str,
    channel: str,
) -> bool:
    """
    Handle a user correction by finding and superseding the original fact.

    Args:
        memory: AgentMemory instance
        session: Neo4j session
        correction: CorrectionResult with original_claim and corrected_claim
        user_id: User ID
        channel: Memory channel

    Returns:
        True if a fact was superseded
    """
    from ..models import Fact

    if not correction.original_claim:
        logger.debug("Correction detected but no original claim extracted")
        return False

    # Find facts that might match the original claim (fuzzy match)
    original_lower = correction.original_claim.strip().lower()

    result = session.run("""
        MATCH (f:Fact)
        WHERE f.user_id = $user_id
          AND (f.channel = $channel OR f.channel = '_global')
          AND f.superseded_at IS NULL
          AND toLower(f.claim) CONTAINS $original_lower
        RETURN f.id AS id, f.claim AS claim
        ORDER BY f.created_at DESC
        LIMIT 5
    """, user_id=user_id, channel=channel, original_lower=original_lower[:50])

    matching_facts = list(result)

    if not matching_facts:
        logger.debug(f"No matching fact found for correction: {original_lower[:50]}...")
        return False

    # Use the most recent match
    original_fact = matching_facts[0]
    logger.info(f"Found fact to correct: {original_fact['claim'][:50]}...")

    # Create new fact with corrected claim
    if correction.corrected_claim:
        new_fact = Fact(
            claim=correction.corrected_claim,
            confidence=0.9,  # High confidence for user corrections
            source="user_correction",
        )

        # Generate embedding for new fact
        embedder = get_embedder()
        new_fact.embedding = embedder.embed_single(new_fact.claim)

        # Supersede via semantic memory
        memory.semantic.supersede_fact(
            original_fact_id=original_fact['id'],
            new_fact=new_fact,
            user_id=user_id,
            channel=channel,
            reason="user_correction",
        )
        return True

    return False


def _handle_contradiction(
    memory: AgentMemory,
    session,
    fact_dict: dict[str, Any],
    contradiction,
    user_id: str,
    channel: str,
) -> str:
    """
    Handle a contradiction between a new fact and an existing fact.

    Args:
        memory: AgentMemory instance
        session: Neo4j session
        fact_dict: The new fact data
        contradiction: ContradictionResult with resolution
        user_id: User ID
        channel: Memory channel

    Returns:
        Action taken: 'superseded', 'skipped', 'flagged', 'stored'
    """
    from ..models import Fact

    resolution = contradiction.resolution.lower() if contradiction.resolution else "flag_review"

    if resolution == "prefer_new":
        # Supersede the old fact with the new one
        if contradiction.contradicting_fact_id:
            new_fact = Fact(
                claim=fact_dict["claim"],
                confidence=fact_dict.get("confidence", 0.7),
                source="extraction",
                source_turn_id=fact_dict.get("source_turn_id"),
            )

            embedder = get_embedder()
            new_fact.embedding = embedder.embed_single(new_fact.claim)

            memory.semantic.supersede_fact(
                original_fact_id=contradiction.contradicting_fact_id,
                new_fact=new_fact,
                user_id=user_id,
                channel=channel,
                reason="contradiction_prefer_new",
            )
            logger.info(f"Superseded contradicting fact: {contradiction.contradicting_fact_id}")
            return "superseded"

    elif resolution == "prefer_old":
        # Don't store the new fact
        logger.info(f"Skipping new fact due to contradiction: {fact_dict['claim'][:50]}...")
        return "skipped"

    else:  # flag_review
        # Store the new fact but flag it for review
        fact_dict["flagged_for_review"] = True
        logger.info(f"Flagging fact for review due to contradiction: {fact_dict['claim'][:50]}...")
        return "flagged"

    return "stored"


def _resolve_entity_semantic(
    memory: AgentMemory,
    name: str,
    etype: str | None,
    embedding: list[float] | None,
    user_id: str,
    channel: str,
    metrics: ConsolidationMetrics,
) -> dict[str, Any] | None:
    """Vector-band entity resolution (§2.10 Slice 1, Bug 3).

    Consults `vector_search_entities` after exact name/alias/slug matching has
    missed. The top *eligible* candidate (same type when etype given; never an
    Agent on either side — INV-3) decides:

      score >= entity_linking_auto_threshold  → return the match (auto-link)
      score >= entity_linking_similarity_threshold → log-only gray zone
      below → None (mint new)

    Note: vector_search_entities bumps access_count as a side effect — one
    write-path bump per candidate lookup is negligible vs recall traffic, but
    decay tuning should know it exists.
    """
    if embedding is None or (etype or "") == "Agent":
        return None
    try:
        candidates = memory.semantic.vector_search_entities(
            embedding, top_k=3, user_id=user_id, channel=channel,
        )
    except Exception as e:
        logger.debug(f"semantic entity lookup failed for '{name}': {e}")
        return None

    for cand in candidates:
        if (cand.get("type") or "") == "Agent":
            continue
        if etype and cand.get("type") and cand["type"] != etype:
            continue
        score = float(cand.get("score") or 0.0)
        if score >= get_settings().entity_linking_auto_threshold:
            metrics.entities_semantic_linked += 1
            logger.debug(
                f"semantic-link: '{name}' → '{cand.get('name')}' "
                f"({cand.get('id')}) score={score:.3f}"
            )
            return cand
        if score >= get_settings().entity_linking_similarity_threshold:
            # Gray zone: never auto-merge — log as adjudication corpus.
            metrics.entities_semantic_candidates += 1
            logger.info(
                f"semantic-link gray zone (log-only): '{name}' ~ "
                f"'{cand.get('name')}' ({cand.get('id')}) score={score:.3f}"
            )
        return None  # top eligible candidate decides; weaker ones can't outrank it
    return None


def _resolve_and_prepare_entities(
    memory: AgentMemory,
    extracted_entities: list[dict[str, Any]],
    entity_map: dict[str, str],
    user_id: str,
    channel: str,
    conv_id: str,
    metrics: ConsolidationMetrics,
    errors: list[str],
) -> tuple[list[Entity], int, int]:
    """
    Resolve each extracted entity dict against the existing store before storage.

    Resolution order:
      1. Honor LLM-supplied `existing_entity_id` if it points to an in-scope entity.
      1.5. Batch-local pending index (name/alias/slug against entities already
           prepared in THIS batch) — without it, two mentions of the same new
           entity in one conversation mint two nodes: the graph lookup below
           can't see unstored batch-mates.
      2. `find_entity_by_name_or_alias` (name → alias → slug).
      3. Otherwise treat as new and prepare an Entity for batch insert.

    On reuse, fold in any new aliases / description / properties without clobbering
    populated values. Populates `entity_map` (lowercase name → entity id) so the
    fact-linking + relationship-linking passes downstream see both new and reused ids.

    Returns (entities_to_store, num_reused, total_entities_resolved).
    """
    from ..memory.semantic import entity_slug

    entities_to_store: list[Entity] = []
    reused = 0
    semantic = memory.semantic
    # Batch-local index over pending (unstored) entities: lowercase name, every
    # alias, and the alnum slug of each — same normalizer the graph lookup uses.
    pending_index: dict[str, Entity] = {}

    def _index_pending(ent: Entity) -> None:
        keys = {ent.name.lower(), entity_slug(ent.name)}
        for alias in ent.aliases:
            keys.add(alias.lower())
            keys.add(entity_slug(alias))
        for key in keys:
            if key:
                pending_index.setdefault(key, ent)

    for entity_dict in extracted_entities:
        try:
            name = entity_dict.get("name")
            etype = entity_dict.get("type")
            if not name or not etype:
                logger.warning(f"Skipping entity with missing name/type: {entity_dict}")
                continue

            llm_aliases = [a for a in entity_dict.get("aliases", []) if a]
            llm_description = entity_dict.get("description")
            llm_properties = entity_dict.get("properties") or None

            # 1) LLM-supplied existing_entity_id
            existing_id = entity_dict.get("existing_entity_id")
            resolved = None
            if existing_id:
                resolved = semantic.get_entity_by_id(existing_id, user_id)
                if not resolved:
                    logger.debug(
                        f"existing_entity_id {existing_id} not found in scope; "
                        f"falling back to name lookup for '{name}'"
                    )

            # 1.5) Batch-local pending match — probe with the candidate's name
            # AND its aliases (an alias may be the form a batch-mate used).
            if not resolved:
                pending = None
                for probe in [name, *llm_aliases]:
                    pending = (pending_index.get(probe.lower())
                               or pending_index.get(entity_slug(probe)))
                    if pending is not None:
                        break
                if pending is not None:
                    for alias in [name, *llm_aliases]:
                        if (alias.lower() != pending.name.lower()
                                and alias.lower() not in {a.lower() for a in pending.aliases}):
                            pending.aliases.append(alias)
                    if llm_description and not pending.description:
                        pending.description = llm_description
                    pending.salience = max(
                        pending.salience, entity_dict.get("confidence", 0.5))
                    entity_map[name.lower()] = pending.id
                    _index_pending(pending)  # new aliases become keys
                    metrics.entities_deduped_in_batch += 1
                    continue

            # 2) Name / alias / slug lookup
            if not resolved:
                resolved = semantic.find_entity_by_name_or_alias(
                    name=name, user_id=user_id, channel=channel,
                )

            # 2.5) Semantic band: embedding similarity against stored entities
            # (catches "RAG" vs "Retrieval-Augmented Generation" — lexically
            # disjoint forms of the same thing). Auto-link only above the
            # conservative threshold; gray zone is logged, never merged.
            candidate_embedding: list[float] | None = None
            if not resolved and get_settings().semantic_entity_linking_enabled:
                try:
                    candidate_embedding = memory.embedder.embed_single(
                        Entity.compute_embedding_text(name, llm_description, etype)
                    )
                except Exception as e:
                    logger.debug(f"candidate embed failed for '{name}': {e}")
                resolved = _resolve_entity_semantic(
                    memory, name, etype, candidate_embedding,
                    user_id, channel, metrics,
                )

            if resolved:
                resolved_id = resolved["id"]
                # Fold in any new aliases / fill description if missing.
                # The LLM's `name` itself becomes an alias if it differs from canonical.
                merged_aliases = list(llm_aliases)
                if name.lower() != (resolved.get("name") or "").lower():
                    merged_aliases.append(name)
                try:
                    semantic.merge_entity_aliases(
                        entity_id=resolved_id,
                        user_id=user_id,
                        aliases=merged_aliases,
                        description=llm_description,
                        properties=llm_properties,
                    )
                except Exception as merge_err:
                    logger.warning(
                        f"merge_entity_aliases failed for {resolved_id}: {merge_err}"
                    )

                entity_map[name.lower()] = resolved_id
                # Also map canonical name for downstream relationship resolution
                canonical = resolved.get("name")
                if canonical:
                    entity_map[canonical.lower()] = resolved_id
                reused += 1
                metrics.entities_reused += 1
                continue

            # 3) New entity (reuse the band-lookup vector — embed once)
            entity = Entity(
                id=str(uuid4()),
                name=name,
                type=etype,
                description=llm_description,
                aliases=llm_aliases,
                properties=llm_properties or {},
                salience=entity_dict.get("confidence", 0.5),
                embedding=candidate_embedding,
            )
            entities_to_store.append(entity)
            entity_map[name.lower()] = entity.id
            _index_pending(entity)

        except Exception as e:
            logger.warning(f"Failed to prepare entity {entity_dict.get('name')}: {e}")
            metrics.storage_errors += 1
            errors.append(f"entity_prep:{conv_id}:{e}")

    return entities_to_store, reused, len(entities_to_store) + reused


def _batch_store_entities(
    session,
    entities: list[Entity],
    conv_id: str,
    user_id: str,
    channel: str,
) -> int:
    """
    Store multiple entities in a single Neo4j transaction using UNWIND.

    Args:
        session: Neo4j session
        entities: List of Entity objects to store
        conv_id: Conversation ID for MENTIONS relationship
        user_id: User ID
        channel: Memory channel

    Returns:
        Number of entities stored
    """
    if not entities:
        return 0

    # Convert entities to dictionaries for Neo4j
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

    # Batch upsert entities and create MENTIONS relationships
    session.run("""
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
    """, entities=entity_data, conv_id=conv_id)

    return len(entities)


def _resolve_relationship_endpoint(
    name: str,
    memory: AgentMemory,
    user_id: str,
    channel: str,
    entity_map: dict[str, str],
    metrics: ConsolidationMetrics | None,
) -> str | None:
    """Recover a relationship endpoint missing from the batch entity_map.

    Exact name/alias/slug lookup first, then the semantic auto-band only
    (no gray-zone linking for endpoints — a wrong edge anchor is worse than a
    dropped edge). Successful resolutions are cached into entity_map so later
    rels in the same batch hit the fast path.
    """
    try:
        found = memory.semantic.find_entity_by_name_or_alias(
            name=name, user_id=user_id, channel=channel,
        )
        if found:
            entity_map[name.lower()] = found["id"]
            return found["id"]
        if get_settings().semantic_entity_linking_enabled and metrics is not None:
            embedding = memory.embedder.embed_single(
                Entity.compute_embedding_text(name, None, "Concept")
            )
            cand = _resolve_entity_semantic(
                memory, name, None, embedding, user_id, channel, metrics,
            )
            if cand:
                entity_map[name.lower()] = cand["id"]
                return cand["id"]
    except Exception as e:
        logger.debug(f"relationship endpoint recovery failed for '{name}': {e}")
    return None


def _batch_store_relationships(
    session,
    relationships: list[dict[str, Any]],
    entity_map: dict[str, str],
    memory: AgentMemory | None = None,
    user_id: str | None = None,
    channel: str | None = None,
    metrics: ConsolidationMetrics | None = None,
) -> int:
    """
    Store multiple relationships in a single Neo4j transaction using UNWIND.

    Endpoints missing from `entity_map` are recovered via store lookup + the
    semantic auto-band when `memory` is provided; irrecoverable relationships
    are counted (`metrics.relationships_dropped`) and logged — previously they
    were dropped silently (a hidden cause of graph edge sparsity).

    Args:
        session: Neo4j session
        relationships: List of relationship dictionaries
        entity_map: Map of lowercase entity name to entity ID
        memory/user_id/channel/metrics: enable endpoint recovery when given

    Returns:
        Number of relationships stored
    """
    if not relationships:
        return 0

    # Filter to only relationships with valid entity mappings
    valid_rels = []
    for rel in relationships:
        source_id = entity_map.get(rel["source"].lower())
        target_id = entity_map.get(rel["target"].lower())
        if memory is not None and user_id is not None and channel is not None:
            if not source_id:
                source_id = _resolve_relationship_endpoint(
                    rel["source"], memory, user_id, channel, entity_map, metrics)
            if not target_id:
                target_id = _resolve_relationship_endpoint(
                    rel["target"], memory, user_id, channel, entity_map, metrics)
        if source_id and target_id:
            valid_rels.append({
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel["type"],
                "confidence": rel.get("confidence", 0.7),
            })
        else:
            logger.debug(
                f"dropping relationship {rel.get('source')!r}-[{rel.get('type')!r}]->"
                f"{rel.get('target')!r}: unresolved endpoint(s)"
            )
            if metrics is not None:
                metrics.relationships_dropped += 1

    if not valid_rels:
        return 0

    session.run("""
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
    """, rels=valid_rels)

    return len(valid_rels)


def _resolve_subject_channel(
    subject: str | None,
    active_channel: str,
    agent_id: str | None,
    subject_agent_id: str | None = None,
) -> str:
    """Map a fact's subject to the channel it belongs in.

    Facts are extracted from both user and assistant turns; routing by *subject*
    (not turn role) keeps the user's memory and each agent's self-knowledge from
    bleeding into each other:
      - a fact attributed to a *specific* agent (``subject_agent_id``, resolved from
        the LLM-supplied name) → that agent's ``_self_{subject_agent_id}`` — so a
        directive aimed at Mobius lands in Mobius's memory, not Atlas's;
      - a bare ``agent`` subject (the conversation's addressed/producing agent) →
        ``_self_{agent_id}`` (legacy fallback);
      - ``user`` / ``third_party`` / agent-without-an-agent → the active channel.
    """
    if subject == "agent" and subject_agent_id:
        return f"_self_{subject_agent_id}"
    if subject == "agent" and agent_id:
        return f"_self_{agent_id}"
    return active_channel


def _get_or_create_memory(
    memory_cache: dict[str, AgentMemory],
    user_id: str,
    channel: str,
    agent_id: str | None = None,
) -> AgentMemory:
    """Return a cached AgentMemory for (user_id, channel), creating it if needed."""
    cache_key = f"{user_id}:{channel}"
    mem = memory_cache.get(cache_key)
    if mem is None:
        from ..memory.interface import AgentMemory as _AM
        mem = _AM(user_id=user_id, channel=channel, agent_id=agent_id)
        memory_cache[cache_key] = mem
    return mem


def _make_subject_router(
    memory_cache: dict[str, AgentMemory],
    user_id: str,
    active_channel: str,
    agent_id: str | None,
) -> Callable[[dict[str, Any]], tuple[AgentMemory, str]]:
    """Build a ``route(fact) -> (memory, channel)`` closure for fact storage.

    The fact dict carries ``subject`` and an optional ``subject_agent_id`` (the
    specific agent the fact concerns, resolved from the LLM-supplied name). The
    active-channel memory is expected to already be cached; cross-subject targets
    (any agent's self-channel) are built lazily on first use, so a single
    conversation can fan facts out to several agents' ``_self_`` channels.
    """
    def route(fact: dict[str, Any]) -> tuple[AgentMemory, str]:
        channel = _resolve_subject_channel(
            fact.get("subject"), active_channel, agent_id,
            subject_agent_id=fact.get("subject_agent_id"),
        )
        return _get_or_create_memory(memory_cache, user_id, channel, agent_id), channel

    return route


def _ensure_agent_entities(
    session,
    memory_cache: dict[str, AgentMemory],
    user_id: str,
    roster: list[dict[str, str]],
) -> None:
    """Upsert one first-class ``Agent`` entity per roster member, in ``_global``.

    Makes agents linkable like any other entity: a fact about "Mobius" resolves
    (via ``find_entity_by_name_or_alias``, which searches the fact's channel +
    ``_global``) to this node, whose canonical key is ``properties.agent_id`` —
    the source of truth, stable across renames. The display name is stamped at
    write-time (the kit can't reach ``ProfileManager``).

    The node lives in ``_global`` so a single entity serves every channel
    (``_self_*`` + active), and the id is scoped ``agent:{user_id}:{agent_id}``
    so it is one-per-user (no cross-user bleed) and idempotent. Failures never
    abort consolidation.
    """
    if not roster:
        return
    from ..models import Entity

    for member in roster:
        agent_id = (member or {}).get("agent_id")
        if not agent_id:
            continue
        ent_id = f"agent:{user_id}:{agent_id}"
        name = member.get("name") or agent_id
        try:
            exists = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e.id LIMIT 1", id=ent_id,
            ).single()
            if exists:
                continue
            mem = _get_or_create_memory(memory_cache, user_id, "_global")
            mem.upsert_entity(Entity(
                id=ent_id,
                name=name,
                type="Agent",
                description=f"AI agent (id {agent_id})",
                properties={"agent_id": agent_id},
            ))
        except Exception as e:  # noqa: BLE001 — never let this abort consolidation
            logger.debug(f"ensure_agent_entity failed for {agent_id}: {e}")


def _resolve_fact_entity_ids(
    memory: AgentMemory,
    session,
    entity_names: list[str],
    entity_map: dict[str, str],
    user_id: str,
    channel: str,
    conv_id: str,
    metrics: ConsolidationMetrics,
) -> list[str]:
    """Resolve a fact's ``entity_names`` to entity ids, bulletproofing the ABOUT link.

    A fact's entity_names previously linked only when the exact lowercased name was in
    the batch-local ``entity_map`` — so cross-batch entities, alias/variant forms, and
    names the LLM mentioned but didn't list as entities were silently dropped, leaving
    facts orphaned (no ``(Fact)-[:ABOUT]->(Entity)`` edge). This resolves each name via:

      1. ``entity_map``                              — fast path (this batch)
      2. ``find_entity_by_name_or_alias``            — name / alias / slug, across the store
      3. auto-create a stub :Entity (if enabled)     — so the link is never lost

    Every resolution is cached back into ``entity_map``. Returns a deduped,
    order-preserving list of entity ids. Records recovery + stub metrics.
    """
    settings = get_settings()
    ids: list[str] = []
    seen: set = set()

    for raw in entity_names or []:
        name = (raw or "").strip()
        if not name:
            continue
        key = name.lower()

        # 1) Batch fast path
        eid = entity_map.get(key)

        # 2) Store lookup (cross-batch / alias / slug)
        if not eid:
            try:
                resolved = memory.semantic.find_entity_by_name_or_alias(
                    name=name, user_id=user_id, channel=channel,
                )
            except Exception as e:  # noqa: BLE001 — never let linking abort fact storage
                logger.debug(f"Entity lookup failed for '{name}': {e}")
                resolved = None
            if resolved:
                eid = resolved["id"]
                entity_map[key] = eid
                metrics.fact_entity_links_recovered += 1

        # 3) Auto-create a stub so the fact is never orphaned
        if not eid and settings.link_autocreate_stub_entities:
            try:
                stub = Entity(
                    id=str(uuid4()),
                    name=name,
                    type="Concept",  # unknown type; enriched on a later real mention
                    salience=0.3,
                )
                # Persist immediately so the subsequent learn_fact MATCH finds it.
                _batch_store_entities(session, [stub], conv_id, user_id, channel)
                eid = stub.id
                entity_map[key] = eid
                metrics.fact_entity_stubs_created += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Stub entity creation failed for '{name}': {e}")
                eid = None

        if eid and eid not in seen:
            seen.add(eid)
            ids.append(eid)

    return ids


# Candidate name regex: capitalized words (incl. inner caps like PostgreSQL),
# acronyms (2-6 caps), and quoted spans. Cheap pre-pass; the LLM normalizes.
_NAME_CANDIDATE_RE = re.compile(
    r'"([^"\n]{2,40})"'                       # double-quoted spans
    r"|'([^'\n]{2,40})'"                      # single-quoted spans
    r"|\b([A-Z][A-Za-z0-9+#.\-]{1,30}"
    r"(?:\s+[A-Z][A-Za-z0-9+#.\-]{1,30}){0,3})\b"  # Capitalized name (1-4 words)
    r"|\b([A-Z]{2,6})\b"                       # ALL-CAPS acronyms
)

# Stopwords stripped from the candidate set — words that match the capitalized
# regex but rarely refer to extractable entities.
_NAME_STOPWORDS = frozenset({
    "I", "I'm", "I'd", "I've", "I'll", "My", "Me", "Mine",
    "You", "Your", "We", "Our", "He", "She", "They", "Their",
    "The", "A", "An", "This", "That", "These", "Those",
    "Yes", "No", "Ok", "Okay", "Sure", "Thanks", "Hi", "Hello", "Hey",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
})


def _extract_name_candidates(text: str, max_candidates: int = 12) -> list[str]:
    """Cheap regex pre-pass: pull proper-noun-shaped tokens from a turn."""
    if not text:
        return []
    # Preserve original casing of first occurrence; dedup by lowercase
    seen_lower: set[str] = set()
    out: list[str] = []
    for match in _NAME_CANDIDATE_RE.finditer(text):
        candidate = next((g for g in match.groups() if g), None)
        if not candidate:
            continue
        candidate = candidate.strip()
        if len(candidate) < 2 or candidate in _NAME_STOPWORDS:
            continue
        key = candidate.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(candidate)
        if len(out) >= max_candidates:
            break
    return out


def _build_scope_context(
    memory: AgentMemory,
    text: str,
    user_id: str,
    channel: str,
    max_entities: int = 8,
    max_facts_per_entity: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build (known_entities, known_facts) for the extraction prompt.

    Pulls existing entities matching capitalized tokens in the turn (case-insensitive
    name/alias/slug), plus the top facts attached to those entities. Caps total
    entities at `max_entities` and total facts at `max_entities * max_facts_per_entity`.
    Returns ([], []) when nothing relevant is in scope — the prompt then renders
    "(none)" blocks and the call behaves like the old stateless extraction.
    """
    candidates = _extract_name_candidates(text)
    if not candidates:
        return [], []

    semantic = memory.semantic
    entities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for name in candidates:
        if len(entities) >= max_entities:
            break
        try:
            match = semantic.find_entity_by_name_or_alias(
                name=name, user_id=user_id, channel=channel,
            )
        except Exception as e:
            logger.debug(f"scope-context lookup failed for {name!r}: {e}")
            continue
        if match and match["id"] not in seen_ids:
            entities.append(match)
            seen_ids.add(match["id"])

    if not entities:
        return [], []

    # Pull a few facts per entity. `get_entity_facts_and_relationships` returns
    # facts ordered by confidence DESC, so a head slice is fine.
    facts: list[dict[str, Any]] = []
    fact_ids: set[str] = set()
    for ent in entities:
        try:
            ctx = semantic.get_entity_facts_and_relationships(ent["id"], user_id)
        except Exception as e:
            logger.debug(f"fact fetch failed for {ent['id']}: {e}")
            continue
        for f in (ctx.get("facts") or [])[:max_facts_per_entity]:
            fid = f.get("id")
            if fid and fid not in fact_ids:
                facts.append(f)
                fact_ids.add(fid)

    return entities, facts


# --- Consolidation pipeline: per-stage helpers + coordinator ---------------
#
# `consolidate_episodic_to_semantic` below is a thin coordinator. The work of a
# single conversation is split across the helpers in this section so each stage
# is independently testable. All helpers share the open Neo4j ``session`` and
# mutate the passed-in ``metrics`` / ``errors`` in place (mirroring the existing
# ``_resolve_and_prepare_entities`` convention) — this keeps the decomposition a
# pure relocation of the original god-function's behavior.


@dataclass
class _ConvExtraction:
    """Result of relevance-filtering + extracting one conversation's user turns."""
    relevant_turns: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    corrections_applied: int = 0
    extraction_failed: bool = False
    # Turn ids that were successfully poured over (window extracted cleanly,
    # relevant OR not). These are marked ``t.consolidated`` so they are never
    # re-processed; turns of a *failed* window are omitted so they retry.
    processed_turn_ids: list[str] = field(default_factory=list)


@dataclass
class _FactStoreResult:
    """Outcome of running the fact verification pipeline for one conversation."""
    stored_fact_ids: list[str] = field(default_factory=list)
    fact_count: int = 0
    skipped_duplicates: int = 0
    contradictions_found: int = 0
    skipped_contradictions: int = 0


def _fetch_pending_conversations(
    session, only_conversation_id: str | None = None,
) -> tuple[list[Any], int]:
    """Discover conversations with unconsolidated user turns.

    Returns ``(records, total_in_neo4j)`` where ``records`` are the
    conversations needing processing (limit 10) and ``total_in_neo4j`` is the
    count seen by the debug census query.
    """
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
    # ``responder_agent_id`` (the agent that produced the immediately following turn)
    # resolves "you"/"your" per user turn in multi-agent conversations.
    # Turn-level idempotency: a turn is consolidated exactly once. Discover only
    # user turns not yet marked ``t.consolidated`` and collect only those — an
    # already-fully-consolidated conversation yields no rows, so an idempotent
    # re-sweep is a no-op (no re-pouring old turns; dedup is not the safety net).
    result = session.run("""
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
    """, only=only_conversation_id)

    records = list(result)
    logger.info(f"Consolidation: {len(records)} conversations need processing")
    return records, len(conversations_found)


def _assemble_windows(
    turns: list[dict[str, Any]], max_tokens: int, max_turns: int,
) -> list[list[dict[str, Any]]]:
    """Greedy token/turn-budgeted window assembly (pure, order-preserving).

    Never splits a turn; a single turn over the token budget becomes its own
    window. Sized so worst-case extraction JSON stays inside the combined
    stage's output cap.
    """
    from ....tokens import estimate_tokens

    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for turn in turns:
        turn_tokens = estimate_tokens(turn.get("content") or "")
        if current and (current_tokens + turn_tokens > max_tokens
                        or len(current) >= max_turns):
            windows.append(current)
            current, current_tokens = [], 0
        current.append(turn)
        current_tokens += turn_tokens
    if current:
        windows.append(current)
    return windows


def _get_conversation_overview(conv_id: str) -> str | None:
    """Rolling conversation summary for the window prompt's OVERVIEW block.

    Lazy kit→agent import, best-effort by design: the summary lives in the
    chat session layer (Redis ``conv_summary:{id}``) and may simply not exist.
    """
    try:
        from agentx_ai.agent.conversation_summary_storage import get_summary
        return get_summary(conv_id)
    except Exception:
        return None


def _accumulate_registry(
    registry_entities: dict[str, dict[str, Any]],
    registry_facts: list[dict[str, Any]],
    result,
) -> None:
    """Fold a window's extractions into the rolling registry (name-lower dedup:
    union aliases, max confidence). Blunts registry echo — a re-emitted entity
    merges here before it ever reaches storage."""
    for ent in result.entities:
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        existing = registry_entities.get(key)
        if existing is None:
            registry_entities[key] = {
                "name": name,
                "type": ent.get("type"),
                "aliases": list(ent.get("aliases") or []),
                "confidence": ent.get("confidence", 0.5),
            }
        else:
            for alias in ent.get("aliases") or []:
                if alias.lower() not in {a.lower() for a in existing["aliases"]}:
                    existing["aliases"].append(alias)
            existing["confidence"] = max(
                existing["confidence"], ent.get("confidence", 0.5))
    for fact in result.facts:
        if fact.get("claim"):
            registry_facts.append({"claim": fact["claim"]})


async def _extract_from_conversation(
    turns, memory, session, extraction_service, user_id, channel,
    conv_id, metrics, errors, roster=None, default_agent_id=None,
) -> _ConvExtraction:
    """Apply correction detection + combined relevance/extraction over a
    conversation's user turns, accumulating entities/facts/relationships.

    Windowed by default (§2.10 Slice 1): turns are pre-filtered for
    pleasantries, assembled into token-budgeted windows, and extracted one
    window per LLM call — with a rolling registry + conversation overview so
    later windows reuse earlier names. `extraction_windowing_enabled=False`
    falls back to the legacy per-turn path.

    ``roster`` ([{agent_id, name}]) lets the extractor attribute facts to specific
    agents by name; the per-turn ``responder_agent_id`` (falling back to
    ``default_agent_id``) tells it which agent "you" addresses.
    """
    if get_settings().extraction_windowing_enabled:
        return await _extract_windowed(
            turns, memory, session, extraction_service, user_id, channel,
            conv_id, metrics, errors, roster=roster,
            default_agent_id=default_agent_id,
        )
    return await _extract_per_turn(
        turns, memory, session, extraction_service, user_id, channel,
        conv_id, metrics, errors, roster=roster,
        default_agent_id=default_agent_id,
    )


async def _extract_windowed(
    turns, memory, session, extraction_service, user_id, channel,
    conv_id, metrics, errors, roster=None, default_agent_id=None,
) -> _ConvExtraction:
    """Windowed extraction: pre-filter → per-turn corrections → windows with
    registry/overview context → split-retry on parse failure."""
    settings = get_settings()
    out = _ConvExtraction()
    relevance_start = time.perf_counter()

    # Pleasantry pre-filter BEFORE windowing (a skip must not consume budget).
    kept: list[dict[str, Any]] = []
    for turn in turns:
        metrics.turns_total += 1
        if extraction_service.is_heuristic_skip(turn.get("content") or ""):
            metrics.turns_skipped_heuristic += 1
            continue
        kept.append(turn)

    # Correction detection stays per-turn (identical semantics to the legacy path).
    if settings.correction_detection_enabled:
        for turn in kept:
            correction = await extraction_service.check_correction(turn["content"])
            metrics.correction_calls += 1
            if correction.is_correction:
                if _handle_user_correction(memory, session, correction, user_id, channel):
                    out.corrections_applied += 1
                    metrics.corrections_applied += 1
                # Still extract from the turn — it may have new info

    if not kept:
        metrics.relevance_latency_ms += int((time.perf_counter() - relevance_start) * 1000)
        return out

    windows = _assemble_windows(
        kept, settings.extraction_window_max_tokens, settings.extraction_window_max_turns)
    overview = _get_conversation_overview(conv_id)
    name_by_agent = {a.get("agent_id"): a.get("name") for a in (roster or [])}
    registry_entities: dict[str, dict[str, Any]] = {}
    registry_facts: list[dict[str, Any]] = []

    async def _run_window(window: list[dict[str, Any]]):
        payload = []
        for i, turn in enumerate(window, start=1):
            addressed_id = turn.get("responder_agent_id") or default_agent_id
            payload.append({
                "index": i,
                "content": turn["content"],
                "turn_id": turn.get("id"),
                "addressed_agent_id": addressed_id,
                "addressed_agent_name": name_by_agent.get(addressed_id),
            })
        scope_entities, scope_facts = _build_scope_context(
            memory=memory,
            text="\n".join(t["content"] for t in window),
            user_id=user_id,
            channel=channel,
        )
        reg_ents = list(registry_entities.values())[-settings.extraction_registry_max_entities:]
        reg_facts = registry_facts[-settings.extraction_registry_max_facts:]
        result = await extraction_service.check_relevance_and_extract_window(
            payload,
            known_entities=scope_entities,
            known_facts=scope_facts,
            registry_entities=reg_ents,
            registry_facts=reg_facts,
            conversation_overview=overview,
            roster=roster,
        )
        metrics.extraction_calls += 1
        metrics.extraction_windows += 1
        metrics.total_tokens_used += result.tokens_used
        return result

    async def _ingest(window: list[dict[str, Any]], depth: int = 0) -> None:
        result = await _run_window(window)
        if not result.success:
            # Parse failure / truncation beyond repair: split once and retry
            # each half (a smaller window ≈ smaller output). Depth-1 only.
            if depth == 0 and len(window) > 1:
                metrics.window_retries += 1
                mid = len(window) // 2
                logger.warning(
                    f"Window extraction failed for {conv_id} "
                    f"({len(window)} turns) — splitting and retrying: {result.error}"
                )
                await _ingest(window[:mid], depth=1)
                await _ingest(window[mid:], depth=1)
                return
            out.extraction_failed = True
            logger.warning(f"Extraction failed for window in {conv_id}: {result.error}")
            return
        # The window extracted cleanly (relevant or not) → its turns are poured
        # over exactly once; mark them so they're never re-consolidated.
        out.processed_turn_ids.extend(t["id"] for t in window if t.get("id"))
        if result.is_relevant:
            out.relevant_turns.extend(window)
            metrics.turns_relevant += len(window)
            out.entities.extend(result.entities)
            out.facts.extend(result.facts)
            out.relationships.extend(result.relationships)
            metrics.entities_extracted += len(result.entities)
            metrics.facts_extracted += len(result.facts)
            metrics.relationships_extracted += len(result.relationships)
            _accumulate_registry(registry_entities, registry_facts, result)
        else:
            metrics.turns_skipped_llm += len(window)
            logger.debug(f"Window judged irrelevant ({len(window)} turns, {result.reason})")

    for window in windows:
        await _ingest(window)

    metrics.relevance_latency_ms += int((time.perf_counter() - relevance_start) * 1000)
    return out


async def _extract_per_turn(
    turns, memory, session, extraction_service, user_id, channel,
    conv_id, metrics, errors, roster=None, default_agent_id=None,
) -> _ConvExtraction:
    """Legacy per-turn extraction (flag-off fallback; pre-Slice-1 behavior)."""
    out = _ConvExtraction()

    relevance_start = time.perf_counter()
    for turn in turns:
        content = turn['content']
        addressed_agent_id = turn.get('responder_agent_id') or default_agent_id
        metrics.turns_total += 1

        # Check for user corrections first (before relevance filter)
        if get_settings().correction_detection_enabled:
            correction = await extraction_service.check_correction(content)
            metrics.correction_calls += 1
            if correction.is_correction:
                if _handle_user_correction(memory, session, correction, user_id, channel):
                    out.corrections_applied += 1
                    metrics.corrections_applied += 1
                # Still extract from the turn - it may have new info

        # Build scope context so the LLM can mark mentions with
        # existing_entity_id and emit refines_fact_id when applicable.
        scope_entities, scope_facts = _build_scope_context(
            memory=memory,
            text=content,
            user_id=user_id,
            channel=channel,
        )

        # Combined relevance + extraction in a single LLM call (~75% fewer calls)
        combined_result = await extraction_service.check_relevance_and_extract(
            content,
            known_entities=scope_entities,
            known_facts=scope_facts,
            roster=roster,
            addressed_agent_id=addressed_agent_id,
        )
        metrics.extraction_calls += 1
        metrics.total_tokens_used += combined_result.tokens_used

        # Track extraction failures - don't mark conversation as consolidated
        if not combined_result.success:
            out.extraction_failed = True
            logger.warning(f"Extraction failed for turn in {conv_id}: {combined_result.error}")
            continue  # Skip this turn but continue with others

        # Extracted cleanly (relevant or not) → mark it processed-once.
        if turn.get("id"):
            out.processed_turn_ids.append(turn["id"])

        if combined_result.is_relevant:
            out.relevant_turns.append(turn)
            metrics.turns_relevant += 1
            out.entities.extend(combined_result.entities)
            out.facts.extend(combined_result.facts)
            out.relationships.extend(combined_result.relationships)
            metrics.entities_extracted += len(combined_result.entities)
            metrics.facts_extracted += len(combined_result.facts)
            metrics.relationships_extracted += len(combined_result.relationships)
        else:
            if combined_result.reason == "heuristic_skip":
                metrics.turns_skipped_heuristic += 1
            else:
                metrics.turns_skipped_llm += 1
            logger.debug(f"Skipping irrelevant turn: {content[:50]}... ({combined_result.reason})")
    metrics.relevance_latency_ms += int((time.perf_counter() - relevance_start) * 1000)

    return out


def _store_conversation_entities(
    session, memory, extracted_entities, entity_map,
    user_id, channel, conv_id, metrics, errors,
) -> int:
    """Resolve extracted entities against existing ones and batch-store the new
    ones. Fills ``entity_map`` (lowercase name -> id) and returns the count."""
    entities_to_store, entities_reused, entity_count = _resolve_and_prepare_entities(
        memory=memory,
        extracted_entities=extracted_entities,
        entity_map=entity_map,
        user_id=user_id,
        channel=channel,
        conv_id=conv_id,
        metrics=metrics,
        errors=errors,
    )

    # Embed new entities at store time so vector_search_entities can see them
    # (consolidation entities historically stored embedding=None, blinding the
    # semantic linking band + --semantic dedupe). Batch call; failure degrades
    # to embedding=None — the entity_linking job's backfill catches those.
    try:
        pending_embed = [e for e in entities_to_store
                         if getattr(e, "embedding", None) is None
                         and hasattr(e, "embedding_text")]
        if pending_embed:
            vectors = memory.embedder.embed([e.embedding_text() for e in pending_embed])
            for ent, vec in zip(pending_embed, vectors, strict=True):
                ent.embedding = vec
    except Exception as embed_err:
        logger.warning(
            f"Entity batch embed failed for {conv_id} (backfill will retry): {embed_err}"
        )

    # Batch store only the genuinely new entities
    try:
        stored_count = _batch_store_entities(
            session, entities_to_store, conv_id, user_id, channel
        )
        metrics.entities_stored += stored_count
        entity_count = stored_count + entities_reused
    except Exception as e:
        logger.warning(f"Batch entity storage failed for {conv_id}: {e}")
        metrics.storage_errors += 1
        errors.append(f"entity_batch:{conv_id}:{e}")
        entity_count = entities_reused

    return entity_count


async def _store_facts_with_verification(
    session, route, extraction_service, extracted_facts, entity_map,
    user_id, active_channel, conv_id, metrics, errors,
) -> _FactStoreResult:
    """Run the four-layer fact pipeline (LLM-refine → hash gate → semantic gate
    → contradiction candidates → temporal/LLM adjudication → store) over the
    extracted facts. Returns the stored fact ids plus per-conversation counters.

    ``route(subject) -> (memory, channel)`` selects where each fact is stored: a
    fact's subject (user/agent/third_party) — not the turn it came from — decides
    its channel, so user facts and agent self-knowledge stay separated. All
    channel-scoped checks (dedup, contradiction, entity linking) run against the
    fact's resolved target channel."""
    settings = get_settings()
    res = _FactStoreResult()

    # Three-layer verification pipeline: hash gate → semantic gate → LLM adjudication
    # Collect fact IDs for batch DERIVED_FROM relationship creation
    embedder = get_embedder()
    # Per-channel entity maps so a routed fact never links to an entity in another
    # channel. The active channel reuses the conversation's prebuilt map.
    entity_maps: dict[str, dict[str, str]] = {active_channel: entity_map}

    for fact_dict in extracted_facts:
        try:
            # Validate required fields
            claim = fact_dict.get("claim")
            if not claim:
                logger.warning(f"Skipping fact with missing claim: {fact_dict}")
                continue

            # Route by subject (+ specific agent): user/third_party → active channel,
            # agent → that agent's self channel.
            fact_dict.setdefault("subject", "user")
            fact_memory, fact_channel = route(fact_dict)
            fact_entity_map = entity_maps.setdefault(fact_channel, {})

            # === Layer 0: LLM-supplied refinement ===
            # If extraction marked this claim as refining a known fact, supersede
            # directly via the existing PREFER_NEW path and skip the rest of the
            # pipeline. Validates the target fact is in scope before acting.
            refines_id = fact_dict.get("refines_fact_id")
            if refines_id:
                target = fact_memory.semantic.get_fact_by_id(refines_id, user_id)
                target_channel = (target or {}).get("channel")
                in_scope = bool(target) and target_channel in (fact_channel, "_global")
                if in_scope:
                    _handle_contradiction(
                        fact_memory, session, fact_dict,
                        type('Result', (), {
                            'has_contradiction': True,
                            'contradicting_fact_id': refines_id,
                            'resolution': 'prefer_new',
                            'reason': 'llm_refinement',
                        })(),
                        user_id, fact_channel,
                    )
                    res.fact_count += 1
                    metrics.facts_stored += 1
                    metrics.facts_superseded_by_refine += 1
                    logger.info(f"Refined fact {refines_id} via llm_refinement: {claim[:50]}...")
                    continue
                else:
                    logger.debug(
                        f"refines_fact_id {refines_id} not in scope; "
                        f"falling through to standard pipeline"
                    )

            # === Layer 1: Fast Gate (no LLM) ===

            # 1a. Exact hash duplicate check
            if _is_duplicate_fact(session, claim, user_id, fact_channel):
                res.skipped_duplicates += 1
                metrics.duplicates_skipped += 1
                logger.debug(f"Skipping hash duplicate: {claim[:50]}...")
                continue

            # Generate embedding early (needed for layers 1b + 2)
            claim_embedding = embedder.embed_single(claim)

            # 1b. Semantic duplicate check (cosine > threshold)
            if _is_semantic_duplicate(
                session, claim_embedding, user_id, fact_channel,
                threshold=settings.semantic_duplicate_threshold,
            ):
                res.skipped_duplicates += 1
                metrics.duplicates_skipped += 1
                metrics.semantic_duplicates_skipped += 1
                logger.debug(f"Skipping semantic duplicate: {claim[:50]}...")
                continue

            # === Layer 2: Semantic Search for Contradiction Candidates (no LLM) ===
            if settings.contradiction_detection_enabled:
                entity_names = fact_dict.get("entity_names", [])
                candidates = _get_contradiction_candidates(
                    session, claim, claim_embedding, entity_names,
                    user_id, fact_channel,
                    similarity_threshold=settings.contradiction_similarity_threshold,
                    max_candidates=settings.contradiction_max_candidates,
                )
                metrics.contradiction_candidates_found += len(candidates)

                # === Layer 3: LLM Adjudication (only if candidates found) ===
                if candidates:
                    # Check for temporal progressions first (no LLM needed)
                    temporal_resolved = False
                    for candidate in candidates:
                        if _is_temporal_progression(fact_dict, candidate):
                            # Auto-supersede without LLM call
                            _handle_contradiction(
                                fact_memory, session, fact_dict,
                                type('Result', (), {
                                    'has_contradiction': True,
                                    'contradicting_fact_id': candidate['id'],
                                    'resolution': 'prefer_new',
                                    'reason': 'temporal_progression',
                                })(),
                                user_id, fact_channel,
                            )
                            res.contradictions_found += 1
                            metrics.contradictions_found += 1
                            metrics.contradictions_resolved += 1
                            metrics.temporal_progressions_resolved += 1
                            res.fact_count += 1
                            metrics.facts_stored += 1
                            temporal_resolved = True
                            logger.info(f"Auto-resolved temporal progression: {claim[:50]}...")
                            break

                    if temporal_resolved:
                        continue

                    # LLM contradiction check against candidates
                    contradiction = await extraction_service.check_contradictions(
                        claim, candidates,
                        new_temporal=fact_dict.get("temporal_context"),
                        new_confidence=fact_dict.get("confidence"),
                    )
                    metrics.contradiction_calls += 1
                    if contradiction.has_contradiction:
                        res.contradictions_found += 1
                        metrics.contradictions_found += 1
                        action = _handle_contradiction(
                            fact_memory, session, fact_dict, contradiction, user_id, fact_channel
                        )
                        if action == "skipped":
                            res.skipped_contradictions += 1
                            continue
                        elif action == "superseded":
                            res.fact_count += 1
                            metrics.facts_stored += 1
                            metrics.contradictions_resolved += 1
                            continue

            # Link fact to mentioned entities — batch map → store lookup → stub.
            entity_ids = _resolve_fact_entity_ids(
                fact_memory, session, fact_dict.get("entity_names", []),
                fact_entity_map, user_id, fact_channel, conv_id, metrics,
            )

            fact = fact_memory.learn_fact(
                claim=claim,
                source="extraction",
                confidence=fact_dict.get("confidence", 0.7),
                source_turn_id=fact_dict.get("source_turn_id"),
                entity_ids=entity_ids if entity_ids else None,
                temporal_context=fact_dict.get("temporal_context"),
            )
            res.fact_count += 1
            metrics.facts_stored += 1
            res.stored_fact_ids.append(fact.id)

        except Exception as e:
            logger.warning(f"Failed to store fact: {e}")
            metrics.storage_errors += 1
            errors.append(f"fact:{conv_id}:{e}")

    return res


def _link_facts_and_relationships(
    session, conv_id, stored_fact_ids, extracted_relationships,
    entity_map, metrics, errors, memory=None, user_id=None, channel=None,
) -> int:
    """Batch-create DERIVED_FROM edges for stored facts and batch-store the
    extracted relationships. Returns the relationship count."""
    # Batch create DERIVED_FROM relationships for all stored facts
    if stored_fact_ids:
        try:
            session.run("""
                UNWIND $fact_ids AS fid
                MATCH (c:Conversation {id: $conv_id}), (f:Fact {id: fid})
                MERGE (f)-[:DERIVED_FROM]->(c)
            """, conv_id=conv_id, fact_ids=stored_fact_ids)
        except Exception as e:
            logger.warning(f"Batch DERIVED_FROM creation failed: {e}")

    # Batch store relationships using UNWIND (with endpoint recovery)
    try:
        rel_count = _batch_store_relationships(
            session, extracted_relationships, entity_map,
            memory=memory, user_id=user_id, channel=channel, metrics=metrics,
        )
        metrics.relationships_stored += rel_count
    except Exception as e:
        logger.warning(f"Batch relationship storage failed for {conv_id}: {e}")
        metrics.storage_errors += 1
        errors.append(f"relationship_batch:{conv_id}:{e}")
        rel_count = 0

    return rel_count


def _mark_turns_consolidated(session, turn_ids: list[str], prop: str) -> None:
    """Stamp ``t.<prop> = datetime()`` on the given turns so they are never
    re-consolidated. ``prop`` is ``consolidated`` (user turns) or
    ``self_consolidated`` (assistant turns). No-op on an empty list.
    """
    if not turn_ids:
        return
    session.run(
        f"UNWIND $ids AS tid MATCH (t:Turn {{id: tid}}) SET t.{prop} = datetime()",
        ids=turn_ids,
    )


async def _consolidate_user_conversation(
    record, conv_idx, total, session, memory_cache,
    extraction_service, metrics, errors, progress_callback,
) -> None:
    """Consolidate one user conversation: extract → store entities → verify and
    store facts → link relationships → mark consolidated."""
    conv_id = record["conversation_id"]
    user_id = record["user_id"]
    channel = record["channel"]
    turns = record["turns"]

    metrics.conversation_id = conv_id  # Track current conversation
    metrics.user_id = user_id
    metrics.channel = channel

    if progress_callback:
        progress_callback("processing", {
            "conversation": f"{conv_idx + 1} of {total}",
            "conversation_id": conv_id,
            "turns": len(turns),
        })

    # Get or create memory instance for this user/channel
    cache_key = f"{user_id}:{channel}"
    if cache_key not in memory_cache:
        memory_cache[cache_key] = _get_memory_for_user(user_id, channel)
    memory = memory_cache[cache_key]

    # Roster of agents that participated, so the extractor can attribute facts to
    # specific agents by name (agent_id is the durable key). Register each as a
    # first-class Agent entity in the active channel + its own self-channel so the
    # ABOUT link resolves regardless of where the fact lands.
    agent_id = record.get("agent_id")
    roster = memory.get_conversation_roster(conv_id)
    if roster:
        _ensure_agent_entities(session, memory_cache, user_id, roster)

    # Filter turns by relevance and extract entities/facts
    extracted = await _extract_from_conversation(
        turns, memory, session, extraction_service, user_id, channel,
        conv_id, metrics, errors, roster=roster, default_agent_id=agent_id,
    )

    # Mark every successfully-poured-over turn (relevant or not) so it is never
    # re-consolidated — independent of storage below, and of whether OTHER
    # windows failed (those turns are simply absent from processed_turn_ids and
    # retry next sweep). Keep c.consolidated as a coarse "last-touched" marker.
    _mark_turns_consolidated(session, extracted.processed_turn_ids, "consolidated")
    if extracted.processed_turn_ids:
        session.run(
            "MATCH (c:Conversation {id: $conv_id}) SET c.consolidated = datetime()",
            conv_id=conv_id,
        )

    if not extracted.relevant_turns:
        logger.debug(f"No relevant turns in conversation {conv_id}, skipping extraction")
        if extracted.extraction_failed:
            logger.warning(f"Some windows in {conv_id} failed extraction; those turns will retry")
        return

    logger.debug(
        f"Combined extraction result: {len(extracted.entities)} entities, "
        f"{len(extracted.facts)} facts, {len(extracted.relationships)} relationships"
    )

    # Use lowercase keys for case-insensitive matching with relationships
    entity_map: dict[str, str] = {}  # lowercase_name -> entity_id for relationship linking

    # Prepare entities for batch storage
    storage_start = time.perf_counter()

    if progress_callback:
        progress_callback("storing", {
            "conversation": f"{conv_idx + 1} of {total}",
            "entities": len(extracted.entities),
            "facts": len(extracted.facts),
            "relationships": len(extracted.relationships),
        })

    entity_count = _store_conversation_entities(
        session, memory, extracted.entities, entity_map,
        user_id, channel, conv_id, metrics, errors,
    )

    # Store facts via AgentMemory interface (three-layer verification pipeline).
    # Route each fact by subject + specific agent: a fact attributed to an agent
    # lands in that agent's self channel, the rest in the active channel. (agent_id
    # and roster were resolved above.)
    route = _make_subject_router(memory_cache, user_id, channel, agent_id)
    fact_result = await _store_facts_with_verification(
        session, route, extraction_service, extracted.facts, entity_map,
        user_id, channel, conv_id, metrics, errors,
    )

    rel_count = _link_facts_and_relationships(
        session, conv_id, fact_result.stored_fact_ids, extracted.relationships,
        entity_map, metrics, errors, memory=memory, user_id=user_id, channel=channel,
    )

    # Track storage latency
    metrics.storage_latency_ms += int((time.perf_counter() - storage_start) * 1000)

    # Log consolidation results
    extras = []
    if fact_result.skipped_duplicates > 0:
        extras.append(f"{fact_result.skipped_duplicates} duplicates")
    if extracted.corrections_applied > 0:
        extras.append(f"{extracted.corrections_applied} corrections")
    if fact_result.contradictions_found > 0:
        extras.append(
            f"{fact_result.contradictions_found} contradictions "
            f"({fact_result.skipped_contradictions} skipped)"
        )
    extras_msg = f" [{', '.join(extras)}]" if extras else ""
    logger.info(
        f"Consolidated conversation {conv_id}: "
        f"{entity_count} entities, {fact_result.fact_count} facts, {rel_count} relationships{extras_msg}"
    )

    # Turn-level marking + the coarse c.consolidated stamp already happened right
    # after extraction (above) — gated on which turns were actually poured over,
    # so a partial-window failure retries only its own turns next sweep.


async def _consolidate_assistant_conversation(
    record, session, self_memory_cache, extraction_service, metrics, errors,
) -> None:
    """Extract assistant self-knowledge into each producing agent's
    ``_self_{agent_id}`` channel and mark the conversation self-consolidated.

    A conversation can have turns from several agents (delegation, @-mentions,
    Agent Alloy). Each assistant turn is routed by *its own* ``agent_id`` so an
    agent's self-knowledge never bleeds into another agent's channel.
    """
    conv_id = record["conversation_id"]
    user_id = record["user_id"]
    conv_agent_id = record["agent_id"]  # conversation default for legacy turns
    active_channel = record.get("channel", "_default")  # the user's channel
    a_turns = record["turns"]

    # Roster + Agent entities so the extractor can name agents ("I" = the speaker,
    # plus any other agent referenced) and facts link to them.
    roster_memory = _get_or_create_memory(self_memory_cache, user_id, active_channel)
    roster = roster_memory.get_conversation_roster(conv_id)
    if roster:
        _ensure_agent_entities(session, self_memory_cache, user_id, roster)

    # Assistant turns marked ``self_consolidated`` (once), so they never re-process:
    # a deterministic skip (no agent / too short) or a clean extraction; a failed
    # extraction is omitted so the turn retries next sweep.
    processed_ids: list[str] = []

    for turn in a_turns:
        content = turn["content"]
        turn_id = turn.get("id")
        metrics.assistant_turns_total += 1

        # Route this turn by the agent that produced it (the "I"); fall back to the
        # conversation's agent for legacy turns missing per-turn attribution.
        turn_agent_id = turn.get("agent_id") or conv_agent_id
        if not turn_agent_id:
            if turn_id:
                processed_ids.append(turn_id)  # nothing to extract → handled
            continue
        self_channel = f"_self_{turn_agent_id}"
        memory = _get_or_create_memory(
            self_memory_cache, user_id, self_channel, turn_agent_id,
        )
        # agent-subject facts → that agent's self channel; user/third-party facts
        # the agent stated/inferred → the user's active channel.
        route = _make_subject_router(
            self_memory_cache, user_id, active_channel, turn_agent_id,
        )

        # Skip short responses (greetings, acknowledgments)
        if len(content.strip()) < 100:
            if turn_id:
                processed_ids.append(turn_id)  # deterministic skip → handled
            continue

        try:
            self_scope_entities, self_scope_facts = _build_scope_context(
                memory=memory,
                text=content,
                user_id=user_id,
                channel=self_channel,
            )
            result = await extraction_service.check_relevance_and_extract_assistant(
                content,
                source_turn_id=turn_id,
                known_entities=self_scope_entities,
                known_facts=self_scope_facts,
                roster=roster,
                addressed_agent_id=turn_agent_id,
            )
        except Exception as e:
            logger.warning(f"Assistant extraction failed for turn in {conv_id}: {e}")
            errors.append(f"assistant_extraction:{conv_id}:{e}")
            continue  # leave unmarked → retries next sweep

        if not result.success:
            continue  # leave unmarked → retries next sweep

        # Extracted cleanly → this assistant turn is poured over exactly once.
        if turn_id:
            processed_ids.append(turn_id)

        if not result.is_relevant:
            continue

        metrics.assistant_turns_relevant += 1
        metrics.assistant_entities_extracted += len(result.entities)
        metrics.assistant_facts_extracted += len(result.facts)

        # Resolve entities against the self-channel store before storing
        entity_map: dict[str, str] = {}
        entities_to_store, _reused, _ = _resolve_and_prepare_entities(
            memory=memory,
            extracted_entities=result.entities,
            entity_map=entity_map,
            user_id=user_id,
            channel=self_channel,
            conv_id=conv_id,
            metrics=metrics,
            errors=errors,
        )

        if entities_to_store:
            try:
                _batch_store_entities(
                    session, entities_to_store, conv_id, user_id, self_channel,
                )
            except Exception as e:
                logger.warning(f"Self-extraction entity storage failed: {e}")
                errors.append(f"self_entity:{conv_id}:{e}")

        # Store facts with self_extraction source. Per-channel entity maps keep a
        # routed user-subject fact from linking to a self-channel entity.
        entity_maps: dict[str, dict[str, str]] = {self_channel: entity_map}
        for fact_dict in result.facts:
            claim = fact_dict.get("claim")
            if not claim:
                continue

            # Route by subject (+ specific agent); default "agent" for the assistant
            # extractor (the speaking agent).
            fact_dict.setdefault("subject", "agent")
            fact_memory, fact_channel = route(fact_dict)
            fact_entity_map = entity_maps.setdefault(fact_channel, {})

            if _is_duplicate_fact(session, claim, user_id, fact_channel):
                continue

            entity_ids = _resolve_fact_entity_ids(
                fact_memory, session, fact_dict.get("entity_names", []),
                fact_entity_map, user_id, fact_channel, conv_id, metrics,
            )

            try:
                fact_memory.learn_fact(
                    claim=claim,
                    source="self_extraction",
                    confidence=fact_dict.get("confidence", 0.7),
                    source_turn_id=fact_dict.get("source_turn_id"),
                    entity_ids=entity_ids if entity_ids else None,
                )
                metrics.assistant_facts_stored += 1
            except Exception as e:
                logger.warning(f"Self-extraction fact storage failed: {e}")
                errors.append(f"self_fact:{conv_id}:{e}")

        # Store relationships (with endpoint recovery in the self channel)
        if result.relationships:
            try:
                _batch_store_relationships(
                    session, result.relationships, entity_map,
                    memory=memory, user_id=user_id, channel=self_channel,
                    metrics=metrics,
                )
            except Exception as e:
                logger.warning(f"Self-extraction relationship storage failed: {e}")

    # Mark the processed assistant turns (never re-consolidated); a failed turn is
    # omitted and retries next sweep. c.self_consolidated is a coarse last-touched
    # marker, no longer the gate.
    _mark_turns_consolidated(session, processed_ids, "self_consolidated")
    if processed_ids:
        session.run(
            "MATCH (c:Conversation {id: $conv_id}) SET c.self_consolidated = datetime()",
            conv_id=conv_id,
        )


async def consolidate_episodic_to_semantic(
    progress_callback: Callable | None = None,
    only_conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    Extract entities, facts, and relationships from recent episodic memory
    and store in semantic memory using the AgentMemory interface.

    Thin coordinator over the ``_consolidate_*`` helpers above: it discovers
    pending conversations, runs the user-turn pipeline (Phase 1) and the
    assistant self-knowledge pipeline (Phase 2), then finalizes metrics.

    Args:
        progress_callback: Optional callback(stage, details) for progress reporting.
        only_conversation_id: If set, restrict both phases to this one conversation
            (the rest of the cluster is left untouched). Used by the
            ``debug_attribution`` harness for a fast, non-destructive single-run.

    Returns:
        Dictionary with consolidation metrics
    """
    job_start = time.perf_counter()
    job_id = str(uuid4())[:8]

    # Cache memory instances per user to avoid repeated initialization
    memory_cache: dict[str, AgentMemory] = {}

    # Aggregated metrics across all conversations
    metrics = ConsolidationMetrics(
        job_id=job_id,
        started_at=datetime.now(UTC),
    )
    errors: list[str] = []

    # =========================================================================
    # Phase 1: User-turn consolidation into semantic memory
    # =========================================================================
    with Neo4jConnection.session() as session:
        records, total_in_neo4j = _fetch_pending_conversations(session, only_conversation_id)

        if progress_callback:
            progress_callback("discovery", {
                "conversations_found": len(records),
                "total_in_neo4j": total_in_neo4j,
            })

        extraction_service = get_extraction_service()
        for conv_idx, record in enumerate(records):
            await _consolidate_user_conversation(
                record, conv_idx, len(records), session,
                memory_cache, extraction_service, metrics, errors,
                progress_callback,
            )

    # Refresh the cached cross-conversation recap for each user/channel we just
    # consolidated, so recall_user_history has a warm summary to surface. Best
    # effort — never let a recap failure abort consolidation.
    from ..config import get_settings as _get_mem_settings
    if getattr(_get_mem_settings(), "user_recap_enabled", True):
        from ..recap import build_and_cache_user_recap
        for cache_key, mem in list(memory_cache.items()):
            try:
                await build_and_cache_user_recap(mem)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"User recap refresh skipped for {cache_key}: {e}")

    # Clean up memory cache to release resources
    memory_cache.clear()

    # =========================================================================
    # Phase 2: Assistant self-knowledge extraction
    # Extract knowledge from assistant turns into per-agent _self_ channels
    # =========================================================================
    self_memory_cache: dict[str, AgentMemory] = {}

    with Neo4jConnection.session() as session:
        # Turn-level idempotency (assistant self-knowledge): only assistant turns
        # not yet marked ``t.self_consolidated``; collect only those.
        assistant_result = session.run("""
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
        """, only=only_conversation_id)

        assistant_records = list(assistant_result)
        logger.info(f"Self-extraction: {len(assistant_records)} conversations with assistant turns")

        extraction_service = get_extraction_service()

        for record in assistant_records:
            await _consolidate_assistant_conversation(
                record, session, self_memory_cache, extraction_service, metrics, errors,
            )

        if assistant_records:
            logger.info(
                f"Self-extraction complete: {metrics.assistant_turns_relevant} relevant "
                f"of {metrics.assistant_turns_total} turns, "
                f"{metrics.assistant_facts_stored} facts stored"
            )

    self_memory_cache.clear()

    # Finalize metrics
    metrics.completed_at = datetime.now(UTC)
    metrics.total_latency_ms = int((time.perf_counter() - job_start) * 1000)
    metrics.total_llm_calls = (
        metrics.relevance_calls + metrics.correction_calls +
        metrics.extraction_calls + metrics.contradiction_calls
    )
    metrics.errors = errors

    if progress_callback:
        progress_callback("complete", {
            "entities_stored": metrics.entities_stored,
            "facts_stored": metrics.facts_stored,
            "relationships_stored": metrics.relationships_stored,
            "conversations_processed": len(records),
            "duration_ms": metrics.total_latency_ms,
        })

    # Log summary
    metrics.log_summary()

    if errors:
        logger.warning(f"Consolidation had {len(errors)} errors: {errors[:5]}")

    return {
        "items_processed": metrics.turns_total,
        "entities": metrics.entities_stored,
        "facts": metrics.facts_stored,
        "relationships": metrics.relationships_stored,
        "metrics": metrics.to_dict(),
        "errors": errors
    }


def detect_patterns() -> dict[str, Any]:
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


def _derive_procedure_scope(signal: str, channel: str | None, agent_id: str | None) -> str:
    """Channel a distilled Procedure should live in.

    An implicit *correction* aimed at a specific agent routes to that agent's
    ``_self_{agent_id}`` (reusing the subject-channel convention) so it doesn't
    leak to other agents via a shared channel. An explicit user *rule* is a
    standing preference that should apply across the channel, so it inherits the
    candidate's channel (often ``_global``).
    """
    if signal == "correction" and agent_id:
        return _resolve_subject_channel("agent", channel or "_global", agent_id)
    return channel or "_global"


def _resolve_user_for_conversation(conversation_id: Any) -> str:
    """Resolve the owning user_id for a conversation (defaults to 'default')."""
    try:
        with Neo4jConnection.session() as session:
            rec = session.run(
                """
                MATCH (u:User)-[:HAS_CONVERSATION]->(c:Conversation {id: $id})
                RETURN u.id AS uid LIMIT 1
            """,
                id=str(conversation_id),
            ).single()
            if rec and rec["uid"]:
                return rec["uid"]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"User resolve for conversation {conversation_id} failed: {e}")
    return "default"


async def distill_procedures(progress_callback: Callable | None = None) -> dict[str, Any]:
    """Distill pending ``procedure_candidates`` into scoped Procedures (Loop 2).

    Reads high-signal candidates (corrections/steers + explicit user rules), drops
    what a competent agent already does by default (baseline-deviation, signal-aware),
    and distills the survivors into durable, scoped Procedure nodes — strengthening a
    near-duplicate instead of creating a duplicate. The dedupe-on-write step makes a
    re-run after a mid-job crash converge rather than double-create (the Procedure
    write is Neo4j, the candidate mark is PostgreSQL — no cross-store transaction).
    """
    from ..memory.procedural import ProceduralMemory

    procedural = ProceduralMemory()
    extraction_service = get_extraction_service()
    settings = get_settings()
    dedupe_threshold = settings.procedural_dedupe_threshold
    batch_limit = settings.procedural_distill_batch_limit

    created = 0
    reinforced = 0
    discarded = 0
    errors: list[str] = []

    with get_postgres_session() as pg:
        from sqlalchemy import text as _sa_text
        rows = pg.execute(
            _sa_text(
                """
            SELECT id, conversation_id, signal, content, channel, agent_id
            FROM procedure_candidates
            WHERE status = 'pending'
            ORDER BY channel, agent_id, id
            LIMIT :lim
        """
            ),
            {"lim": batch_limit},
        ).fetchall()

    if not rows:
        return {
            "items_processed": 0,
            "procedures_created": 0,
            "procedures_reinforced": 0,
            "discarded": 0,
        }

    # Group candidates by the scope (channel) their procedure will live in.
    groups: dict[str, list[Any]] = {}
    for r in rows:
        scope = _derive_procedure_scope(r.signal, r.channel, r.agent_id)
        groups.setdefault(scope, []).append(r)

    if progress_callback:
        progress_callback("discovery", {"candidates": len(rows), "scopes": len(groups)})

    for scope, cands in groups.items():
        cand_ids = [int(c.id) for c in cands]
        batch = [{"signal": c.signal, "content": c.content} for c in cands]
        signals = sorted({c.signal for c in cands if c.signal})

        try:
            result = await extraction_service.distill_procedure(batch, scope)
        except Exception as e:  # noqa: BLE001
            errors.append(f"distill[{scope}]: {e}")
            continue

        if not result.success:
            # Transient provider issue — leave candidates pending so a later run retries.
            errors.append(f"distill[{scope}]: {result.error}")
            continue

        if not result.keep:
            procedural.mark_candidates(cand_ids, "discarded")
            discarded += len(cand_ids)
            continue

        user_id = _resolve_user_for_conversation(cands[0].conversation_id)
        evidence = [f"cand:{cid}" for cid in cand_ids] + [
            f"conv:{c.conversation_id}" for c in cands
        ]
        conv_ids = list({str(c.conversation_id) for c in cands})
        agent_id = next((c.agent_id for c in cands if c.agent_id), None)

        # Dedupe-on-write: a cosine-similar existing procedure in scope is
        # strengthened instead of duplicated.
        existing = procedural.find_procedures(
            f"{result.trigger}\n{result.body}",
            channels=[scope],
            top_k=1,
            min_score=dedupe_threshold,
        )
        try:
            if existing and procedural.reinforce_procedure(
                existing[0].id,
                evidence_refs=evidence,
                signal_kinds=signals,
                channel=scope,
            ):
                reinforced += 1
                procedural.mark_candidates(cand_ids, "distilled", distilled_into=existing[0].id)
            else:
                proc = procedural.learn_procedure(
                    trigger=result.trigger,
                    body=result.body,
                    rationale=result.rationale,
                    scope=scope,
                    agent_id=agent_id,
                    signal_kinds=signals,
                    evidence_refs=evidence,
                    conversation_ids=conv_ids,
                    user_id=user_id,
                )
                created += 1
                procedural.mark_candidates(cand_ids, "distilled", distilled_into=proc.id)
        except Exception as e:  # noqa: BLE001
            errors.append(f"write[{scope}]: {e}")
            continue

    logger.info(
        f"Procedure distillation: {created} created, {reinforced} reinforced, "
        f"{discarded} discarded from {len(rows)} candidates"
    )
    return {
        "items_processed": len(rows),
        "procedures_created": created,
        "procedures_reinforced": reinforced,
        "discarded": discarded,
        "errors": errors,
    }


def apply_memory_decay() -> dict[str, Any]:
    """
    Apply time-based decay to memory salience scores.

    Returns:
        Dictionary with decay metrics
    """
    decay_rate = get_settings().salience_decay_rate
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


def promote_to_global() -> dict[str, Any]:
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

    settings = get_settings()
    min_confidence = settings.promotion_min_confidence
    min_access = settings.promotion_min_access_count
    min_conversations = settings.promotion_min_conversations

    entities_promoted = 0
    facts_promoted = 0
    entities_updated = 0
    facts_updated = 0
    errors: list[str] = []

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


def cleanup_old_memories() -> dict[str, Any]:
    """Archive or delete old, low-salience memories."""
    retention_days = get_settings().episodic_retention_days
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


def reset_consolidation(delete_memories: bool = False) -> dict[str, Any]:
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
        """)

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
    outcome: dict[str, Any]
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
        "triggered_at": datetime.now(UTC).isoformat()
    }

    redis.xadd("reflection_jobs", {"data": json.dumps(job_data)})
    logger.info(f"Queued reflection for conversation {conversation_id}")


def manage_audit_partitions() -> dict[str, Any]:
    """
    Manage audit log partitions.
    Creates future partitions and drops old ones based on retention settings.

    Returns:
        Dictionary with partition management results
    """
    from sqlalchemy import text

    settings = get_settings()
    retention_days = settings.audit_retention_days
    ahead_days = settings.audit_partition_ahead_days

    partitions_created = 0
    partitions_dropped = 0
    errors: list[str] = []

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


# Tokens that lead third-person claims ("User works at…") or carry no entity weight;
# excluded from the n-gram candidates so they never spuriously match an entity name.
_BACKFILL_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "is", "are", "was", "were", "be", "has", "have", "had", "does", "do", "did",
    "user", "agent", "their", "they", "it", "this", "that", "his", "her",
})


def _slug(text: str) -> str:
    """Lowercased, non-alphanumerics stripped — mirrors find_entity_by_name_or_alias."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _build_entity_index(session, user_id: str, channel: str) -> dict[str, str]:
    """Map name/alias (lowercased) and slug → entity id for a (user, channel) scope.

    Scoped to ``channel`` + ``_global`` and ranked by salience so a name shared by
    two entities resolves to the more salient one (mirrors the live resolver's
    ordering). Built once per channel and reused across that channel's facts.
    """
    rows = session.run("""
        MATCH (e:Entity)
        WHERE e.user_id = $user_id
          AND (e.channel = $channel OR e.channel = '_global')
        RETURN e.id AS id, e.name AS name,
               coalesce(e.aliases, []) AS aliases,
               coalesce(e.salience, 0.5) AS salience
        ORDER BY coalesce(e.salience, 0.5) ASC
    """, user_id=user_id, channel=channel)

    # Ascending salience so higher-salience entries overwrite lower ones on key collision.
    index: dict[str, str] = {}
    for r in rows:
        names = [r["name"], *(r["aliases"] or [])]
        for nm in names:
            if not nm:
                continue
            index[nm.lower()] = r["id"]
            slug = _slug(nm)
            if slug:
                index[slug] = r["id"]
    return index


def _claim_entity_candidates(claim: str, max_ngram: int) -> list[str]:
    """Contiguous 1..max_ngram-word n-grams from a claim, for entity-name matching.

    Punctuation is stripped per word; stopwords are dropped from the unigrams but
    kept inside multi-word grams (e.g. "Bank of America"). Returns lowercased,
    deduped candidates ordered longest-first so the most specific match wins.
    """
    words = [w.strip(".,;:!?\"'()[]{}").strip() for w in claim.split()]
    words = [w for w in words if w]
    candidates: list[str] = []
    seen: set = set()
    for n in range(min(max_ngram, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            low = gram.lower()
            if low in seen:
                continue
            # Drop a lone stopword / too-short unigram; keep multi-word grams.
            if n == 1 and (low in _BACKFILL_STOPWORDS or len(low) < 3):
                continue
            seen.add(low)
            candidates.append(gram)
    return candidates


def _backfill_entity_embeddings(limit: int) -> int:
    """Embed entities stored with ``embedding IS NULL`` (bounded batch).

    Consolidation historically stored entities without embeddings, blinding
    `vector_search_entities` (and thus the §2.10 semantic band + `--semantic`
    dedupe) to them. The live path now embeds at store time; this self-healing
    sweep converges the backlog. Returns the number embedded.
    """
    from ..connections import Neo4jConnection
    with Neo4jConnection.session() as session:
        rows = list(session.run(
            """
            MATCH (e:Entity)
            WHERE e.embedding IS NULL AND e.name IS NOT NULL
            RETURN e.id AS id, e.name AS name, e.description AS description,
                   coalesce(e.type, 'Concept') AS type
            LIMIT $limit
            """, limit=limit,
        ))
        if not rows:
            return 0
        texts = [Entity.compute_embedding_text(r["name"], r["description"], r["type"])
                 for r in rows]
        vectors = get_embedder().embed(texts)
        session.run(
            """
            UNWIND $items AS item
            MATCH (e:Entity {id: item.id})
            SET e.embedding = item.embedding
            """,
            items=[{"id": r["id"], "embedding": v}
                   for r, v in zip(rows, vectors, strict=True)],
        ).consume()
        return len(rows)


def link_facts_to_entities() -> dict[str, Any]:
    """
    Backfill: link orphaned facts (no ``ABOUT`` edge) to existing entities.

    Deterministic, full-history repair for facts stored before/around the live
    linking fix. For each orphan fact it matches claim n-grams against a per-channel
    index of entity names/aliases/slugs (same semantics as
    ``find_entity_by_name_or_alias``) and creates ``(Fact)-[:ABOUT]->(:Entity)``
    edges. No LLM; it only links to entities that already exist (unlike the live
    path, it does not create stubs). Also backfills missing entity *embeddings*
    (bounded per run) so the write-time semantic band and ``--semantic`` dedupe
    can see pre-Slice-1 entities.

    Returns:
        Dict with items_processed, links_created, facts_still_orphan,
        embeddings_backfilled, errors.
    """
    settings = get_settings()
    if not settings.entity_linking_enabled:
        logger.debug("Entity linking is disabled")
        return {"items_processed": 0, "links_created": 0,
                "facts_still_orphan": 0, "skipped": "disabled"}

    embeddings_backfilled = 0
    try:
        embeddings_backfilled = _backfill_entity_embeddings(
            settings.entity_embedding_backfill_batch)
        if embeddings_backfilled:
            logger.info(f"🔗 Backfilled {embeddings_backfilled} entity embedding(s)")
    except Exception as e:
        logger.warning(f"Entity embedding backfill failed (will retry next run): {e}")

    max_ngram = settings.entity_linking_max_ngram
    links_created = 0
    facts_processed = 0
    facts_still_orphan = 0
    errors: list[str] = []

    with Neo4jConnection.session() as session:
        # All orphan facts (no 7-day window) — bounded by a config cap.
        facts_to_link = list(session.run("""
            MATCH (f:Fact)
            WHERE NOT (f)-[:ABOUT]->(:Entity)
            RETURN f.id AS fact_id,
                   f.claim AS claim,
                   coalesce(f.user_id, 'default') AS user_id,
                   coalesce(f.channel, '_default') AS channel
            ORDER BY f.user_id, f.channel, f.id
            LIMIT $cap
        """, cap=settings.entity_linking_max_facts))
        logger.info(f"Entity-link backfill: found {len(facts_to_link)} unlinked facts")

        # Cache one entity index per (user_id, channel) across the batch.
        index_cache: dict[tuple[str, str], dict[str, str]] = {}

        for record in facts_to_link:
            fact_id = record["fact_id"]
            claim = record["claim"] or ""
            user_id = record["user_id"]
            channel = record["channel"]
            facts_processed += 1

            try:
                cache_key = (user_id, channel)
                index = index_cache.get(cache_key)
                if index is None:
                    index = _build_entity_index(session, user_id, channel)
                    index_cache[cache_key] = index

                matched_ids: list[str] = []
                seen_ids: set = set()
                for phrase in _claim_entity_candidates(claim, max_ngram):
                    eid = index.get(phrase.lower()) or index.get(_slug(phrase))
                    if eid and eid not in seen_ids:
                        seen_ids.add(eid)
                        matched_ids.append(eid)

                if not matched_ids:
                    facts_still_orphan += 1
                    continue

                session.run("""
                    MATCH (f:Fact {id: $fact_id})
                    UNWIND $entity_ids AS eid
                    MATCH (e:Entity {id: eid})
                    MERGE (f)-[r:ABOUT]->(e)
                    ON CREATE SET r.confidence = 0.85,
                                  r.linked_at = datetime(),
                                  r.method = 'backfill_namematch'
                """, fact_id=fact_id, entity_ids=matched_ids)
                links_created += len(matched_ids)
                logger.debug(f"Backfill linked fact '{claim[:50]}...' to {len(matched_ids)} entities")

            except Exception as e:
                logger.warning(f"Failed to link fact {fact_id}: {e}")
                errors.append(f"{fact_id}:{e}")

    logger.info(
        f"Entity-link backfill complete: {facts_processed} processed, "
        f"{links_created} links created, {facts_still_orphan} still orphan"
    )

    return {
        "items_processed": facts_processed,
        "links_created": links_created,
        "facts_still_orphan": facts_still_orphan,
        "embeddings_backfilled": embeddings_backfilled,
        "errors": errors,
    }

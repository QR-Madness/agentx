"""
RecallLayer - Enhanced retrieval with multiple techniques.

Implements 5 retrieval enhancement techniques:
1. Hybrid Search (BM25 + Vector) - Keyword + semantic fusion
2. Entity-Centric - Graph traversal from entities to linked facts
3. Query Expansion - Question→statement transforms
4. HyDE - Hypothetical document embedding
5. Self-Query - LLM-based filter extraction

Each technique can be enabled/disabled via config settings.
Comprehensive debug logging shows exactly what's happening.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import get_settings
from ..models import MemoryBundle
from ....utils.async_bridge import run_coro_sync as _run_coro_sync

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger
    from .retrieval import MemoryRetriever
    from .interface import AgentMemory

logger = logging.getLogger(__name__)

# First-person query detection for the attribution guard (§2.11). Word-bounded
# so "in my opinion"-style hits are intended; contractions ("I'm", "I've")
# match via the bare "i" token since the apostrophe is a word boundary.
_FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)


@dataclass
class RecallMetrics:
    """Metrics for a single recall operation with technique breakdown."""

    query: str
    user_id: str
    channel: str
    techniques_enabled: dict[str, bool] = field(default_factory=dict)

    # Base retrieval
    base_results: int = 0
    base_latency_ms: int = 0

    # HyDE
    hyde_hypothetical: str = ""
    hyde_results: int = 0
    hyde_latency_ms: int = 0

    # Hybrid search
    hybrid_bm25_results: int = 0
    hybrid_bm25_top_scores: list[tuple[str, float]] = field(default_factory=list)
    hybrid_vector_results: int = 0
    hybrid_merged_results: int = 0
    hybrid_latency_ms: int = 0

    # Entity-centric
    entity_centric_entities: list[str] = field(default_factory=list)
    entity_centric_facts: int = 0
    entity_centric_latency_ms: int = 0

    # Query expansion
    expansion_variants: list[str] = field(default_factory=list)
    expansion_results: int = 0
    expansion_latency_ms: int = 0

    # Self-query
    self_query_filters: dict[str, Any] = field(default_factory=dict)
    self_query_results: int = 0
    self_query_latency_ms: int = 0

    # Cross-encoder stage (post-fusion, §2.11)
    rerank_stage_facts: int = 0
    rerank_stage_turns: int = 0
    rerank_stage_latency_ms: int = 0

    # First-person attribution guard
    first_person_guard_applied: bool = False
    first_person_penalized: int = 0

    # Final
    final_results: int = 0
    duplicates_removed: int = 0
    total_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for audit logging."""
        return {
            "query": self.query[:100],
            "user_id": self.user_id,
            "channel": self.channel,
            "techniques_enabled": self.techniques_enabled,
            "base_results": self.base_results,
            "base_latency_ms": self.base_latency_ms,
            "hyde_hypothetical": self.hyde_hypothetical[:100] if self.hyde_hypothetical else "",
            "hyde_results": self.hyde_results,
            "hyde_latency_ms": self.hyde_latency_ms,
            "hybrid_bm25_results": self.hybrid_bm25_results,
            "hybrid_vector_results": self.hybrid_vector_results,
            "hybrid_merged_results": self.hybrid_merged_results,
            "hybrid_latency_ms": self.hybrid_latency_ms,
            "entity_centric_entities": self.entity_centric_entities,
            "entity_centric_facts": self.entity_centric_facts,
            "entity_centric_latency_ms": self.entity_centric_latency_ms,
            "expansion_variants": self.expansion_variants,
            "expansion_results": self.expansion_results,
            "expansion_latency_ms": self.expansion_latency_ms,
            "self_query_filters": self.self_query_filters,
            "self_query_results": self.self_query_results,
            "self_query_latency_ms": self.self_query_latency_ms,
            "rerank_stage_facts": self.rerank_stage_facts,
            "rerank_stage_turns": self.rerank_stage_turns,
            "rerank_stage_latency_ms": self.rerank_stage_latency_ms,
            "first_person_guard_applied": self.first_person_guard_applied,
            "first_person_penalized": self.first_person_penalized,
            "final_results": self.final_results,
            "duplicates_removed": self.duplicates_removed,
            "total_latency_ms": self.total_latency_ms,
        }


class RecallLayer:
    """
    Enhanced retrieval layer with multiple techniques.

    Wraps the base MemoryRetriever and augments results using:
    - Hybrid search (BM25 + vector with RRF fusion)
    - Entity-centric retrieval (graph traversal)
    - Query expansion (question→statement transforms)
    - HyDE (hypothetical document embedding)
    - Self-query (LLM filter extraction)

    Usage:
        recall = RecallLayer(memory, base_retriever, audit_logger)
        bundle = recall.recall("When is my birthday?", user_id="user123")
    """

    def __init__(
        self,
        memory: AgentMemory,
        base_retriever: MemoryRetriever,
        audit_logger: MemoryAuditLogger | None = None,
    ):
        self.memory = memory
        self.base_retriever = base_retriever
        self.audit_logger = audit_logger
        # Optional settings override (tests). When None, settings resolve live
        # via get_settings() so runtime updates propagate — see the property.
        self._settings: Any = None

        # Question words for query expansion
        self._question_words = {
            "what", "when", "where", "who", "why", "how",
            "which", "whose", "whom", "is", "are", "was",
            "were", "do", "does", "did", "can", "could",
            "will", "would", "should", "have", "has", "had",
        }

        # Common synonyms for expansion
        self._synonyms = {
            "prefer": ["like", "favor", "enjoy", "want"],
            "birthday": ["birth date", "date of birth", "born"],
            "favorite": ["preferred", "best", "top"],
            "work": ["job", "employment", "career", "occupation"],
            "live": ["reside", "stay", "home", "location"],
            "email": ["e-mail", "mail address"],
            "phone": ["telephone", "mobile", "cell"],
        }

    @property
    def settings(self):
        # Live-resolved unless explicitly overridden (mirrors
        # ExtractionService.settings) — get_settings() has its own TTL cache.
        if self._settings is not None:
            return self._settings
        return get_settings()

    def recall(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        channels: list[str] | None = None,
        time_window_hours: int | None = None,
        **kwargs,
    ) -> MemoryBundle:
        """
        Enhanced retrieval with multiple techniques.

        Args:
            query: The search query
            user_id: User ID for scoping
            top_k: Number of results per category
            channels: Channels to search (default: current + _global)
            time_window_hours: Optional time filter
            **kwargs: Additional arguments passed to base retriever

        Returns:
            MemoryBundle with aggregated results from all techniques
        """
        start_time = time.perf_counter()
        settings = self.settings
        channel = channels[0] if channels else self.memory.channel

        # Initialize metrics
        metrics = RecallMetrics(
            query=query,
            user_id=user_id,
            channel=channel,
            techniques_enabled={
                "hybrid": settings.recall_enable_hybrid,
                "entity_centric": settings.recall_enable_entity_centric,
                "query_expansion": settings.recall_enable_query_expansion,
                "hyde": settings.recall_enable_hyde,
                "self_query": settings.recall_enable_self_query,
            },
        )

        logger.debug(
            f"[RecallLayer] Query: \"{query[:50]}...\" user={user_id} channel={channel}"
        )
        logger.debug(
            f"[RecallLayer] Techniques: hybrid={settings.recall_enable_hybrid} "
            f"entity={settings.recall_enable_entity_centric} "
            f"expansion={settings.recall_enable_query_expansion} "
            f"hyde={settings.recall_enable_hyde} "
            f"self_query={settings.recall_enable_self_query}"
        )

        # Generate embedding for the query (used by multiple techniques)
        embedding = self.memory.embedder.embed_single(query)

        # Step 1: Base retrieval (always runs). The pre-fusion cross-encoder
        # pass is deferred — this layer runs its own post-fusion stage (§2.11)
        # and the encoder must never run twice per recall.
        base_start = time.perf_counter()
        base_bundle = self.base_retriever.retrieve(
            query=query,
            user_id=user_id,
            top_k=top_k,
            channels=channels,
            time_window_hours=time_window_hours,
            defer_cross_encoder=True,
            **kwargs,
        )
        metrics.base_latency_ms = int((time.perf_counter() - base_start) * 1000)
        metrics.base_results = (
            len(base_bundle.facts) +
            len(base_bundle.entities) +
            len(base_bundle.relevant_turns)
        )
        logger.debug(
            f"[RecallLayer:Base] Retrieved {len(base_bundle.facts)} facts, "
            f"{len(base_bundle.entities)} entities, "
            f"{len(base_bundle.relevant_turns)} turns in {metrics.base_latency_ms}ms"
        )

        # Track all results for merging
        all_bundles = [base_bundle]

        # Step 2: Hybrid search (BM25 + Vector)
        if settings.recall_enable_hybrid:
            hybrid_bundle, hybrid_metrics = self._hybrid_retrieval(
                query=query,
                embedding=embedding,
                user_id=user_id,
                channels=channels or [channel, "_global"],
                top_k=top_k,
            )
            metrics.hybrid_bm25_results = hybrid_metrics["bm25_results"]
            metrics.hybrid_bm25_top_scores = hybrid_metrics["bm25_top_scores"]
            metrics.hybrid_vector_results = hybrid_metrics["vector_results"]
            metrics.hybrid_merged_results = hybrid_metrics["merged_results"]
            metrics.hybrid_latency_ms = hybrid_metrics["latency_ms"]
            all_bundles.append(hybrid_bundle)

        # Step 3: Entity-centric retrieval
        if settings.recall_enable_entity_centric:
            entity_bundle, entity_metrics = self._entity_centric_retrieval(
                query=query,
                embedding=embedding,
                user_id=user_id,
                channels=channels or [channel, "_global"],
                top_k=top_k,
            )
            metrics.entity_centric_entities = entity_metrics["entities"]
            metrics.entity_centric_facts = entity_metrics["facts"]
            metrics.entity_centric_latency_ms = entity_metrics["latency_ms"]
            all_bundles.append(entity_bundle)

        # Step 4: Query expansion
        if settings.recall_enable_query_expansion:
            expansion_bundle, expansion_metrics = self._expansion_retrieval(
                query=query,
                user_id=user_id,
                channels=channels,
                top_k=top_k,
                time_window_hours=time_window_hours,
                **kwargs,
            )
            metrics.expansion_variants = expansion_metrics["variants"]
            metrics.expansion_results = expansion_metrics["results"]
            metrics.expansion_latency_ms = expansion_metrics["latency_ms"]
            all_bundles.append(expansion_bundle)

        # Step 5: HyDE (if enabled)
        if settings.recall_enable_hyde:
            hyde_bundle, hyde_metrics = self._hyde_retrieval(
                query=query,
                user_id=user_id,
                channels=channels,
                top_k=top_k,
                time_window_hours=time_window_hours,
                **kwargs,
            )
            metrics.hyde_hypothetical = hyde_metrics["hypothetical"]
            metrics.hyde_results = hyde_metrics["results"]
            metrics.hyde_latency_ms = hyde_metrics["latency_ms"]
            all_bundles.append(hyde_bundle)

        # Step 6: Self-query (if enabled)
        if settings.recall_enable_self_query:
            self_query_bundle, self_query_metrics = self._self_query_retrieval(
                query=query,
                user_id=user_id,
                channels=channels,
                top_k=top_k,
                **kwargs,
            )
            metrics.self_query_filters = self_query_metrics["filters"]
            metrics.self_query_results = self_query_metrics["results"]
            metrics.self_query_latency_ms = self_query_metrics["latency_ms"]
            all_bundles.append(self_query_bundle)

        # Step 7: Merge all bundles
        final_bundle, merge_stats = self._merge_bundles(*all_bundles)

        # Step 8: Cross-encoder rerank stage (post-fusion, §2.11). Scores the
        # fused candidate pool against the query and cuts back to top_k, so
        # final metrics + audit reflect what the prompt will actually see.
        if settings.cross_encoder_enabled:
            final_bundle, ce_stats = self._cross_encoder_stage(query, final_bundle, top_k)
            metrics.rerank_stage_facts = ce_stats["facts"]
            metrics.rerank_stage_turns = ce_stats["turns"]
            metrics.rerank_stage_latency_ms = ce_stats["latency_ms"]

        # Step 9: First-person attribution guard — a subject filter, not a
        # relevance signal, so it runs after the cross-encoder. Penalizes
        # (never drops) facts about other people on first-person queries.
        if settings.recall_first_person_guard:
            final_bundle, guard_stats = self._first_person_guard(query, final_bundle)
            metrics.first_person_guard_applied = guard_stats["applied"]
            metrics.first_person_penalized = guard_stats["penalized"]

        metrics.final_results = (
            len(final_bundle.facts) +
            len(final_bundle.entities) +
            len(final_bundle.relevant_turns)
        )
        metrics.duplicates_removed = merge_stats["duplicates_removed"]

        # Calculate total latency
        metrics.total_latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Log summary
        self._log_recall_summary(metrics)

        return final_bundle

    def _hybrid_retrieval(
        self,
        query: str,
        embedding: list[float],
        user_id: str,
        channels: list[str],
        top_k: int,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Hybrid search combining BM25 (keyword) and vector (semantic).

        Uses Reciprocal Rank Fusion (RRF) to combine rankings:
        score = bm25_weight/(k + bm25_rank) + vector_weight/(k + vector_rank)
        """
        start_time = time.perf_counter()
        settings = self.settings

        bm25_weight = settings.recall_hybrid_bm25_weight
        vector_weight = settings.recall_hybrid_vector_weight
        rrf_k = settings.recall_hybrid_rrf_k

        # Stage-1 candidate pool (§2.11): each arm over-fetches to the pool
        # width; RRF emits the full pool only when the cross-encoder stage
        # will cut it back to top_k (otherwise the prompt would blow up).
        fetch_k = max(top_k * 2, settings.recall_candidate_pool)
        output_k = self._pool_output_width(top_k)

        # Extract keywords for BM25
        keywords = self._extract_keywords(query)
        logger.debug(f"[RecallLayer:Hybrid] BM25 keywords: {keywords}")

        bm25_results = []
        bm25_top_scores = []

        if keywords:
            # BM25 search using Neo4j full-text index
            bm25_results = self._bm25_search(
                keywords=keywords,
                user_id=user_id,
                channels=channels,
                limit=fetch_k,  # Over-fetch for fusion
            )
            logger.debug(
                f"[RecallLayer:Hybrid] BM25 found {len(bm25_results)} facts"
            )
            if bm25_results:
                bm25_top_scores = [
                    (r["claim"][:50], r["score"])
                    for r in bm25_results[:3]
                ]
                for claim, score in bm25_top_scores:
                    logger.debug(f"[RecallLayer:Hybrid]   - \"{claim}...\" (score={score:.2f})")

        # Vector search (already done in base, but we need rankings)
        vector_results = self.memory.semantic.vector_search_facts(
            query_embedding=embedding,
            user_id=user_id,
            channel=channels[0] if channels else "_global",
            top_k=fetch_k,
            min_confidence=0.0,  # Don't filter yet, RRF will handle ranking
        )
        logger.debug(f"[RecallLayer:Hybrid] Vector found {len(vector_results)} facts")

        # Apply RRF fusion
        merged = self._rrf_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rrf_k=rrf_k,
            top_k=output_k,
        )
        logger.debug(f"[RecallLayer:Hybrid] RRF merged to {len(merged)} facts")

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(f"[RecallLayer:Hybrid] Completed in {latency_ms}ms")

        # Build bundle with merged facts
        bundle = MemoryBundle(facts=merged)

        return bundle, {
            "bm25_results": len(bm25_results),
            "bm25_top_scores": bm25_top_scores,
            "vector_results": len(vector_results),
            "merged_results": len(merged),
            "latency_ms": latency_ms,
        }

    def _bm25_search(
        self,
        keywords: str,
        user_id: str,
        channels: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Execute BM25 search using Neo4j full-text index."""
        from ..connections import Neo4jConnection

        # Escape special Lucene characters
        escaped_keywords = self._escape_lucene(keywords)

        query = """
        CALL db.index.fulltext.queryNodes('fact_search', $keywords)
        YIELD node AS f, score
        WHERE f.user_id = $user_id
          AND f.channel IN $channels
          AND f.superseded_at IS NULL
          // Exclude forgotten/retired facts (forget_fact: past + salience≈0.05).
          AND NOT (coalesce(f.temporal_context, 'current') = 'past'
                   AND coalesce(f.salience, 0.5) < 0.1)
        RETURN f.id AS id,
               f.claim AS claim,
               f.confidence AS confidence,
               f.source AS source,
               f.channel AS channel,
               f.salience AS salience,
               f.temporal_context AS temporal_context,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            with Neo4jConnection.session() as session:
                result = session.run(
                    query,
                    keywords=escaped_keywords,
                    user_id=user_id,
                    channels=channels,
                    limit=limit,
                )
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"[RecallLayer:Hybrid] BM25 search failed: {e}")
            return []

    def _escape_lucene(self, text: str) -> str:
        """Escape special Lucene query characters."""
        special_chars = r'+-&|!(){}[]^"~*?:\/'
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    def _extract_keywords(self, query: str) -> str:
        """Extract meaningful keywords from query for BM25."""
        # Remove question words and common stopwords
        stopwords = self._question_words | {
            "the", "a", "an", "my", "your", "his", "her", "its",
            "our", "their", "this", "that", "these", "those",
            "i", "me", "you", "he", "she", "it", "we", "they",
            "am", "be", "been", "being", "about", "for", "with",
            "at", "by", "from", "to", "of", "in", "on", "up",
        }

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) >= 3]

        return " ".join(keywords)

    def _rrf_fusion(
        self,
        bm25_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        bm25_weight: float,
        vector_weight: float,
        rrf_k: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Combine BM25 and vector results using Reciprocal Rank Fusion."""
        # Build rank maps
        bm25_ranks = {r["id"]: i + 1 for i, r in enumerate(bm25_results)}
        vector_ranks = {r["id"]: i + 1 for i, r in enumerate(vector_results)}

        # Collect all unique IDs
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # Build result map for lookup
        result_map = {}
        for r in bm25_results + vector_results:
            if r["id"] not in result_map:
                result_map[r["id"]] = r

        # Calculate RRF scores
        scored = []
        for id_ in all_ids:
            bm25_rank = bm25_ranks.get(id_, len(bm25_results) + rrf_k)
            vector_rank = vector_ranks.get(id_, len(vector_results) + rrf_k)

            rrf_score = (
                bm25_weight / (rrf_k + bm25_rank) +
                vector_weight / (rrf_k + vector_rank)
            )

            result = result_map[id_].copy()
            result["rrf_score"] = rrf_score
            result["bm25_rank"] = bm25_ranks.get(id_)
            result["vector_rank"] = vector_ranks.get(id_)
            scored.append(result)

            # Log RRF contribution for top results
            if bm25_ranks.get(id_) and bm25_ranks[id_] <= 3:
                logger.debug(
                    f"[RecallLayer:Hybrid:RRF] BM25 rank={bm25_rank} "
                    f"vector rank={vector_rank} id={id_[:8]} "
                    f"rrf={rrf_score:.4f}"
                )

        # Sort by RRF score and take top_k
        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored[:top_k]

    def _entity_centric_retrieval(
        self,
        query: str,
        embedding: list[float],
        user_id: str,
        channels: list[str],
        top_k: int,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Entity-centric retrieval via graph traversal.

        1. Find entities matching query via vector search
        2. Traverse ABOUT relationships to get linked facts
        """
        start_time = time.perf_counter()
        settings = self.settings

        threshold = settings.recall_entity_similarity_threshold
        max_entities = settings.recall_entity_max_entities

        # Find matching entities
        entities = self.memory.semantic.vector_search_entities(
            query_embedding=embedding,
            user_id=user_id,
            channel=channels[0] if channels else None,
            top_k=max_entities,
        )

        # Filter by similarity threshold
        entities = [e for e in entities if e.get("score", 0) >= threshold]

        entity_names = [e["name"] for e in entities]
        logger.debug(
            f"[RecallLayer:EntityCentric] Found {len(entities)} entities: {entity_names}"
        )

        # Get facts linked to these entities
        linked_facts = []
        if entities:
            entity_ids = [e["id"] for e in entities]
            linked_facts = self._get_facts_for_entities(
                entity_ids=entity_ids,
                user_id=user_id,
                channels=channels,
                limit=top_k,
            )
            logger.debug(
                f"[RecallLayer:EntityCentric] Graph traversal found {len(linked_facts)} linked facts"
            )
            for fact in linked_facts[:3]:
                logger.debug(
                    f"[RecallLayer:EntityCentric]   - \"{fact['claim'][:50]}...\""
                )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(f"[RecallLayer:EntityCentric] Completed in {latency_ms}ms")

        bundle = MemoryBundle(
            facts=linked_facts,
            entities=[{"id": e["id"], "name": e["name"], "type": e.get("type", "Unknown")}
                      for e in entities],
        )

        return bundle, {
            "entities": entity_names,
            "facts": len(linked_facts),
            "latency_ms": latency_ms,
        }

    def _get_facts_for_entities(
        self,
        entity_ids: list[str],
        user_id: str,
        channels: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get facts linked to entities via ABOUT relationship."""
        from ..connections import Neo4jConnection

        query = """
        MATCH (f:Fact)-[:ABOUT]->(e:Entity)
        WHERE e.id IN $entity_ids
          AND f.user_id = $user_id
          AND f.channel IN $channels
          AND f.superseded_at IS NULL
          // Exclude forgotten/retired facts (forget_fact: past + salience≈0.05).
          AND NOT (coalesce(f.temporal_context, 'current') = 'past'
                   AND coalesce(f.salience, 0.5) < 0.1)
        RETURN DISTINCT f.id AS id,
               f.claim AS claim,
               f.confidence AS confidence,
               f.source AS source,
               f.channel AS channel,
               f.salience AS salience,
               f.temporal_context AS temporal_context,
               collect(e.name) AS linked_entities
        ORDER BY f.confidence DESC, f.salience DESC
        LIMIT $limit
        """

        try:
            with Neo4jConnection.session() as session:
                result = session.run(
                    query,
                    entity_ids=entity_ids,
                    user_id=user_id,
                    channels=channels,
                    limit=limit,
                )
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning(f"[RecallLayer:EntityCentric] Graph query failed: {e}")
            return []

    def _expansion_retrieval(
        self,
        query: str,
        user_id: str,
        channels: list[str] | None,
        top_k: int,
        time_window_hours: int | None,
        **kwargs,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Query expansion retrieval.

        Transforms question form to statement form and generates variants.
        """
        start_time = time.perf_counter()
        settings = self.settings

        max_variants = settings.recall_expansion_max_variants

        # Generate query variants
        variants = self._expand_query(query)
        variants = variants[:max_variants]

        logger.debug(f"[RecallLayer:Expansion] Generated {len(variants)} variants:")
        for v in variants:
            logger.debug(f"[RecallLayer:Expansion]   - \"{v}\"")

        # Search with each variant
        all_results = []
        for variant in variants:
            variant_embedding = self.memory.embedder.embed_single(variant)
            channel = channels[0] if channels else self.memory.channel
            results = self.memory.semantic.vector_search_facts(
                query_embedding=variant_embedding,
                user_id=user_id,
                channel=channel,
                top_k=top_k,
                min_confidence=self.settings.recall_min_confidence,
            )
            logger.debug(
                f"[RecallLayer:Expansion] Variant \"{variant[:30]}...\" found {len(results)} facts"
            )
            all_results.extend(results)

        # Deduplicate by ID (keep first occurrence with higher score)
        seen_ids: set[str] = set()
        unique_results = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                unique_results.append(r)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(
            f"[RecallLayer:Expansion] Found {len(unique_results)} unique results in {latency_ms}ms"
        )

        bundle = MemoryBundle(facts=unique_results[:top_k])

        return bundle, {
            "variants": variants,
            "results": len(unique_results),
            "latency_ms": latency_ms,
        }

    def _expand_query(self, query: str) -> list[str]:
        """Generate query variants using rule-based transforms."""
        variants = []
        query_lower = query.lower().strip()

        # Transform 1: Question to statement
        statement = self._question_to_statement(query_lower)
        if statement and statement != query_lower:
            variants.append(statement)

        # Transform 2: Extract core noun phrase
        core = self._extract_core_phrase(query_lower)
        if core and core not in variants:
            variants.append(core)

        # Transform 3: Synonym expansion
        for word, synonyms in self._synonyms.items():
            if word in query_lower:
                for syn in synonyms[:2]:  # Limit synonyms
                    variant = query_lower.replace(word, syn)
                    if variant not in variants and variant != query_lower:
                        variants.append(variant)
                        break  # Only one synonym per word

        return variants

    def _question_to_statement(self, query: str) -> str:
        """Transform question form to statement form."""
        # "When is my birthday?" -> "birthday is"
        # "What is my favorite color?" -> "favorite color is"
        # "Where do I work?" -> "work at" or "works at"

        patterns = [
            (r"^when is (?:my|the|your) (.+?)\??$", r"\1 is"),
            (r"^what is (?:my|the|your) (.+?)\??$", r"\1 is"),
            (r"^where (?:do|does|did) (?:i|you|they) (.+?)\??$", r"\1 at"),
            (r"^who is (?:my|the|your) (.+?)\??$", r"\1 is"),
            (r"^how (?:do|does|did) (?:i|you) (.+?)\??$", r"\1"),
            (r"^(?:do|does|did) (?:i|you) (?:have|like|prefer|want) (.+?)\??$", r"prefer \1"),
        ]

        for pattern, replacement in patterns:
            match = re.match(pattern, query)
            if match:
                return re.sub(pattern, replacement, query)

        return ""

    def _extract_core_phrase(self, query: str) -> str:
        """Extract the core noun phrase from a query."""
        # Remove question words and common verbs
        words = query.split()
        filtered = []
        skip_next = False

        for _i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue

            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in self._question_words:
                continue
            if word_clean in {"is", "are", "was", "were", "do", "does", "did"}:
                continue
            if word_clean in {"my", "your", "the", "a", "an"}:
                continue

            filtered.append(word_clean)

        return " ".join(filtered)

    def _hyde_retrieval(
        self,
        query: str,
        user_id: str,
        channels: list[str] | None,
        top_k: int,
        time_window_hours: int | None,
        **kwargs,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        HyDE: Hypothetical Document Embedding.

        1. Use LLM to generate hypothetical answer
        2. Embed the hypothetical answer
        3. Search with that embedding (closer to stored facts)
        """
        start_time = time.perf_counter()
        settings = self.settings

        # Generate hypothetical document
        hypothetical = self._generate_hypothetical(query)

        if not hypothetical:
            logger.debug("[RecallLayer:HyDE] Failed to generate hypothetical, skipping")
            return MemoryBundle(), {
                "hypothetical": "",
                "results": 0,
                "latency_ms": int((time.perf_counter() - start_time) * 1000),
            }

        logger.debug(f"[RecallLayer:HyDE] Generated: \"{hypothetical[:100]}...\"")

        # Embed hypothetical and search
        hyde_embedding = self.memory.embedder.embed_single(hypothetical)
        channel = channels[0] if channels else self.memory.channel
        results = self.memory.semantic.vector_search_facts(
            query_embedding=hyde_embedding,
            user_id=user_id,
            channel=channel,
            top_k=top_k,
            min_confidence=settings.recall_min_confidence,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(f"[RecallLayer:HyDE] Found {len(results)} results in {latency_ms}ms")

        bundle = MemoryBundle(facts=results)

        return bundle, {
            "hypothetical": hypothetical,
            "results": len(results),
            "latency_ms": latency_ms,
        }

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical answer using LLM."""
        settings = self.settings

        try:
            from ....model_roles import resolve_member_model
            from ....providers.registry import get_registry

            registry = get_registry()
            # An empty explicit value follows the `fast_utility` model role.
            model_pref = (resolve_member_model("recall_hyde", settings.recall_hyde_model)
                          or settings.recall_hyde_model)
            provider, model_id, _ = registry.resolve_with_fallback(model_pref)

            from ....providers.base import Message, MessageRole

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Generate a single factual statement that would answer the user's question. "
                        "Write as if you're stating a known fact. Be concise and specific. "
                        "Do not include phrases like 'I think' or 'It might be'. "
                        "Just state the fact directly."
                    ),
                ),
                Message(
                    role=MessageRole.USER,
                    content=f"Question: {query}\n\nFactual statement:",
                ),
            ]

            result = _run_coro_sync(
                provider.complete(
                    messages,
                    model_id,
                    temperature=settings.recall_hyde_temperature,
                    max_tokens=settings.recall_hyde_max_tokens,
                )
            )

            return result.content.strip()

        except Exception as e:
            logger.warning(f"[RecallLayer:HyDE] LLM call failed: {e}")
            return ""

    def _self_query_retrieval(
        self,
        query: str,
        user_id: str,
        channels: list[str] | None,
        top_k: int,
        **kwargs,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Self-query: LLM extracts structured filters.

        Parses natural language to extract:
        - time_window_hours: "last week" -> 168
        - keywords: explicit terms to filter by
        - entity_type: filter by entity type
        """
        start_time = time.perf_counter()

        # Extract filters using LLM
        filters = self._extract_filters(query)

        if not filters:
            logger.debug("[RecallLayer:SelfQuery] No filters extracted, skipping")
            return MemoryBundle(), {
                "filters": {},
                "results": 0,
                "latency_ms": int((time.perf_counter() - start_time) * 1000),
            }

        logger.debug(f"[RecallLayer:SelfQuery] Extracted filters: {filters}")

        # Apply filters to retrieval. NOTE: time_window_hours is extracted by
        # the LLM but not yet applied here — only keyword filtering is wired up.
        keywords = filters.get("keywords", [])

        fetch_k = max(top_k * 2, self.settings.recall_candidate_pool)
        embedding = self.memory.embedder.embed_single(query)
        results = self.memory.semantic.vector_search_facts(
            query_embedding=embedding,
            user_id=user_id,
            channel=channels[0] if channels else self.memory.channel,
            top_k=fetch_k,  # Over-fetch to allow filtering
            min_confidence=self.settings.recall_min_confidence,
        )

        # Filter by keywords if present
        if keywords:
            filtered = []
            for r in results:
                claim_lower = r.get("claim", "").lower()
                if any(kw.lower() in claim_lower for kw in keywords):
                    filtered.append(r)
            results = filtered
            logger.debug(
                f"[RecallLayer:SelfQuery] Keyword filter reduced to {len(results)} results"
            )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(f"[RecallLayer:SelfQuery] Found {len(results)} results in {latency_ms}ms")

        bundle = MemoryBundle(facts=results[:self._pool_output_width(top_k)])

        return bundle, {
            "filters": filters,
            "results": len(results),
            "latency_ms": latency_ms,
        }

    def _extract_filters(self, query: str) -> dict[str, Any]:
        """Extract structured filters from query using LLM."""
        settings = self.settings

        try:
            from ....model_roles import resolve_member_model
            from ....providers.registry import get_registry

            registry = get_registry()
            # An empty explicit value follows the `fast_utility` model role.
            model_pref = (resolve_member_model(
                "recall_self_query", settings.recall_self_query_model)
                or settings.recall_self_query_model)
            provider, model_id, _ = registry.resolve_with_fallback(model_pref)

            from ....providers.base import Message, MessageRole

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Extract search filters from the user's query. "
                        "Return a JSON object with these optional fields:\n"
                        "- time_window_hours: number of hours to look back "
                        "(e.g., 'last week' = 168, 'yesterday' = 24, 'last month' = 720)\n"
                        "- keywords: list of specific terms to search for\n"
                        "- entity_type: type of entity (Person, Organization, etc.)\n\n"
                        "Return ONLY the JSON object, no explanation.\n"
                        "If no filters apply, return {}"
                    ),
                ),
                Message(
                    role=MessageRole.USER,
                    content=query,
                ),
            ]

            result = _run_coro_sync(
                provider.complete(
                    messages,
                    model_id,
                    temperature=settings.recall_self_query_temperature,
                    max_tokens=settings.recall_self_query_max_tokens,
                )
            )

            # Parse JSON response
            import json

            content = result.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            return json.loads(content)

        except Exception as e:
            logger.warning(f"[RecallLayer:SelfQuery] Filter extraction failed: {e}")
            return {}

    def _pool_output_width(self, top_k: int) -> int:
        """Technique output width: the full candidate pool when the post-fusion
        cross-encoder stage will cut it back to top_k, else top_k as before."""
        if self.settings.cross_encoder_enabled:
            return max(top_k, self.settings.recall_candidate_pool)
        return top_k

    def _cross_encoder_stage(
        self,
        query: str,
        bundle: MemoryBundle,
        top_k: int,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Post-fusion cross-encoder rerank (§2.11 stage 2).

        Scores (query, claim) pairs over the fused fact pool and (query,
        content) pairs over the turn list, re-sorts by cross-encoder score,
        and cuts both back to top_k. Always-include turns (current-conversation
        context) stay pinned ahead of the reranked rest. A load failure never
        breaks recall — the bundle is trimmed to top_k and returned as-is.
        """
        start_time = time.perf_counter()
        stats: dict[str, Any] = {"facts": 0, "turns": 0, "latency_ms": 0}

        encoder: Any = self.base_retriever._get_cross_encoder()
        if encoder is not None:
            try:
                if bundle.facts:
                    pairs = [(query, f.get("claim") or "") for f in bundle.facts]
                    scores = [float(s) for s in encoder.predict(pairs)]
                    for fact, score in zip(bundle.facts, scores, strict=True):
                        fact["cross_encoder_score"] = score
                    bundle.facts = self._ce_blend_order(bundle.facts, scores)
                    stats["facts"] = len(bundle.facts)

                pinned = [t for t in bundle.relevant_turns if t.get("always_include")]
                rest = [t for t in bundle.relevant_turns if not t.get("always_include")]
                if rest:
                    pairs = [(query, t.get("content") or "") for t in rest]
                    scores = [float(s) for s in encoder.predict(pairs)]
                    for turn, score in zip(rest, scores, strict=True):
                        turn["cross_encoder_score"] = score
                    rest = self._ce_blend_order(rest, scores)
                    stats["turns"] = len(rest)
                bundle.relevant_turns = pinned + rest
            except Exception as e:
                logger.warning(f"[RecallLayer:CrossEncoder] Rerank failed: {e}")

        # Cut back to top_k regardless — the widened pool must never reach the
        # prompt un-trimmed (encoder unavailable/failed included).
        bundle.facts = bundle.facts[:top_k]
        if len(bundle.relevant_turns) > top_k:
            pinned = [t for t in bundle.relevant_turns if t.get("always_include")]
            rest = [t for t in bundle.relevant_turns if not t.get("always_include")]
            bundle.relevant_turns = pinned + rest[: max(0, top_k - len(pinned))]

        stats["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        logger.debug(
            f"[RecallLayer:CrossEncoder] Reranked {stats['facts']} facts, "
            f"{stats['turns']} turns in {stats['latency_ms']}ms"
        )
        return bundle, stats

    def _ce_blend_order(self, items: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
        """
        Order items by cross-encoder rank with bounded demotion: the encoder
        may promote a candidate freely, but can demote it at most
        ``recall_ce_max_demotion`` positions below its incoming (stage-1
        fused) rank. Hedges encoder blind spots — the eval showed MiniLM
        burying a fused-rank-2 negation-heavy claim below top-10 — without
        diluting the promotion wins that carry the stage's MRR gain.
        ``recall_ce_max_demotion <= 0`` disables the cap (pure CE order).
        """
        cap = self.settings.recall_ce_max_demotion
        ce_order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        ce_rank = {idx: r + 1 for r, idx in enumerate(ce_order)}
        keyed = []
        for i, item in enumerate(items):
            fused_rank = i + 1
            effective = ce_rank[i] if cap <= 0 else min(ce_rank[i], fused_rank + cap)
            item["rerank_effective_rank"] = effective
            # Ties break toward the better cross-encoder rank, then incoming
            keyed.append((effective, ce_rank[i], i))
        keyed.sort()
        return [items[i] for _, _, i in keyed]

    def _first_person_guard(
        self,
        query: str,
        bundle: MemoryBundle,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Read-side attribution guard (§2.11 stage 3).

        On first-person queries ("what do I…"), facts whose claim is not about
        the user (claims start with their subject by extraction convention:
        "User …" vs "<Name> …") get a rank penalty. Rank-based, not
        score-based — fused/cross-encoder scores are heterogeneous (raw CE
        logits can be negative, where a multiplier would *promote*). A fact at
        rank r moves to effective rank r/penalty (0.5 → doubled). INV: recall
        may rerank, never hide — a third-party fact can still be the answer.
        """
        if not _FIRST_PERSON_RE.search(query):
            return bundle, {"applied": False, "penalized": 0}

        penalty = min(max(self.settings.recall_first_person_penalty, 0.01), 1.0)
        penalized = 0
        keyed: list[tuple[float, int, int]] = []
        for i, fact in enumerate(bundle.facts):
            claim = fact.get("claim") or ""
            about_user = claim.startswith(("User ", "User's "))
            rank = float(i + 1)
            if not about_user:
                rank /= penalty
                penalized += 1
                fact["first_person_penalized"] = True
            # Effective-rank ties break toward the user's own facts (a rank-1
            # distractor at penalty 0.5 lands exactly on the rank-2 user fact).
            keyed.append((rank, 1 if not about_user else 0, i))

        if penalized and penalized < len(bundle.facts):
            keyed.sort()
            bundle.facts = [bundle.facts[i] for _, _, i in keyed]
            logger.debug(
                f"[RecallLayer:FirstPersonGuard] Penalized {penalized}/"
                f"{len(bundle.facts)} non-user facts (penalty={penalty})"
            )
        return bundle, {"applied": True, "penalized": penalized}

    def _merge_bundles(
        self,
        *bundles: MemoryBundle,
    ) -> tuple[MemoryBundle, dict[str, Any]]:
        """
        Merge multiple MemoryBundles, deduplicating by ID.

        Keeps the version with the higher score when duplicates found.

        NOTE (§2.11): ``relevant_turns`` is a parallel path by design — the
        bundle contract keeps verbatim episodic turns separate from the fused
        fact pool (only the base retriever populates turns; techniques fuse
        facts). The cross-encoder stage reranks turns in their own list. A
        deliberate fuse-vs-parallel decision is parked in Memory-Roadmap §2.11.
        """
        seen_fact_ids: set[str] = set()
        seen_entity_ids: set[str] = set()
        seen_turn_ids: set[str] = set()

        merged_facts = []
        merged_entities = []
        merged_turns = []

        duplicates_removed = 0
        total_before = 0

        for bundle in bundles:
            # Merge facts
            for fact in bundle.facts:
                total_before += 1
                fact_id = fact.get("id")
                if fact_id and fact_id not in seen_fact_ids:
                    seen_fact_ids.add(fact_id)
                    merged_facts.append(fact)
                elif fact_id:
                    duplicates_removed += 1

            # Merge entities
            for entity in bundle.entities:
                total_before += 1
                entity_id = entity.get("id")
                if entity_id and entity_id not in seen_entity_ids:
                    seen_entity_ids.add(entity_id)
                    merged_entities.append(entity)
                elif entity_id:
                    duplicates_removed += 1

            # Merge turns
            for turn in bundle.relevant_turns:
                total_before += 1
                turn_id = turn.get("id")
                if turn_id and turn_id not in seen_turn_ids:
                    seen_turn_ids.add(turn_id)
                    merged_turns.append(turn)
                elif turn_id:
                    duplicates_removed += 1

        logger.debug(
            f"[RecallLayer:Merge] Combined {total_before} items, "
            f"removed {duplicates_removed} duplicates → "
            f"{len(merged_facts)} facts, {len(merged_entities)} entities, "
            f"{len(merged_turns)} turns"
        )

        return MemoryBundle(
            facts=merged_facts,
            entities=merged_entities,
            relevant_turns=merged_turns,
            strategies=bundles[0].strategies if bundles else [],
            active_goals=bundles[0].active_goals if bundles else [],
            user_context=bundles[0].user_context if bundles else {},
        ), {
            "duplicates_removed": duplicates_removed,
        }

    def _log_recall_summary(self, metrics: RecallMetrics) -> None:
        """Log a summary of the recall operation."""
        techniques_used = [
            name for name, enabled in metrics.techniques_enabled.items()
            if enabled
        ]

        logger.debug(
            f"[RecallLayer:Summary] query=\"{metrics.query[:50]}...\" "
            f"user={metrics.user_id}"
        )
        logger.debug(
            f"[RecallLayer:Summary] Techniques: {', '.join(techniques_used)}"
        )
        logger.debug(
            f"[RecallLayer:Summary] Base: {metrics.base_results} results in "
            f"{metrics.base_latency_ms}ms"
        )

        if metrics.techniques_enabled.get("hybrid"):
            logger.debug(
                f"[RecallLayer:Summary] Hybrid: {metrics.hybrid_merged_results} results "
                f"({metrics.hybrid_bm25_results} BM25, {metrics.hybrid_vector_results} vector) "
                f"in {metrics.hybrid_latency_ms}ms"
            )

        if metrics.techniques_enabled.get("entity_centric"):
            logger.debug(
                f"[RecallLayer:Summary] Entity-Centric: {metrics.entity_centric_facts} facts "
                f"({len(metrics.entity_centric_entities)} entities) "
                f"in {metrics.entity_centric_latency_ms}ms"
            )

        if metrics.techniques_enabled.get("query_expansion"):
            logger.debug(
                f"[RecallLayer:Summary] Expansion: {metrics.expansion_results} results "
                f"({len(metrics.expansion_variants)} variants) "
                f"in {metrics.expansion_latency_ms}ms"
            )

        if metrics.techniques_enabled.get("hyde"):
            logger.debug(
                f"[RecallLayer:Summary] HyDE: {metrics.hyde_results} results "
                f"(doc=\"{metrics.hyde_hypothetical[:30]}...\") "
                f"in {metrics.hyde_latency_ms}ms"
            )

        if metrics.techniques_enabled.get("self_query"):
            logger.debug(
                f"[RecallLayer:Summary] SelfQuery: {metrics.self_query_results} results "
                f"(filters={metrics.self_query_filters}) "
                f"in {metrics.self_query_latency_ms}ms"
            )

        logger.debug(
            f"[RecallLayer:Summary] Final: {metrics.final_results} results "
            f"(removed {metrics.duplicates_removed} duplicates) "
            f"in {metrics.total_latency_ms}ms"
        )

        # Also log to audit logger if available
        if self.audit_logger:
            try:
                from ..audit import OperationType, MemoryType

                self.audit_logger.log_read(
                    operation=OperationType.SEARCH.value,
                    memory_type=MemoryType.SEMANTIC.value,
                    user_id=metrics.user_id,
                    channels=[metrics.channel],
                    result_count=metrics.final_results,
                    latency_ms=metrics.total_latency_ms,
                    metadata=metrics.to_dict(),
                )
            except Exception as e:
                logger.debug(f"[RecallLayer] Audit logging failed: {e}")

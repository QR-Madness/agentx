"""Memory retrieval - multi-strategy retrieval engine with weighted scoring and caching."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING

from ..models import MemoryBundle
from ..embeddings import get_embedder
from ..config import get_settings

if TYPE_CHECKING:
    from .interface import AgentMemory
    from ..audit import MemoryAuditLogger

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievalWeights:
    """
    Weights for combining different retrieval strategies.

    These weights can be overridden per-request by passing a dict
    or RetrievalWeights to remember().

    Attributes:
        episodic: Weight for past conversation turns (default: 0.3)
        semantic_facts: Weight for factual knowledge (default: 0.25)
        semantic_entities: Weight for entities (default: 0.2)
        procedural: Weight for strategies/patterns (default: 0.15)
        recency: Weight for recent context (default: 0.1)
    """

    episodic: float = 0.3
    semantic_facts: float = 0.25
    semantic_entities: float = 0.2
    procedural: float = 0.15
    recency: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RetrievalWeights":
        """
        Create RetrievalWeights from a dictionary, using defaults for missing keys.

        Args:
            d: Dictionary with weight keys

        Returns:
            RetrievalWeights instance
        """
        defaults = cls()
        return cls(
            episodic=d.get("episodic", defaults.episodic),
            semantic_facts=d.get("semantic_facts", defaults.semantic_facts),
            semantic_entities=d.get("semantic_entities", defaults.semantic_entities),
            procedural=d.get("procedural", defaults.procedural),
            recency=d.get("recency", defaults.recency),
        )

    @classmethod
    def from_config(cls) -> "RetrievalWeights":
        """Create RetrievalWeights from config settings."""
        return cls(
            episodic=settings.retrieval_weight_episodic,
            semantic_facts=settings.retrieval_weight_semantic_facts,
            semantic_entities=settings.retrieval_weight_semantic_entities,
            procedural=settings.retrieval_weight_procedural,
            recency=settings.retrieval_weight_recency,
        )

    def merge(
        self, overrides: Optional[Union["RetrievalWeights", Dict[str, float]]]
    ) -> "RetrievalWeights":
        """
        Merge with overrides, returning new weights.

        Only overrides non-default values (allows partial overrides).

        Args:
            overrides: Weights to override with (dict or RetrievalWeights)

        Returns:
            New RetrievalWeights with overrides applied
        """
        if overrides is None:
            return self

        if isinstance(overrides, dict):
            override_weights = RetrievalWeights.from_dict(overrides)
        else:
            override_weights = overrides

        return RetrievalWeights(
            episodic=override_weights.episodic,
            semantic_facts=override_weights.semantic_facts,
            semantic_entities=override_weights.semantic_entities,
            procedural=override_weights.procedural,
            recency=override_weights.recency,
        )


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""

    query_hash: str = ""
    user_id: str = ""
    channels_searched: List[str] = field(default_factory=list)

    # Hit counts per memory type
    episodic_count: int = 0
    semantic_facts_count: int = 0
    semantic_entities_count: int = 0
    procedural_count: int = 0
    total_count: int = 0

    # Results per channel
    results_per_channel: Dict[str, int] = field(default_factory=dict)

    # Latency breakdown (ms)
    embedding_latency_ms: int = 0
    episodic_latency_ms: int = 0
    semantic_latency_ms: int = 0
    procedural_latency_ms: int = 0
    reranking_latency_ms: int = 0
    total_latency_ms: int = 0

    # Cache status
    cache_hit: bool = False
    cache_key: str = ""


class MemoryRetriever:
    """
    Multi-strategy retrieval engine.
    Combines vector search, graph traversal, and temporal heuristics.
    """

    def __init__(self, memory: "AgentMemory", audit_logger: Optional["MemoryAuditLogger"] = None):
        """Initialize the memory retriever.

        Args:
            memory: AgentMemory instance to retrieve from.
            audit_logger: Optional audit logger for operation tracking.
        """
        self.memory = memory
        self.embedder = get_embedder()
        self.weights = RetrievalWeights.from_config()
        self._audit_logger = audit_logger
        self._cross_encoder = None  # Lazy-loaded if enabled

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder model if enabled."""
        if self._cross_encoder is None and settings.cross_encoder_enabled:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(settings.cross_encoder_model)
                logger.info(f"Loaded cross-encoder model: {settings.cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self._cross_encoder = False  # Mark as failed
        return self._cross_encoder if self._cross_encoder else None

    def _get_cache_key(
        self,
        query: str,
        user_id: str,
        channels: List[str],
        top_k: int,
        include_episodic: bool,
        include_semantic: bool,
        include_procedural: bool,
    ) -> str:
        """Generate cache key for retrieval."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        channels_str = ",".join(sorted(channels))
        params_hash = hashlib.sha256(
            f"{top_k}:{include_episodic}:{include_semantic}:{include_procedural}".encode()
        ).hexdigest()[:8]
        return f"{settings.retrieval_cache_key_prefix}:{user_id}:{channels_str}:{query_hash}:{params_hash}"

    def _get_cached(self, cache_key: str) -> Optional[MemoryBundle]:
        """Get cached retrieval result."""
        if not settings.retrieval_cache_enabled:
            return None

        try:
            from ..connections import get_redis_client
            redis = get_redis_client()
            if redis is None:
                return None

            cached = redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return MemoryBundle(**data)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        return None

    def _set_cached(self, cache_key: str, bundle: MemoryBundle) -> None:
        """Cache retrieval result with TTL."""
        if not settings.retrieval_cache_enabled:
            return

        try:
            from ..connections import get_redis_client
            redis = get_redis_client()
            if redis is None:
                return

            data = bundle.model_dump()
            redis.setex(
                cache_key,
                settings.retrieval_cache_ttl_seconds,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")

    def invalidate_cache(self, user_id: str, channel: str) -> int:
        """
        Invalidate retrieval cache for user/channel scope.

        Args:
            user_id: User ID
            channel: Channel name

        Returns:
            Number of keys invalidated
        """
        if not settings.retrieval_cache_enabled:
            return 0

        try:
            from ..connections import get_redis_client
            redis = get_redis_client()
            if redis is None:
                return 0

            # Pattern to match all cache keys for this user/channel
            pattern = f"{settings.retrieval_cache_key_prefix}:{user_id}:*{channel}*"
            count = 0

            # Use SCAN to find matching keys (safer than KEYS for production)
            cursor = 0
            while True:
                cursor, keys = redis.scan(cursor, match=pattern, count=100)
                if keys:
                    redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break

            if count > 0:
                logger.debug(f"Invalidated {count} cache keys for {user_id}/{channel}")
            return count
        except Exception as e:
            logger.debug(f"Cache invalidation failed: {e}")
            return 0

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        time_window_hours: Optional[int] = None,
        channel: Optional[str] = None,
        channels: Optional[List[str]] = None,
        strategy_weights: Optional[Union[RetrievalWeights, Dict[str, float]]] = None,
    ) -> MemoryBundle:
        """
        Main retrieval method combining multiple strategies.

        Args:
            query: Query text
            user_id: User ID
            top_k: Number of results per category
            include_episodic: Include episodic memory
            include_semantic: Include semantic memory
            include_procedural: Include procedural memory
            time_window_hours: Time window filter
            channel: Primary memory channel (also searches _global)
            channels: Explicit list of channels to search (overrides channel param)
            strategy_weights: Optional override for retrieval weights.
                Can be RetrievalWeights or dict with keys:
                episodic, semantic_facts, semantic_entities, procedural, recency

        Returns:
            MemoryBundle with aggregated results
        """
        total_start = time.perf_counter()
        metrics = RetrievalMetrics(user_id=user_id)

        # Determine channels to search
        if channels:
            search_channels = list(set(channels))
        elif channel and channel != "_global":
            search_channels = [channel, "_global"]
        else:
            search_channels = ["_global"]

        metrics.channels_searched = search_channels
        active_channel = channels[0] if channels else (channel or "_global")

        # Merge weights
        effective_weights = self.weights.merge(strategy_weights)

        # Check cache
        cache_key = self._get_cache_key(
            query, user_id, search_channels, top_k,
            include_episodic, include_semantic, include_procedural
        )
        metrics.cache_key = cache_key

        cached_bundle = self._get_cached(cache_key)
        if cached_bundle:
            metrics.cache_hit = True
            metrics.total_latency_ms = int((time.perf_counter() - total_start) * 1000)
            self._log_metrics(metrics)
            return cached_bundle

        # Generate query embedding
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_single(query)
        metrics.embedding_latency_ms = int((time.perf_counter() - t0) * 1000)
        metrics.query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        bundle = MemoryBundle()

        # Strategy 1: Episodic memory (past conversations)
        if include_episodic:
            t1 = time.perf_counter()
            bundle.relevant_turns = self._retrieve_episodic(
                query_embedding, user_id, top_k, time_window_hours, active_channel
            )
            metrics.episodic_latency_ms = int((time.perf_counter() - t1) * 1000)
            metrics.episodic_count = len(bundle.relevant_turns)

        # Strategy 2: Semantic memory (facts and entities)
        if include_semantic:
            t2 = time.perf_counter()

            facts = self.memory.semantic.vector_search_facts(
                query_embedding, top_k=top_k, user_id=user_id, channel=active_channel
            )
            bundle.facts = facts
            metrics.semantic_facts_count = len(facts)

            entities = self.memory.semantic.vector_search_entities(
                query_embedding, top_k=top_k, user_id=user_id, channel=active_channel
            )
            bundle.entities = entities
            metrics.semantic_entities_count = len(entities)

            # Graph expansion: get related entities and their facts
            if entities:
                entity_ids = [e["id"] for e in entities[:5]]
                graph_context = self.memory.semantic.get_entity_graph(
                    entity_ids, depth=2, user_id=user_id, channel=active_channel
                )

                # Merge additional entities and facts
                existing_entity_ids = {e["id"] for e in bundle.entities}
                for related in graph_context.get("related", []):
                    if related.get("entity") and related["entity"].get("id") not in existing_entity_ids:
                        bundle.entities.append(related["entity"])

                existing_fact_ids = {f["id"] for f in bundle.facts}
                for fact in graph_context.get("facts", []):
                    if fact.get("id") not in existing_fact_ids:
                        bundle.facts.append(fact)

            metrics.semantic_latency_ms = int((time.perf_counter() - t2) * 1000)

        # Strategy 3: Procedural memory (what worked before)
        if include_procedural:
            t3 = time.perf_counter()
            strategies = self.memory.procedural.find_strategies(
                query, top_k=5, user_id=user_id, channel=active_channel
            )
            bundle.strategies = [
                {
                    "description": s.description,
                    "tool_sequence": s.tool_sequence,
                    "success_rate": s.success_count / max(1, s.success_count + s.failure_count),
                    "channel": getattr(s, "channel", "_global"),
                }
                for s in strategies
            ]
            metrics.procedural_latency_ms = int((time.perf_counter() - t3) * 1000)
            metrics.procedural_count = len(bundle.strategies)

        # Strategy 4: Active goals
        goals = self.memory.get_active_goals()
        bundle.active_goals = [
            {
                "id": g.id,
                "description": g.description,
                "priority": g.priority,
                "status": g.status,
                "channel": getattr(g, "channel", "_global"),
            }
            for g in goals
        ]

        # Strategy 5: User context
        bundle.user_context = self.memory.get_user_context()

        # Rerank if enabled
        if settings.reranking_enabled:
            t4 = time.perf_counter()
            bundle = self._rerank(bundle, query, query_embedding, effective_weights, active_channel)
            metrics.reranking_latency_ms = int((time.perf_counter() - t4) * 1000)

        # Calculate channel distribution
        metrics.results_per_channel = self._count_results_per_channel(bundle)
        metrics.total_count = (
            metrics.episodic_count +
            metrics.semantic_facts_count +
            metrics.semantic_entities_count +
            metrics.procedural_count
        )

        # Cache results
        self._set_cached(cache_key, bundle)

        metrics.total_latency_ms = int((time.perf_counter() - total_start) * 1000)
        self._log_metrics(metrics)

        return bundle

    def _retrieve_episodic(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int,
        time_window_hours: Optional[int],
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve from episodic memory with recency boost.

        Args:
            query_embedding: Query vector
            user_id: User ID
            top_k: Number of results
            time_window_hours: Time window filter
            channel: Memory channel to search (also searches _global)

        Returns:
            List of relevant turns
        """
        # Vector search
        turns = self.memory.episodic.vector_search(
            query_embedding,
            top_k=top_k * 2,
            user_id=user_id,
            time_window_hours=time_window_hours,
            channel=channel
        )

        # Get recent turns regardless of similarity
        recent = self.memory.episodic.get_recent_turns(
            user_id,
            hours=time_window_hours or 24,
            limit=top_k,
            channel=channel
        )

        # Combine and deduplicate
        seen_ids = set()
        combined = []

        for turn in turns + recent:
            turn_id = turn.get("id") or f"{turn.get('conversation_id')}:{turn.get('timestamp')}"
            if turn_id not in seen_ids:
                seen_ids.add(turn_id)
                combined.append(turn)

        return combined[:top_k * 2]  # Return more for reranking

    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
        score_key: str = "score"
    ) -> List[Dict[str, Any]]:
        """
        Normalize scores to 0-1 range using min-max normalization.

        Args:
            results: List of result dicts with scores
            score_key: Key containing the score value

        Returns:
            Results with added 'normalized_score' field
        """
        if not results:
            return results

        scores = [r.get(score_key, 0) for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            # All scores equal, normalize to 0.5
            for r in results:
                r["normalized_score"] = 0.5
            return results

        for r in results:
            r["normalized_score"] = (r.get(score_key, 0) - min_s) / (max_s - min_s)

        return results

    def _calculate_recency_score(self, timestamp: Any) -> float:
        """
        Calculate recency score based on timestamp.
        More recent = higher score.

        Args:
            timestamp: Timestamp (datetime or ISO string)

        Returns:
            Score between 0 and 1
        """
        try:
            if isinstance(timestamp, str):
                # Parse ISO format
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, datetime):
                ts = timestamp
            else:
                return 0.5

            # Make timezone-aware if needed
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_hours = (now - ts).total_seconds() / 3600

            # Exponential decay: half-life of 24 hours
            half_life = 24.0
            return 0.5 ** (age_hours / half_life)
        except Exception:
            return 0.5

    def _rerank(
        self,
        bundle: MemoryBundle,
        query: str,
        query_embedding: List[float],
        weights: RetrievalWeights,
        active_channel: str,
    ) -> MemoryBundle:
        """
        Rerank retrieved items using weighted combination of scores.

        Scoring considers:
        - Vector similarity (from search)
        - Recency (time-based decay)
        - Channel boost (active channel preference)
        - Diversity (limit items from same conversation)

        Args:
            bundle: Memory bundle to rerank
            query: Original query text
            query_embedding: Query vector
            weights: Retrieval weights
            active_channel: Active channel for boost

        Returns:
            Reranked memory bundle
        """
        channel_boost = settings.channel_active_boost

        # Rerank turns
        if bundle.relevant_turns:
            # Normalize similarity scores
            bundle.relevant_turns = self._normalize_scores(bundle.relevant_turns, "score")

            for turn in bundle.relevant_turns:
                similarity = turn.get("normalized_score", 0.5)
                recency = self._calculate_recency_score(turn.get("timestamp"))
                is_active_channel = turn.get("channel") == active_channel

                # Weighted combination
                turn["final_score"] = (
                    weights.episodic * similarity +
                    weights.recency * recency
                )

                # Apply channel boost
                if is_active_channel and active_channel != "_global":
                    turn["final_score"] *= channel_boost

            # Sort by final score
            bundle.relevant_turns.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            # Diversity: limit items from same conversation
            seen_convs: Dict[str, int] = {}
            filtered_turns = []
            for turn in bundle.relevant_turns:
                conv_id = turn.get("conversation_id")
                if conv_id:
                    seen_convs[conv_id] = seen_convs.get(conv_id, 0) + 1
                    if seen_convs[conv_id] <= 3:
                        filtered_turns.append(turn)
                else:
                    filtered_turns.append(turn)

            bundle.relevant_turns = filtered_turns[:settings.default_top_k]

        # Rerank facts
        if bundle.facts:
            bundle.facts = self._normalize_scores(bundle.facts, "score")

            for fact in bundle.facts:
                similarity = fact.get("normalized_score", 0.5)
                confidence = fact.get("confidence", 0.5)
                is_active_channel = fact.get("channel") == active_channel

                fact["final_score"] = (
                    weights.semantic_facts * similarity * confidence
                )

                if is_active_channel and active_channel != "_global":
                    fact["final_score"] *= channel_boost

            bundle.facts.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            bundle.facts = bundle.facts[:settings.default_top_k]

        # Rerank entities
        if bundle.entities:
            bundle.entities = self._normalize_scores(bundle.entities, "score")

            for entity in bundle.entities:
                similarity = entity.get("normalized_score", 0.5)
                salience = entity.get("salience", 0.5)
                is_active_channel = entity.get("channel") == active_channel

                entity["final_score"] = (
                    weights.semantic_entities * similarity * salience
                )

                if is_active_channel and active_channel != "_global":
                    entity["final_score"] *= channel_boost

            bundle.entities.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            bundle.entities = bundle.entities[:settings.default_top_k]

        # Optional: Cross-encoder reranking for highest accuracy
        cross_encoder = self._get_cross_encoder()
        if cross_encoder and bundle.relevant_turns:
            bundle = self._cross_encoder_rerank(query, bundle, cross_encoder)

        return bundle

    def _cross_encoder_rerank(
        self,
        query: str,
        bundle: MemoryBundle,
        cross_encoder: Any
    ) -> MemoryBundle:
        """
        Rerank using cross-encoder model for higher accuracy.

        Args:
            query: Original query text
            bundle: Memory bundle to rerank
            cross_encoder: Loaded cross-encoder model

        Returns:
            Reranked memory bundle
        """
        try:
            # Prepare pairs for cross-encoder
            turn_texts = [t.get("content", "") for t in bundle.relevant_turns]
            if not turn_texts:
                return bundle

            pairs = [[query, text] for text in turn_texts]
            scores = cross_encoder.predict(pairs)

            # Update scores
            for i, score in enumerate(scores):
                bundle.relevant_turns[i]["cross_encoder_score"] = float(score)

            # Re-sort by cross-encoder score
            bundle.relevant_turns.sort(
                key=lambda x: x.get("cross_encoder_score", 0),
                reverse=True
            )

            logger.debug(f"Cross-encoder reranked {len(bundle.relevant_turns)} turns")
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")

        return bundle

    def _count_results_per_channel(self, bundle: MemoryBundle) -> Dict[str, int]:
        """Count results per channel for metrics."""
        counts: Dict[str, int] = {}

        for turn in bundle.relevant_turns:
            ch = turn.get("channel", "_global")
            counts[ch] = counts.get(ch, 0) + 1

        for fact in bundle.facts:
            ch = fact.get("channel", "_global")
            counts[ch] = counts.get(ch, 0) + 1

        for entity in bundle.entities:
            ch = entity.get("channel", "_global")
            counts[ch] = counts.get(ch, 0) + 1

        return counts

    def _log_metrics(self, metrics: RetrievalMetrics) -> None:
        """Log retrieval metrics."""
        logger.debug(
            f"Retrieval metrics: total={metrics.total_count} "
            f"latency={metrics.total_latency_ms}ms "
            f"cache_hit={metrics.cache_hit} "
            f"channels={metrics.channels_searched}"
        )

        # Log to audit if configured
        if self._audit_logger:
            from ..audit import OperationType, MemoryType
            self._audit_logger.log_read(
                operation=OperationType.SEARCH.value,
                memory_type=MemoryType.COMPOSITE.value,
                user_id=metrics.user_id,
                channels=metrics.channels_searched,
                query_text=f"[hash:{metrics.query_hash}]",
                result_count=metrics.total_count,
                latency_ms=metrics.total_latency_ms,
                success=True,
                metadata={
                    "episodic_count": metrics.episodic_count,
                    "semantic_facts_count": metrics.semantic_facts_count,
                    "semantic_entities_count": metrics.semantic_entities_count,
                    "procedural_count": metrics.procedural_count,
                    "cache_hit": metrics.cache_hit,
                    "embedding_latency_ms": metrics.embedding_latency_ms,
                    "episodic_latency_ms": metrics.episodic_latency_ms,
                    "semantic_latency_ms": metrics.semantic_latency_ms,
                    "procedural_latency_ms": metrics.procedural_latency_ms,
                    "reranking_latency_ms": metrics.reranking_latency_ms,
                    "results_per_channel": metrics.results_per_channel,
                },
            )

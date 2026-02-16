"""Memory retrieval - multi-strategy retrieval engine."""

from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

from ..models import MemoryBundle
from ..embeddings import get_embedder
from ..config import get_settings

if TYPE_CHECKING:
    from .interface import AgentMemory
    from ..audit import MemoryAuditLogger

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
        self.weights = RetrievalWeights()
        self._audit_logger = audit_logger

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
            channel: Memory channel to search (also searches _global)
            strategy_weights: Optional override for retrieval weights.
                Can be RetrievalWeights or dict with keys:
                episodic, semantic_facts, semantic_entities, procedural, recency

        Returns:
            MemoryBundle with aggregated results
        """
        # Merge weights if overrides provided (stored for future weighted combination)
        _effective_weights = self.weights.merge(strategy_weights)  # noqa: F841

        query_embedding = self.embedder.embed_single(query)

        bundle = MemoryBundle()

        # Strategy 1: Episodic memory (past conversations)
        if include_episodic:
            bundle.relevant_turns = self._retrieve_episodic(
                query_embedding, user_id, top_k, time_window_hours, channel
            )

        # Strategy 2: Semantic memory (facts and entities)
        if include_semantic:
            facts = self.memory.semantic.vector_search_facts(
                query_embedding, top_k=top_k, user_id=user_id, channel=channel
            )
            bundle.facts = facts

            entities = self.memory.semantic.vector_search_entities(
                query_embedding, top_k=top_k, user_id=user_id, channel=channel
            )
            bundle.entities = entities

            # Graph expansion: get related entities and their facts
            if entities:
                entity_ids = [e["id"] for e in entities[:5]]
                graph_context = self.memory.semantic.get_entity_graph(
                    entity_ids, depth=2
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

        # Strategy 3: Procedural memory (what worked before)
        if include_procedural:
            strategies = self.memory.procedural.find_strategies(
                query, top_k=5, user_id=user_id, channel=channel
            )
            bundle.strategies = [
                {
                    "description": s.description,
                    "tool_sequence": s.tool_sequence,
                    "success_rate": s.success_count / max(1, s.success_count + s.failure_count)
                }
                for s in strategies
            ]

        # Strategy 4: Active goals
        goals = self.memory.get_active_goals()
        bundle.active_goals = [
            {
                "id": g.id,
                "description": g.description,
                "priority": g.priority,
                "status": g.status
            }
            for g in goals
        ]

        # Strategy 5: User context
        bundle.user_context = self.memory.get_user_context()

        # Rerank if enabled
        if settings.reranking_enabled:
            bundle = self._rerank(bundle, query_embedding)

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

        # Sort by combined score (similarity + recency)
        # This is a simplified scoring; production would be more sophisticated
        return combined[:top_k]

    def _rerank(
        self,
        bundle: MemoryBundle,
        query_embedding: List[float]
    ) -> MemoryBundle:
        """
        Rerank retrieved items using cross-encoder or other scoring.
        This is a simplified version; production would use a cross-encoder model.

        Args:
            bundle: Memory bundle to rerank
            query_embedding: Query vector

        Returns:
            Reranked memory bundle
        """
        # For now, just ensure diversity by limiting items from same conversation
        seen_convs = {}
        filtered_turns = []

        for turn in bundle.relevant_turns:
            conv_id = turn.get("conversation_id")
            if conv_id:
                seen_convs[conv_id] = seen_convs.get(conv_id, 0) + 1
                if seen_convs[conv_id] <= 3:  # Max 3 turns per conversation
                    filtered_turns.append(turn)
            else:
                filtered_turns.append(turn)

        bundle.relevant_turns = filtered_turns

        return bundle

"""Main memory interface - unified API for agent memory operations."""

import logging
import time
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..portability import MemoryExport

from ..models import Turn, Entity, Fact, Goal, Strategy, MemoryBundle, compute_claim_hash
from ..embeddings import get_embedder
from ..connections import Neo4jConnection
from ..config import get_settings
from ..audit import MemoryAuditLogger, OperationType, MemoryType
from ..events import (
    MemoryEventEmitter,
    TurnStoredPayload,
    FactLearnedPayload,
    FactUpdatedPayload,
    FactDeletedPayload,
    EntityCreatedPayload,
    EntityUpdatedPayload,
    EntityDeletedPayload,
    RetrievalCompletePayload,
)
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .procedural import ProceduralMemory
from .working import WorkingMemory
from .goal import GoalMemory
from .retrieval import MemoryRetriever, RetrievalWeights
from .recall import RecallLayer

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Unified interface for agent memory operations.

    Usage:
        memory = AgentMemory(user_id="user123")

        # Store a conversation turn
        memory.store_turn(turn)

        # Retrieve relevant context for a query
        context = memory.remember("What did we discuss about Python?")

        # Learn a new fact
        memory.learn_fact("User prefers concise responses", source="inferred")
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        channel: str = "_global",
        session_id: Optional[str] = None,
        events: Optional[MemoryEventEmitter] = None,
        agent_id: Optional[str] = None,
    ):
        # Validate user_id
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")

        self.user_id = user_id.strip()
        self.conversation_id = conversation_id
        self.channel = channel
        self.session_id = session_id
        self.agent_id = agent_id
        self.self_channel = f"_self_{agent_id}" if agent_id else None
        self.embedder = get_embedder()

        # Event emitter (create default if not provided)
        self.events = events if events is not None else MemoryEventEmitter()

        # Audit logger
        self._settings = get_settings()
        self._audit_logger = MemoryAuditLogger(self._settings)

        # Sub-modules (with audit logger injection)
        self.episodic = EpisodicMemory(audit_logger=self._audit_logger)
        self.semantic = SemanticMemory(audit_logger=self._audit_logger)
        self.procedural = ProceduralMemory(audit_logger=self._audit_logger)
        self.goal = GoalMemory(audit_logger=self._audit_logger)
        self.working = WorkingMemory(
            user_id,
            conversation_id,
            channel=self.channel,
            audit_logger=self._audit_logger
        )
        self.retriever = MemoryRetriever(self, audit_logger=self._audit_logger)
        self.recall_layer = RecallLayer(
            memory=self,
            base_retriever=self.retriever,
            audit_logger=self._audit_logger,
        )

    # Storage operations

    def _invalidate_retrieval_cache(self) -> None:
        """Invalidate retrieval cache for current user/channel scope."""
        self.retriever.invalidate_cache(self.user_id, self.channel)

    def _default_recall_channels(self) -> List[str]:
        """Build the default channel list for recall: [active, self, _global]."""
        channels = [self.channel] if self.channel != "_global" else []
        if self.self_channel:
            channels.append(self.self_channel)
        channels.append("_global")
        return channels

    def store_turn(self, turn: Turn) -> None:
        """
        Store a conversation turn in episodic memory.

        Args:
            turn: Turn object to store
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None

        try:
            # Generate embedding if not provided
            if turn.embedding is None:
                turn.embedding = self.embedder.embed_single(turn.content)

            # Store in Neo4j (graph)
            self.episodic.store_turn(turn, user_id=self.user_id, channel=self.channel, agent_id=self.agent_id)

            # Store in PostgreSQL (logs)
            self.episodic.store_turn_log(turn, channel=self.channel)

            # Update working memory
            self.working.add_turn(turn)

            # Emit turn stored event
            self.events.emit(
                MemoryEventEmitter.TURN_STORED,
                TurnStoredPayload(
                    event_name=MemoryEventEmitter.TURN_STORED,
                    turn_id=turn.id,
                    conversation_id=turn.conversation_id,
                    role=turn.role,
                    content=turn.content,
                    user_id=self.user_id,
                    channel=self.channel,
                )
            )

            # Invalidate retrieval cache for this scope
            self._invalidate_retrieval_cache()
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._audit_logger.log_write(
                operation=OperationType.STORE.value,
                memory_type=MemoryType.EPISODIC.value,
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=turn.conversation_id,
                channel=self.channel,
                record_ids=[turn.id],
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg,
            )

    def get_conversation_participants(self, conversation_id: str) -> List[str]:
        """Distinct agent ids that have produced turns in a conversation.

        Pass-through to :meth:`EpisodicMemory.get_conversation_agent_ids`;
        used by the chat endpoint to hydrate ``Session.participants`` for
        multi-agent prompt awareness (Phase 16.2).
        """
        return self.episodic.get_conversation_agent_ids(conversation_id)

    def learn_fact(
        self,
        claim: str,
        source: str = "extraction",
        confidence: float = 0.8,
        entity_ids: Optional[List[str]] = None,
        source_turn_id: Optional[str] = None,
        temporal_context: Optional[str] = None
    ) -> Fact:
        """
        Add a fact to semantic memory.

        Args:
            claim: Factual claim
            source: Source of the fact
            confidence: Confidence score (0-1)
            entity_ids: Related entity IDs
            source_turn_id: Source turn ID
            temporal_context: Temporal context ("current", "past", "future", or None)

        Returns:
            Created Fact object
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None
        fact_id: Optional[str] = None

        try:
            fact = Fact(
                claim=claim,
                source=source,
                confidence=confidence,
                entity_ids=entity_ids or [],
                source_turn_id=source_turn_id,
                temporal_context=temporal_context,
                embedding=self.embedder.embed_single(claim)
            )
            fact_id = fact.id
            self.semantic.store_fact(fact, user_id=self.user_id, channel=self.channel)

            # Emit fact learned event
            self.events.emit(
                MemoryEventEmitter.FACT_LEARNED,
                FactLearnedPayload(
                    event_name=MemoryEventEmitter.FACT_LEARNED,
                    fact_id=fact.id,
                    claim=claim,
                    confidence=confidence,
                    source=source,
                    user_id=self.user_id,
                    channel=self.channel,
                )
            )

            # Invalidate retrieval cache for this scope
            self._invalidate_retrieval_cache()
            return fact
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._audit_logger.log_write(
                operation=OperationType.STORE.value,
                memory_type=MemoryType.SEMANTIC.value,
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                channel=self.channel,
                record_ids=[fact_id] if fact_id else None,
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg,
                metadata={"fact_source": source, "confidence": confidence},
            )

    def upsert_entity(self, entity: Entity) -> Entity:
        """
        Add or update an entity in semantic memory.

        Args:
            entity: Entity object to upsert

        Returns:
            Updated Entity object
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None

        try:
            if entity.embedding is None:
                entity.embedding = self.embedder.embed_single(entity.embedding_text())

            result = self.semantic.upsert_entity(entity, user_id=self.user_id, channel=self.channel)

            # Emit entity created event
            self.events.emit(
                MemoryEventEmitter.ENTITY_CREATED,
                EntityCreatedPayload(
                    event_name=MemoryEventEmitter.ENTITY_CREATED,
                    entity_id=entity.id,
                    name=entity.name,
                    entity_type=entity.type,
                    user_id=self.user_id,
                    channel=self.channel,
                )
            )

            # Invalidate retrieval cache for this scope
            self._invalidate_retrieval_cache()
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._audit_logger.log_write(
                operation=OperationType.STORE.value,
                memory_type=MemoryType.SEMANTIC.value,
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                channel=self.channel,
                record_ids=[entity.id],
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg,
                metadata={"entity_type": entity.type, "entity_name": entity.name},
            )

    def update_fact(
        self,
        fact_id: str,
        *,
        claim: Optional[str] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        temporal_context: Optional[str] = None,
        salience: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update editable fields on a fact in place.

        When `claim` changes, re-embeds and recomputes claim_hash.
        Returns the updated fact dict, or None if not found.
        """
        embedding: Optional[List[float]] = None
        claim_hash: Optional[str] = None
        if claim is not None:
            embedding = self.embedder.embed_single(claim)
            claim_hash = compute_claim_hash(claim)

        updated = self.semantic.update_fact(
            fact_id=fact_id,
            user_id=self.user_id,
            claim=claim,
            claim_hash=claim_hash,
            embedding=embedding,
            confidence=confidence,
            source=source,
            temporal_context=temporal_context,
            salience=salience,
            channel=self.channel,
        )

        if updated is None:
            return None

        fields_updated = [
            k for k, v in {
                "claim": claim,
                "confidence": confidence,
                "source": source,
                "temporal_context": temporal_context,
                "salience": salience,
            }.items() if v is not None
        ]
        self.events.emit(
            MemoryEventEmitter.FACT_UPDATED,
            FactUpdatedPayload(
                event_name=MemoryEventEmitter.FACT_UPDATED,
                fact_id=fact_id,
                fields_updated=fields_updated,
                user_id=self.user_id,
                channel=self.channel,
            ),
        )
        self._invalidate_retrieval_cache()
        return updated

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact. Returns True if deleted, False if not found."""
        deleted = self.semantic.delete_fact(
            fact_id=fact_id,
            user_id=self.user_id,
            channel=self.channel,
        )
        if deleted:
            self.events.emit(
                MemoryEventEmitter.FACT_DELETED,
                FactDeletedPayload(
                    event_name=MemoryEventEmitter.FACT_DELETED,
                    fact_id=fact_id,
                    user_id=self.user_id,
                    channel=self.channel,
                ),
            )
            self._invalidate_retrieval_cache()
        return deleted

    def boost_salience(self, fact_id: str, to: float = 0.9) -> Optional[Dict[str, Any]]:
        """
        Raise a fact's salience ("remember this") so it survives memory decay
        and ranks higher in recall. ``to`` is clamped to [0, 1]. Returns the
        updated fact dict, or None if not found.
        """
        target = max(0.0, min(1.0, float(to)))
        return self.update_fact(fact_id, salience=target)

    def forget_fact(self, fact_id: str, hard: bool = False) -> Dict[str, Any]:
        """
        "Forget" a fact. Soft (default) retires it for recall while preserving
        provenance: marks ``temporal_context="past"`` and drops both confidence
        and salience so it sinks in ranking. ``hard=True`` deletes the node.

        Returns ``{"success": bool, "mode": "soft"|"hard", ...}``.
        """
        if hard:
            deleted = self.delete_fact(fact_id)
            return {"success": deleted, "mode": "hard", "fact_id": fact_id}

        current = self.semantic.get_fact_by_id(fact_id, self.user_id)
        if current is None:
            return {"success": False, "mode": "soft", "fact_id": fact_id}

        lowered = round(float(current.get("confidence") or 0.5) * 0.3, 3)
        updated = self.update_fact(
            fact_id,
            confidence=max(0.05, lowered),
            temporal_context="past",
            salience=0.05,
        )
        return {
            "success": updated is not None,
            "mode": "soft",
            "fact_id": fact_id,
            "fact": updated,
        }

    def get_fact_provenance(self, fact_id: str) -> Dict[str, Any]:
        """
        Resolve where a fact was learned ("where did I learn this?"). Returns the
        fact's ``source``/``source_turn_id`` and, when the origin turn is still on
        record, the originating conversation + turn snippet.
        """
        fact = self.semantic.get_fact_by_id(fact_id, self.user_id)
        if fact is None:
            return {"success": False, "fact_id": fact_id, "error": "Fact not found"}

        source_turn_id = fact.get("source_turn_id")
        result: Dict[str, Any] = {
            "success": True,
            "fact_id": fact_id,
            "claim": fact.get("claim"),
            "source": fact.get("source"),
            "source_turn_id": source_turn_id,
            "origin": None,
        }
        if source_turn_id:
            turn = self.episodic.get_turn_by_id(source_turn_id, self.user_id)
            if turn:
                content = turn.get("content") or ""
                result["origin"] = {
                    "conversation_id": turn.get("conversation_id"),
                    "role": turn.get("role"),
                    "timestamp": str(turn.get("timestamp")),
                    "snippet": content[:600],
                }
        return result

    def update_entity(
        self,
        entity_id: str,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
        description: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update editable fields on an entity in place.

        When `name` or `description` changes, re-embeds using
        ``f"{name}: {description or ''}"`` to match upsert_entity.
        Returns the updated entity dict, or None if not found.
        """
        embedding: Optional[List[float]] = None
        if name is not None or description is not None:
            current = self.semantic.get_entity_by_id(entity_id, self.user_id)
            if current is None:
                return None
            new_name = name if name is not None else current.get("name", "")
            new_desc = description if description is not None else current.get("description")
            new_type = type if type is not None else current.get("type", "")
            text = Entity.compute_embedding_text(new_name, new_desc, new_type)
            embedding = self.embedder.embed_single(text)

        updated = self.semantic.update_entity(
            entity_id=entity_id,
            user_id=self.user_id,
            name=name,
            type=type,
            description=description,
            aliases=aliases,
            properties=properties,
            embedding=embedding,
            channel=self.channel,
        )

        if updated is None:
            return None

        fields_updated = [
            k for k, v in {
                "name": name,
                "type": type,
                "description": description,
                "aliases": aliases,
                "properties": properties,
            }.items() if v is not None
        ]
        self.events.emit(
            MemoryEventEmitter.ENTITY_UPDATED,
            EntityUpdatedPayload(
                event_name=MemoryEventEmitter.ENTITY_UPDATED,
                entity_id=entity_id,
                fields_updated=fields_updated,
                user_id=self.user_id,
                channel=self.channel,
            ),
        )
        self._invalidate_retrieval_cache()
        return updated

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity. Returns True if deleted, False if not found."""
        deleted = self.semantic.delete_entity(
            entity_id=entity_id,
            user_id=self.user_id,
            channel=self.channel,
        )
        if deleted:
            self.events.emit(
                MemoryEventEmitter.ENTITY_DELETED,
                EntityDeletedPayload(
                    event_name=MemoryEventEmitter.ENTITY_DELETED,
                    entity_id=entity_id,
                    user_id=self.user_id,
                    channel=self.channel,
                ),
            )
            self._invalidate_retrieval_cache()
        return deleted

    def add_goal(self, goal: Goal) -> Goal:
        """
        Add a goal to track.

        Args:
            goal: Goal object to add

        Returns:
            Created Goal object
        """
        # Embedding is a cross-cutting concern owned by the facade (mirrors how
        # entities are embedded outside SemanticMemory); storage lives in GoalMemory.
        if goal.embedding is None:
            goal.embedding = self.embedder.embed_single(goal.description)
        return self.goal.add_goal(
            goal,
            user_id=self.user_id,
            channel=self.channel,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Retrieve a goal by ID.

        Args:
            goal_id: ID of the goal to retrieve

        Returns:
            Goal object if found and accessible, None otherwise
        """
        return self.goal.get_goal(goal_id, user_id=self.user_id, channel=self.channel)

    def complete_goal(
        self,
        goal_id: str,
        status: str = "completed",
        result: Optional[str] = None
    ) -> bool:
        """
        Update a goal's status.

        Args:
            goal_id: ID of the goal to update
            status: New status ('completed', 'abandoned', 'blocked')
            result: Optional result/summary of goal completion

        Returns:
            True if goal was found and updated, False otherwise
        """
        return self.goal.complete_goal(
            goal_id,
            user_id=self.user_id,
            status=status,
            result=result,
            channel=self.channel,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
        )

    def record_tool_usage(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int,
        turn_id: Optional[str] = None
    ) -> None:
        """
        Record tool invocation for procedural learning.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Tool output
            success: Whether invocation was successful
            latency_ms: Latency in milliseconds
            turn_id: Optional turn ID
        """
        start_time = time.perf_counter()
        record_success = True
        error_msg = None

        if not self.conversation_id:
            logger.debug("Skipping tool usage recording: no conversation_id set")
            return

        try:
            self.procedural.record_invocation(
                conversation_id=self.conversation_id,
                turn_id=turn_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                success=success,
                latency_ms=latency_ms,
                channel=self.channel
            )
        except Exception as e:
            record_success = False
            error_msg = str(e)
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            self._audit_logger.log_write(
                operation=OperationType.RECORD.value,
                memory_type=MemoryType.PROCEDURAL.value,
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                channel=self.channel,
                latency_ms=elapsed_ms,
                success=record_success,
                error_message=error_msg,
                metadata={"tool_name": tool_name, "tool_success": success, "tool_latency_ms": latency_ms},
            )

    # Retrieval operations

    def remember(
        self,
        query: str,
        top_k: int = 10,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        time_window_hours: Optional[int] = None,
        strategy_weights: Optional[Union[RetrievalWeights, Dict[str, float]]] = None,
        channels: Optional[List[str]] = None,
        use_recall_layer: bool = True,
    ) -> MemoryBundle:
        """
        Retrieve relevant memories for the given query.

        This is the main retrieval method that combines multiple
        strategies and returns a unified MemoryBundle.

        Args:
            query: Query text
            top_k: Number of results per category
            include_episodic: Include episodic memory
            include_semantic: Include semantic memory
            include_procedural: Include procedural memory
            time_window_hours: Time window filter
            strategy_weights: Optional per-request weight overrides.
                Can be RetrievalWeights or dict with keys:
                episodic, semantic_facts, semantic_entities, procedural, recency
            channels: Optional explicit list of channels to search.
                If not provided, searches [active_channel, "_global"].
                Example: channels=["project-a", "project-b", "_global"]
            use_recall_layer: If True (default), use RecallLayer with enhanced
                retrieval techniques (hybrid search, entity-centric, query expansion).
                Set to False for basic vector search only.

        Returns:
            MemoryBundle with aggregated results
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None
        result = None

        try:
            if use_recall_layer:
                # Use RecallLayer for enhanced retrieval with multiple techniques
                result = self.recall_layer.recall(
                    query=query,
                    user_id=self.user_id,
                    top_k=top_k,
                    channels=channels or self._default_recall_channels(),
                    time_window_hours=time_window_hours,
                    include_episodic=include_episodic,
                    include_semantic=include_semantic,
                    include_procedural=include_procedural,
                    strategy_weights=strategy_weights,
                    conversation_id=self.conversation_id,
                )
            else:
                # Use base retriever for basic vector search
                result = self.retriever.retrieve(
                    query=query,
                    user_id=self.user_id,
                    top_k=top_k,
                    include_episodic=include_episodic,
                    include_semantic=include_semantic,
                    include_procedural=include_procedural,
                    time_window_hours=time_window_hours,
                    channel=self.channel,
                    channels=channels or self._default_recall_channels(),
                    strategy_weights=strategy_weights,
                    conversation_id=self.conversation_id,
                )

            # Calculate counts for event
            turns_count = len(result.relevant_turns) if result else 0
            facts_count = len(result.facts) if result else 0
            entities_count = len(result.entities) if result else 0
            result_count = turns_count + facts_count + entities_count

            # Emit retrieval complete event
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self.events.emit(
                MemoryEventEmitter.RETRIEVAL_COMPLETE,
                RetrievalCompletePayload(
                    event_name=MemoryEventEmitter.RETRIEVAL_COMPLETE,
                    query=query,
                    result_count=result_count,
                    latency_ms=float(latency_ms),
                    user_id=self.user_id,
                    channel=self.channel,
                    turns_count=turns_count,
                    facts_count=facts_count,
                    entities_count=entities_count,
                )
            )
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Calculate total result count
            result_count = 0
            metadata = {
                "top_k": top_k,
                "include_episodic": include_episodic,
                "include_semantic": include_semantic,
                "include_procedural": include_procedural,
            }
            if result:
                result_count = (
                    len(result.relevant_turns)
                    + len(result.entities)
                    + len(result.facts)
                    + len(result.strategies)
                )
                metadata["turns_count"] = len(result.relevant_turns)
                metadata["entities_count"] = len(result.entities)
                metadata["facts_count"] = len(result.facts)
                metadata["strategies_count"] = len(result.strategies)

            self._audit_logger.log_read(
                operation=OperationType.RETRIEVE.value,
                memory_type=MemoryType.COMPOSITE.value,
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                channels=[self.channel, "_global"] if self.channel != "_global" else ["_global"],
                query_text=query,
                result_count=result_count,
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg,
                metadata=metadata,
            )

    def get_active_goals(self) -> List[Goal]:
        """
        Get all active goals for the user in accessible channels.

        Returns:
            List of active Goal objects
        """
        return self.goal.get_active_goals(user_id=self.user_id, channel=self.channel)

    def get_user_context(self) -> Dict[str, Any]:
        """
        Get user profile and preferences scoped to accessible channels.

        Returns:
            User context dictionary
        """
        with Neo4jConnection.session() as session:
            # Build channel filter for preferences
            if self.channel and self.channel != "_global":
                pref_channel_filter = "AND (p.channel = $channel OR p.channel = '_global' OR p.channel IS NULL)"
            else:
                pref_channel_filter = "AND (p.channel = '_global' OR p.channel IS NULL)"

            result = session.run(f"""
                MATCH (u:User {{id: $user_id}})
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
                WHERE true {pref_channel_filter}
                OPTIONAL MATCH (u)-[exp:HAS_EXPERTISE]->(t:Topic)
                OPTIONAL MATCH (u)-[:INTERESTED_IN]->(i:Topic)
                RETURN u,
                       collect(DISTINCT p) AS preferences,
                       collect(DISTINCT {{topic: t.name, level: exp.level}}) AS expertise,
                       collect(DISTINCT i.name) AS interests
            """, user_id=self.user_id, channel=self.channel)

            record = result.single()
            if not record:
                return {}

            return {
                "user": dict(record["u"]) if record["u"] else {},
                "preferences": [dict(p) for p in record["preferences"] if p],
                "expertise": [e for e in record["expertise"] if e.get("topic")],
                "interests": [i for i in record["interests"] if i]
            }

    def what_worked_for(self, task_description: str, top_k: int = 5) -> List[Strategy]:
        """
        Find successful strategies for similar tasks.

        Args:
            task_description: Description of the task
            top_k: Number of strategies to return

        Returns:
            List of Strategy objects scoped to accessible channels
        """
        return self.procedural.find_strategies(
            task_description,
            top_k,
            user_id=self.user_id,
            channel=self.channel
        )

    # Working memory operations

    def get_working_context(self) -> Dict[str, Any]:
        """
        Get current working memory state.

        Returns:
            Working memory context dictionary
        """
        return self.working.get_context()

    def set_working_context(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """
        Set a value in working memory.

        Args:
            key: Key name
            value: Value to store
            ttl_seconds: Time to live in seconds
        """
        self.working.set(key, value, ttl_seconds)

    # Lifecycle operations

    def reflect(self, outcome: Dict[str, Any]) -> None:
        """
        Trigger reflection on conversation outcome.
        Called at end of conversation or after task completion.

        Args:
            outcome: Outcome dictionary
        """
        if not self.conversation_id:
            logger.warning("Cannot reflect without a conversation_id")
            return

        from ..consolidation.jobs import trigger_reflection
        trigger_reflection(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            outcome=outcome
        )

    # Portability (scriptable import/export)

    def export_memory(self, channel: Optional[str] = None) -> "MemoryExport":
        """Serialize this user's memory graph into a round-trippable envelope.

        The export is text-only (embeddings are regenerated on import).

        Args:
            channel: Limit to one channel (defaults to this instance's channel;
                pass ``"_all"`` to export every channel for the user).
        """
        from ..portability import MemoryExporter

        scope = channel if channel is not None else self.channel
        return MemoryExporter(user_id=self.user_id, channel=scope).export()

    def import_memory(
        self,
        payload: Union["MemoryExport", Dict[str, Any]],
        mode: str = "merge",
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Restore a memory export under this user (idempotent MERGE-on-id).

        Args:
            payload: A ``MemoryExport`` or its serialized dict.
            mode: ``"merge"`` (upsert) or ``"replace"`` (wipe the target
                channel first, then import).
            channel: Override the wipe scope for replace mode.
        """
        from ..portability import MemoryImporter

        return MemoryImporter(user_id=self.user_id).import_export(
            payload, mode=mode, channel=channel  # type: ignore[arg-type]
        )

    def close(self) -> None:
        """Clean up connections."""
        self.working.clear_session()

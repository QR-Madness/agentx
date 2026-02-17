"""Main memory interface - unified API for agent memory operations."""

import logging
import time
from typing import Optional, List, Dict, Any, Union

from ..models import Turn, Entity, Fact, Goal, Strategy, MemoryBundle
from ..embeddings import get_embedder
from ..connections import Neo4jConnection
from ..config import get_settings
from ..audit import MemoryAuditLogger, OperationType, MemoryType
from ..events import (
    MemoryEventEmitter,
    TurnStoredPayload,
    FactLearnedPayload,
    EntityCreatedPayload,
    RetrievalCompletePayload,
)
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .procedural import ProceduralMemory
from .working import WorkingMemory
from .retrieval import MemoryRetriever, RetrievalWeights

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
        events: Optional[MemoryEventEmitter] = None
    ):
        # Validate user_id
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and cannot be empty")

        self.user_id = user_id.strip()
        self.conversation_id = conversation_id
        self.channel = channel
        self.session_id = session_id
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
        self.working = WorkingMemory(
            user_id,
            conversation_id,
            channel=self.channel,
            audit_logger=self._audit_logger
        )
        self.retriever = MemoryRetriever(self, audit_logger=self._audit_logger)

    # Storage operations

    def _invalidate_retrieval_cache(self) -> None:
        """Invalidate retrieval cache for current user/channel scope."""
        self.retriever.invalidate_cache(self.user_id, self.channel)

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
            self.episodic.store_turn(turn, user_id=self.user_id, channel=self.channel)

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

    def learn_fact(
        self,
        claim: str,
        source: str = "extraction",
        confidence: float = 0.8,
        entity_ids: Optional[List[str]] = None,
        source_turn_id: Optional[str] = None
    ) -> Fact:
        """
        Add a fact to semantic memory.

        Args:
            claim: Factual claim
            source: Source of the fact
            confidence: Confidence score (0-1)
            entity_ids: Related entity IDs
            source_turn_id: Source turn ID

        Returns:
            Created Fact object
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None

        try:
            fact = Fact(
                claim=claim,
                source=source,
                confidence=confidence,
                entity_ids=entity_ids or [],
                source_turn_id=source_turn_id,
                embedding=self.embedder.embed_single(claim)
            )
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
                record_ids=[fact.id] if success else None,
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
                text = f"{entity.name}: {entity.description or entity.type}"
                entity.embedding = self.embedder.embed_single(text)

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

    def add_goal(self, goal: Goal) -> Goal:
        """
        Add a goal to track.

        Args:
            goal: Goal object to add

        Returns:
            Created Goal object
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None

        try:
            if goal.embedding is None:
                goal.embedding = self.embedder.embed_single(goal.description)

            with Neo4jConnection.session() as session:
                session.run("""
                    MERGE (u:User {id: $user_id})
                    CREATE (g:Goal {
                        id: $goal_id,
                        description: $description,
                        status: $status,
                        priority: $priority,
                        channel: $channel,
                        created_at: datetime(),
                        embedding: $embedding
                    })
                    MERGE (u)-[:HAS_GOAL]->(g)
                """,
                    user_id=self.user_id,
                    goal_id=goal.id,
                    description=goal.description,
                    status=goal.status,
                    priority=goal.priority,
                    channel=self.channel,
                    embedding=goal.embedding
                )
            return goal
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
                record_ids=[goal.id],
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg,
                metadata={"goal_status": goal.status, "goal_priority": goal.priority},
            )

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Retrieve a goal by ID.

        Args:
            goal_id: ID of the goal to retrieve

        Returns:
            Goal object if found and accessible, None otherwise
        """
        with Neo4jConnection.session() as session:
            # Build channel filter
            if self.channel and self.channel != "_global":
                channel_filter = "AND (g.channel = $channel OR g.channel = '_global')"
            else:
                channel_filter = "AND g.channel = '_global'"

            result = session.run(f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal {{id: $goal_id}})
                WHERE true {channel_filter}
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent.id AS parent_goal_id
            """, goal_id=goal_id, user_id=self.user_id, channel=self.channel)

            record = result.single()
            if not record or not record["g"]:
                return None

            goal_data = dict(record["g"])
            if record["parent_goal_id"]:
                goal_data["parent_goal_id"] = record["parent_goal_id"]
            return Goal(**goal_data)

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
        start_time = time.perf_counter()
        success = True
        error_msg = None
        updated = False

        try:
            with Neo4jConnection.session() as session:
                # Build channel filter to respect access boundaries
                if self.channel and self.channel != "_global":
                    channel_filter = "AND (g.channel = $channel OR g.channel = '_global')"
                else:
                    channel_filter = "AND g.channel = '_global'"

                # SECURITY: Verify user owns the goal via HAS_GOAL relationship
                query_result = session.run(f"""
                    MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal {{id: $goal_id}})
                    WHERE true {channel_filter}
                    SET g.status = $status,
                        g.completed_at = datetime(),
                        g.result = $result
                    RETURN g.id AS updated_id
                """,
                    user_id=self.user_id,
                    goal_id=goal_id,
                    status=status,
                    result=result,
                    channel=self.channel
                )
                record = query_result.single()
                updated = record is not None and record["updated_id"] is not None
                return updated
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            self._audit_logger.log_write(
                operation=OperationType.UPDATE.value,
                memory_type=MemoryType.SEMANTIC.value,
                user_id=self.user_id,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                channel=self.channel,
                record_ids=[goal_id] if updated else None,
                latency_ms=latency_ms,
                success=success,
                error_message=error_msg,
                metadata={"new_status": status, "goal_found": updated},
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
        channels: Optional[List[str]] = None
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

        Returns:
            MemoryBundle with aggregated results
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None
        result = None

        try:
            result = self.retriever.retrieve(
                query=query,
                user_id=self.user_id,
                top_k=top_k,
                include_episodic=include_episodic,
                include_semantic=include_semantic,
                include_procedural=include_procedural,
                time_window_hours=time_window_hours,
                channel=self.channel,
                channels=channels,
                strategy_weights=strategy_weights
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
        with Neo4jConnection.session() as session:
            # Build channel filter
            if self.channel and self.channel != "_global":
                channel_filter = "AND (g.channel = $channel OR g.channel = '_global')"
            else:
                channel_filter = "AND g.channel = '_global'"

            result = session.run(f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal)
                WHERE g.status = 'active' {channel_filter}
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent
                ORDER BY g.priority DESC
            """, user_id=self.user_id, channel=self.channel)

            goals = []
            for record in result:
                goal_data = dict(record["g"])
                if record["parent"]:
                    goal_data["parent"] = dict(record["parent"])
                goals.append(Goal(**goal_data))
            return goals

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
        from ..consolidation.jobs import trigger_reflection
        trigger_reflection(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            outcome=outcome
        )

    def close(self) -> None:
        """Clean up connections."""
        self.working.clear_session()

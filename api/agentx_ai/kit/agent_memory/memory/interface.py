"""Main memory interface - unified API for agent memory operations."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from ..models import Turn, Entity, Fact, Goal, Strategy, MemoryBundle
from ..embeddings import get_embedder
from ..connections import Neo4jConnection
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .procedural import ProceduralMemory
from .working import WorkingMemory
from .retrieval import MemoryRetriever


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

    def __init__(self, user_id: str, conversation_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.embedder = get_embedder()

        # Sub-modules
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.working = WorkingMemory(user_id, conversation_id)
        self.retriever = MemoryRetriever(self)

    # Storage operations

    def store_turn(self, turn: Turn) -> None:
        """
        Store a conversation turn in episodic memory.

        Args:
            turn: Turn object to store
        """
        # Generate embedding if not provided
        if turn.embedding is None:
            turn.embedding = self.embedder.embed_single(turn.content)

        # Store in Neo4j (graph)
        self.episodic.store_turn(turn)

        # Store in PostgreSQL (logs)
        self.episodic.store_turn_log(turn)

        # Update working memory
        self.working.add_turn(turn)

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
        fact = Fact(
            claim=claim,
            source=source,
            confidence=confidence,
            entity_ids=entity_ids or [],
            source_turn_id=source_turn_id,
            embedding=self.embedder.embed_single(claim)
        )
        self.semantic.store_fact(fact)
        return fact

    def upsert_entity(self, entity: Entity) -> Entity:
        """
        Add or update an entity in semantic memory.

        Args:
            entity: Entity object to upsert

        Returns:
            Updated Entity object
        """
        if entity.embedding is None:
            text = f"{entity.name}: {entity.description or entity.type}"
            entity.embedding = self.embedder.embed_single(text)

        return self.semantic.upsert_entity(entity)

    def add_goal(self, goal: Goal) -> Goal:
        """
        Add a goal to track.

        Args:
            goal: Goal object to add

        Returns:
            Created Goal object
        """
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
                embedding=goal.embedding
            )
        return goal

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
        self.procedural.record_invocation(
            conversation_id=self.conversation_id,
            turn_id=turn_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            success=success,
            latency_ms=latency_ms
        )

    # Retrieval operations

    def remember(
        self,
        query: str,
        top_k: int = 10,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        time_window_hours: Optional[int] = None
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

        Returns:
            MemoryBundle with aggregated results
        """
        return self.retriever.retrieve(
            query=query,
            user_id=self.user_id,
            top_k=top_k,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            time_window_hours=time_window_hours
        )

    def get_active_goals(self) -> List[Goal]:
        """
        Get all active goals for the user.

        Returns:
            List of active Goal objects
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_GOAL]->(g:Goal)
                WHERE g.status = 'active'
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent
                ORDER BY g.priority DESC
            """, user_id=self.user_id)

            goals = []
            for record in result:
                goal_data = dict(record["g"])
                if record["parent"]:
                    goal_data["parent"] = dict(record["parent"])
                goals.append(Goal(**goal_data))
            return goals

    def get_user_context(self) -> Dict[str, Any]:
        """
        Get user profile and preferences.

        Returns:
            User context dictionary
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
                OPTIONAL MATCH (u)-[exp:HAS_EXPERTISE]->(t:Topic)
                OPTIONAL MATCH (u)-[:INTERESTED_IN]->(i:Topic)
                RETURN u,
                       collect(DISTINCT p) AS preferences,
                       collect(DISTINCT {topic: t.name, level: exp.level}) AS expertise,
                       collect(DISTINCT i.name) AS interests
            """, user_id=self.user_id)

            record = result.single()
            if not record:
                return {}

            return {
                "user": dict(record["u"]) if record["u"] else {},
                "preferences": [dict(p) for p in record["preferences"]],
                "expertise": record["expertise"],
                "interests": record["interests"]
            }

    def what_worked_for(self, task_description: str, top_k: int = 5) -> List[Strategy]:
        """
        Find successful strategies for similar tasks.

        Args:
            task_description: Description of the task
            top_k: Number of strategies to return

        Returns:
            List of Strategy objects
        """
        return self.procedural.find_strategies(task_description, top_k)

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

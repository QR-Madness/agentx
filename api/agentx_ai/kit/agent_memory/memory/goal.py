"""Goal memory - user goals and objectives, stored as a Neo4j graph.

Mirrors the SemanticMemory/EpisodicMemory sub-module pattern: the constructor
takes only an optional audit logger, and every method receives ``user_id`` /
``channel`` per call. The ``AgentMemory`` facade owns embedding and delegates
storage here so goal operations are independently testable.
"""

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..connections import Neo4jConnection
from ..audit import OperationType, MemoryType
from ..models import Goal
from .query_utils import get_channel_filter_cypher

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger

logger = logging.getLogger(__name__)


class GoalMemory:
    """Stores and retrieves user goals as ``(:Goal)`` nodes linked via
    ``(:User)-[:HAS_GOAL]->(:Goal)`` with optional ``[:SUBGOAL_OF]`` hierarchy."""

    def __init__(self, audit_logger: Optional["MemoryAuditLogger"] = None):
        """Initialize goal memory.

        Args:
            audit_logger: Optional audit logger for operation tracking.
        """
        self._audit_logger = audit_logger

    def add_goal(
        self,
        goal: Goal,
        user_id: str,
        channel: str = "_global",
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Goal:
        """Create a goal node and link it to the user (and parent, if any).

        The caller is responsible for populating ``goal.embedding`` (the facade
        does this, mirroring how entities are embedded outside SemanticMemory).
        """
        start_time = time.perf_counter()
        success = True
        error_msg = None

        try:
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
                    user_id=user_id,
                    goal_id=goal.id,
                    description=goal.description,
                    status=goal.status,
                    priority=goal.priority,
                    channel=channel,
                    embedding=goal.embedding
                )
                if goal.parent_goal_id:
                    session.run("""
                        MATCH (child:Goal {id: $child_id})
                        MATCH (parent:Goal {id: $parent_id})
                        MERGE (child)-[:SUBGOAL_OF]->(parent)
                    """,
                        child_id=goal.id,
                        parent_id=goal.parent_goal_id,
                    )
            return goal
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._audit_logger:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                self._audit_logger.log_write(
                    operation=OperationType.STORE.value,
                    memory_type=MemoryType.SEMANTIC.value,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    channel=channel,
                    record_ids=[goal.id],
                    latency_ms=latency_ms,
                    success=success,
                    error_message=error_msg,
                    metadata={"goal_status": goal.status, "goal_priority": goal.priority},
                )

    def get_goal(
        self,
        goal_id: str,
        user_id: str,
        channel: str = "_global",
    ) -> Optional[Goal]:
        """Retrieve a goal by ID if the user owns it and it's in scope."""
        with Neo4jConnection.session() as session:
            channel_filter = get_channel_filter_cypher(channel)

            result = session.run(f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal {{id: $goal_id}})
                WHERE true {channel_filter}
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent.id AS parent_goal_id
            """, goal_id=goal_id, user_id=user_id, channel=channel)

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
        user_id: str,
        status: str = "completed",
        result: Optional[str] = None,
        channel: str = "_global",
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> bool:
        """Update a goal's status. Returns True if the goal was found and updated."""
        start_time = time.perf_counter()
        success = True
        error_msg = None
        updated = False

        try:
            with Neo4jConnection.session() as session:
                channel_filter = get_channel_filter_cypher(channel)

                # SECURITY: Verify user owns the goal via HAS_GOAL relationship
                query_result = session.run(f"""
                    MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal {{id: $goal_id}})
                    WHERE true {channel_filter}
                    SET g.status = $status,
                        g.completed_at = datetime(),
                        g.result = $result
                    RETURN g.id AS updated_id
                """,
                    user_id=user_id,
                    goal_id=goal_id,
                    status=status,
                    result=result,
                    channel=channel
                )
                record = query_result.single()
                updated = record is not None and record["updated_id"] is not None
                return updated
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._audit_logger:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                self._audit_logger.log_write(
                    operation=OperationType.UPDATE.value,
                    memory_type=MemoryType.SEMANTIC.value,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    channel=channel,
                    record_ids=[goal_id] if updated else None,
                    latency_ms=latency_ms,
                    success=success,
                    error_message=error_msg,
                    metadata={"new_status": status, "goal_found": updated},
                )

    def get_active_goals(
        self,
        user_id: str,
        channel: str = "_global",
    ) -> List[Goal]:
        """Get all active goals for the user in accessible channels."""
        with Neo4jConnection.session() as session:
            if channel and channel != "_global":
                channel_filter = "AND (g.channel = $channel OR g.channel = '_global')"
            else:
                channel_filter = "AND g.channel = '_global'"

            result = session.run(f"""
                MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal)
                WHERE g.status = 'active' {channel_filter}
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent
                ORDER BY g.priority DESC
            """, user_id=user_id, channel=channel)

            goals: List[Goal] = []
            for record in result:
                goal_data: Dict[str, Any] = dict(record["g"])
                if record["parent"]:
                    goal_data["parent"] = dict(record["parent"])
                goals.append(Goal(**goal_data))
            return goals

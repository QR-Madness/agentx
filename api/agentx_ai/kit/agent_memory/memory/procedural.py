"""Procedural memory - tool usage patterns and successful strategies."""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, TYPE_CHECKING, cast
import json

from typing_extensions import LiteralString
from sqlalchemy import text

from ..models import Strategy
from ..connections import Neo4jConnection, get_postgres_session
from ..embeddings import get_embedder
from ..query_utils import CypherFilterBuilder, convert_record_datetimes

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger


class ProceduralMemory:
    """Handles tool usage patterns and successful strategies."""

    def __init__(self, audit_logger: Optional["MemoryAuditLogger"] = None):
        """Initialize procedural memory.

        Args:
            audit_logger: Optional audit logger for operation tracking.
        """
        self.embedder = get_embedder()
        self._audit_logger = audit_logger

    def record_invocation(
        self,
        conversation_id: str,
        turn_id: Optional[str],
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int,
        channel: str = "_global",
        turn_index: Optional[int] = None,
    ) -> None:
        """
        Record a tool invocation.

        Args:
            conversation_id: Conversation ID
            turn_id: Turn ID (optional)
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Tool output
            success: Whether invocation was successful
            latency_ms: Latency in milliseconds
            channel: Memory channel
            turn_index: Turn index within conversation (optional)
        """
        # PostgreSQL for audit log
        with get_postgres_session() as session:
            session.execute(
                text(
                    """
                INSERT INTO tool_invocations
                (conversation_id, turn_index, tool_name, tool_input, tool_output, success, latency_ms, channel)
                VALUES (:conv_id, :turn_idx, :tool, :input, :output, :success, :latency, :channel)
            """
                ),
                {
                    "conv_id": conversation_id,
                    "turn_idx": turn_index if turn_index is not None else 0,
                    "tool": tool_name,
                    "input": json.dumps(tool_input),
                    "output": json.dumps(tool_output) if tool_output else None,
                    "success": success,
                    "latency": latency_ms,
                    "channel": channel,
                },
            )

        # Neo4j for graph relationships
        # FIX: Calculate running average correctly
        # Formula: new_avg = old_avg + (new_value - old_avg) / new_count
        # We need to increment count AFTER calculating the average update
        with Neo4jConnection.session() as neo_session:
            neo_session.run(
                """
                MERGE (tool:Tool {name: $tool_name})
                ON CREATE SET tool.usage_count = 0,
                              tool.success_count = 0,
                              tool.avg_latency_ms = $latency
                SET tool.avg_latency_ms = CASE
                        WHEN tool.usage_count = 0 THEN $latency
                        ELSE tool.avg_latency_ms + ($latency - tool.avg_latency_ms) / (tool.usage_count + 1)
                    END,
                    tool.usage_count = tool.usage_count + 1,
                    tool.success_count = tool.success_count + CASE WHEN $success THEN 1 ELSE 0 END

                WITH tool
                MATCH (c:Conversation {id: $conv_id})
                CREATE (inv:ToolInvocation {
                    timestamp: datetime(),
                    tool_name: $tool_name,
                    success: $success,
                    latency_ms: $latency,
                    channel: $channel,
                    turn_index: $turn_index
                })
                MERGE (c)-[:USED_TOOL]->(inv)
                MERGE (inv)-[:INVOKED]->(tool)
            """,
                conv_id=conversation_id,
                tool_name=tool_name,
                success=success,
                latency=latency_ms,
                channel=channel,
                turn_index=turn_index,
            )

    def learn_strategy(
        self,
        description: str,
        context_pattern: str,
        tool_sequence: List[str],
        from_conversation_id: Optional[str] = None,
        success: bool = True,
        user_id: Optional[str] = None,
        channel: str = "_global",
    ) -> Strategy:
        """
        Record a successful (or failed) strategy pattern.

        Args:
            description: Strategy description
            context_pattern: Context pattern for matching
            tool_sequence: Sequence of tools used
            from_conversation_id: Source conversation ID
            success: Whether strategy was successful
            user_id: User ID for linking
            channel: Memory channel

        Returns:
            Strategy object
        """
        embedding = self.embedder.embed_single(description)

        strategy = Strategy(
            description=description,
            context_pattern=context_pattern,
            tool_sequence=tool_sequence,
            embedding=embedding,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            last_used=datetime.now(timezone.utc),
        )

        with Neo4jConnection.session() as session:
            session.run(
                """
                CREATE (s:Strategy {
                    id: $id,
                    description: $description,
                    context_pattern: $context_pattern,
                    tool_sequence: $tool_sequence,
                    embedding: $embedding,
                    success_count: $success_count,
                    failure_count: $failure_count,
                    user_id: $user_id,
                    channel: $channel,
                    last_used: datetime()
                })

                // Link to user
                WITH s
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:HAS_STRATEGY]->(s)

                // Link to tools
                WITH s
                UNWIND $tool_sequence AS tool_name
                MATCH (t:Tool {name: tool_name})
                MERGE (s)-[:USES_TOOL]->(t)

                // Link to conversation if provided
                WITH s
                OPTIONAL MATCH (c:Conversation {id: $conv_id})
                FOREACH (_ IN CASE WHEN c IS NOT NULL AND $success THEN [1] ELSE [] END |
                    MERGE (s)-[:SUCCEEDED_IN]->(c)
                )
                FOREACH (_ IN CASE WHEN c IS NOT NULL AND NOT $success THEN [1] ELSE [] END |
                    MERGE (s)-[:FAILED_IN]->(c)
                )
            """,
                id=strategy.id,
                description=strategy.description,
                context_pattern=strategy.context_pattern,
                tool_sequence=strategy.tool_sequence,
                embedding=strategy.embedding,
                success_count=strategy.success_count,
                failure_count=strategy.failure_count,
                conv_id=from_conversation_id,
                success=success,
                user_id=user_id,
                channel=channel,
            )

        return strategy

    def find_strategies(
        self,
        task_description: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Strategy]:
        """
        Find strategies that worked for similar tasks.

        Args:
            task_description: Description of the task
            top_k: Number of strategies to return
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of Strategy objects
        """
        embedding = self.embedder.embed_single(task_description)

        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("s")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            result = session.run(
                cast(LiteralString, f"""
                CALL db.index.vector.queryNodes('strategy_embeddings', $k, $embedding)
                YIELD node AS s, score
                WHERE s.success_count > 0 {filters.build_inline()}
                RETURN s.id AS id,
                       s.description AS description,
                       s.context_pattern AS context_pattern,
                       s.tool_sequence AS tool_sequence,
                       s.success_count AS success_count,
                       s.failure_count AS failure_count,
                       s.channel AS channel,
                       CASE WHEN (s.success_count + s.failure_count) > 0
                            THEN s.success_count * 1.0 / (s.success_count + s.failure_count)
                            ELSE 0.5
                       END AS success_rate,
                       score
                ORDER BY success_rate * score DESC
            """),
                k=top_k * 2,
                embedding=embedding,
                user_id=user_id,
                channel=channel,
            )

            strategies = []
            for record in result:
                strategies.append(
                    Strategy(
                        id=record["id"],
                        description=record["description"],
                        context_pattern=record["context_pattern"],
                        tool_sequence=record["tool_sequence"],
                        success_count=record["success_count"],
                        failure_count=record["failure_count"],
                    )
                )

            return strategies[:top_k]

    def reinforce_strategy(
        self,
        strategy_id: str,
        success: bool,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> bool:
        """
        Update strategy success/failure counts.

        Args:
            strategy_id: Strategy ID
            success: Whether execution was successful
            user_id: User ID (for validation)
            channel: Channel scope (validates strategy is accessible)

        Returns:
            True if strategy was found and updated, False otherwise
        """
        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("s")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            if success:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy {{id: $id}})
                    WHERE true {filters.build_inline()}
                    SET s.success_count = s.success_count + 1,
                        s.last_used = datetime()
                    RETURN s.id AS updated_id
                """,
                    ),
                    id=strategy_id,
                    user_id=user_id,
                    channel=channel,
                )
            else:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy {{id: $id}})
                    WHERE true {filters.build_inline()}
                    SET s.failure_count = s.failure_count + 1,
                        s.last_used = datetime()
                    RETURN s.id AS updated_id
                """,
                    ),
                    id=strategy_id,
                    user_id=user_id,
                    channel=channel,
                )

            record = result.single()
            return record is not None and record["updated_id"] is not None

    def get_tool_stats(
        self,
        task_type: Optional[str] = None,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get tool usage statistics, optionally filtered by task type.

        Args:
            task_type: Optional task type filter
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of tool statistics
        """
        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("s")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            if task_type:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy)-[:USES_TOOL]->(t:Tool)
                    WHERE s.context_pattern CONTAINS $task_type
                          {filters.build_inline()}
                    WITH t,
                         sum(s.success_count) AS successes,
                         sum(s.failure_count) AS failures
                    RETURN t.name AS tool,
                           successes,
                           failures,
                           CASE WHEN (successes + failures) > 0
                                THEN successes * 1.0 / (successes + failures)
                                ELSE 0.5
                           END AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY success_rate DESC
                """,
                    ),
                    task_type=task_type,
                    user_id=user_id,
                    channel=channel,
                )
            else:
                # For global stats without task_type, still scope to user's strategies
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy)-[:USES_TOOL]->(t:Tool)
                    WHERE true {filters.build_inline()}
                    WITH t,
                         sum(s.success_count) AS successes,
                         sum(s.failure_count) AS failures,
                         count(DISTINCT s) AS strategy_count
                    RETURN t.name AS tool,
                           successes + failures AS usage_count,
                           successes AS success_count,
                           CASE WHEN (successes + failures) > 0
                                THEN successes * 1.0 / (successes + failures)
                                ELSE 0.5
                           END AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY usage_count DESC
                """,
                    ),
                    user_id=user_id,
                    channel=channel,
                )

            return [dict(record) for record in result]

    def list_strategies(
        self, user_id: str, channel: str = "_global", offset: int = 0, limit: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List strategies with pagination.

        Args:
            user_id: User ID to filter by
            channel: Filter by channel (searches channel + _global)
            offset: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (strategies list, total count)
        """
        with Neo4jConnection.session() as session:
            # Build WHERE conditions
            conditions = ["s.user_id = $user_id"]

            # Channel filter - _all means no filter, otherwise search both specified channel and _global
            if channel == "_all":
                pass  # No channel filter - show all channels
            elif channel and channel != "_global":
                conditions.append("(s.channel = $channel OR s.channel = '_global')")
            else:
                conditions.append("s.channel = '_global'")

            where_clause = " AND ".join(conditions)

            # Get total count
            count_result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (s:Strategy)
                WHERE {where_clause}
                RETURN count(s) AS total
            """,
                ),
                user_id=user_id,
                channel=channel,
            )
            record = count_result.single()
            total = record["total"] if record else 0

            # Get paginated results with success rate
            result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (s:Strategy)
                WHERE {where_clause}
                RETURN s.id AS id,
                       s.description AS description,
                       s.tool_sequence AS tool_sequence,
                       s.success_count AS success_count,
                       s.failure_count AS failure_count,
                       CASE WHEN (s.success_count + s.failure_count) > 0
                            THEN s.success_count * 1.0 / (s.success_count + s.failure_count)
                            ELSE 0.5
                       END AS success_rate,
                       s.channel AS channel,
                       s.last_used AS last_used
                ORDER BY s.success_count DESC, s.last_used DESC
                SKIP $offset
                LIMIT $limit
            """,
                ),
                user_id=user_id,
                channel=channel,
                offset=offset,
                limit=limit,
            )

            strategies = []
            for record in result:
                strategy = convert_record_datetimes(dict(record))
                strategies.append(strategy)
            return strategies, total

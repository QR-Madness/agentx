"""Procedural memory - tool usage patterns and successful strategies."""

from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from ..models import Strategy
from ..connections import Neo4jConnection, get_postgres_session
from ..embeddings import get_embedder


class ProceduralMemory:
    """Handles tool usage patterns and successful strategies."""

    def __init__(self):
        self.embedder = get_embedder()

    def record_invocation(
        self,
        conversation_id: str,
        turn_id: Optional[str],
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int
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
        """
        # PostgreSQL for audit log
        with get_postgres_session() as session:
            session.execute("""
                INSERT INTO tool_invocations
                (conversation_id, turn_index, tool_name, tool_input, tool_output, success, latency_ms)
                VALUES (:conv_id, :turn_idx, :tool, :input, :output, :success, :latency)
            """, {
                "conv_id": conversation_id,
                "turn_idx": 0,  # Would need proper turn index
                "tool": tool_name,
                "input": json.dumps(tool_input),
                "output": json.dumps(tool_output) if tool_output else None,
                "success": success,
                "latency": latency_ms
            })

        # Neo4j for graph relationships
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (tool:Tool {name: $tool_name})
                ON CREATE SET tool.usage_count = 0, tool.success_count = 0
                SET tool.usage_count = tool.usage_count + 1,
                    tool.success_count = tool.success_count + CASE WHEN $success THEN 1 ELSE 0 END,
                    tool.avg_latency_ms = coalesce(
                        (tool.avg_latency_ms * (tool.usage_count - 1) + $latency) / tool.usage_count,
                        $latency
                    )

                WITH tool
                MATCH (c:Conversation {id: $conv_id})
                CREATE (inv:ToolInvocation {
                    timestamp: datetime(),
                    tool_name: $tool_name,
                    success: $success,
                    latency_ms: $latency
                })
                MERGE (c)-[:USED_TOOL]->(inv)
                MERGE (inv)-[:INVOKED]->(tool)
            """,
                conv_id=conversation_id,
                tool_name=tool_name,
                success=success,
                latency=latency_ms
            )

    def learn_strategy(
        self,
        description: str,
        context_pattern: str,
        tool_sequence: List[str],
        from_conversation_id: Optional[str] = None,
        success: bool = True
    ) -> Strategy:
        """
        Record a successful (or failed) strategy pattern.

        Args:
            description: Strategy description
            context_pattern: Context pattern for matching
            tool_sequence: Sequence of tools used
            from_conversation_id: Source conversation ID
            success: Whether strategy was successful

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
            last_used=datetime.utcnow()
        )

        with Neo4jConnection.session() as session:
            session.run("""
                CREATE (s:Strategy {
                    id: $id,
                    description: $description,
                    context_pattern: $context_pattern,
                    tool_sequence: $tool_sequence,
                    embedding: $embedding,
                    success_count: $success_count,
                    failure_count: $failure_count,
                    last_used: datetime()
                })

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
                success=success
            )

        return strategy

    def find_strategies(
        self,
        task_description: str,
        top_k: int = 5
    ) -> List[Strategy]:
        """
        Find strategies that worked for similar tasks.

        Args:
            task_description: Description of the task
            top_k: Number of strategies to return

        Returns:
            List of Strategy objects
        """
        embedding = self.embedder.embed_single(task_description)

        with Neo4jConnection.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('strategy_embeddings', $k, $embedding)
                YIELD node AS s, score
                WHERE s.success_count > 0
                RETURN s.id AS id,
                       s.description AS description,
                       s.context_pattern AS context_pattern,
                       s.tool_sequence AS tool_sequence,
                       s.success_count AS success_count,
                       s.failure_count AS failure_count,
                       s.success_count * 1.0 / (s.success_count + s.failure_count) AS success_rate,
                       score
                ORDER BY success_rate * score DESC
            """,
                k=top_k * 2,
                embedding=embedding
            )

            strategies = []
            for record in result:
                strategies.append(Strategy(
                    id=record["id"],
                    description=record["description"],
                    context_pattern=record["context_pattern"],
                    tool_sequence=record["tool_sequence"],
                    success_count=record["success_count"],
                    failure_count=record["failure_count"]
                ))

            return strategies[:top_k]

    def reinforce_strategy(self, strategy_id: str, success: bool) -> None:
        """
        Update strategy success/failure counts.

        Args:
            strategy_id: Strategy ID
            success: Whether execution was successful
        """
        with Neo4jConnection.session() as session:
            if success:
                session.run("""
                    MATCH (s:Strategy {id: $id})
                    SET s.success_count = s.success_count + 1,
                        s.last_used = datetime()
                """, id=strategy_id)
            else:
                session.run("""
                    MATCH (s:Strategy {id: $id})
                    SET s.failure_count = s.failure_count + 1,
                        s.last_used = datetime()
                """, id=strategy_id)

    def get_tool_stats(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get tool usage statistics, optionally filtered by task type.

        Args:
            task_type: Optional task type filter

        Returns:
            List of tool statistics
        """
        with Neo4jConnection.session() as session:
            if task_type:
                result = session.run("""
                    MATCH (s:Strategy)-[:USES_TOOL]->(t:Tool)
                    WHERE s.context_pattern CONTAINS $task_type
                    WITH t,
                         sum(s.success_count) AS successes,
                         sum(s.failure_count) AS failures
                    RETURN t.name AS tool,
                           successes,
                           failures,
                           successes * 1.0 / (successes + failures + 0.1) AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY success_rate DESC
                """, task_type=task_type)
            else:
                result = session.run("""
                    MATCH (t:Tool)
                    RETURN t.name AS tool,
                           t.usage_count AS usage_count,
                           t.success_count AS success_count,
                           t.success_count * 1.0 / (t.usage_count + 0.1) AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY usage_count DESC
                """)

            return [dict(record) for record in result]

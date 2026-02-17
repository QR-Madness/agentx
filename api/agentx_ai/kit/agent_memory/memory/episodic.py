"""Episodic memory - conversation history storage and retrieval."""

import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from sqlalchemy import text

from ..models import Turn
from ..connections import Neo4jConnection, get_postgres_session
from ..config import get_settings

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger

settings = get_settings()


class EpisodicMemory:
    """Handles storage and retrieval of conversation history."""

    def __init__(self, audit_logger: Optional["MemoryAuditLogger"] = None):
        """Initialize episodic memory.

        Args:
            audit_logger: Optional audit logger for operation tracking.
        """
        self._audit_logger = audit_logger

    def store_turn(self, turn: Turn, user_id: Optional[str] = None, channel: str = "_global") -> None:
        """
        Store turn in Neo4j graph.

        Args:
            turn: Turn object to store
            user_id: User ID for linking
            channel: Memory channel
        """
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (c:Conversation {id: $conv_id})
                ON CREATE SET c.started_at = datetime(),
                              c.user_id = $user_id,
                              c.channel = $channel

                CREATE (t:Turn {
                    id: $turn_id,
                    index: $index,
                    timestamp: datetime($timestamp),
                    role: $role,
                    content: $content,
                    embedding: $embedding,
                    token_count: $token_count,
                    channel: $channel
                })

                MERGE (c)-[:HAS_TURN]->(t)

                // Link user to conversation
                WITH c, t
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:HAS_CONVERSATION]->(c)

                // Link to previous turn if exists
                WITH c, t
                OPTIONAL MATCH (c)-[:HAS_TURN]->(prev:Turn)
                WHERE prev.index = $index - 1
                FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (prev)-[:FOLLOWED_BY]->(t)
                )
            """,
                conv_id=turn.conversation_id,
                turn_id=turn.id,
                index=turn.index,
                timestamp=turn.timestamp.isoformat(),
                role=turn.role,
                content=turn.content,
                embedding=turn.embedding,
                token_count=turn.token_count,
                user_id=user_id,
                channel=channel
            )

    def store_turn_log(self, turn: Turn, channel: str = "_global") -> None:
        """
        Store turn in PostgreSQL for audit/backup.

        Args:
            turn: Turn object to store
            channel: Memory channel
        """
        with get_postgres_session() as session:
            # Store embedding as JSON array (not Python str representation) for proper parsing
            embedding_json = json.dumps(turn.embedding) if turn.embedding else None

            session.execute(text("""
                INSERT INTO conversation_logs
                (conversation_id, turn_index, timestamp, role, content, token_count, embedding, channel)
                VALUES (:conv_id, :index, :timestamp, :role, :content, :tokens, :embedding, :channel)
                ON CONFLICT (conversation_id, turn_index) DO UPDATE
                SET content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding,
                    channel = EXCLUDED.channel
            """), {
                "conv_id": turn.conversation_id,
                "index": turn.index,
                "timestamp": turn.timestamp,
                "role": turn.role,
                "content": turn.content,
                "tokens": turn.token_count,
                "embedding": embedding_json,
                "channel": channel
            })

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
        time_window_hours: Optional[int] = None,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memory by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            user_id: Filter by user ID
            time_window_hours: Filter by time window (must be positive, max 8760 = 1 year)
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of matching turns
        """
        with Neo4jConnection.session() as session:
            # Build time filter with bounds validation
            time_filter = ""
            if time_window_hours is not None:
                # Validate time_window_hours to prevent invalid queries
                validated_hours = max(1, min(int(time_window_hours), 8760))  # 1 hour to 1 year
                time_filter = f"AND t.timestamp > datetime() - duration('PT{validated_hours}H')"

            user_filter = ""
            if user_id:
                user_filter = "AND c.user_id = $user_id"

            # Channel filter - search both specified channel and _global
            channel_filter = ""
            if channel and channel != "_global":
                channel_filter = "AND (t.channel = $channel OR t.channel = '_global')"
            elif channel == "_global":
                channel_filter = "AND t.channel = '_global'"

            result = session.run(f"""
                CALL db.index.vector.queryNodes('turn_embeddings', $k, $embedding)
                YIELD node AS t, score
                MATCH (c:Conversation)-[:HAS_TURN]->(t)
                WHERE true {time_filter} {user_filter} {channel_filter}
                RETURN t.id AS id,
                       t.content AS content,
                       t.role AS role,
                       t.timestamp AS timestamp,
                       t.channel AS channel,
                       c.id AS conversation_id,
                       score
                ORDER BY score DESC
            """,
                k=top_k * 2,  # Over-fetch for filtering
                embedding=query_embedding,
                user_id=user_id,
                channel=channel
            )

            return [dict(record) for record in result][:top_k]

    def get_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        channel: Optional[str] = None
    ) -> List[Turn]:
        """
        Get all turns in a conversation.

        Args:
            conversation_id: Conversation ID
            user_id: Filter by user ID (security check)
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of Turn objects
        """
        with Neo4jConnection.session() as session:
            # Build user filter
            user_filter = ""
            if user_id:
                user_filter = "AND c.user_id = $user_id"

            # Build channel filter - search both specified channel and _global
            channel_filter = ""
            if channel and channel != "_global":
                channel_filter = "AND (t.channel = $channel OR t.channel = '_global')"
            elif channel == "_global":
                channel_filter = "AND t.channel = '_global'"

            result = session.run(f"""
                MATCH (c:Conversation {{id: $conv_id}})-[:HAS_TURN]->(t:Turn)
                WHERE true {user_filter} {channel_filter}
                RETURN t
                ORDER BY t.index
            """, conv_id=conversation_id, user_id=user_id, channel=channel)

            return [Turn(**dict(record["t"])) for record in result]

    def get_recent_turns(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 50,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent turns across all conversations.

        Args:
            user_id: User ID
            hours: Time window in hours (must be positive, max 8760 = 1 year)
            limit: Maximum number of results (capped to 500)
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of turn dictionaries
        """
        # Validate hours and limit to prevent resource exhaustion
        validated_hours = max(1, min(int(hours), 8760))  # 1 hour to 1 year
        validated_limit = max(1, min(int(limit), 500))   # Cap at 500 results

        with Neo4jConnection.session() as session:
            # Channel filter - search both specified channel and _global
            channel_filter = ""
            if channel and channel != "_global":
                channel_filter = "AND (t.channel = $channel OR t.channel = '_global')"
            elif channel == "_global":
                channel_filter = "AND t.channel = '_global'"

            result = session.run(f"""
                MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
                WHERE c.user_id = $user_id
                  AND t.timestamp > datetime() - duration('PT' + $hours + 'H')
                  {channel_filter}
                RETURN t.content AS content,
                       t.role AS role,
                       t.timestamp AS timestamp,
                       t.channel AS channel,
                       c.id AS conversation_id
                ORDER BY t.timestamp DESC
                LIMIT $limit
            """,
                user_id=user_id,
                hours=str(validated_hours),
                limit=validated_limit,
                channel=channel
            )

            return [dict(record) for record in result]

"""Episodic memory - conversation history storage and retrieval."""

from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import text

from ..models import Turn
from ..connections import Neo4jConnection, get_postgres_session


class EpisodicMemory:
    """Handles storage and retrieval of conversation history."""

    def store_turn(self, turn: Turn) -> None:
        """
        Store turn in Neo4j graph.

        Args:
            turn: Turn object to store
        """
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (c:Conversation {id: $conv_id})
                ON CREATE SET c.started_at = datetime()

                CREATE (t:Turn {
                    id: $turn_id,
                    index: $index,
                    timestamp: datetime($timestamp),
                    role: $role,
                    content: $content,
                    embedding: $embedding,
                    token_count: $token_count
                })

                MERGE (c)-[:HAS_TURN]->(t)

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
                token_count=turn.token_count
            )

    def store_turn_log(self, turn: Turn) -> None:
        """
        Store turn in PostgreSQL for audit/backup.

        Args:
            turn: Turn object to store
        """
        with get_postgres_session() as session:
            session.execute(text("""
                INSERT INTO conversation_logs
                (conversation_id, turn_index, timestamp, role, content, token_count, embedding)
                VALUES (:conv_id, :index, :timestamp, :role, :content, :tokens, :embedding)
                ON CONFLICT (conversation_id, turn_index) DO UPDATE
                SET content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding
            """), {
                "conv_id": turn.conversation_id,
                "index": turn.index,
                "timestamp": turn.timestamp,
                "role": turn.role,
                "content": turn.content,
                "tokens": turn.token_count,
                "embedding": str(turn.embedding) if turn.embedding else None
            })

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memory by vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            user_id: Filter by user ID
            time_window_hours: Filter by time window

        Returns:
            List of matching turns
        """
        with Neo4jConnection.session() as session:
            # Build time filter
            time_filter = ""
            if time_window_hours:
                time_filter = f"AND t.timestamp > datetime() - duration('PT{time_window_hours}H')"

            user_filter = ""
            if user_id:
                user_filter = "AND c.user_id = $user_id"

            result = session.run(f"""
                CALL db.index.vector.queryNodes('turn_embeddings', $k, $embedding)
                YIELD node AS t, score
                MATCH (c:Conversation)-[:HAS_TURN]->(t)
                WHERE true {time_filter} {user_filter}
                RETURN t.id AS id,
                       t.content AS content,
                       t.role AS role,
                       t.timestamp AS timestamp,
                       c.id AS conversation_id,
                       score
                ORDER BY score DESC
            """,
                k=top_k * 2,  # Over-fetch for filtering
                embedding=query_embedding,
                user_id=user_id
            )

            return [dict(record) for record in result][:top_k]

    def get_conversation(self, conversation_id: str) -> List[Turn]:
        """
        Get all turns in a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of Turn objects
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:HAS_TURN]->(t:Turn)
                RETURN t
                ORDER BY t.index
            """, conv_id=conversation_id)

            return [Turn(**dict(record["t"])) for record in result]

    def get_recent_turns(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent turns across all conversations.

        Args:
            user_id: User ID
            hours: Time window in hours
            limit: Maximum number of results

        Returns:
            List of turn dictionaries
        """
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
                WHERE c.user_id = $user_id
                  AND t.timestamp > datetime() - duration('PT' + $hours + 'H')
                RETURN t.content AS content,
                       t.role AS role,
                       t.timestamp AS timestamp,
                       c.id AS conversation_id
                ORDER BY t.timestamp DESC
                LIMIT $limit
            """,
                user_id=user_id,
                hours=str(hours),
                limit=limit
            )

            return [dict(record) for record in result]

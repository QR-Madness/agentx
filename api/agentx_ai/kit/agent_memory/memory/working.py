"""Working memory - fast, ephemeral memory for current session state using Redis."""

from typing import Any, Dict, List, Optional
import json
from datetime import datetime

from ..connections import RedisConnection
from ..config import get_settings

settings = get_settings()


class WorkingMemory:
    """
    Fast, ephemeral memory for current session state.
    Uses Redis for sub-millisecond access.
    """

    def __init__(self, user_id: str, conversation_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.redis = RedisConnection.get_client()

        # Key prefixes
        self.session_key = f"working:{user_id}:{conversation_id or 'global'}"
        self.turns_key = f"{self.session_key}:turns"
        self.context_key = f"{self.session_key}:context"

    def add_turn(self, turn) -> None:
        """
        Add a turn to working memory (keeps last N turns).

        Args:
            turn: Turn object to add
        """
        turn_data = {
            "id": turn.id,
            "index": turn.index,
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat()
        }

        # Push to list, trim to max size
        self.redis.lpush(self.turns_key, json.dumps(turn_data))
        self.redis.ltrim(self.turns_key, 0, settings.max_working_memory_items - 1)

        # Set TTL (1 hour default)
        self.redis.expire(self.turns_key, 3600)

    def get_recent_turns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent turns from working memory.

        Args:
            limit: Maximum number of turns to return

        Returns:
            List of turn dictionaries
        """
        turns = self.redis.lrange(self.turns_key, 0, limit - 1)
        return [json.loads(t) for t in turns]

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """
        Set a value in working memory.

        Args:
            key: Key name
            value: Value to store
            ttl_seconds: Time to live in seconds
        """
        full_key = f"{self.context_key}:{key}"
        self.redis.setex(full_key, ttl_seconds, json.dumps(value))

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from working memory.

        Args:
            key: Key name

        Returns:
            Stored value or None
        """
        full_key = f"{self.context_key}:{key}"
        value = self.redis.get(full_key)
        return json.loads(value) if value else None

    def delete(self, key: str) -> None:
        """
        Delete a value from working memory.

        Args:
            key: Key name
        """
        full_key = f"{self.context_key}:{key}"
        self.redis.delete(full_key)

    def get_context(self) -> Dict[str, Any]:
        """
        Get all working memory context.

        Returns:
            Dictionary of all context values
        """
        # Get all keys matching the context pattern
        pattern = f"{self.context_key}:*"
        keys = self.redis.keys(pattern)

        context = {}
        for key in keys:
            short_key = key.replace(f"{self.context_key}:", "")
            value = self.redis.get(key)
            if value:
                context[short_key] = json.loads(value)

        # Include recent turns
        context["recent_turns"] = self.get_recent_turns()

        return context

    def clear_session(self) -> None:
        """Clear all working memory for this session."""
        pattern = f"{self.session_key}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

    # Specialized working memory operations

    def set_active_goal(self, goal_id: str, goal_description: str) -> None:
        """
        Set the currently active goal.

        Args:
            goal_id: Goal ID
            goal_description: Goal description
        """
        self.set("active_goal", {
            "id": goal_id,
            "description": goal_description,
            "set_at": datetime.utcnow().isoformat()
        })

    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active goal.

        Returns:
            Active goal dictionary or None
        """
        return self.get("active_goal")

    def push_thought(self, thought: str) -> None:
        """
        Push a reasoning step (for chain-of-thought tracking).

        Args:
            thought: Reasoning step text
        """
        thoughts_key = f"{self.session_key}:thoughts"
        self.redis.lpush(thoughts_key, json.dumps({
            "thought": thought,
            "timestamp": datetime.utcnow().isoformat()
        }))
        self.redis.ltrim(thoughts_key, 0, 99)  # Keep last 100 thoughts
        self.redis.expire(thoughts_key, 3600)

    def get_thoughts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent reasoning steps.

        Args:
            limit: Maximum number of thoughts to return

        Returns:
            List of thought dictionaries
        """
        thoughts_key = f"{self.session_key}:thoughts"
        thoughts = self.redis.lrange(thoughts_key, 0, limit - 1)
        return [json.loads(t) for t in thoughts]

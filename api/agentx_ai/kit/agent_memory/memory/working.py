"""Working memory - fast, ephemeral memory for current session state using Redis."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
from datetime import datetime, timezone

from ..connections import RedisConnection
from ..config import get_settings

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger

settings = get_settings()


class WorkingMemory:
    """
    Fast, ephemeral memory for current session state.
    Uses Redis for sub-millisecond access.
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        channel: str = "_global",
        audit_logger: Optional["MemoryAuditLogger"] = None
    ):
        """Initialize working memory.

        Args:
            user_id: User ID
            conversation_id: Conversation ID
            channel: Memory channel (default: _global)
            audit_logger: Optional audit logger for operation tracking.
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.channel = channel
        self.redis = RedisConnection.get_client()
        self._audit_logger = audit_logger

        # Key prefixes - include channel for isolation
        self.session_key = f"working:{user_id}:{channel}:{conversation_id or 'global'}"
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
            limit: Maximum number of turns to return (capped to max_working_memory_items)

        Returns:
            List of turn dictionaries
        """
        # Cap limit to configured maximum to prevent resource exhaustion
        effective_limit = min(max(1, limit), settings.max_working_memory_items)
        turns = self.redis.lrange(self.turns_key, 0, effective_limit - 1)

        # Refresh TTL on access (sliding window expiry)
        if turns:
            self.redis.expire(self.turns_key, 3600)

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
        # Use SCAN instead of KEYS to avoid blocking Redis (O(N) blocking)
        pattern = f"{self.context_key}:*"
        keys = []
        cursor = 0
        while True:
            cursor, batch = self.redis.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        context = {}
        for key in keys:
            # Handle both bytes and string keys depending on Redis config
            key_str = key.decode() if isinstance(key, bytes) else key
            short_key = key_str.replace(f"{self.context_key}:", "")
            value = self.redis.get(key_str)
            if value:
                context[short_key] = json.loads(value)

        # Include recent turns
        context["recent_turns"] = self.get_recent_turns()

        return context

    def clear_session(self) -> None:
        """Clear all working memory for this session."""
        # Use SCAN instead of KEYS to avoid blocking Redis
        pattern = f"{self.session_key}:*"
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                # Handle both bytes and string keys
                key_strs = [k.decode() if isinstance(k, bytes) else k for k in keys]
                self.redis.delete(*key_strs)
            if cursor == 0:
                break

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
            "set_at": datetime.now(timezone.utc).isoformat()
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }))
        # Use configured limit instead of hard-coded value
        self.redis.ltrim(thoughts_key, 0, settings.max_working_memory_items - 1)
        self.redis.expire(thoughts_key, 3600)

    def get_thoughts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent reasoning steps.

        Args:
            limit: Maximum number of thoughts to return (capped to max_working_memory_items)

        Returns:
            List of thought dictionaries
        """
        # Cap limit to configured maximum
        effective_limit = min(max(1, limit), settings.max_working_memory_items)
        thoughts_key = f"{self.session_key}:thoughts"
        thoughts = self.redis.lrange(thoughts_key, 0, effective_limit - 1)
        return [json.loads(t) for t in thoughts]

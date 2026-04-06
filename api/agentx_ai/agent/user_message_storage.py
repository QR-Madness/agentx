"""
User Message Storage for large user inputs.

Stores oversized user messages in Redis to avoid context overflow,
providing a reference key for later retrieval via internal tools.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Key prefix for user messages in Redis
USER_MESSAGE_PREFIX = "user_message:"


def _get_redis_client():
    """Get Redis client from memory connections (lazy import)."""
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


def store_user_message(
    message_id: str,
    content: str,
    session_id: Optional[str] = None,
    ttl_seconds: int = 3600,
) -> Optional[str]:
    """
    Store a large user message in Redis.

    Args:
        message_id: Unique identifier for this message
        content: The full message content
        session_id: Optional session ID for context
        ttl_seconds: Time-to-live in seconds (default 1 hour)

    Returns:
        Storage key for retrieval, or None on failure
    """
    # Generate a short, readable key
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    timestamp = datetime.utcnow().strftime("%H%M%S")
    storage_key = f"msg_{timestamp}_{content_hash}"

    redis_key = f"{USER_MESSAGE_PREFIX}{storage_key}"

    # Store metadata + content as JSON
    data = {
        "message_id": message_id,
        "content": content,
        "size_chars": len(content),
        "session_id": session_id,
        "stored_at": datetime.utcnow().isoformat(),
    }

    try:
        logger.debug(f"Attempting to store user message: {storage_key} ({len(content):,} chars)")
        client = _get_redis_client()
        client.setex(redis_key, ttl_seconds, json.dumps(data))
        logger.info(f"Stored user message: {storage_key} ({len(content):,} chars, TTL={ttl_seconds}s)")
        return storage_key
    except Exception as e:
        logger.warning(f"Failed to store user message in Redis: {e}")
        return None


def get_user_message(storage_key: str) -> Optional[dict]:
    """
    Retrieve a stored user message from Redis.

    Args:
        storage_key: The storage key returned by store_user_message

    Returns:
        Dict with message_id, content, size_chars, session_id, stored_at
        or None if not found/expired
    """
    redis_key = f"{USER_MESSAGE_PREFIX}{storage_key}"

    try:
        client = _get_redis_client()
        data = client.get(redis_key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve user message from Redis: {e}")
        return None


def get_user_message_content(
    storage_key: str,
    offset: int = 0,
    limit: Optional[int] = None,
) -> Optional[str]:
    """
    Retrieve just the content of a stored user message, with optional pagination.

    Args:
        storage_key: The storage key
        offset: Start position in content (default 0)
        limit: Max characters to return (default: all)

    Returns:
        Content string or None if not found
    """
    data = get_user_message(storage_key)
    if not data:
        return None

    content = data.get("content", "")
    if limit:
        return content[offset:offset + limit]
    return content[offset:]


def list_user_messages(pattern: str = "*") -> list[dict]:
    """
    List stored user messages matching a pattern.

    Args:
        pattern: Redis key pattern (e.g., "msg_*")

    Returns:
        List of metadata dicts (without full content)
    """
    try:
        client = _get_redis_client()
        keys = client.keys(f"{USER_MESSAGE_PREFIX}{pattern}")

        results = []
        for key in keys:
            # Decode key if bytes
            key_str = key.decode() if isinstance(key, bytes) else key
            storage_key = key_str.replace(USER_MESSAGE_PREFIX, "")

            data = client.get(key)
            if data:
                parsed = json.loads(data)
                # Return metadata without full content
                results.append({
                    "key": storage_key,
                    "message_id": parsed.get("message_id"),
                    "session_id": parsed.get("session_id"),
                    "size_chars": parsed.get("size_chars"),
                    "stored_at": parsed.get("stored_at"),
                })

        return sorted(results, key=lambda x: x.get("stored_at", ""), reverse=True)
    except Exception as e:
        logger.error(f"Failed to list user messages: {e}")
        return []


def delete_user_message(storage_key: str) -> bool:
    """
    Delete a stored user message.

    Args:
        storage_key: The storage key

    Returns:
        True if deleted, False otherwise
    """
    redis_key = f"{USER_MESSAGE_PREFIX}{storage_key}"

    try:
        client = _get_redis_client()
        return client.delete(redis_key) > 0
    except Exception as e:
        logger.error(f"Failed to delete user message: {e}")
        return False

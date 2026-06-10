"""
User Message Storage for large user inputs.

Stores oversized user messages in Redis to avoid context overflow,
providing a reference key for later retrieval via internal tools.

The Redis mechanics live in :mod:`redis_blob_storage`; this module defines the
user-message-specific key scheme, payload, and list projection.
"""

from datetime import datetime

from .redis_blob_storage import RedisBlobStorage, make_storage_key

# Key prefix for user messages in Redis
USER_MESSAGE_PREFIX = "user_message:"

_storage = RedisBlobStorage(USER_MESSAGE_PREFIX, label="user message")


def store_user_message(
    message_id: str,
    content: str,
    session_id: str | None = None,
    ttl_seconds: int = 3600,
) -> str | None:
    """
    Store a large user message in Redis.

    Args:
        message_id: Unique identifier for this message
        content: The full message content
        session_id: Optional session ID for context
        ttl_seconds: Time-to-live in seconds (default 1 hour)

    Returns:
        Storage key for retrieval, or None on failure.
    """
    storage_key = make_storage_key("msg", content)
    data = {
        "message_id": message_id,
        "content": content,
        "size_chars": len(content),
        "session_id": session_id,
        "stored_at": datetime.utcnow().isoformat(),
    }
    return _storage.store(storage_key, data, ttl_seconds)


def get_user_message(storage_key: str) -> dict | None:
    """
    Retrieve a stored user message from Redis.

    Returns:
        Dict with message_id, content, size_chars, session_id, stored_at
        or None if not found/expired
    """
    return _storage.get(storage_key)


def get_user_message_content(
    storage_key: str,
    offset: int = 0,
    limit: int | None = None,
) -> str | None:
    """Retrieve just the content of a stored user message, with pagination."""
    return _storage.get_content(storage_key, offset, limit)


def list_user_messages(pattern: str = "*") -> list[dict]:
    """
    List stored user messages matching a pattern (metadata only, newest first).

    Args:
        pattern: Redis key pattern (e.g., "msg_*")
    """
    def project(storage_key: str, parsed: dict) -> dict:
        return {
            "key": storage_key,
            "message_id": parsed.get("message_id"),
            "session_id": parsed.get("session_id"),
            "size_chars": parsed.get("size_chars"),
            "stored_at": parsed.get("stored_at"),
        }

    return _storage.list_items(project, pattern)


def delete_user_message(storage_key: str) -> bool:
    """Delete a stored user message. Returns True if deleted."""
    return _storage.delete(storage_key)

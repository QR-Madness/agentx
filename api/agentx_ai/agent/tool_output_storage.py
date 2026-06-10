"""
Tool Output Storage for large tool results.

Stores oversized tool outputs in Redis to avoid context overflow,
providing a reference key for later retrieval.

The Redis mechanics live in :mod:`redis_blob_storage`; this module defines the
tool-output-specific key scheme, payload, and list projection.
"""

from datetime import datetime

from .redis_blob_storage import RedisBlobStorage, make_storage_key

# Key prefix for tool outputs in Redis
TOOL_OUTPUT_PREFIX = "tool_output:"

_storage = RedisBlobStorage(TOOL_OUTPUT_PREFIX, label="tool output")


def store_tool_output(
    tool_call_id: str,
    tool_name: str,
    content: str,
    ttl_seconds: int = 3600,
) -> str | None:
    """
    Store a large tool output in Redis.

    Args:
        tool_call_id: The tool call ID from the LLM
        tool_name: Name of the tool that produced the output
        content: The full output content
        ttl_seconds: Time-to-live in seconds (default 1 hour)

    Returns:
        Storage key for retrieval, or None on failure (caller should fall back
        to truncation).
    """
    storage_key = make_storage_key(tool_name, content)
    data = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "content": content,
        "size_chars": len(content),
        "stored_at": datetime.utcnow().isoformat(),
    }
    return _storage.store(storage_key, data, ttl_seconds)


def get_tool_output(storage_key: str) -> dict | None:
    """
    Retrieve a stored tool output from Redis.

    Returns:
        Dict with tool_call_id, tool_name, content, size_chars, stored_at
        or None if not found/expired
    """
    return _storage.get(storage_key)


def get_tool_output_content(
    storage_key: str,
    offset: int = 0,
    limit: int | None = None,
) -> str | None:
    """Retrieve just the content of a stored tool output, with pagination."""
    return _storage.get_content(storage_key, offset, limit)


def list_tool_outputs(pattern: str = "*") -> list[dict]:
    """
    List stored tool outputs matching a pattern (metadata only, newest first).

    Args:
        pattern: Redis key pattern (e.g., "read_file_*")
    """
    def project(storage_key: str, parsed: dict) -> dict:
        return {
            "key": storage_key,
            "tool_name": parsed.get("tool_name"),
            "tool_call_id": parsed.get("tool_call_id"),
            "size_chars": parsed.get("size_chars"),
            "stored_at": parsed.get("stored_at"),
        }

    return _storage.list_items(project, pattern)


def delete_tool_output(storage_key: str) -> bool:
    """Delete a stored tool output. Returns True if deleted."""
    return _storage.delete(storage_key)

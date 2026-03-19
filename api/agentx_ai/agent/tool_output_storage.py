"""
Tool Output Storage for large tool results.

Stores oversized tool outputs in Redis to avoid context overflow,
providing a reference key for later retrieval.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Key prefix for tool outputs in Redis
TOOL_OUTPUT_PREFIX = "tool_output:"


def _get_redis_client():
    """Get Redis client from memory connections (lazy import)."""
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


def store_tool_output(
    tool_call_id: str,
    tool_name: str,
    content: str,
    ttl_seconds: int = 3600,
) -> Optional[str]:
    """
    Store a large tool output in Redis.

    Args:
        tool_call_id: The tool call ID from the LLM
        tool_name: Name of the tool that produced the output
        content: The full output content
        ttl_seconds: Time-to-live in seconds (default 1 hour)

    Returns:
        Storage key for retrieval
    """
    # Generate a short, readable key
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    timestamp = datetime.utcnow().strftime("%H%M%S")
    storage_key = f"{tool_name}_{timestamp}_{content_hash}"

    redis_key = f"{TOOL_OUTPUT_PREFIX}{storage_key}"

    # Store metadata + content as JSON
    data = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "content": content,
        "size_chars": len(content),
        "stored_at": datetime.utcnow().isoformat(),
    }

    try:
        logger.debug(f"Attempting to store tool output: {storage_key} ({len(content):,} chars)")
        client = _get_redis_client()
        client.setex(redis_key, ttl_seconds, json.dumps(data))
        logger.info(f"Stored tool output: {storage_key} ({len(content):,} chars, TTL={ttl_seconds}s)")
        return storage_key
    except Exception as e:
        logger.warning(f"Failed to store tool output in Redis (falling back to truncation): {e}")
        # Return None to signal storage failure - caller should fall back to truncation
        return None


def get_tool_output(storage_key: str) -> Optional[dict]:
    """
    Retrieve a stored tool output from Redis.

    Args:
        storage_key: The storage key returned by store_tool_output

    Returns:
        Dict with tool_call_id, tool_name, content, size_chars, stored_at
        or None if not found/expired
    """
    redis_key = f"{TOOL_OUTPUT_PREFIX}{storage_key}"

    try:
        client = _get_redis_client()
        data = client.get(redis_key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve tool output from Redis: {e}")
        return None


def get_tool_output_content(
    storage_key: str,
    offset: int = 0,
    limit: Optional[int] = None,
) -> Optional[str]:
    """
    Retrieve just the content of a stored tool output, with optional pagination.

    Args:
        storage_key: The storage key
        offset: Start position in content (default 0)
        limit: Max characters to return (default: all)

    Returns:
        Content string or None if not found
    """
    data = get_tool_output(storage_key)
    if not data:
        return None

    content = data.get("content", "")
    if limit:
        return content[offset:offset + limit]
    return content[offset:]


def list_tool_outputs(pattern: str = "*") -> list[dict]:
    """
    List stored tool outputs matching a pattern.

    Args:
        pattern: Redis key pattern (e.g., "read_file_*")

    Returns:
        List of metadata dicts (without full content)
    """
    try:
        client = _get_redis_client()
        keys = client.keys(f"{TOOL_OUTPUT_PREFIX}{pattern}")

        results = []
        for key in keys:
            # Decode key if bytes
            key_str = key.decode() if isinstance(key, bytes) else key
            storage_key = key_str.replace(TOOL_OUTPUT_PREFIX, "")

            data = client.get(key)
            if data:
                parsed = json.loads(data)
                # Return metadata without full content
                results.append({
                    "key": storage_key,
                    "tool_name": parsed.get("tool_name"),
                    "tool_call_id": parsed.get("tool_call_id"),
                    "size_chars": parsed.get("size_chars"),
                    "stored_at": parsed.get("stored_at"),
                })

        return sorted(results, key=lambda x: x.get("stored_at", ""), reverse=True)
    except Exception as e:
        logger.error(f"Failed to list tool outputs: {e}")
        return []


def delete_tool_output(storage_key: str) -> bool:
    """
    Delete a stored tool output.

    Args:
        storage_key: The storage key

    Returns:
        True if deleted, False otherwise
    """
    redis_key = f"{TOOL_OUTPUT_PREFIX}{storage_key}"

    try:
        client = _get_redis_client()
        return client.delete(redis_key) > 0
    except Exception as e:
        logger.error(f"Failed to delete tool output: {e}")
        return False

"""
Shared Redis blob storage.

A small base for the "store a large blob under a generated key, with metadata,
and retrieve/paginate/list/delete it" pattern. Used by tool output storage and
user message storage, which previously carried near-identical copies of this
logic. Each consumer supplies its own key prefix, key-generation scheme, stored
payload, and list projection; the Redis mechanics live here.

Note: checkpoint storage uses a different access pattern (Redis lists keyed by
conversation) and intentionally does not build on this base.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional, cast

logger = logging.getLogger(__name__)


def make_storage_key(prefix_token: str, content: str) -> str:
    """Build a short, readable key: ``{prefix_token}_{HHMMSS}_{hash8}``.

    ``prefix_token`` is a human-readable label (e.g. a tool name or ``"msg"``).
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    timestamp = datetime.utcnow().strftime("%H%M%S")
    return f"{prefix_token}_{timestamp}_{content_hash}"


class RedisBlobStorage:
    """Key/value blob store backed by Redis with JSON metadata payloads."""

    def __init__(self, key_prefix: str, label: str):
        self.key_prefix = key_prefix
        self.label = label  # human-readable noun for log messages

    @staticmethod
    def _client():
        """Get Redis client from memory connections (lazy import)."""
        from ..kit.agent_memory.connections import RedisConnection
        return RedisConnection.get_client()

    def _redis_key(self, storage_key: str) -> str:
        return f"{self.key_prefix}{storage_key}"

    def store(self, storage_key: str, data: dict, ttl_seconds: int) -> Optional[str]:
        """Store ``data`` (JSON-serialized) under ``storage_key``.

        Returns the storage key on success, or None on failure so the caller
        can fall back (e.g. to truncation).
        """
        size = data.get("size_chars", "?")
        try:
            logger.debug(f"Attempting to store {self.label}: {storage_key} ({size} chars)")
            client = self._client()
            client.setex(self._redis_key(storage_key), ttl_seconds, json.dumps(data))
            logger.info(f"Stored {self.label}: {storage_key} ({size} chars, TTL={ttl_seconds}s)")
            return storage_key
        except Exception as e:
            logger.warning(f"Failed to store {self.label} in Redis: {e}")
            return None

    def get(self, storage_key: str) -> Optional[dict]:
        """Retrieve the full payload dict, or None if missing/expired."""
        try:
            client = self._client()
            data = cast(Optional[bytes], client.get(self._redis_key(storage_key)))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve {self.label} from Redis: {e}")
            return None

    def get_content(
        self,
        storage_key: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Optional[str]:
        """Retrieve just the ``content`` field, with optional pagination."""
        data = self.get(storage_key)
        if not data:
            return None
        content = data.get("content", "")
        if limit:
            return content[offset:offset + limit]
        return content[offset:]

    def list_items(
        self,
        projection: Callable[[str, dict], dict],
        pattern: str = "*",
    ) -> list[dict]:
        """List stored items matching ``pattern``, newest first.

        ``projection(storage_key, parsed_payload)`` builds the per-item dict
        (so each consumer controls which metadata fields surface).
        """
        try:
            client = self._client()
            keys = cast(list[Any], client.keys(f"{self.key_prefix}{pattern}"))

            results = []
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                storage_key = key_str.replace(self.key_prefix, "")
                data = cast(Optional[bytes], client.get(key))
                if data:
                    results.append(projection(storage_key, json.loads(data)))

            return sorted(results, key=lambda x: x.get("stored_at", ""), reverse=True)
        except Exception as e:
            logger.error(f"Failed to list {self.label}s: {e}")
            return []

    def delete(self, storage_key: str) -> bool:
        """Delete a stored item. Returns True if a key was removed."""
        try:
            client = self._client()
            return cast(int, client.delete(self._redis_key(storage_key))) > 0
        except Exception as e:
            logger.error(f"Failed to delete {self.label}: {e}")
            return False

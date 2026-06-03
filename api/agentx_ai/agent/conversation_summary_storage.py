"""
Per-conversation rolling summary, persisted in Redis.

``Session.summary`` is in-memory only, so it's lost whenever the session is
rebuilt cold (process restart, eviction, restore-from-history). Persisting it
here — keyed by conversation, like checkpoints/scratchpad — lets the summary of
aged-out turns survive rehydration so a long conversation stays coherent.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "conv_summary:"
SUMMARY_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days


def _redis():
    from ..kit.agent_memory.connections import RedisConnection

    return RedisConnection.get_client()


def _key(conversation_id: str) -> str:
    return f"{SUMMARY_PREFIX}{conversation_id}"


def get_summary(conversation_id: str) -> Optional[str]:
    """Return the persisted rolling summary, or None."""
    try:
        val = _redis().get(_key(conversation_id))
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"summary read failed: {e}")
        return None
    if val is None:
        return None
    return val.decode() if isinstance(val, (bytes, bytearray)) else str(val)


def set_summary(conversation_id: str, summary: str) -> None:
    """Persist (replace) the rolling summary for a conversation."""
    if not summary or not summary.strip():
        return
    try:
        client = _redis()
        client.set(_key(conversation_id), summary)
        client.expire(_key(conversation_id), SUMMARY_TTL_SECONDS)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.warning(f"summary write failed: {e}")


def clear_summary(conversation_id: str) -> None:
    """Delete the persisted summary for a conversation."""
    try:
        _redis().delete(_key(conversation_id))
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"summary clear failed: {e}")

"""
Conversation checkpoint storage.

Checkpoints are short, model-authored anchors describing where a long task
stands. They are kept in Redis keyed by conversation, and re-injected as a
SYSTEM message at the top of the chat context every turn. That placement is
what lets them survive trajectory compression (which only drops tool-call
rounds, never SYSTEM messages built fresh per turn).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "checkpoint:"
CHECKPOINT_TTL_SECONDS = 60 * 60 * 24 * 7  # one week
MAX_CHECKPOINTS_PER_CONVERSATION = 8


def _redis():
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


def _key(conversation_id: str) -> str:
    return f"{CHECKPOINT_PREFIX}{conversation_id}"


def add_checkpoint(
    conversation_id: str,
    summary: str,
    decisions: Optional[list[str]] = None,
    next_step: str = "",
) -> dict[str, Any]:
    """Append a checkpoint to the conversation. Returns the stored entry."""
    entry = {
        "summary": summary.strip(),
        "decisions": [d.strip() for d in (decisions or []) if d and d.strip()],
        "next_step": next_step.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client = _redis()
        client.rpush(_key(conversation_id), json.dumps(entry))
        client.ltrim(_key(conversation_id), -MAX_CHECKPOINTS_PER_CONVERSATION, -1)
        client.expire(_key(conversation_id), CHECKPOINT_TTL_SECONDS)
    except Exception as e:  # pragma: no cover — Redis offline
        logger.warning(f"Failed to store checkpoint: {e}")

    return entry


def list_checkpoints(conversation_id: str) -> list[dict[str, Any]]:
    """Return checkpoints for a conversation, oldest first. Empty on errors."""
    try:
        raw = _redis().lrange(_key(conversation_id), 0, -1) or []
    except Exception as e:  # pragma: no cover
        logger.debug(f"checkpoint read failed: {e}")
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except (TypeError, ValueError):
            continue
    return out


def render_checkpoints_block(conversation_id: str) -> str:
    """Render checkpoints as a system-prompt block, or "" if none."""
    items = list_checkpoints(conversation_id)
    if not items:
        return ""

    lines: list[str] = ["## Checkpoints (model-authored, survive compression)"]
    for i, cp in enumerate(items, 1):
        lines.append(f"### {i}. {cp.get('summary', '')}")
        decisions = cp.get("decisions") or []
        if decisions:
            lines.append("Decisions:")
            for d in decisions:
                lines.append(f"  - {d}")
        if cp.get("next_step"):
            lines.append(f"Next: {cp['next_step']}")
    return "\n".join(lines)

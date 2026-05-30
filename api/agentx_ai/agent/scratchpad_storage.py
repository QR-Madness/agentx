"""
Conversation scratchpad storage.

Scratchpad notes are short, free-form, model-authored working notes for the
current conversation — lighter-weight than :mod:`checkpoint_storage` checkpoints
(which capture structured task anchors). Like checkpoints, they are kept in
Redis keyed by conversation and re-injected as a SYSTEM message at the top of
the chat context every turn, so they survive trajectory compression (which only
drops tool-call rounds, never SYSTEM messages built fresh per turn).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, cast

logger = logging.getLogger(__name__)

SCRATCHPAD_PREFIX = "scratchpad:"
SCRATCHPAD_TTL_SECONDS = 60 * 60 * 24 * 7  # one week
MAX_NOTES_PER_CONVERSATION = 20


def _redis():
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


def _key(conversation_id: str) -> str:
    return f"{SCRATCHPAD_PREFIX}{conversation_id}"


def add_note(conversation_id: str, note: str) -> dict[str, Any]:
    """Append a scratchpad note to the conversation. Returns the stored entry."""
    entry = {
        "note": note.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client = _redis()
        client.rpush(_key(conversation_id), json.dumps(entry))
        client.ltrim(_key(conversation_id), -MAX_NOTES_PER_CONVERSATION, -1)
        client.expire(_key(conversation_id), SCRATCHPAD_TTL_SECONDS)
    except Exception as e:  # pragma: no cover — Redis offline
        logger.warning(f"Failed to store scratchpad note: {e}")

    return entry


def list_notes(conversation_id: str) -> list[dict[str, Any]]:
    """Return scratchpad notes for a conversation, oldest first. Empty on errors."""
    try:
        raw = cast(list, _redis().lrange(_key(conversation_id), 0, -1) or [])
    except Exception as e:  # pragma: no cover
        logger.debug(f"scratchpad read failed: {e}")
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except (TypeError, ValueError):
            continue
    return out


def clear_notes(conversation_id: str) -> int:
    """Delete all scratchpad notes for a conversation. Returns the prior count."""
    try:
        client = _redis()
        count = int(cast(int, client.llen(_key(conversation_id))) or 0)
        client.delete(_key(conversation_id))
        return count
    except Exception as e:  # pragma: no cover — Redis offline
        logger.warning(f"Failed to clear scratchpad: {e}")
        return 0


def render_scratchpad_block(conversation_id: str) -> str:
    """Render scratchpad notes as a system-prompt block, or "" if none."""
    items = list_notes(conversation_id)
    if not items:
        return ""

    lines: list[str] = ["## Scratchpad (model-authored notes, survive compression)"]
    for i, entry in enumerate(items, 1):
        lines.append(f"{i}. {entry.get('note', '')}")
    return "\n".join(lines)

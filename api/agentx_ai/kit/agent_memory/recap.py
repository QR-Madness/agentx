"""
Cached cross-conversation user recap.

A short natural-language summary of who a user is and what they're working on,
built from their recent turns + known facts and cached in Redis. It is refreshed
during consolidation (not per-turn) and surfaced through
``recall_user_history`` — filling that tool's previously-empty ``summary`` field.

Distinct from the per-session rolling summary (one conversation) and from the
structured ``user_context`` (preferences/expertise nodes): this is a durable,
free-text profile spanning conversations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, UTC
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .memory.interface import AgentMemory

RECAP_PREFIX = "user_recap:"
RECAP_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

_RECAP_INSTRUCTIONS = (
    "You are maintaining a durable profile of a user for an AI assistant. "
    "From the user's known facts and recent messages below, write a concise "
    "recap (3-5 sentences): who they are, what they're working on, key "
    "preferences, and any ongoing goals. State only what the material supports; "
    "do not invent details. Write in the third person about the user."
)


def _redis():
    from .connections import RedisConnection
    return RedisConnection.get_client()


def _key(user_id: str, channel: str) -> str:
    return f"{RECAP_PREFIX}{user_id}:{channel}"


def get_cached_recap(user_id: str, channel: str = "_default") -> dict[str, Any] | None:
    """Return the cached recap entry ``{summary, updated_at}`` or None."""
    try:
        raw = _redis().get(_key(user_id, channel))
    except Exception as e:  # pragma: no cover — Redis offline
        logger.debug(f"recap read failed: {e}")
        return None
    if not raw or not isinstance(raw, (str, bytes)):
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def set_cached_recap(user_id: str, channel: str, summary: str) -> dict[str, Any]:
    """Cache a recap summary with TTL. Returns the stored entry."""
    entry = {
        "summary": summary.strip(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    try:
        _redis().setex(_key(user_id, channel), RECAP_TTL_SECONDS, json.dumps(entry))
    except Exception as e:  # pragma: no cover — Redis offline
        logger.warning(f"Failed to cache user recap: {e}")
    return entry


async def build_and_cache_user_recap(
    memory: AgentMemory,
    *,
    max_turns: int = 30,
    max_facts: int = 20,
) -> str | None:
    """
    Build a fresh recap for ``memory``'s user/channel from recent turns + facts,
    cache it, and return the summary. Returns None when there's no material or
    no model provider is configured (degrades gracefully).
    """
    user_id = memory.user_id
    channel = memory.channel

    try:
        turns = memory.episodic.get_recent_turns(
            user_id, hours=24 * 30, limit=max_turns * 2, channel=channel
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"recap: recent turns unavailable: {e}")
        turns = []
    user_msgs = [t for t in turns if t.get("role") == "user"][:max_turns]

    try:
        facts, _ = memory.semantic.list_facts(
            user_id=user_id, channel=channel, offset=0, limit=max_facts, min_confidence=0.3
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"recap: facts unavailable: {e}")
        facts = []

    if not user_msgs and not facts:
        return None

    fact_lines = "\n".join(f"- {f.get('claim', '')}" for f in facts if f.get("claim"))
    msg_lines = "\n".join(
        f"- {(t.get('content') or '')[:300]}" for t in user_msgs if t.get("content")
    )
    material = "\n\n".join(filter(None, [
        f"Known facts:\n{fact_lines}" if fact_lines else "",
        f"Recent user messages:\n{msg_lines}" if msg_lines else "",
    ]))

    from ...providers.registry import get_registry
    from ...providers.base import Message, MessageRole
    from ...config import get_config_manager
    from ...model_roles import resolve_member_model

    explicit_model = get_config_manager().get("session.rolling_summary.model", "")
    # An empty explicit value follows the `summarizer` model role; the concrete
    # model is a last-resort floor only when neither an explicit value nor the
    # role is configured.
    model = (
        resolve_member_model("rolling_summary", explicit_model)
        or "anthropic:claude-haiku-4-5-20251001"
    )
    try:
        provider, model_id, _ = get_registry().resolve_with_fallback(model)
    except Exception as e:  # noqa: BLE001 — no provider configured
        logger.debug(f"recap: no provider for model {model}: {e}")
        return None

    messages = [
        Message(role=MessageRole.SYSTEM, content=_RECAP_INSTRUCTIONS),
        Message(role=MessageRole.USER, content=material),
    ]
    try:
        result = await provider.complete(messages, model_id, temperature=0.3, max_tokens=400)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"recap: summarization failed: {e}")
        return None

    summary = (result.content or "").strip()
    if not summary:
        return None
    set_cached_recap(user_id, channel, summary)
    return summary

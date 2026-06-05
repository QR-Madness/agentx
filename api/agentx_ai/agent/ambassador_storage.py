"""
Per-turn Ambassador briefings, persisted in Redis (a conversation *sidecar*).

The Ambassador (Phase 16.6) runs parallel to a conversation and briefs the user
on individual turns. Its output is deliberately kept **out** of the main agent's
world: it lives under the dedicated ``ambassador:`` key prefix, which is read by
*nothing* in the main-agent path (``hydrate_session_from_history``,
``load_recent_turns``, ``ContextManager.assemble_turn_context`` read only the
Postgres ``conversation_logs`` turn stream and the ``conv_summary:`` Redis key).
This prefix isolation is the load-bearing no-pollution guarantee — never write a
briefing into ``conversation_logs`` or ``conv_summary:``.

Briefings are keyed by the **client message id** (stable, persisted in the
client's localStorage) rather than a server turn_index, so a just-finished live
turn can be briefed without depending on DB persistence timing or turn ordering.

Keys:
    Briefing:  ambassador:{conversation_id}:msg:{message_id}   (Redis string, JSON, TTL)
    Index:     ambassador:{conversation_id}:index              (Redis set of message_ids, TTL)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

AMBASSADOR_PREFIX = "ambassador:"
AMBASSADOR_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days, mirrors conv_summary

# status ∈ streaming | done | error | empty_provider | cancelled


def _redis():
    from ..kit.agent_memory.connections import RedisConnection

    return RedisConnection.get_client()


def _briefing_key(conversation_id: str, message_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:msg:{message_id}"


def _index_key(conversation_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:index"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def _write(conversation_id: str, record: dict) -> None:
    """Persist (replace) a briefing record + index it. Best-effort."""
    message_id = record.get("message_id")
    if not message_id:
        return
    try:
        client = _redis()
        key = _briefing_key(conversation_id, message_id)
        client.set(key, json.dumps(record))
        client.expire(key, AMBASSADOR_TTL_SECONDS)
        idx = _index_key(conversation_id)
        client.sadd(idx, message_id)
        client.expire(idx, AMBASSADOR_TTL_SECONDS)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.warning(f"ambassador write failed: {e}")


def get_briefing(conversation_id: str, message_id: str) -> Optional[dict]:
    """Return the briefing record for one message, or None."""
    try:
        raw = _redis().get(_briefing_key(conversation_id, message_id))
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador read failed: {e}")
        return None
    if raw is None:
        return None
    try:
        return json.loads(_decode(raw))
    except (ValueError, TypeError):
        return None


def list_briefings(conversation_id: str) -> list[dict]:
    """Return all briefing records for a conversation (unordered)."""
    try:
        client = _redis()
        ids = [_decode(m) for m in client.smembers(_index_key(conversation_id))]
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador list failed: {e}")
        return []

    out: list[dict] = []
    for message_id in ids:
        rec = get_briefing(conversation_id, message_id)
        if rec is not None:
            out.append(rec)
        else:
            # Briefing string expired but the index still points at it — prune.
            try:
                client.srem(_index_key(conversation_id), message_id)
            except Exception:
                pass
    return out


def set_status(
    conversation_id: str,
    message_id: str,
    status: str,
    *,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Create/update a briefing record's status (preserving any summary text)."""
    record = get_briefing(conversation_id, message_id) or {
        "message_id": message_id,
        "summary": "",
        "created_at": _now(),
    }
    record["status"] = status
    record["updated_at"] = _now()
    if run_id is not None:
        record["run_id"] = run_id
    if error is not None:
        record["error"] = error
    _write(conversation_id, record)


def append_chunk(conversation_id: str, message_id: str, text: str) -> None:
    """Append streamed text to a briefing (status stays whatever it was)."""
    if not text:
        return
    record = get_briefing(conversation_id, message_id) or {
        "message_id": message_id,
        "status": "streaming",
        "summary": "",
        "created_at": _now(),
    }
    record["summary"] = (record.get("summary") or "") + text
    record["updated_at"] = _now()
    _write(conversation_id, record)


def set_summary(
    conversation_id: str,
    message_id: str,
    summary: str,
    status: str = "done",
) -> None:
    """Replace the full briefing text and settle the status."""
    record = get_briefing(conversation_id, message_id) or {
        "message_id": message_id,
        "created_at": _now(),
    }
    record["summary"] = summary or ""
    record["status"] = status
    record["updated_at"] = _now()
    record.pop("error", None)
    _write(conversation_id, record)


def clear(conversation_id: str) -> None:
    """Delete all briefings for a conversation."""
    try:
        client = _redis()
        idx = _index_key(conversation_id)
        ids = [_decode(m) for m in client.smembers(idx)]
        for message_id in ids:
            client.delete(_briefing_key(conversation_id, message_id))
        client.delete(idx)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador clear failed: {e}")

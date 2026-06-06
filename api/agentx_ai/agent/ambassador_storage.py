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
    Q&A:       ambassador:{conversation_id}:qa:{qa_id}         (Redis string, JSON, TTL)
    Q&A index: ambassador:{conversation_id}:qa_index           (Redis set of qa_ids, TTL)

Per-turn briefings and free-form Q&A share this sidecar (and the no-pollution
invariant) but live under disjoint key families so they replay independently.
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


def _qa_key(conversation_id: str, qa_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:qa:{qa_id}"


def _qa_index_key(conversation_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:qa_index"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


# --- Generic record engine (shared by the briefing + Q&A families) ----------


def _persist(item_key: str, index_key: str, item_id: str, record: dict) -> None:
    """Persist (replace) one record under ``item_key`` and index its id. Best-effort."""
    if not item_id:
        return
    try:
        client = _redis()
        client.set(item_key, json.dumps(record))
        client.expire(item_key, AMBASSADOR_TTL_SECONDS)
        client.sadd(index_key, item_id)
        client.expire(index_key, AMBASSADOR_TTL_SECONDS)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.warning(f"ambassador write failed: {e}")


def _read(item_key: str) -> Optional[dict]:
    """Return the JSON record at ``item_key``, or None."""
    try:
        raw = _redis().get(item_key)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador read failed: {e}")
        return None
    if raw is None:
        return None
    try:
        return json.loads(_decode(raw))
    except (ValueError, TypeError):
        return None


def _list(index_key: str, item_key_for) -> list[dict]:
    """Return all records indexed by ``index_key``, pruning expired stragglers."""
    try:
        client = _redis()
        ids = [_decode(m) for m in client.smembers(index_key)]
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador list failed: {e}")
        return []

    out: list[dict] = []
    for item_id in ids:
        rec = _read(item_key_for(item_id))
        if rec is not None:
            out.append(rec)
        else:
            # Record string expired but the index still points at it — prune.
            try:
                client.srem(index_key, item_id)
            except Exception:
                pass
    return out


def _write(conversation_id: str, record: dict) -> None:
    """Persist (replace) a briefing record + index it. Best-effort."""
    message_id = record.get("message_id")
    if not message_id:
        return
    _persist(
        _briefing_key(conversation_id, message_id),
        _index_key(conversation_id),
        message_id,
        record,
    )


def get_briefing(conversation_id: str, message_id: str) -> Optional[dict]:
    """Return the briefing record for one message, or None."""
    return _read(_briefing_key(conversation_id, message_id))


def list_briefings(conversation_id: str) -> list[dict]:
    """Return all briefing records for a conversation (unordered)."""
    return _list(
        _index_key(conversation_id),
        lambda message_id: _briefing_key(conversation_id, message_id),
    )


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
    """Delete all briefings + Q&A for a conversation."""
    try:
        client = _redis()
        for index_key, key_for in (
            (_index_key(conversation_id), lambda i: _briefing_key(conversation_id, i)),
            (_qa_index_key(conversation_id), lambda i: _qa_key(conversation_id, i)),
        ):
            for item_id in [_decode(m) for m in client.smembers(index_key)]:
                client.delete(key_for(item_id))
            client.delete(index_key)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador clear failed: {e}")


# --- Free-form Q&A records --------------------------------------------------
#
# A Q&A entry is conversation-scoped (not turn-scoped): the user asks the
# ambassador anything about the conversation. Same status lifecycle + streaming
# shape as a briefing, but carries the `question` and lives under the `qa:` family
# so it replays independently of per-turn briefings.


def _write_qa(conversation_id: str, record: dict) -> None:
    qa_id = record.get("qa_id")
    if not qa_id:
        return
    _persist(
        _qa_key(conversation_id, qa_id),
        _qa_index_key(conversation_id),
        qa_id,
        record,
    )


def get_qa(conversation_id: str, qa_id: str) -> Optional[dict]:
    """Return one Q&A record, or None."""
    return _read(_qa_key(conversation_id, qa_id))


def list_qa(conversation_id: str) -> list[dict]:
    """Return all Q&A records for a conversation (unordered; sort by created_at)."""
    return _list(
        _qa_index_key(conversation_id),
        lambda qa_id: _qa_key(conversation_id, qa_id),
    )


def create_qa(
    conversation_id: str,
    qa_id: str,
    question: str,
    *,
    run_id: Optional[str] = None,
) -> None:
    """Open a Q&A entry in the ``streaming`` state, stamping the question."""
    _write_qa(
        conversation_id,
        {
            "qa_id": qa_id,
            "question": question,
            "answer": "",
            "status": "streaming",
            "run_id": run_id,
            "created_at": _now(),
            "updated_at": _now(),
        },
    )


def set_qa_status(
    conversation_id: str,
    qa_id: str,
    status: str,
    *,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update a Q&A record's status (preserving its question + any answer text)."""
    record = get_qa(conversation_id, qa_id) or {
        "qa_id": qa_id,
        "question": "",
        "answer": "",
        "created_at": _now(),
    }
    record["status"] = status
    record["updated_at"] = _now()
    if run_id is not None:
        record["run_id"] = run_id
    if error is not None:
        record["error"] = error
    _write_qa(conversation_id, record)


def append_qa_chunk(conversation_id: str, qa_id: str, text: str) -> None:
    """Append streamed answer text to a Q&A record."""
    if not text:
        return
    record = get_qa(conversation_id, qa_id) or {
        "qa_id": qa_id,
        "question": "",
        "answer": "",
        "status": "streaming",
        "created_at": _now(),
    }
    record["answer"] = (record.get("answer") or "") + text
    record["updated_at"] = _now()
    _write_qa(conversation_id, record)


def set_qa_answer(
    conversation_id: str,
    qa_id: str,
    answer: str,
    status: str = "done",
) -> None:
    """Replace the full answer text and settle the status."""
    record = get_qa(conversation_id, qa_id) or {
        "qa_id": qa_id,
        "question": "",
        "created_at": _now(),
    }
    record["answer"] = answer or ""
    record["status"] = status
    record["updated_at"] = _now()
    record.pop("error", None)
    _write_qa(conversation_id, record)

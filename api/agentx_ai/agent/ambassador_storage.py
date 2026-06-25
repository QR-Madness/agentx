"""
The Ambassador's durable conversation, persisted in Redis (a conversation *sidecar*).

The Ambassador (Phase 16.7) runs parallel to a conversation and talks *with* the
user about it — per-turn **briefings** and free-form **Q&A** — without ever writing
into the main agent's world. Its output lives under a dedicated Redis prefix read by
*nothing* in the main-agent path (``hydrate_session_from_history``,
``load_recent_turns``, ``ContextManager.assemble_turn_context`` read only the Postgres
``conversation_logs`` stream and the ``conv_summary:`` key). This prefix isolation is
the load-bearing no-pollution guarantee — never write the ambassador's output into
``conversation_logs`` or ``conv_summary:``.

**Slice 1b — unified thread model.** Briefings and Q&A are now one ordered **thread**
of *entries* (an "Inquiry") keyed by a ``thread_id`` (defaults to the conversation id),
so the thread can carry its own ``title`` and the panel renders one conversation. Each
entry is one ambassador turn carrying its optional prompting ``question`` (entry-oriented
rather than message-split — this keeps in-place streaming updates simple and preserves
briefing idempotency, which is keyed by the briefed turn's ``message_id``).

The previous per-family public API (``set_summary``/``create_qa``/``get_qa``/…) is
preserved as **thin projections** over the entry store, so callers and replay shapes are
unchanged. Pre-1b records under the legacy key families still **replay** (a one-place
fold in ``list_thread``) and age out via the TTL.

Keys:
    Thread meta:  amb_thread:{thread_id}:meta            (Redis string, JSON, TTL)
    Entry:        amb_thread:{thread_id}:entry:{entry_id}(Redis string, JSON, TTL)
    Entry index:  amb_thread:{thread_id}:index           (Redis set of entry_ids, TTL)
    Legacy (read-only, pre-1b): ambassador:{cid}:msg:{mid} / :index / :qa:{qid} / :qa_index
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, UTC
from typing import cast

logger = logging.getLogger(__name__)

# Legacy prefix — still asserted by the no-pollution test and read for back-compat.
AMBASSADOR_PREFIX = "ambassador:"
# Unified thread family (Slice 1b).
THREAD_PREFIX = "amb_thread:"
AMBASSADOR_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days, mirrors conv_summary

# Sentinel: a patch value meaning "remove this field" (e.g. clear `error` on settle).
_DELETE = object()

# status ∈ streaming | done | error | empty_provider | cancelled
# kind   ∈ briefing | qa


def _redis():
    from ..kit.agent_memory.connections import RedisConnection

    return RedisConnection.get_client()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _decode(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


# --- keys -------------------------------------------------------------------


def _entry_key(thread_id: str, entry_id: str) -> str:
    return f"{THREAD_PREFIX}{thread_id}:entry:{entry_id}"


def _thread_index_key(thread_id: str) -> str:
    return f"{THREAD_PREFIX}{thread_id}:index"


def _thread_meta_key(thread_id: str) -> str:
    return f"{THREAD_PREFIX}{thread_id}:meta"


# Legacy key families (pre-1b) — read for back-compat fold + cleared by clear().
def _briefing_key(conversation_id: str, message_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:msg:{message_id}"


def _index_key(conversation_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:index"


def _qa_key(conversation_id: str, qa_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:qa:{qa_id}"


def _qa_index_key(conversation_id: str) -> str:
    return f"{AMBASSADOR_PREFIX}{conversation_id}:qa_index"


# --- generic record engine --------------------------------------------------


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


def _read(item_key: str) -> dict | None:
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
        ids = [_decode(m) for m in cast(set, client.smembers(index_key))]
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


# --- entry core (the unified store) -----------------------------------------


def get_entry(thread_id: str, entry_id: str) -> dict | None:
    """Return the raw thread entry, or None."""
    return _read(_entry_key(thread_id, entry_id))


def _set_entry(thread_id: str, entry_id: str, kind: str, **patch) -> None:
    """Upsert one entry: get-or-create (preserving ``created_at``), apply ``patch``
    (a value of ``None`` is skipped; ``_DELETE`` pops the field), restamp
    ``updated_at``. Best-effort."""
    if not entry_id:
        return
    rec = get_entry(thread_id, entry_id) or {
        "id": entry_id,
        "kind": kind,
        "question": "",
        "content": "",
        "tool_calls": [],
        "status": "streaming",
        "created_at": _now(),
    }
    rec.setdefault("kind", kind)
    for key, value in patch.items():
        if value is _DELETE:
            rec.pop(key, None)
        elif value is not None:
            rec[key] = value
    rec["updated_at"] = _now()
    _persist(_entry_key(thread_id, entry_id), _thread_index_key(thread_id), entry_id, rec)


def set_entry_tool_calls(thread_id: str, entry_id: str, tool_calls: list) -> None:
    """Persist the live tool-call chips on an entry so they survive a reload. The entry
    is always created first (by ``create_qa`` / briefing ``set_status``), so a missing
    entry is a no-op rather than a phantom create."""
    rec = get_entry(thread_id, entry_id)
    if rec is None:
        return
    rec["tool_calls"] = list(tool_calls or [])
    rec["updated_at"] = _now()
    _persist(_entry_key(thread_id, entry_id), _thread_index_key(thread_id), entry_id, rec)


def _legacy_briefing_to_entry(rec: dict) -> dict:
    return {
        "id": rec.get("message_id"),
        "kind": "briefing",
        "question": "",
        "content": rec.get("summary") or "",
        "tool_calls": rec.get("tool_calls") or rec.get("toolCalls") or [],
        "status": rec.get("status"),
        "run_id": rec.get("run_id"),
        "error": rec.get("error"),
        "message_id": rec.get("message_id"),
        "created_at": rec.get("created_at"),
        "updated_at": rec.get("updated_at"),
    }


def _legacy_qa_to_entry(rec: dict) -> dict:
    return {
        "id": rec.get("qa_id"),
        "kind": "qa",
        "question": rec.get("question") or "",
        "content": rec.get("answer") or "",
        "tool_calls": rec.get("tool_calls") or rec.get("toolCalls") or [],
        "status": rec.get("status"),
        "run_id": rec.get("run_id"),
        "error": rec.get("error"),
        "created_at": rec.get("created_at"),
        "updated_at": rec.get("updated_at"),
    }


def list_thread(thread_id: str) -> list[dict]:
    """All entries for a thread, oldest-first by ``created_at``.

    Folds in pre-1b legacy briefing/Q&A records (the one-place migration bridge) so old
    sidecars still replay; a new entry with the same id wins. Removable once the legacy
    TTL window (30 days) has fully aged out.
    """
    entries: dict[str, dict] = {}
    # Legacy first, so a re-written new entry of the same id supersedes it below.
    for rec in _list(_index_key(thread_id), lambda i: _briefing_key(thread_id, i)):
        e = _legacy_briefing_to_entry(rec)
        if e.get("id"):
            entries[e["id"]] = e
    for rec in _list(_qa_index_key(thread_id), lambda i: _qa_key(thread_id, i)):
        e = _legacy_qa_to_entry(rec)
        if e.get("id"):
            entries[e["id"]] = e
    for rec in _list(_thread_index_key(thread_id), lambda i: _entry_key(thread_id, i)):
        if rec.get("id"):
            entries[rec["id"]] = rec
    out = list(entries.values())
    out.sort(key=lambda e: e.get("created_at") or "")
    return out


# --- thread meta (title) ----------------------------------------------------


def get_thread_meta(thread_id: str) -> dict | None:
    return _read(_thread_meta_key(thread_id))


def get_thread_title(thread_id: str) -> str:
    """The Inquiry's own title, or "" (the client falls back to the chat title)."""
    return (get_thread_meta(thread_id) or {}).get("title") or ""


def set_thread_title(thread_id: str, title: str, *, auto: bool = False) -> dict:
    """Set/rename the thread's title. Best-effort; returns the meta record.

    ``auto`` marks a machine-derived title (vs. a user rename). Auto-titling only ever
    fills a thread that has no manual title, so a person's rename is never clobbered —
    see ``AmbassadorService._maybe_autotitle``."""
    meta = get_thread_meta(thread_id) or {"thread_id": thread_id, "created_at": _now()}
    meta["title"] = (title or "").strip()[:200]
    meta["title_auto"] = bool(auto)
    meta["updated_at"] = _now()
    try:
        client = _redis()
        client.set(_thread_meta_key(thread_id), json.dumps(meta))
        client.expire(_thread_meta_key(thread_id), AMBASSADOR_TTL_SECONDS)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.warning(f"ambassador title write failed: {e}")
    return meta


def derive_title(question: str, *, limit: int = 48) -> str:
    """A short, human title derived from an Inquiry's first question — model-free.

    Collapses whitespace and truncates on a word boundary. Pure (no I/O) so it's
    testable and cheap; the UI already frames the string as an "Inquiry", so no prefix."""
    text = " ".join((question or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip() or text[:limit].rstrip()
    return cut + "…"


# --- per-user thread registry (standalone "Inquiries") ----------------------
#
# Threads are otherwise reachable only by a known thread_id. This ZSET lets the
# command deck enumerate a user's *standalone* Inquiries (the home `deck:{user}`
# thread + minted `inq:{user}:{uuid}` threads). Written from the views (which hold
# the user id — storage itself is thread-keyed, with no user dimension). Score is
# a recency timestamp so the list is newest-first.


def _user_threads_key(user_id: str) -> str:
    return f"amb_user:{user_id}:threads"


def register_thread(user_id: str, thread_id: str) -> None:
    """Record a standalone Inquiry under a user (idempotent; bumps recency). Best-effort."""
    if not user_id or not thread_id:
        return
    try:
        client = _redis()
        key = _user_threads_key(user_id)
        client.zadd(key, {thread_id: datetime.now(UTC).timestamp()})
        client.expire(key, AMBASSADOR_TTL_SECONDS)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.warning(f"ambassador thread register failed: {e}")


def unregister_thread(user_id: str, thread_id: str) -> None:
    """Drop a thread from a user's registry (on delete). Best-effort."""
    if not user_id or not thread_id:
        return
    try:
        _redis().zrem(_user_threads_key(user_id), thread_id)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador thread unregister failed: {e}")


def list_user_threads(user_id: str) -> list[dict]:
    """A user's standalone Inquiries, newest-first: ``[{thread_id, title, created_at,
    updated_at}]``. Self-healing — an id whose meta has aged out (TTL) or been cleared
    is dropped from the registry so the list never shows ghosts. Never raises."""
    try:
        client = _redis()
        ids = [_decode(t) for t in client.zrevrange(_user_threads_key(user_id), 0, -1)]
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador thread list failed: {e}")
        return []
    out: list[dict] = []
    for tid in ids:
        meta = get_thread_meta(tid)
        if not meta:
            unregister_thread(user_id, tid)  # ghost — meta gone
            continue
        out.append({
            "thread_id": tid,
            "title": meta.get("title") or "",
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
        })
    return out


# --- wire serialization (replay endpoint) -----------------------------------


def _entry_to_wire(e: dict) -> dict:
    """Serialize an entry for the thread replay endpoint (camelCase ``toolCalls`` to
    match the client's live SSE shape)."""
    out = {
        "id": e.get("id"),
        "kind": e.get("kind"),
        "question": e.get("question") or "",
        "content": e.get("content") or "",
        "status": e.get("status"),
        "toolCalls": e.get("tool_calls") or [],
        "created_at": e.get("created_at"),
        "updated_at": e.get("updated_at"),
    }
    if e.get("message_id"):
        out["message_id"] = e["message_id"]
    if e.get("run_id"):
        out["run_id"] = e["run_id"]
    if e.get("error"):
        out["error"] = e["error"]
    return out


def thread_payload(thread_id: str) -> dict:
    """The full replay payload for one Inquiry: ``{thread_id, title, entries}``."""
    return {
        "thread_id": thread_id,
        "title": get_thread_title(thread_id),
        "entries": [_entry_to_wire(e) for e in list_thread(thread_id)],
    }


# --- per-turn briefings (kind="briefing"): public API over the entry store ---


def _entry_to_briefing(e: dict) -> dict:
    out = {
        "message_id": e.get("message_id") or e.get("id"),
        "status": e.get("status"),
        "summary": e.get("content") or "",
        "created_at": e.get("created_at"),
        "updated_at": e.get("updated_at"),
        "toolCalls": e.get("tool_calls") or [],
    }
    if e.get("run_id"):
        out["run_id"] = e["run_id"]
    if e.get("error"):
        out["error"] = e["error"]
    return out


def get_briefing(conversation_id: str, message_id: str) -> dict | None:
    """Return the briefing record for one message, or None."""
    e = get_entry(conversation_id, message_id)
    if e is not None and e.get("kind") == "briefing":
        return _entry_to_briefing(e)
    legacy = _read(_briefing_key(conversation_id, message_id))
    return _entry_to_briefing(_legacy_briefing_to_entry(legacy)) if legacy else None


def list_briefings(conversation_id: str) -> list[dict]:
    """Return all briefing records for a conversation (oldest-first)."""
    return [
        _entry_to_briefing(e)
        for e in list_thread(conversation_id)
        if e.get("kind") == "briefing"
    ]


def set_status(
    conversation_id: str,
    message_id: str,
    status: str,
    *,
    run_id: str | None = None,
    error: str | None = None,
) -> None:
    """Create/update a briefing entry's status (preserving any summary text)."""
    _set_entry(
        conversation_id,
        message_id,
        "briefing",
        status=status,
        message_id=message_id,
        run_id=run_id,
        error=error,
    )


def append_chunk(conversation_id: str, message_id: str, text: str) -> None:
    """Append streamed text to a briefing (status stays whatever it was)."""
    if not text:
        return
    cur = get_entry(conversation_id, message_id)
    content = ((cur or {}).get("content") or "") + text
    _set_entry(conversation_id, message_id, "briefing", content=content, message_id=message_id)


def set_summary(
    conversation_id: str,
    message_id: str,
    summary: str,
    status: str = "done",
) -> None:
    """Replace the full briefing text and settle the status."""
    _set_entry(
        conversation_id,
        message_id,
        "briefing",
        content=summary or "",
        status=status,
        message_id=message_id,
        error=_DELETE,
    )


# --- free-form Q&A (kind="qa"): public API over the entry store -------------


def _entry_to_qa(e: dict) -> dict:
    out = {
        "qa_id": e.get("id"),
        "question": e.get("question") or "",
        "answer": e.get("content") or "",
        "status": e.get("status"),
        "created_at": e.get("created_at"),
        "updated_at": e.get("updated_at"),
        "toolCalls": e.get("tool_calls") or [],
    }
    if e.get("run_id"):
        out["run_id"] = e["run_id"]
    if e.get("error"):
        out["error"] = e["error"]
    return out


def get_qa(conversation_id: str, qa_id: str) -> dict | None:
    """Return one Q&A record, or None."""
    e = get_entry(conversation_id, qa_id)
    if e is not None and e.get("kind") == "qa":
        return _entry_to_qa(e)
    legacy = _read(_qa_key(conversation_id, qa_id))
    return _entry_to_qa(_legacy_qa_to_entry(legacy)) if legacy else None


def list_qa(conversation_id: str) -> list[dict]:
    """Return all Q&A records for a conversation (oldest-first)."""
    return [
        _entry_to_qa(e)
        for e in list_thread(conversation_id)
        if e.get("kind") == "qa"
    ]


def create_qa(
    conversation_id: str,
    qa_id: str,
    question: str,
    *,
    run_id: str | None = None,
) -> None:
    """Open a Q&A entry in the ``streaming`` state, stamping the question."""
    _set_entry(
        conversation_id,
        qa_id,
        "qa",
        question=question,
        content="",
        status="streaming",
        run_id=run_id,
    )


def set_qa_status(
    conversation_id: str,
    qa_id: str,
    status: str,
    *,
    run_id: str | None = None,
    error: str | None = None,
) -> None:
    """Update a Q&A entry's status (preserving its question + any answer text)."""
    _set_entry(conversation_id, qa_id, "qa", status=status, run_id=run_id, error=error)


def append_qa_chunk(conversation_id: str, qa_id: str, text: str) -> None:
    """Append streamed answer text to a Q&A entry."""
    if not text:
        return
    cur = get_entry(conversation_id, qa_id)
    content = ((cur or {}).get("content") or "") + text
    _set_entry(conversation_id, qa_id, "qa", content=content)


def set_qa_answer(
    conversation_id: str,
    qa_id: str,
    answer: str,
    status: str = "done",
) -> None:
    """Replace the full answer text and settle the status."""
    _set_entry(conversation_id, qa_id, "qa", content=answer or "", status=status, error=_DELETE)


# --- lifecycle --------------------------------------------------------------


def clear(conversation_id: str) -> None:
    """Delete the whole thread (entries + meta) and any legacy briefing/Q&A records."""
    try:
        client = _redis()
        for index_key, key_for in (
            (_thread_index_key(conversation_id), lambda i: _entry_key(conversation_id, i)),
            (_index_key(conversation_id), lambda i: _briefing_key(conversation_id, i)),
            (_qa_index_key(conversation_id), lambda i: _qa_key(conversation_id, i)),
        ):
            for item_id in [_decode(m) for m in cast(set, client.smembers(index_key))]:
                client.delete(key_for(item_id))
            client.delete(index_key)
        client.delete(_thread_meta_key(conversation_id))
    except Exception as e:  # pragma: no cover - Redis offline
        logger.debug(f"ambassador clear failed: {e}")


# --- aide-swarm digest cache ------------------------------------------------
# The aide swarm condenses ONE conversation read-only into a short digest. Caching
# those digests keeps a repeat survey cheap. The cache lives under its own `amb_aide:`
# key — part of the ambassador sidecar family, NEVER `conv_summary:`/`conversation_logs`
# — so the no-pollution invariant (INV-2) holds. A digest is keyed by conversation +
# focus and validated by a `fingerprint` (the conversation's message_count + last_at);
# a grown conversation no longer matches → cache miss → re-digest.

AIDE_PREFIX = "amb_aide:"


def _aide_key(conversation_id: str, focus_hash: str) -> str:
    return f"{AIDE_PREFIX}{conversation_id}:{focus_hash}"


def _focus_hash(focus: str) -> str:
    f = (focus or "").strip()
    # Not security-sensitive — just a stable short key for the focus string.
    return hashlib.sha1(f.encode(), usedforsecurity=False).hexdigest()[:8] if f else "_"


def get_aide_digest(conversation_id: str, fingerprint: str, focus: str = "") -> str | None:
    """Return a cached aide digest IF its fingerprint still matches (else None — a
    changed conversation re-digests). Read-only over the sidecar; never raises."""
    rec = _read(_aide_key(conversation_id, _focus_hash(focus)))
    if not rec or rec.get("fingerprint") != fingerprint:
        return None
    digest = rec.get("digest")
    return digest if isinstance(digest, str) and digest else None


def set_aide_digest(
    conversation_id: str, fingerprint: str, digest: str, focus: str = "", ttl: int | None = None,
) -> None:
    """Write-through cache one aide digest into the ambassador sidecar (`amb_aide:`),
    keyed by conversation + focus and validated by `fingerprint`. Best-effort, TTL'd."""
    if not conversation_id or not digest:
        return
    try:
        client = _redis()
        key = _aide_key(conversation_id, _focus_hash(focus))
        client.set(key, json.dumps({
            "digest": digest, "fingerprint": fingerprint, "created_at": _now(),
        }))
        client.expire(key, ttl if ttl is not None else AMBASSADOR_TTL_SECONDS)
    except Exception as e:  # pragma: no cover - Redis offline
        logger.warning(f"aide digest cache write failed: {e}")

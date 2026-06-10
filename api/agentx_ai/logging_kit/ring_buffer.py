"""In-memory ring buffer + bounded pub/sub for the client Log panel.

A :class:`logging.Handler` that keeps the last N **structured, redacted,
ANSI-free** records (the message is already redacted by the QueueHandler-side
``RedactionFilter``). It also fans new records to live subscribers via bounded
queues that *drop oldest* under back-pressure so a slow SSE client can neither
stall the logging listener nor grow memory without bound.

Lives on the ``QueueListener`` side, so capture is off the request/stream path.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from itertools import count
from typing import Any

from .categories import category_for

# Records from these logger prefixes are never captured into the ring — otherwise
# logging *about* serving the log stream would feed itself (amplification loop).
_SELF_PREFIXES = ("agentx_ai.logging_kit", "agentx_ai.views_logs")

_MAX_SUBSCRIBERS = 32
_SUBSCRIBER_QUEUE_MAX = 1000


def _record_to_dict(record: logging.LogRecord) -> dict[str, Any]:
    cat = category_for(record.name)
    out: dict[str, Any] = {
        "id": None,  # filled by the handler under lock for stable ordering
        "ts": record.created,
        "level": record.levelname,
        "logger": record.name,
        "category": cat.key,
        "run_id": getattr(record, "run_id", None),
        "conversation_id": getattr(record, "conversation_id", None),
        "agent_id": getattr(record, "agent_id", None),
        "message": record.getMessage(),
    }
    if record.exc_text:
        out["exc"] = record.exc_text
    # Oversized, already-redacted payloads (e.g. a full LLM request) ride a
    # separate field so the message stays a one-line summary but the in-app Log
    # panel can expand the full content on demand.
    detail = getattr(record, "llm_detail", None)
    if detail:
        out["detail"] = detail
    return out


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._ids = count(1)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_SELF_PREFIXES):
            return
        try:
            entry = _record_to_dict(record)
        except Exception:  # noqa: BLE001 — never let capture break logging
            return
        with self._lock:
            entry["id"] = next(self._ids)
            self._buf.append(entry)
            subs = list(self._subscribers)
        for q in subs:
            self._offer(q, entry)

    @staticmethod
    def _offer(q: queue.Queue, entry: dict[str, Any]) -> None:
        try:
            q.put_nowait(entry)
        except queue.Full:
            try:
                q.get_nowait()  # drop oldest
                q.put_nowait(entry)
            except queue.Empty:
                pass

    # --- query API (Phase 4 endpoints) ---------------------------------------

    def snapshot(
        self,
        *,
        level: str | None = None,
        category: str | None = None,
        run_id: str | None = None,
        search: str | None = None,
        since_id: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        lvl = (level or "").upper() or None
        needle = (search or "").lower() or None
        with self._lock:
            items = list(self._buf)
        out = []
        for e in items:
            if since_id is not None and e["id"] <= since_id:
                continue
            if lvl and e["level"] != lvl:
                continue
            if category and e["category"] != category:
                continue
            if run_id and e.get("run_id") != run_id:
                continue
            if needle and needle not in e["message"].lower():
                continue
            out.append(e)
        return out[-limit:]

    def subscribe(self) -> queue.Queue | None:
        q: queue.Queue = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        with self._lock:
            if len(self._subscribers) >= _MAX_SUBSCRIBERS:
                return None
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


# Process-wide singleton, installed by ``setup.configure_logging``.
_handler: RingBufferHandler | None = None


def get_ring_handler() -> RingBufferHandler | None:
    return _handler


def set_ring_handler(handler: RingBufferHandler | None) -> None:
    global _handler
    _handler = handler

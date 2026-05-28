"""
Detached chat runs — decouple SSE generation from the client connection.

The chat generator (``generate_sse`` in ``views.py``) used to be driven directly
by the ASGI server pulling the ``StreamingHttpResponse``. When the client
disconnected (tab close, or even a tab switch), the server stopped pulling, the
LLM stream was abandoned, and — because turns are persisted only at the *end* of
the generator — nothing was saved. New chats vanished entirely.

This module runs the generator in a detached daemon thread (own event loop),
fanning every SSE event into a Redis Stream and marking run state. The HTTP
endpoint becomes a *tail* over that stream: it can be detached from (close the
connection) and re-attached to (replay from 0 + continue live) without affecting
the run, which always plays to completion and persists its turns.

Mirrors the patterns in ``agent/plan_state.py`` and ``background/chat_jobs.py``
(Redis hash for state + Redis Stream for events + daemon thread + best-effort
degradation when Redis is down).

Keys:
    Events:  chat_run:{run_id}:events   (Redis Stream, MAXLEN-capped, TTL)
    State:   chat_run:{run_id}          (Redis Hash, TTL)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

RUN_KEY_PREFIX = "chat_run"
RUN_TTL_SECONDS = 2 * 60 * 60          # 2h — outlives a reasonable reopen window
EVENTS_MAXLEN = 5000                   # cap buffered events per run
TAIL_BLOCK_MS = 2000                   # XREAD block timeout while tailing
STALE_RUNNING_SECONDS = 15 * 60        # a "running" run older than this is orphaned

# Sentinel appended after the generator finishes so the tail knows to stop even
# though the underlying generator doesn't emit a terminal event itself.
CLOSE_EVENT = "event: close\ndata: {}\n\n"


def _redis():
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


def _events_key(run_id: str) -> str:
    return f"{RUN_KEY_PREFIX}:{run_id}:events"


def _state_key(run_id: str) -> str:
    return f"{RUN_KEY_PREFIX}:{run_id}"


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class ChatRunStore:
    """Redis-backed state + event buffer for a detached chat run."""

    def create(self, run_id: str) -> None:
        try:
            client = _redis()
            now = datetime.now(timezone.utc).isoformat()
            client.hset(
                _state_key(run_id),
                mapping={"status": "running", "created_at": now, "updated_at": now},
            )
            client.expire(_state_key(run_id), RUN_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"chat_run create failed: {e}")

    def append_event(self, run_id: str, sse_event: str) -> None:
        try:
            client = _redis()
            client.xadd(
                _events_key(run_id),
                {"data": sse_event},
                maxlen=EVENTS_MAXLEN,
                approximate=True,
            )
            client.expire(_events_key(run_id), RUN_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"chat_run append_event failed: {e}")

    def mark(self, run_id: str, status: str) -> None:
        try:
            client = _redis()
            now = datetime.now(timezone.utc).isoformat()
            client.hset(_state_key(run_id), mapping={"status": status, "updated_at": now})
            client.expire(_state_key(run_id), RUN_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"chat_run mark failed: {e}")

    def get_state(self, run_id: str) -> Optional[dict]:
        try:
            raw = _redis().hgetall(_state_key(run_id))
            if not raw:
                return None
            return {_decode(k): _decode(v) for k, v in raw.items()}
        except Exception as e:
            logger.warning(f"chat_run get_state failed: {e}")
            return None

    def request_cancel(self, run_id: str) -> bool:
        try:
            client = _redis()
            if not client.exists(_state_key(run_id)):
                return False
            client.hset(_state_key(run_id), "cancel_requested", "1")
            client.expire(_state_key(run_id), RUN_TTL_SECONDS)
            return True
        except Exception as e:
            logger.warning(f"chat_run request_cancel failed: {e}")
            return False

    def is_cancel_requested(self, run_id: str) -> bool:
        try:
            value = _redis().hget(_state_key(run_id), "cancel_requested")
            return _decode(value) == "1"
        except Exception:
            return False


store = ChatRunStore()


# Factory yielding the SSE async generator to drive. Kept as a zero-arg callable
# so the (request-free) ``generate_sse`` closure can be handed over as-is.
GenFactory = Callable[[], AsyncGenerator[str, None]]


def start_chat_run(gen_factory: GenFactory) -> str:
    """Create run state and spawn the detached runner thread. Returns run_id."""
    run_id = uuid4().hex[:16]
    store.create(run_id)
    threading.Thread(
        target=_drive_run,
        args=(run_id, gen_factory),
        name=f"chat-run-{run_id}",
        daemon=True,
    ).start()
    logger.info(f"Started detached chat run {run_id}")
    return run_id


def _drive_run(run_id: str, gen_factory: GenFactory) -> None:
    """Consume the generator in a fresh event loop, fanning events into Redis."""

    async def _run() -> None:
        gen = gen_factory()
        cancelled = False
        try:
            async for sse_event in gen:
                store.append_event(run_id, sse_event)
                if store.is_cancel_requested(run_id):
                    cancelled = True
                    await gen.aclose()
                    break
        finally:
            # Always terminate the tail and settle status, even on aclose/raise.
            store.append_event(run_id, CLOSE_EVENT)
            store.mark(run_id, "cancelled" if cancelled else "done")

    try:
        asyncio.run(_run())
        logger.info(f"Detached chat run {run_id} finished")
    except Exception as e:  # noqa: BLE001 — runner thread must not crash the worker
        logger.exception(f"Detached chat run {run_id} failed: {e}")
        try:
            store.append_event(
                run_id,
                f"event: error\ndata: {json.dumps({'error': str(e)[:500]})}\n\n",
            )
            store.append_event(run_id, CLOSE_EVENT)
            store.mark(run_id, "failed")
        except Exception:
            pass


def _is_stale_running(state: dict) -> bool:
    if state.get("status") != "running":
        return False
    updated = state.get("updated_at")
    if not updated:
        return False
    try:
        ts = datetime.fromisoformat(updated)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age > STALE_RUNNING_SECONDS
    except Exception:
        return False


async def tail_chat_run(run_id: str, last_id: str = "0") -> AsyncGenerator[str, None]:
    """
    Yield SSE event strings for a run, replaying from ``last_id`` then following
    live. Stops on the CLOSE sentinel or a terminal/orphaned state. Emits a
    ``run_missing`` event when the run's state/events are gone (TTL expiry) so
    the client can fall back to restoring from conversation history.
    """
    loop = asyncio.get_event_loop()
    key = _events_key(run_id)

    state = store.get_state(run_id)
    if state is None:
        yield f"event: run_missing\ndata: {json.dumps({'run_id': run_id})}\n\n"
        return
    if _is_stale_running(state):
        store.mark(run_id, "failed")
        yield f"event: run_missing\ndata: {json.dumps({'run_id': run_id, 'reason': 'stale'})}\n\n"
        return

    redis = _redis()
    while True:
        try:
            entries = await loop.run_in_executor(
                None,
                lambda lid=last_id: redis.xread({key: lid}, block=TAIL_BLOCK_MS, count=100),
            )
        except Exception as e:
            logger.warning(f"tail_chat_run xread failed: {e}")
            yield f"event: error\ndata: {json.dumps({'error': 'stream read failed'})}\n\n"
            return

        if not entries:
            # No new events in the block window — stop if the run has settled.
            st = store.get_state(run_id)
            if st is None or st.get("status") != "running":
                return
            continue

        for _stream_key, msgs in entries:
            for msg_id, fields in msgs:
                last_id = _decode(msg_id)
                sse_event = _decode(fields.get(b"data") or fields.get("data") or "")
                if sse_event == CLOSE_EVENT:
                    yield CLOSE_EVENT
                    return
                if sse_event:
                    yield sse_event

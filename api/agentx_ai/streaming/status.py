"""
Per-phase status emission for a streaming chat turn.

Between "message sent" and the answer, the chat would otherwise show a single
"Thinking…" spinner. This module publishes a granular `status` SSE event so the
client can show a live activity line ("Recalling memory…", "Running web_search…",
…).

Delivery rides the **run event bus**, not generator yields. The chat generator is
driven in a detached daemon thread (`chat_run._drive_run`) that fans every event
into a per-run Redis Stream; the live client and `/attach` both *tail* that stream
(never the generator directly). So anything that can resolve the current `run_id`
can publish a status by appending straight to the bus — which is the *only* way
deep, sync, nested, or cross-thread phases (e.g. embedding inside `remember()`,
which runs on a separate daemon thread) could ever surface, since the generator is
blocked in those calls and cannot yield. Routing **all** status through one ambient
emitter keeps a single mechanism for the coarse phases now and deep phases later.

The run is resolved ambiently via a `ContextVar` set at the top of `_drive_run`
(contextvars propagate through `asyncio.run` + the tasks it spawns). Callers that
cross a thread boundary the contextvar can't reach (the embedding daemon) pass an
explicit `run_id=`.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Set by `chat_run._drive_run` for the lifetime of one detached run; None outside
# a streaming turn (background jobs, plan exec driven standalone, tests) — in which
# case `emit_status` is a safe no-op.
current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)

# Stable phase slug -> default human label. `phase` is the client's stable key
# (phase -> icon/affordance); `label` is the display text. Reserved slugs at the
# bottom are not emitted yet — they're the drop-in points for the deferred deep
# sub-phases, kept here so the vocabulary lives in one place.
STATUS_PHASES: dict[str, str] = {
    "recalling": "Recalling memory…",
    "composing": "Composing context…",
    "thinking": "Thinking…",
    "running_tool": "Running tool…",
    "reading": "Reading results…",
    # --- reserved for deep sub-phases (not emitted yet) ---
    "embedding": "Embedding…",
    "reranking": "Reranking…",
    "reasoning_step": "Reasoning…",
}

# Min interval between two emits of the *same* (run_id, phase) with the same
# label/detail. Protects the capped event buffer + the UI from deep-status spam
# (per-chunk embedding, per reasoning step). Coarse's handful of transitions never
# trip it.
_THROTTLE_SECONDS = 0.15

# (run_id, phase) -> (last_label, last_detail, last_ts). Guarded by a lock since
# the embedding daemon thread emits concurrently with the run's async task.
_last_emit: dict[tuple[str, str], tuple[str, str | None, float]] = {}
_lock = threading.Lock()


def emit_status(
    phase: str,
    label: str | None = None,
    *,
    detail: str | None = None,
    group: str | None = None,
    progress: float | None = None,
    run_id: str | None = None,
) -> None:
    """Publish a `status` event to the current run's event bus.

    No-op when no run can be resolved (no `run_id` and no contextvar) so call
    sites stay unconditional. `phase` is a stable slug (see `STATUS_PHASES`);
    `label` defaults to that slug's text. `detail`/`group`/`progress` are optional
    headroom for deep sub-phases (a breadcrumb group + progress bar) and are
    omitted from the payload when unset.
    """
    rid = run_id or current_run_id.get()
    if not rid:
        return

    text = label or STATUS_PHASES.get(phase, phase)

    # Coalesce repeats of the same phase+label within the throttle window.
    now = time.monotonic()
    key = (rid, phase)
    with _lock:
        prev = _last_emit.get(key)
        if (
            prev is not None
            and prev[0] == text
            and prev[1] == detail
            and (now - prev[2]) < _THROTTLE_SECONDS
        ):
            return
        _last_emit[key] = (text, detail, now)

    payload: dict[str, object] = {"phase": phase, "label": text}
    if detail is not None:
        payload["detail"] = detail
    if group is not None:
        payload["group"] = group
    if progress is not None:
        payload["progress"] = progress

    try:
        from .chat_run import store
        from .tool_loop import _sse

        store.append_event(rid, _sse("status", payload))
    except Exception as e:  # noqa: BLE001 — status is best-effort, never break a turn
        logger.debug(f"emit_status failed (phase={phase}): {e}")


def clear_run_throttle(run_id: str) -> None:
    """Drop a finished run's throttle entries so the dict doesn't grow unbounded."""
    with _lock:
        for key in [k for k in _last_emit if k[0] == run_id]:
            _last_emit.pop(key, None)

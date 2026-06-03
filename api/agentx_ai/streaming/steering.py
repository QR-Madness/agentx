"""
Live steering — fold a user's mid-turn message into the running turn.

A streaming turn used to be fire-and-forget: once it started you could only let
it finish or hard-cancel it. Steering lets the user enqueue a message (via
`POST /api/agent/chat/runs/{run_id}/steer`) that the streaming tool loop drains
at the next **safe boundary** (after a tool round, or instead of ending) and
folds in as a fresh user turn, so the agent course-corrects mid-trajectory
without the run being thrown away.

Delivery mirrors the SSE status emitter: the queue lives on the run's Redis
state (`chat_run`), and the loop resolves *which* run it is via the same
`current_run_id` ContextVar set in `chat_run._drive_run`. So the loop drains
ambiently with no run_id threaded through its signature.
"""

from __future__ import annotations

import logging

from .status import current_run_id

logger = logging.getLogger(__name__)


def drain_steer_messages() -> list[str]:
    """Drain the current run's pending steer messages (in order).

    No-op (empty list) when no run is resolvable — e.g. the tool loop is driven
    standalone (plan exec) or in tests. Destructive: each message is returned
    exactly once.
    """
    rid = current_run_id.get()
    if not rid:
        return []
    try:
        from .chat_run import store

        return store.drain_steer(rid)
    except Exception as e:  # noqa: BLE001 — steering is best-effort, never break a turn
        logger.debug(f"drain_steer_messages failed: {e}")
        return []

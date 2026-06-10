"""Correlation context for log records.

The streaming layer already sets a ``current_run_id`` ContextVar at the top of
``chat_run._drive_run`` (see ``streaming/status.py``); we reuse it as the turn
correlation id rather than maintaining a second source of truth. Two extra vars
(conversation / agent) are owned here for callers that have them.

``ContextFilter`` stamps each record. It is attached to the **QueueHandler**, so
it runs in the *emitting* thread where the ContextVars are visible — the
``QueueListener`` worker thread that fans records to the console/ring/archive
would not see them.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

conversation_id_var: ContextVar[str | None] = ContextVar("log_conversation_id", default=None)
agent_id_var: ContextVar[str | None] = ContextVar("log_agent_id", default=None)

# Resolved lazily once; the streaming module is stdlib-light so importing it from
# a log filter is cheap and import-safe (no Django at module load).
_run_id_var: ContextVar | None = None


def _run_id() -> str | None:
    global _run_id_var
    if _run_id_var is None:
        try:
            from agentx_ai.streaming.status import current_run_id

            _run_id_var = current_run_id
        except Exception:  # noqa: BLE001 — never let logging break on an import hiccup
            return None
    try:
        return _run_id_var.get()
    except Exception:  # noqa: BLE001
        return None


def _short(run_id: str | None) -> str | None:
    """A compact run tag for the console (full id stays on the structured record)."""
    if not run_id:
        return None
    tail = run_id.split("_")[-1]
    return tail[:6] if tail else run_id[:6]


class ContextFilter(logging.Filter):
    """Stamp ``run_id`` / ``conversation_id`` / ``agent_id`` onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = _run_id()
        record.run_id = rid  # type: ignore[attr-defined]
        record.run_tag = _short(rid)  # type: ignore[attr-defined]
        record.conversation_id = conversation_id_var.get()  # type: ignore[attr-defined]
        record.agent_id = agent_id_var.get()  # type: ignore[attr-defined]
        return True


def set_turn_context(*, conversation_id: str | None = None, agent_id: str | None = None) -> None:
    """Opportunistically tag the current context (best-effort; safe to call anywhere)."""
    if conversation_id is not None:
        conversation_id_var.set(conversation_id)
    if agent_id is not None:
        agent_id_var.set(agent_id)

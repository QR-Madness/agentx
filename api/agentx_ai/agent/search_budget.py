"""Per-turn web-search budget (Foundation #5).

Bounds how many ``web_search`` / ``web_research`` calls a single user turn may
make before the tool short-circuits with a budget error instead of spending
another Tavily/Brave credit. One *window* is opened per turn at the top of
``streaming_tool_loop``; the synchronous tool executor runs inside that same
async context, so the window is resolvable ambiently via a ``ContextVar`` — no
threading the counter through every signature (mirrors ``current_run_id``).

**No active window ⇒ unlimited.** Background/non-turn callers (alloy specialists,
the planner, reflection jobs) never open a window, so they're never gated — the
cap is specifically a runaway-tool-loop guard for an interactive turn.

The window also carries best-effort ledger attribution (conversation/agent id)
so a search-spend row can be tied back to the turn that made it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from collections.abc import Iterator


@dataclass
class _Window:
    """Mutable per-turn search-budget state. ``limit`` of 0 ⇒ unlimited."""

    limit: int = 0
    conversation_id: str | None = None
    agent_id: str | None = None
    used: int = field(default=0)


_window: ContextVar[_Window | None] = ContextVar("search_budget_window", default=None)


@contextmanager
def search_budget_window(
    limit: int,
    *,
    conversation_id: str | None = None,
    agent_id: str | None = None,
) -> Iterator[None]:
    """Open a per-turn budget window. Restores any prior window on exit."""
    token = _window.set(
        _Window(limit=max(0, int(limit or 0)), conversation_id=conversation_id, agent_id=agent_id)
    )
    try:
        yield
    finally:
        _window.reset(token)


def consume(weight: int = 1) -> tuple[bool, int, int]:
    """Charge ``weight`` calls against the active window.

    Returns ``(allowed, used, limit)``. ``allowed`` is False only when a window
    is active, has a non-zero limit, and is already at/over it — in which case
    the counter is *not* advanced (the call won't happen). No window or a 0
    limit ⇒ always allowed (``limit`` reported as 0 = unlimited).
    """
    win = _window.get()
    if win is None or win.limit <= 0:
        return True, 0, 0
    if win.used >= win.limit:
        return False, win.used, win.limit
    win.used += weight
    return True, win.used, win.limit


def attribution() -> tuple[str | None, str | None]:
    """Best-effort ``(conversation_id, agent_id)`` for the active window."""
    win = _window.get()
    if win is None:
        return None, None
    return win.conversation_id, win.agent_id

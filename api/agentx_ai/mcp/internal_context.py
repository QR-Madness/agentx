"""
Per-request context for internal tool handlers.

Internal tools registered in :mod:`internal_tools` execute as plain functions
without access to the calling agent. Some tools (``recall_user_history``,
``checkpoint``) need to know which user/channel/conversation invoked them. This
module exposes a :class:`contextvars.ContextVar` that the chat endpoint sets
before dispatching tools, and tool handlers read.

Using a :mod:`contextvars` ``ContextVar`` keeps the binding per-task even when
multiple chat streams run concurrently in the same process.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InternalToolContext:
    user_id: str
    channel: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None


_active: contextvars.ContextVar[Optional[InternalToolContext]] = contextvars.ContextVar(
    "agentx_internal_tool_context",
    default=None,
)


def set_context(ctx: Optional[InternalToolContext]) -> contextvars.Token:
    """Bind the active context. Returns a token to pass to :func:`reset_context`."""
    return _active.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    """Restore the previous binding."""
    _active.reset(token)


def current_context() -> Optional[InternalToolContext]:
    """Return the active context, or ``None`` when no chat is in flight."""
    return _active.get()

"""Interactive OAuth flow bridge between the MCP event loop and Django views.

The SDK's ``OAuthClientProvider`` drives the authorization-code flow with two
async hooks that run on the MCP manager's background loop:

- ``redirect_handler(url)`` — the user must open ``url`` in a browser;
- ``callback_handler() -> (code, state)`` — awaited until the authorization
  server redirects back to us.

The browser redirect lands on the Django view ``GET /api/mcp/oauth/callback``
(a different thread). This module is the bridge: a per-server ``PendingFlow``
holds an ``asyncio.Future`` on the manager loop; the redirect handler parses
the OAuth ``state`` out of the authorization URL and indexes the flow by it;
the Django view resolves the future thread-safely via
``loop.call_soon_threadsafe``. Single-flight per server: starting a new flow
cancels a stale one. Flows expire after ``FLOW_TTL_S``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

FLOW_TTL_S = 600.0  # a consent screen left open longer than this is abandoned


@dataclass
class PendingFlow:
    server_name: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future  # resolves to (code, state)
    created_at: float = field(default_factory=time.monotonic)
    authorization_url: str | None = None
    state: str | None = None
    # Set once the redirect handler has published the authorization URL —
    # the connect endpoint waits on this to answer "auth_required" quickly.
    url_ready: threading.Event = field(default_factory=threading.Event)
    # Terminal error from the background connect task (shown in server status).
    error: str | None = None

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.created_at > FLOW_TTL_S


_LOCK = threading.Lock()
_BY_SERVER: dict[str, PendingFlow] = {}
_BY_STATE: dict[str, PendingFlow] = {}
# Last terminal auth error per server (shown in the Toolkit until a new attempt).
_LAST_ERROR: dict[str, str] = {}


def _prune_locked() -> None:
    for state in [s for s, f in _BY_STATE.items() if f.expired]:
        _BY_STATE.pop(state, None)
    for name in [n for n, f in _BY_SERVER.items() if f.expired]:
        flow = _BY_SERVER.pop(name)
        flow.loop.call_soon_threadsafe(_cancel_future, flow.future)


def _cancel_future(future: asyncio.Future) -> None:
    if not future.done():
        future.cancel()


def begin_flow(server_name: str, loop: asyncio.AbstractEventLoop) -> PendingFlow:
    """Register a fresh flow for a server (cancelling any stale one)."""
    with _LOCK:
        _prune_locked()
        old = _BY_SERVER.pop(server_name, None)
        if old is not None:
            if old.state:
                _BY_STATE.pop(old.state, None)
            old.loop.call_soon_threadsafe(_cancel_future, old.future)
        _LAST_ERROR.pop(server_name, None)
        flow = PendingFlow(server_name=server_name, loop=loop, future=loop.create_future())
        _BY_SERVER[server_name] = flow
        return flow


def publish_authorization_url(flow: PendingFlow, url: str) -> None:
    """Called by the redirect handler (manager loop) once the URL is known."""
    flow.authorization_url = url
    try:
        flow.state = (parse_qs(urlparse(url).query).get("state") or [None])[0]
    except ValueError:
        flow.state = None
    with _LOCK:
        if flow.state:
            _BY_STATE[flow.state] = flow
    flow.url_ready.set()


def resolve_callback(state: str, code: str) -> PendingFlow | None:
    """Resolve the flow for an OAuth callback (Django view thread).

    Returns the flow on success, None for an unknown/expired state.
    """
    with _LOCK:
        _prune_locked()
        flow = _BY_STATE.pop(state, None)
        if flow is not None and _BY_SERVER.get(flow.server_name) is flow:
            _BY_SERVER.pop(flow.server_name, None)
    if flow is None:
        return None

    def _set() -> None:
        if not flow.future.done():
            flow.future.set_result((code, state))

    flow.loop.call_soon_threadsafe(_set)
    return flow


def fail_by_state(state: str, error: str) -> PendingFlow | None:
    """Fail the flow for a callback that carries `error` (e.g. denied consent)."""
    with _LOCK:
        flow = _BY_STATE.pop(state, None)
        if flow is not None and _BY_SERVER.get(flow.server_name) is flow:
            _BY_SERVER.pop(flow.server_name, None)
    if flow is not None:
        flow.error = error
        with _LOCK:
            _LAST_ERROR[flow.server_name] = error
        flow.loop.call_soon_threadsafe(_cancel_future, flow.future)
    return flow


def fail_flow(flow: PendingFlow, error: str) -> None:
    """Mark ONE flow failed (denied consent / connect error).

    Identity-guarded: a superseded attempt failing late (its future was
    cancelled by a newer ``begin_flow``) must not pop, cancel, or error-mark
    the newer flow registered under the same server name — keying by name
    alone caused exactly that cascade in the first live run (retry #2's
    pending consent killed by retry #1's death rattle).
    """
    flow.error = error
    with _LOCK:
        is_current = _BY_SERVER.get(flow.server_name) is flow
        if is_current:
            _BY_SERVER.pop(flow.server_name, None)
        if flow.state and _BY_STATE.get(flow.state) is flow:
            _BY_STATE.pop(flow.state, None)
        superseded = (not is_current) and flow.server_name in _BY_SERVER
        if not superseded:  # don't clobber a live retry's clean slate
            _LAST_ERROR[flow.server_name] = error
    flow.loop.call_soon_threadsafe(_cancel_future, flow.future)


def last_error(server_name: str) -> str | None:
    with _LOCK:
        return _LAST_ERROR.get(server_name)


def get_flow(server_name: str) -> PendingFlow | None:
    with _LOCK:
        _prune_locked()
        return _BY_SERVER.get(server_name)


def finish_flow(flow: PendingFlow) -> None:
    """Drop ONE flow's bookkeeping after its connect settles (identity-guarded)."""
    with _LOCK:
        if _BY_SERVER.get(flow.server_name) is flow:
            _BY_SERVER.pop(flow.server_name, None)
        if flow.state and _BY_STATE.get(flow.state) is flow:
            _BY_STATE.pop(flow.state, None)

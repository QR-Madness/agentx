"""Quiet one specific, benign asyncio log.

Python 3.14's ``asyncio.shield`` reports ``"<Exc> exception in shielded future"``
through the loop's exception handler when a *shielded* inner future finishes with an
exception after the outer task was cancelled. In practice this fires when a client
**aborts an in-flight request** — e.g. barge-in stops a text-to-speech fetch, or you
navigate away mid request — and the (now-discarded) HTTP call inside the OpenAI/httpx
stack is cancelled. Nothing is wrong: the result was thrown away on purpose. But it
logs at ERROR, which reads like a real failure.

``install()`` swaps in a loop exception handler that downgrades exactly that case
(``CancelledError`` + ``"shielded future"`` in the message) to debug and **delegates
everything else** to the previous handler — so real errors are untouched. Idempotent;
safe to call from a request path (installs on the running loop, once)."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_installed = False


def install() -> None:
    """Install the benign-cancellation filter on the running event loop (once)."""
    global _installed
    if _installed:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no running loop here — nothing to filter
    previous = loop.get_exception_handler()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        message = context.get("message", "")
        if isinstance(exc, asyncio.CancelledError) and "shielded future" in message:
            logger.debug("Ignoring benign cancelled-request noise: %s", message)
            return
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    _installed = True

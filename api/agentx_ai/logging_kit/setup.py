"""``configure_logging()`` — wire the whole pipeline.

Topology::

    every logger ─► root ─► QueueHandler ──(queue)──► QueueListener thread
                              │                          ├─ console (rich/plain/json)
                              │                          ├─ ring buffer (API)
                              ContextFilter               └─ archive (Phase 6)
                              RedactionFilter

The filters sit on the **QueueHandler** so they run in the *emitting* thread
(where the run-id ContextVar is visible and where redaction happens once); the
expensive console render + IO happen once, off-thread, in the listener. Call this
from ``settings`` with ``LOGGING_CONFIG = None`` so Django doesn't re-configure.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue

from .context import ContextFilter
from .flags import LogFlags, read_flags
from .handler import build_console_handler
from .redaction import RedactionFilter
from .ring_buffer import RingBufferHandler, set_ring_handler

_NOISY_THIRD_PARTY = ("httpx", "httpcore", "neo4j", "urllib3", "asyncio", "openai", "anthropic")

_configured = False
_listener: logging.handlers.QueueListener | None = None


class _PassThroughQueueHandler(logging.handlers.QueueHandler):
    """Enqueue the record unchanged.

    The stock ``QueueHandler.prepare`` pre-formats the message and drops
    ``exc_info`` (needed for pickling across processes). We never cross a process
    boundary, so pass the record through intact — this preserves ``exc_info`` for
    rich tracebacks and ``args`` for downstream formatters.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return record


def configure_logging(flags: LogFlags | None = None) -> None:
    global _configured, _listener
    if _configured:
        return
    flags = flags or read_flags()

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    targets: list[logging.Handler] = [build_console_handler(flags)]

    if flags.api_enabled or flags.archive_enabled:
        ring = RingBufferHandler(capacity=flags.ring_size)
        set_ring_handler(ring)
        targets.append(ring)

    # Phase 6 archive handler is appended here (see archive.attach_archive).
    try:
        from .archive import build_archive_handler

        archive = build_archive_handler(flags)
        if archive is not None:
            targets.append(archive)
    except Exception:  # noqa: BLE001 — archive is best-effort, never block logging
        pass

    log_queue: queue.Queue = queue.Queue(-1)
    qh = _PassThroughQueueHandler(log_queue)
    qh.addFilter(ContextFilter())
    qh.addFilter(RedactionFilter())
    root.addHandler(qh)
    root.setLevel(logging.INFO)

    _listener = logging.handlers.QueueListener(log_queue, *targets, respect_handler_level=True)
    _listener.start()
    atexit.register(_stop_listener)

    logging.getLogger("agentx_ai").setLevel(logging.DEBUG)
    logging.getLogger("django").setLevel(logging.INFO)
    for name in _NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def _stop_listener() -> None:
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:  # noqa: BLE001
            pass
        _listener = None

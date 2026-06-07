"""AgentX logging kit — centralized, color-coded, run-correlated logging.

A single ``configure_logging()`` call (from ``settings``) transforms every
existing ``logging.getLogger(__name__)`` call site: category badges, semantic
highlighting, per-turn run tags, a startup banner, compact LLM cards, an
in-memory ring buffer behind ``/api/logs``, and a compressed on-disk archive.
All decorations are governed by ``AGENTX_LOG_*`` flags (decorations on by
default; off restores the historical plain output).
"""

from .flags import LogFlags, read_flags
from .setup import configure_logging

__all__ = ["configure_logging", "read_flags", "LogFlags"]

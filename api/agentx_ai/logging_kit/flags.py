"""Typed reads of the ``AGENTX_LOG_*`` environment flags.

Kept import-light (stdlib only) so it can be imported from ``settings.py`` while
Django is still wiring up. Every knob defaults so the platform behaves well with
no configuration: decorations ON, archive ON, log API ON.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

LlmLogLevel = Literal["off", "summary", "full"]
LogFormat = Literal["pretty", "json"]

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _legacy_llm_debug() -> bool:
    """The pre-existing ``DEBUG_LOG_LLM_REQUESTS`` switch (full-dump)."""
    return os.environ.get("DEBUG_LOG_LLM_REQUESTS", "").strip().lower() not in _FALSE


@dataclass(frozen=True)
class LogFlags:
    decorations: bool
    banner: bool
    force_color: bool
    llm_level: LlmLogLevel
    fmt: LogFormat
    api_enabled: bool
    ring_size: int
    archive_enabled: bool
    archive_max_mb: int
    archive_backups: int


def read_flags() -> LogFlags:
    """Snapshot the current environment into a typed flags object."""
    decorations = _env_bool("AGENTX_LOG_DECORATIONS", True)

    # LLM level: explicit env wins; else legacy full-dump switch; else a quiet
    # summary when decorations are on, fully off otherwise.
    raw_level = os.environ.get("AGENTX_LLM_LOG_LEVEL", "").strip().lower()
    if raw_level in ("off", "summary", "full"):
        llm_level: LlmLogLevel = raw_level  # type: ignore[assignment]
    elif _legacy_llm_debug():
        llm_level = "full"
    else:
        llm_level = "summary" if decorations else "off"

    raw_fmt = os.environ.get("AGENTX_LOG_FORMAT", "").strip().lower()
    fmt: LogFormat = "json" if raw_fmt == "json" else "pretty"

    return LogFlags(
        decorations=decorations,
        banner=_env_bool("AGENTX_LOG_BANNER", decorations),
        force_color=_env_bool("AGENTX_LOG_FORCE_COLOR", False),
        llm_level=llm_level,
        fmt=fmt,
        api_enabled=_env_bool("AGENTX_LOG_API_ENABLED", True),
        ring_size=_env_int("AGENTX_LOG_RING_SIZE", 2000),
        archive_enabled=_env_bool("AGENTX_LOG_ARCHIVE_ENABLED", True),
        archive_max_mb=_env_int("AGENTX_LOG_ARCHIVE_MAX_MB", 10),
        archive_backups=_env_int("AGENTX_LOG_ARCHIVE_BACKUPS", 10),
    )

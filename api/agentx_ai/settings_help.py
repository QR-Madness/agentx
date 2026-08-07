"""
Per-setting help — authored once, rendered everywhere.

``settings_help.yaml`` is the single source for what a setting *is*, how it
works, why you'd change it, and how to manage it. The settings manifest serves
it to the client (which renders it beside the control) and
``scripts/gen_settings_reference.py`` renders it into the docs site. Neither
side authors prose of its own, so the UI and the documentation cannot disagree.

Keys are ``store`` → ``key``: dotted paths for the config store
(``search.max_results``), bare field names for the memory store
(``recall_candidate_pool``), and settings-screen ids for the ``sections`` store
(``memory-recall``), which describes a whole screen rather than one key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HELP_PATH = Path(__file__).parent / "settings_help.yaml"

#: The prose fields a fully-authored setting carries, in reading order.
HELP_FIELDS: tuple[str, ...] = ("summary", "what", "how", "why", "manage")

#: The ``sections`` store answers a smaller question — a screen has no "how it
#: works at request time" and nothing to manage — so it carries three fields.
SECTION_HELP_FIELDS: tuple[str, ...] = ("summary", "what", "why")

SECTIONS_STORE = "sections"

_cache: dict[str, dict[str, dict[str, str]]] | None = None


def _fields_for(store: str) -> tuple[str, ...]:
    return SECTION_HELP_FIELDS if store == SECTIONS_STORE else HELP_FIELDS


def _load() -> dict[str, dict[str, dict[str, str]]]:
    """Parse the help file. Never raises — help is presentation, so a broken
    file degrades to "no help" rather than taking the manifest down with it."""
    global _cache
    if _cache is not None:
        return _cache

    data: dict[str, Any] = {}
    try:
        import yaml

        if HELP_PATH.exists():
            with open(HELP_PATH, encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover — defensive
        logger.error(f"settings help: failed to load {HELP_PATH}: {exc}")
        data = {}

    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for store, keys in data.items():
        if not isinstance(keys, dict):
            continue
        wanted = _fields_for(str(store))
        store_entries: dict[str, dict[str, str]] = {}
        for key, entry in keys.items():
            if not isinstance(entry, dict):
                continue
            fields = {
                name: str(entry[name]).strip()
                for name in wanted
                if entry.get(name)
            }
            if fields:
                store_entries[str(key)] = fields
        normalized[str(store)] = store_entries

    _cache = normalized
    return _cache


def get_help(store: str, key: str) -> dict[str, str] | None:
    """Authored help for one setting, or None when it hasn't been written yet."""
    return _load().get(store, {}).get(key) or None


def get_section_help(section_id: str) -> dict[str, str] | None:
    """Authored blurb for a settings screen, or None if unwritten."""
    return _load().get(SECTIONS_STORE, {}).get(section_id) or None


def all_help() -> dict[str, dict[str, dict[str, str]]]:
    """The whole help corpus, keyed store → key → field."""
    return _load()


def documented_keys(store: str) -> set[str]:
    return set(_load().get(store, {}))


def reset_help_cache() -> None:
    """Drop the parsed cache (tests, and the docs generator between runs)."""
    global _cache
    _cache = None

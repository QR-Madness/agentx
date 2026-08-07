"""
Settings Manifest v3 — canonical, machine-readable registry of every
user-tunable setting across the platform's stores.

One endpoint that answers, for any setting: what exists, where it lives, its
type, default, current value (secrets redacted), how it can be changed, which
surface renders it, what values are legal, and — in prose — what it does and
when you'd touch it. The client renders help from here rather than carrying its
own copy, and ``scripts/gen_settings_reference.py`` generates the docs page from
the same source, so the UI and the documentation cannot drift apart.

Stores covered:
- ``memory`` — the memory kit's pydantic ``Settings`` (~150 keys; overrides in
  ``data/memory_settings.json``), split between the two ``/api/memory/*``
  settings endpoints by their key whitelists.
- ``config`` — ``DEFAULT_CONFIG`` leaves (``data/config.json``).

Writability, constraints, tiers and null/empty semantics all derive from
``settings_registry.CONFIG_SECTIONS`` — the same declaration
``views.config_update`` walks — so this can no longer claim a key is writable
that the write path drops, or vice versa. Prose comes from
``settings_help.yaml``. Nothing here is authored twice.

v1 shipped registry-only and deferred prose descriptions to the docs-site plus
validation ranges to the UI. v2 reverses that: the manifest is the source and
both surfaces render from it. v3 adds a ``sections`` block — one entry per
settings screen, with its authored blurb and how many settings it holds — so the
Overview and the generated reference can describe a screen without either of
them counting its contents by hand.
"""

import logging
from datetime import datetime, UTC
from typing import Any

from .settings_help import get_help, get_section_help

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 3

def _is_secret(name: str) -> bool:
    from .settings_registry import is_secret_path

    return is_secret_path(name)


def _redact(name: str, value: Any) -> Any:
    if not _is_secret(name):
        return value
    return "***" if value else ""


def _jsonable(value: Any) -> Any:
    """Best-effort plain-JSON projection of a default/current value."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _apply_spec(entry: dict[str, Any], spec: Any, *, ui_section: str | None) -> None:
    """Attach the v2 axes to an entry.

    Only declared axes are emitted — an entry with no tier carries no ``tier``
    key rather than a null, so a consumer can tell "not classified" from
    "classified as nothing".
    """
    if ui_section:
        entry["ui_section"] = ui_section

    if spec is not None:
        if spec.nullable:
            entry["nullable"] = True
        if spec.empty_means:
            entry["empty_means"] = spec.empty_means
        if spec.whole_write:
            entry["write_mode"] = "whole"
        if spec.tier:
            entry["tier"] = spec.tier
        if spec.secret is not None:
            entry["secret"] = spec.secret
        constraints = {
            k: v
            for k, v in (
                ("min", spec.min), ("max", spec.max), ("step", spec.step),
                ("enum", list(spec.enum) if spec.enum else None), ("unit", spec.unit),
            )
            if v is not None
        }
        if constraints:
            entry["constraints"] = constraints

    help_entry = get_help(entry["store"], entry["key"])
    if help_entry:
        entry["help"] = help_entry


def _type_name(value: Any, annotation: Any = None) -> str:
    if annotation is not None:
        name = getattr(annotation, "__name__", None)
        if name:
            return name
        return str(annotation)
    return type(value).__name__


def _flatten_config(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten DEFAULT_CONFIG to leaves.

    A dict declared ``whole_write`` in the registry stops the descent: it is
    written as one value, so it is one entry here rather than a set of leaves
    the client could be misled into patching individually.
    """
    from .settings_registry import whole_write_paths

    atomic = set(whole_write_paths())

    leaves: list[tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v and path not in atomic:
            leaves.extend(_flatten_config(v, path + "."))
        else:
            leaves.append((path, v))
    return leaves


def _memory_entries() -> list[dict[str, Any]]:
    from .kit.agent_memory.config import (
        CONSOLIDATION_READONLY_KEYS,
        Settings,
        get_consolidation_settings,
        get_recall_settings,
        get_settings,
    )
    from .model_roles import ROLE_MEMBERS
    from .settings_registry import memory_key_spec

    current = get_settings()
    # Mirror the POST handlers: a key the GET builder shows for display only is
    # not writable, and saying so here is what keeps the manifest honest.
    consolidation_keys = set(get_consolidation_settings().keys()) - CONSOLIDATION_READONLY_KEYS
    recall_keys = set(get_recall_settings().keys())
    role_by_source = {
        meta["source"]: (member, meta["role"])
        for member, meta in ROLE_MEMBERS.items()
        if meta["kind"] == "memory"
    }

    entries: list[dict[str, Any]] = []
    for name, field in Settings.model_fields.items():
        if name in recall_keys:
            writable_via = "/api/memory/recall-settings"
            ui_section = "memory-recall"
        elif name in consolidation_keys:
            writable_via = "/api/memory/settings"
            ui_section = "memory-consolidation"
        else:
            # Connection/embedding/workspace plumbing — .env / settings-file only.
            writable_via = None
            ui_section = None
        entry: dict[str, Any] = {
            "key": name,
            "store": "memory",
            "type": _type_name(field.default, field.annotation),
            "default": _redact(name, _jsonable(field.default)),
            "value": _redact(name, _jsonable(getattr(current, name, None))),
            "secret": _is_secret(name),
            "writable_via": writable_via,
        }
        _apply_spec(entry, memory_key_spec(name), ui_section=ui_section)
        if name in role_by_source:
            entry["role_member"], entry["role"] = role_by_source[name]
        entries.append(entry)
    return entries


def _config_entries() -> list[dict[str, Any]]:
    from .config import DEFAULT_CONFIG, get_config_manager
    from .model_roles import ROLE_MEMBERS
    from .settings_registry import config_key_spec, config_ui_section, config_writable_via

    cfg = get_config_manager()
    role_by_source = {
        meta["source"]: (member, meta["role"])
        for member, meta in ROLE_MEMBERS.items()
        if meta["kind"] == "config"
    }

    entries: list[dict[str, Any]] = []
    for path, default in _flatten_config(DEFAULT_CONFIG):
        entry: dict[str, Any] = {
            "key": path,
            "store": "config",
            "type": _type_name(default),
            "default": _redact(path, _jsonable(default)),
            "value": _redact(path, _jsonable(cfg.get(path, default))),
            "secret": _is_secret(path),
            "writable_via": config_writable_via(path),
        }
        _apply_spec(entry, config_key_spec(path), ui_section=config_ui_section(path))
        if path in role_by_source:
            entry["role_member"], entry["role"] = role_by_source[path]
        entries.append(entry)
    return entries


def build_sections(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per settings screen: its label, its authored blurb, and how
    many settings it holds.

    Counts are derived from the entries just built, never tallied by hand, so a
    screen's "22 settings" moves the moment a key joins or leaves it. Screens
    that own nothing (the template library, themes) report zero rather than
    being omitted — the Overview lists every screen, and a missing count would
    read as a load failure rather than as "this one has no knobs".
    """
    from .settings_registry import ALL_SECTIONS

    writable = [e for e in entries if e.get("writable_via")]
    counts: dict[str, int] = {}
    for entry in writable:
        section = entry.get("ui_section")
        if section:
            counts[section] = counts.get(section, 0) + 1

    sections: list[dict[str, Any]] = []
    for section_id, label in ALL_SECTIONS.items():
        section: dict[str, Any] = {
            "id": section_id,
            "label": label,
            "writable_count": counts.get(section_id, 0),
        }
        blurb = get_section_help(section_id)
        if blurb:
            section["help"] = blurb
        sections.append(section)
    return sections


def build_manifest() -> dict[str, Any]:
    """Assemble the full manifest. Never raises — a store that fails to
    introspect is reported in `errors` instead of breaking the endpoint."""
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in (_memory_entries, _config_entries):
        try:
            entries.extend(source())
        except Exception as e:  # pragma: no cover — defensive
            logger.error(f"settings manifest: {source.__name__} failed: {e}")
            errors.append(f"{source.__name__}: {e}")

    counts = {"total": len(entries)}
    for entry in entries:
        counts[entry["store"]] = counts.get(entry["store"], 0) + 1
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": counts,
        "sections": build_sections(entries),
        "entries": entries,
    }
    if errors:
        manifest["errors"] = errors
    return manifest


def build_reference_entries() -> list[dict[str, Any]]:
    """Declared metadata for every setting, with no live values — what the docs
    generator renders.

    `build_manifest` reads the running install's config so the UI can show what
    you've actually set. A committed docs page must not: it would bake one
    machine's values into the repo and make the generated file depend on whose
    laptop produced it. This builds the same entries from declarations alone —
    defaults, types, bounds, tiers, help — and omits `value` entirely.
    """
    from .config import DEFAULT_CONFIG
    from .kit.agent_memory.config import (
        CONSOLIDATION_READONLY_KEYS,
        Settings,
        get_consolidation_settings,
        get_recall_settings,
    )
    from .settings_registry import (
        config_key_spec,
        config_ui_section,
        config_writable_via,
        memory_key_spec,
    )

    consolidation_keys = set(get_consolidation_settings().keys()) - CONSOLIDATION_READONLY_KEYS
    recall_keys = set(get_recall_settings().keys())

    entries: list[dict[str, Any]] = []

    for name, field in Settings.model_fields.items():
        if name in recall_keys:
            writable_via, ui_section = "/api/memory/recall-settings", "memory-recall"
        elif name in consolidation_keys:
            writable_via, ui_section = "/api/memory/settings", "memory-consolidation"
        else:
            writable_via, ui_section = None, None
        entry: dict[str, Any] = {
            "key": name,
            "store": "memory",
            "type": _type_name(field.default, field.annotation),
            "default": _redact(name, _jsonable(field.default)),
            "secret": _is_secret(name),
            "writable_via": writable_via,
        }
        _apply_spec(entry, memory_key_spec(name), ui_section=ui_section)
        entries.append(entry)

    for path, default in _flatten_config(DEFAULT_CONFIG):
        entry = {
            "key": path,
            "store": "config",
            "type": _type_name(default),
            "default": _redact(path, _jsonable(default)),
            "secret": _is_secret(path),
            "writable_via": config_writable_via(path),
        }
        _apply_spec(entry, config_key_spec(path), ui_section=config_ui_section(path))
        entries.append(entry)

    return entries

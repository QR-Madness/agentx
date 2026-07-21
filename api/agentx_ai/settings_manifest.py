"""
Settings Manifest v1 — canonical, machine-readable registry of every
user-tunable setting across the platform's stores.

The keystone substrate for the future settings agent (explain settings,
suggest configurations, validate changes — todo/backlog/genome-advisor.md):
one endpoint a model can read to learn what exists, where it lives, its type,
default, current value (secrets redacted), how it can be changed, and whether
a model role governs it.

Stores covered:
- ``memory`` — the memory kit's pydantic ``Settings`` (~150 keys; overrides in
  ``data/memory_settings.json``), split between the two ``/api/memory/*``
  settings endpoints by their key whitelists.
- ``config`` — ``DEFAULT_CONFIG`` leaves (``data/config.json``); writability
  mirrors the ``config_update`` section handlers (that view stays the source
  of truth — extend ``_CONFIG_WRITE_ROUTES`` when it grows a section).

v1 is deliberately registry-only: no prose descriptions (the docs-site is the
narrative source until key-level descriptions are authored) and no validation
ranges — both are the manifest's planned v2 axes.
"""

import logging
from datetime import datetime, UTC
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

# Any key whose name smells like a credential is value-redacted. URIs redact
# too — connection strings embed passwords (postgres_uri).
_SECRET_MARKERS = ("key", "password", "token", "secret", "uri")

# config_update's writable surface, mirrored per section root. Nested tuples
# are the per-key whitelists where the handler has one; True = every sub-key
# under the root is writable. Keep in lockstep with views.config_update.
_CONFIG_WRITE_ROUTES: dict[str, Any] = {
    "providers": True,
    "preferences": True,
    "llm_settings": True,
    "context_limits": True,
    "prompt_enhancement": True,
    "planner": True,
    "search": ("backend", "fallback_enabled", "max_results",
               "cache_ttl_seconds", "tavily_api_key", "brave_api_key"),
    "alloy": ("allow_adhoc_delegation", "max_parallel_delegations",
              "max_delegation_depth", "delegation_timeout_seconds",
              "non_blocking_delegations", "chain_of_command"),
    "ambassador": True,
    "images": ("enabled", "default_model", "avatar_model", "avatar_style_prompt"),
    "vision": ("enabled", "refeed_recent_turns"),
    "models": ("roles.fast_utility", "roles.deep_reasoning", "roles.summarizer"),
    # The Conversation Context settings section (one home for the in-conversation
    # context techniques).
    "context": ("summary_trigger_ratio", "verbatim_budget_ratio", "recent_floor",
                "preassembly_summary_enabled", "conversation_state_enabled",
                "conversation_state_compaction_enabled", "rehydrate_max_turns",
                "max_input_tokens"),
    "session": ("rolling_summary.enabled", "rolling_summary.model",
                "rolling_summary.max_tokens"),
    "trajectory_compression": ("enabled", "threshold_ratio",
                               "preserve_recent_rounds", "model",
                               "max_knowledge_chars"),
    "compression": ("enabled", "model", "max_summary_chars"),
    "memory": ("episodic_leads_enabled", "project_channels"),
    # Thinking Patterns (Settings → Intelligence → Thinking).
    "reasoning": ("chat_patterns_enabled", "auto_classifier_enabled",
                  "classifier_model", "classifier_min_chars",
                  "step_back_model", "step_back_timeout_seconds",
                  "cot_enabled", "step_back_enabled", "reflection_enabled",
                  "self_consistency_enabled", "sc_model", "sc_k",
                  "min_output_tokens"),
}


def _is_secret(name: str) -> bool:
    n = name.lower()
    return any(marker in n for marker in _SECRET_MARKERS)


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


def _type_name(value: Any, annotation: Any = None) -> str:
    if annotation is not None:
        name = getattr(annotation, "__name__", None)
        if name:
            return name
        return str(annotation)
    return type(value).__name__


def _config_writable_via(path: str) -> str | None:
    root, _, rest = path.partition(".")
    route = _CONFIG_WRITE_ROUTES.get(root)
    if route is True:
        return "/api/config/update"
    if isinstance(route, tuple):
        # Match either the immediate key or an explicit nested path.
        head = rest.partition(".")[0]
        if rest in route or head in route:
            return "/api/config/update"
    return None


def _flatten_config(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            leaves.extend(_flatten_config(v, path + "."))
        else:
            leaves.append((path, v))
    return leaves


def _memory_entries() -> list[dict[str, Any]]:
    from .kit.agent_memory.config import (
        Settings,
        get_consolidation_settings,
        get_recall_settings,
        get_settings,
    )
    from .model_roles import ROLE_MEMBERS

    current = get_settings()
    consolidation_keys = set(get_consolidation_settings().keys())
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
        elif name in consolidation_keys:
            writable_via = "/api/memory/settings"
        else:
            # Connection/embedding/workspace plumbing — .env / settings-file only.
            writable_via = None
        entry: dict[str, Any] = {
            "key": name,
            "store": "memory",
            "type": _type_name(field.default, field.annotation),
            "default": _redact(name, _jsonable(field.default)),
            "value": _redact(name, _jsonable(getattr(current, name, None))),
            "secret": _is_secret(name),
            "writable_via": writable_via,
        }
        if name in role_by_source:
            entry["role_member"], entry["role"] = role_by_source[name]
        entries.append(entry)
    return entries


def _config_entries() -> list[dict[str, Any]]:
    from .config import DEFAULT_CONFIG, get_config_manager
    from .model_roles import ROLE_MEMBERS

    cfg = get_config_manager()
    role_by_source = {
        meta["source"]: (member, meta["role"])
        for member, meta in ROLE_MEMBERS.items()
        if meta["kind"] == "config"
    }
    # trajectory_compression.* is now written directly via /api/config/update
    # (Settings → Conversation Context); the legacy memory-settings bridge
    # (trajectory_compression_* keys on /api/memory/settings) still accepts
    # writes for back-compat but is no longer the canonical route.

    entries: list[dict[str, Any]] = []
    for path, default in _flatten_config(DEFAULT_CONFIG):
        writable_via = _config_writable_via(path)
        entry: dict[str, Any] = {
            "key": path,
            "store": "config",
            "type": _type_name(default),
            "default": _redact(path, _jsonable(default)),
            "value": _redact(path, _jsonable(cfg.get(path, default))),
            "secret": _is_secret(path),
            "writable_via": writable_via,
        }
        if path in role_by_source:
            entry["role_member"], entry["role"] = role_by_source[path]
        entries.append(entry)
    return entries


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
        "entries": entries,
    }
    if errors:
        manifest["errors"] = errors
    return manifest

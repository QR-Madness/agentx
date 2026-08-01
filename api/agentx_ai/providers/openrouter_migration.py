"""Find and repair model references that OpenRouter's alias rename broke.

OpenRouter moved its "always the newest" aliases to a **tilde prefix**: the
catalog lists ``~anthropic/claude-sonnet-latest`` and the plain
``anthropic/claude-sonnet-latest`` is gone. Ids stored before that rename still
*run* — OpenRouter resolves them server-side — but they miss our catalog cache,
so capabilities fall through to conservative defaults. An 8192 window reported
for a 1M model shows up much later as premature compaction and "spotty memory".

``openrouter_provider.resolve_catalog_id`` fixes the *reading* half at lookup
time. This module fixes the *stored* half — and does so *opt-in*, because these
values are the user's own model choices. Silently rewriting them would be
exactly the "don't change the user's defaults" failure: the scan reports what it
would change, and nothing moves until the user says so.

A reference is only ever proposed when the replacement is **verified present in
the live catalog** — we never suggest rewriting one broken id into another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER = "openrouter"


@dataclass
class StaleRef:
    """One stored model id that should gain the ``~`` prefix."""

    #: Where it lives — `profile`, `config` or `memory_settings`.
    store: str
    #: Machine-addressable location (profile id, config dot-path, settings field).
    location: str
    #: Human label for the confirmation list.
    label: str
    current: str
    suggested: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "location": self.location,
            "label": self.label,
            "current": self.current,
            "suggested": self.suggested,
        }


@dataclass
class MigrationReport:
    refs: list[StaleRef] = field(default_factory=list)
    #: True when the catalog couldn't be read — an empty result then means
    #: "don't know", not "nothing to fix", and the UI must say so.
    catalog_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.refs),
            "catalog_available": self.catalog_available,
            "refs": [ref.to_dict() for ref in self.refs],
        }


def _candidate(model: str | None) -> str | None:
    """The `~`-prefixed form of an un-prefixed OpenRouter `*-latest` id."""
    if not model or not isinstance(model, str):
        return None
    if not model.startswith(f"{PROVIDER}:"):
        return None
    _, _, model_id = model.partition(":")
    if not model_id.endswith("-latest") or model_id.startswith("~"):
        return None
    return f"{PROVIDER}:~{model_id}"


def _catalog_ids(registry: Any) -> set[str] | None:
    """Model ids OpenRouter currently lists, or None when unreadable."""
    try:
        provider = registry.get_provider(PROVIDER)
    except Exception as e:  # noqa: BLE001 — unconfigured is a normal state
        logger.debug(f"OpenRouter not configured; skipping alias scan: {e}")
        return None
    listed = provider.list_models()
    # An empty catalog means "not fetched yet", not "no models" — treating it as
    # authoritative would report every alias as unfixable.
    return set(listed) if listed else None


async def scan_async(registry: Any = None, *, cfg: Any = None) -> MigrationReport:
    """``scan`` with the catalog warmed first.

    A cold provider lists nothing, which this module correctly reports as "can't
    check" rather than "all clear" — but that would make the scan useless on the
    first call of every process. Warming first is what makes the honest default
    safe to keep.
    """
    from ..providers.registry import get_registry

    registry = registry or get_registry()
    try:
        provider = registry.get_provider(PROVIDER)
        if not provider.list_models():
            await provider.fetch_models()
    except Exception as e:  # noqa: BLE001 — unconfigured/unreachable is normal
        logger.debug(f"Could not warm the OpenRouter catalog for the alias scan: {e}")
    return scan(registry, cfg=cfg)


def scan(registry: Any = None, *, cfg: Any = None) -> MigrationReport:
    """Report every stored OpenRouter `*-latest` id whose `~` form exists.

    Reads agent profiles, the global model settings, and the memory-stage model
    fields. Never writes.
    """
    from ..providers.registry import get_registry

    registry = registry or get_registry()
    catalog = _catalog_ids(registry)
    if catalog is None:
        return MigrationReport(catalog_available=False)

    def usable(model: str | None) -> str | None:
        suggested = _candidate(model)
        if suggested and suggested.partition(":")[2] in catalog:
            return suggested
        return None

    refs: list[StaleRef] = []
    refs.extend(_scan_profiles(usable))
    refs.extend(_scan_config(usable, cfg))
    refs.extend(_scan_memory_settings(usable))
    return MigrationReport(refs=refs)


def _scan_profiles(usable: Any) -> list[StaleRef]:
    """Agent profiles' own model choices."""
    from ..agent.profiles import get_profile_manager

    fields = (
        ("default_model", "model"),
        ("speech_model", "voice model"),
        ("transcription_model", "transcription model"),
    )
    refs: list[StaleRef] = []
    try:
        profiles = get_profile_manager().list_profiles()
    except Exception as e:  # noqa: BLE001 — a scan never breaks the caller
        logger.warning(f"Could not scan profiles for stale model ids: {e}")
        return refs

    for profile in profiles:
        for attr, noun in fields:
            suggested = usable(getattr(profile, attr, None))
            if suggested:
                refs.append(StaleRef(
                    store="profile",
                    location=f"{profile.id}.{attr}",
                    label=f"{profile.name} — {noun}",
                    current=getattr(profile, attr),
                    suggested=suggested,
                ))
    return refs


#: Global model settings that hold a `provider:model` reference, by dot-path.
_CONFIG_MODEL_PATHS: tuple[tuple[str, str], ...] = (
    ("preferences.default_model", "Default model"),
    ("models.defaults.chat", "Default — chat"),
    ("models.defaults.reasoning", "Default — reasoning"),
    ("models.defaults.extraction", "Default — extraction"),
    ("models.roles.fast_utility", "Role — fast utility"),
    ("models.roles.deep_reasoning", "Role — deep reasoning"),
    ("models.roles.summarizer", "Role — summarizer"),
    ("images.default_model", "Images — default model"),
)


def _scan_config(usable: Any, cfg: Any = None) -> list[StaleRef]:
    """Global settings plus every config-backed role member."""
    from ..config import get_config_manager
    from ..model_roles import ROLE_MEMBERS

    config = cfg or get_config_manager()
    paths: list[tuple[str, str]] = list(_CONFIG_MODEL_PATHS)
    # Role members whose value lives in config.json (the memory-kind ones are
    # handled by _scan_memory_settings).
    paths.extend(
        (member["source"], member["label"])
        for member in ROLE_MEMBERS.values()
        if member.get("kind") == "config"
    )

    refs: list[StaleRef] = []
    seen: set[str] = set()
    for path, label in paths:
        if path in seen:
            continue
        seen.add(path)
        suggested = usable(config.get(path))
        if suggested:
            refs.append(StaleRef(
                store="config",
                location=path,
                label=label,
                current=config.get(path),
                suggested=suggested,
            ))
    return refs


def _scan_memory_settings(usable: Any) -> list[StaleRef]:
    """Per-stage memory model fields (their own settings store)."""
    from ..model_roles import ROLE_MEMBERS

    refs: list[StaleRef] = []
    try:
        from ..kit.agent_memory.config import get_settings

        settings = get_settings()
    except Exception as e:  # noqa: BLE001 — memory settings are optional
        logger.debug(f"Memory settings unavailable for alias scan: {e}")
        return refs

    for member in ROLE_MEMBERS.values():
        if member.get("kind") != "memory":
            continue
        source = member["source"]
        suggested = usable(getattr(settings, source, None))
        if suggested:
            refs.append(StaleRef(
                store="memory_settings",
                location=source,
                label=member["label"],
                current=getattr(settings, source),
                suggested=suggested,
            ))
    return refs


def apply(refs: list[StaleRef], *, cfg: Any = None) -> dict[str, Any]:
    """Rewrite the given references. Returns per-store counts and any failures.

    Only ever called with refs the user confirmed. Each store is written through
    its own normal writer, so validation and persistence behave exactly as a
    hand edit would.
    """
    from ..config import get_config_manager

    applied: list[str] = []
    failed: list[dict[str, str]] = []

    profile_refs = [r for r in refs if r.store == "profile"]
    config_refs = [r for r in refs if r.store == "config"]
    memory_refs = [r for r in refs if r.store == "memory_settings"]

    if profile_refs:
        from ..agent.profiles import get_profile_manager

        manager = get_profile_manager()
        for ref in profile_refs:
            profile_id, _, attr = ref.location.partition(".")
            try:
                manager.update_profile(profile_id, {attr: ref.suggested})
                applied.append(ref.location)
            except Exception as e:  # noqa: BLE001 — report, don't abort the batch
                failed.append({"location": ref.location, "error": str(e)})

    if config_refs:
        config = cfg or get_config_manager()
        for ref in config_refs:
            try:
                config.set(ref.location, ref.suggested)
                applied.append(ref.location)
            except Exception as e:  # noqa: BLE001
                failed.append({"location": ref.location, "error": str(e)})
        config.save()

    if memory_refs:
        try:
            from ..kit.agent_memory.config import save_memory_settings

            save_memory_settings({r.location: r.suggested for r in memory_refs})
            applied.extend(r.location for r in memory_refs)
        except Exception as e:  # noqa: BLE001
            failed.extend(
                {"location": r.location, "error": str(e)} for r in memory_refs
            )

    if applied:
        logger.info(f"Repaired {len(applied)} OpenRouter alias reference(s): {applied}")
    return {"applied": applied, "failed": failed, "count": len(applied)}


def refs_from_payload(payload: list[dict[str, Any]]) -> list[StaleRef]:
    """Rebuild refs from a client confirmation, dropping malformed entries."""
    refs: list[StaleRef] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        store = str(item.get("store") or "")
        location = str(item.get("location") or "")
        suggested = str(item.get("suggested") or "")
        if store not in ("profile", "config", "memory_settings") or not location or not suggested:
            continue
        refs.append(StaleRef(
            store=store,
            location=location,
            label=str(item.get("label") or location),
            current=str(item.get("current") or ""),
            suggested=suggested,
        ))
    return refs

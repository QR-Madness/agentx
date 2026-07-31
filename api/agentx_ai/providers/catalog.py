"""Provider catalog — what model backends this install can reach.

Providers used to be a hardcoded five-name enum repeated in four places
(``config.DEFAULT_CONFIG``, ``registry._load_default_config``, the whitelist in
``views.config_update``, and the client's ``PROVIDERS`` array). This module is
the single definition, and it opens the list up: alongside the five **built-ins**
a user can register any number of **custom** entries pointing at an
OpenAI-compatible endpoint (Groq, Together, DeepSeek, Fireworks, xAI, Mistral,
Cerebras, Ollama, vLLM, llama.cpp, a private gateway…).

Two config shapes, deliberately kept apart:

- **Built-ins** keep their historical ``providers.<name>.*`` paths *exactly* —
  no migration, and ``get_provider_value``'s env-var fallbacks still apply.
- **Custom** entries live under ``providers.custom.<id>`` as self-describing
  records. Their ``id`` is user-facing: it's the left half of a
  ``provider:model`` reference, so ``groq:llama-3.3-70b`` resolves through the
  entry registered as ``groq``.

The section accessor reads defaults from ``DEFAULT_CONFIG`` rather than
duplicating a literal at each call site — ``ConfigManager`` does **not** merge
new defaults into an existing ``config.json``, so a literal default drifts the
moment the shipped one changes (see ``Decisions.md`` ADR-13).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import DEFAULT_CONFIG, ConfigManager, get_config_manager


class ProviderKind(str, Enum):
    """The wire protocol a provider speaks.

    Custom entries may only be ``OPENAI_COMPATIBLE`` — the other kinds name the
    bespoke built-in implementations and are not user-selectable.
    """

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LMSTUDIO = "lmstudio"
    VERCEL = "vercel"


#: The only kind a user may register. Everything else is a built-in shape.
CUSTOM_KINDS: frozenset[ProviderKind] = frozenset({ProviderKind.OPENAI_COMPATIBLE})


@dataclass(frozen=True)
class BuiltinSpec:
    """Static description of a shipped provider (values still come from config)."""

    id: str
    kind: ProviderKind
    label: str
    #: Which field decides "configured" — an API key for cloud, a URL for local.
    credential: str = "api_key"


BUILTINS: tuple[BuiltinSpec, ...] = (
    BuiltinSpec("openrouter", ProviderKind.OPENROUTER, "OpenRouter"),
    BuiltinSpec("anthropic", ProviderKind.ANTHROPIC, "Anthropic"),
    BuiltinSpec("openai", ProviderKind.OPENAI_COMPATIBLE, "OpenAI"),
    BuiltinSpec("vercel", ProviderKind.VERCEL, "Vercel AI Gateway"),
    BuiltinSpec("lmstudio", ProviderKind.LMSTUDIO, "LM Studio", credential="base_url"),
)

BUILTIN_IDS: frozenset[str] = frozenset(spec.id for spec in BUILTINS)

#: Config keys under ``providers.*`` that are not provider ids. Every child of
#: ``providers`` stays a mapping so the config redactor's shape never breaks.
RESERVED_IDS: frozenset[str] = frozenset({"custom", "policy"})

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")

#: Fields persisted for a custom entry (anything else in a payload is dropped).
_CUSTOM_FIELDS = ("kind", "label", "base_url", "api_key", "headers", "enabled")


@dataclass
class ProviderEntry:
    """One reachable backend — built-in or user-registered."""

    id: str
    kind: ProviderKind
    label: str
    base_url: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    builtin: bool = False
    #: Which field decides `configured` (mirrors BuiltinSpec.credential).
    credential: str = "api_key"

    @property
    def configured(self) -> bool:
        """Whether this entry carries enough to instantiate a provider."""
        if not self.enabled:
            return False
        return bool(self.base_url if self.credential == "base_url" else self.api_key)

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize for the API — **never** leaks the key or header values.

        The key is reduced to a fingerprint (``····3f21``) so the client can show
        "a key is set" without ever holding the secret. Header *names* survive
        because they're structural; their values are secrets by assumption.
        """
        return {
            "id": self.id,
            "kind": self.kind.value,
            "label": self.label,
            "base_url": self.base_url,
            "key_fingerprint": fingerprint(self.api_key),
            "header_names": sorted(self.headers),
            "enabled": self.enabled,
            "builtin": self.builtin,
            "credential": self.credential,
            "configured": self.configured,
        }


def fingerprint(secret: str | None) -> str | None:
    """``····3f21`` — the last four characters of a secret, or None if unset.

    Short secrets collapse to ``····`` entirely rather than exposing most of a
    weak key.
    """
    if not secret:
        return None
    tail = str(secret)[-4:]
    return f"····{tail}" if len(str(secret)) > 4 else "····"


def validate_id(provider_id: str) -> str:
    """Return a normalized custom-provider id, or raise ``ValueError``.

    Ids are lowercase and URL/CLI-safe because they appear verbatim in model
    references (``groq:llama-3.3-70b``) and in config paths.
    """
    normalized = (provider_id or "").strip().lower()
    if not _ID_RE.match(normalized):
        raise ValueError(
            "Provider id must be 2–32 characters: lowercase letters, digits, "
            "'-' or '_', starting with a letter or digit."
        )
    if normalized in BUILTIN_IDS:
        raise ValueError(f"'{normalized}' is a built-in provider — pick another id.")
    if normalized in RESERVED_IDS:
        raise ValueError(f"'{normalized}' is reserved — pick another id.")
    return normalized


def _config(cfg: ConfigManager | None = None) -> ConfigManager:
    return cfg or get_config_manager()


def custom_section(cfg: ConfigManager | None = None) -> dict[str, Any]:
    """The ``providers.custom`` block, defaulted from ``DEFAULT_CONFIG``.

    Installs predating this feature have no ``custom`` key; they read as the
    shipped default (an empty mapping) instead of a duplicated literal.
    """
    shipped = DEFAULT_CONFIG.get("providers", {}).get("custom", {})
    section = _config(cfg).get("providers.custom", shipped)
    return section if isinstance(section, dict) else {}


def _entry_from_record(provider_id: str, record: dict[str, Any]) -> ProviderEntry | None:
    """Build an entry from a persisted record, or None if it's unusable.

    Tolerant by design: a hand-edited ``config.json`` with one bad entry must not
    take down provider loading for everything else.
    """
    try:
        kind = ProviderKind(str(record.get("kind") or ProviderKind.OPENAI_COMPATIBLE.value))
    except ValueError:
        return None
    if kind not in CUSTOM_KINDS:
        return None
    headers = record.get("headers")
    return ProviderEntry(
        id=provider_id,
        kind=kind,
        label=str(record.get("label") or provider_id),
        base_url=record.get("base_url") or None,
        api_key=record.get("api_key") or None,
        headers={str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {},
        enabled=bool(record.get("enabled", True)),
        builtin=False,
    )


def custom_entries(cfg: ConfigManager | None = None) -> dict[str, ProviderEntry]:
    """Every registered custom provider, keyed by id (malformed records dropped)."""
    entries: dict[str, ProviderEntry] = {}
    for provider_id, record in custom_section(cfg).items():
        if not isinstance(record, dict) or provider_id in BUILTIN_IDS:
            continue
        entry = _entry_from_record(str(provider_id), record)
        if entry is not None:
            entries[entry.id] = entry
    return entries


def builtin_entries(cfg: ConfigManager | None = None) -> dict[str, ProviderEntry]:
    """The five shipped providers, hydrated from their historical config paths."""
    config = _config(cfg)
    entries: dict[str, ProviderEntry] = {}
    for spec in BUILTINS:
        entries[spec.id] = ProviderEntry(
            id=spec.id,
            kind=spec.kind,
            label=spec.label,
            base_url=config.get(f"providers.{spec.id}.base_url"),
            api_key=config.get(f"providers.{spec.id}.api_key"),
            enabled=True,
            builtin=True,
            credential=spec.credential,
        )
    return entries


def all_entries(cfg: ConfigManager | None = None) -> dict[str, ProviderEntry]:
    """Built-ins first, then custom entries.

    A custom entry can never shadow a built-in: ``custom_entries`` already drops
    ids in ``BUILTIN_IDS``, so a provider promoted to built-in in a later release
    quietly wins over a same-named custom one rather than colliding.
    """
    entries = builtin_entries(cfg)
    entries.update(custom_entries(cfg))
    return entries


def get_entry(provider_id: str, cfg: ConfigManager | None = None) -> ProviderEntry | None:
    """Look up one entry by id (built-in or custom)."""
    return all_entries(cfg).get(provider_id)


def upsert_custom(
    provider_id: str,
    payload: dict[str, Any],
    cfg: ConfigManager | None = None,
) -> ProviderEntry:
    """Create or update a custom provider and persist it.

    Merges onto any existing record so a partial update (e.g. renaming the label)
    doesn't wipe the stored key. Raises ``ValueError`` on a bad id, kind, or a
    missing base URL.
    """
    provider_id = validate_id(provider_id)
    config = _config(cfg)

    existing = custom_section(config).get(provider_id)
    record: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    for key in _CUSTOM_FIELDS:
        if key in payload and payload[key] is not None:
            record[key] = payload[key]

    record.setdefault("kind", ProviderKind.OPENAI_COMPATIBLE.value)
    record.setdefault("label", provider_id)
    record.setdefault("enabled", True)

    entry = _entry_from_record(provider_id, record)
    if entry is None:
        raise ValueError(
            f"Unsupported provider kind '{record.get('kind')}'. "
            f"Custom providers must be one of: {', '.join(k.value for k in CUSTOM_KINDS)}."
        )
    if not entry.base_url:
        raise ValueError("A custom provider needs a base URL (e.g. https://api.groq.com/openai/v1).")

    config.set(f"providers.custom.{provider_id}", record)
    config.save()
    return entry


def delete_custom(provider_id: str, cfg: ConfigManager | None = None) -> bool:
    """Remove a custom provider. Returns False when it wasn't registered."""
    config = _config(cfg)
    if provider_id not in custom_section(config):
        return False
    config.unset(f"providers.custom.{provider_id}")
    config.save()
    return True


def known_ids(cfg: ConfigManager | None = None) -> frozenset[str]:
    """Every id ``config_update`` will accept under ``providers.*``."""
    return BUILTIN_IDS | frozenset(custom_entries(cfg))

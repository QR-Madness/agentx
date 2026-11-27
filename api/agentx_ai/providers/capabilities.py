"""One warm-once capability probe — the shared brain behind every modality gate.

The idiom "check caps → warm the provider catalog once if they look cold →
re-check" had grown four near-identical copies (`model_outputs_image`, the
vision/audio input gates in views, `core._model_supports_tools`). This module
is the single definition; each gate is now a one-line predicate over it.

Never raises: a probe failure returns ``False`` so callers degrade to the
conservative path (text-only / no tools) rather than sending a payload the
model would 400 on.

The same module owns the catalog's "is this model known?" answer
(``catalog_lists``) and the turn-time warm built on it (``warm_catalog_for``):
capabilities are read synchronously from a provider's cached catalog, and that
cache starts empty after every registry reload.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Upper bound on the turn-time catalog fetch. Matches the 5s the model list
#: (`/api/providers/models`) and the route description allow a provider: a slow
#: catalog may delay a turn this much, never fail it.
CATALOG_WARM_TIMEOUT = 5.0


def catalog_lists(provider: Any, model_id: str) -> bool | None:
    """Whether ``provider``'s catalog lists ``model_id``.

    ``None`` means "can't tell": the catalog is empty because it hasn't been
    fetched since the provider was built, its fetch failed, or the provider has
    no catalog at all. A provider whose ids aren't literal catalog keys
    (OpenRouter's ``~`` alias family) answers through ``resolve_catalog_id``.
    One definition shared by the route description's ``known`` flag and
    ``warm_catalog_for``. May raise if the provider's ``list_models`` does.
    """
    listed = provider.list_models()
    if not listed:
        return None
    resolver = getattr(provider, "resolve_catalog_id", None)
    if callable(resolver):
        return resolver(model_id) is not None
    return model_id in listed


async def warm_catalog_for(provider: Any, model_id: str) -> bool | None:
    """Fetch the provider's catalog once unless it already lists ``model_id``.

    Call before reading capabilities that size a turn. Every registry reload
    (each Settings save, each provider change) builds fresh provider instances
    with empty catalogs, and a capability read from an empty catalog returns the
    provider's fallback defaults, such as OpenRouter's and the AI Gateway's 8192
    window for a 1M-token model. That number then sized the whole turn and was
    reported to the client's context meter as fact.

    Cheap when warm: providers that cache their catalog only refetch once it is
    stale. Bounded by ``CATALOG_WARM_TIMEOUT``. Returns ``catalog_lists`` after
    the attempt. Never raises; a failed or slow fetch leaves the defaults in
    place, exactly as before.
    """
    try:
        if catalog_lists(provider, model_id):
            return True
        fetch = getattr(provider, "fetch_models", None)
        if fetch is not None:
            async with asyncio.timeout(CATALOG_WARM_TIMEOUT):
                await fetch()
        return catalog_lists(provider, model_id)
    except Exception as e:  # noqa: BLE001 — a cold catalog degrades the turn, never breaks it
        logger.debug(f"[catalog] warm for {model_id!r} failed, keeping provider defaults: {e}")
        return None


async def probe_model_capability(
    provider: Any,
    model_id: str,
    predicate: Callable[[Any], bool],
    caps: Any = None,
    *,
    tag: str = "capability",
) -> bool:
    """True when ``predicate(caps)`` holds for the model's capabilities.

    An uncached model reports provider defaults (modalities ``["text"]``, flags
    off), so when the predicate fails we warm the provider catalog **once**
    (``fetch_models``) and re-check before concluding. ``caps`` may be passed
    when the caller already resolved it. ``tag`` labels the debug log line.
    """
    try:
        if caps is None:
            caps = provider.get_capabilities(model_id)
        if predicate(caps):
            return True
        warm = getattr(provider, "fetch_models", None)
        if warm is not None:
            await warm()
            caps = provider.get_capabilities(model_id)
        return predicate(caps)
    except Exception as e:  # noqa: BLE001 — degrade, never break the turn
        logger.debug(f"[{tag}] capability probe failed, treating as unsupported: {e}")
        return False


def has_input_modality(caps: Any, modality: str) -> bool:
    """Whether ``caps.input_modalities`` includes ``modality`` (case-insensitive)."""
    mods = [str(m).lower() for m in (getattr(caps, "input_modalities", None) or [])]
    return modality in mods


def has_output_modality(caps: Any, modality: str) -> bool:
    """Whether ``caps.output_modalities`` includes ``modality`` (case-insensitive)."""
    mods = [str(m).lower() for m in (getattr(caps, "output_modalities", None) or [])]
    return modality in mods

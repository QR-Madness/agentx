"""One warm-once capability probe — the shared brain behind every modality gate.

The idiom "check caps → warm the provider catalog once if they look cold →
re-check" had grown four near-identical copies (`model_outputs_image`, the
vision/audio input gates in views, `core._model_supports_tools`). This module
is the single definition; each gate is now a one-line predicate over it.

Never raises: a probe failure returns ``False`` so callers degrade to the
conservative path (text-only / no tools) rather than sending a payload the
model would 400 on.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


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

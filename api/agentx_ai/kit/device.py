"""Shared compute-device selection for on-device models.

Both the translation kit (NLLB-200 + language detection) and the local
embedding model (`sentence-transformers`) run on-device. Historically the
translation models were never moved off the CPU and the embedder relied on
`sentence-transformers`' silent auto-detect, so there was no single, visible,
configurable answer to "what device are we running on?".

`resolve_device()` is that single answer. It reads the ``AGENTX_DEVICE`` env var
(``auto`` | ``cpu`` | ``cuda`` | ``cuda:N``), defaulting to ``auto`` which picks
CUDA when available and CPU otherwise. The resolved device is cached and logged
once so startup makes the choice obvious in the logs.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger("agentx")

_DEVICE_ENV_VAR = "AGENTX_DEVICE"


@lru_cache(maxsize=None)
def resolve_device(override: str | None = None) -> str:
    """Resolve the torch device string to run on-device models.

    Args:
        override: Explicit device (``cpu``/``cuda``/``cuda:N``/``auto``). When
            ``None``, the ``AGENTX_DEVICE`` env var is consulted, defaulting to
            ``auto``.

    Returns:
        A torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """
    requested = (override or os.environ.get(_DEVICE_ENV_VAR, "auto")).strip().lower()

    if requested in ("", "auto"):
        device = "cuda" if _cuda_available() else "cpu"
    elif requested.startswith("cuda"):
        device = requested if _cuda_available() else "cpu"
        if device != requested:
            logger.warning(
                "AGENTX_DEVICE=%s requested but CUDA is unavailable; falling back to CPU.",
                requested,
            )
    else:
        device = "cpu"

    logger.info(
        "Compute device resolved to '%s' (requested=%s, cuda_available=%s).",
        device,
        requested,
        _cuda_available(),
    )
    return device


def cuda_available() -> bool:
    """Public, non-cached CUDA availability check (for health/diagnostics)."""
    return _cuda_available()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken — treat as CPU-only
        return False

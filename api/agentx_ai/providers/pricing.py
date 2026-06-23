"""Per-turn cost estimation from model pricing metadata.

The pricing snapshot frozen in the return payload is intended to be persisted
alongside the turn so historical costs remain stable if provider pricing
changes later.
"""

from __future__ import annotations

from typing import TypedDict

from .base import ModelCapabilities


class PricingSnapshot(TypedDict):
    cost_per_1k_input: float
    cost_per_1k_output: float


class CostEstimate(TypedDict):
    cost_input: float
    cost_output: float
    cost_total: float
    currency: str
    pricing_snapshot: PricingSnapshot


def estimate_cost(
    caps: ModelCapabilities,
    tokens_in: int,
    tokens_out: int,
) -> CostEstimate | None:
    """Return an absolute-dollar cost estimate for one turn, or None if the
    model has no pricing metadata (e.g. local LM Studio models).
    """
    in_rate = caps.cost_per_1k_input
    out_rate = caps.cost_per_1k_output
    if in_rate is None and out_rate is None:
        return None

    in_rate = in_rate or 0.0
    out_rate = out_rate or 0.0

    cost_in = (tokens_in / 1000.0) * in_rate
    cost_out = (tokens_out / 1000.0) * out_rate

    return {
        "cost_input": cost_in,
        "cost_output": cost_out,
        "cost_total": cost_in + cost_out,
        "currency": caps.pricing_currency or "USD",
        "pricing_snapshot": {
            "cost_per_1k_input": in_rate,
            "cost_per_1k_output": out_rate,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Audio pricing (Foundation #5) — TTS billed per 1k input chars, STT per minute
# ──────────────────────────────────────────────────────────────────────────────
# Provider APIs don't reliably expose audio pricing, so rates live here (code) and
# are overridable via `config.pricing.audio.{provider:model}` — an override map,
# because an added default wouldn't reach installs with a pre-existing config.json.
_DEFAULT_AUDIO_PRICING: dict[str, dict[str, float | None]] = {
    # The shipped ambassador voices. TTS ≈ $0.015 / 1k chars; STT (whisper) $0.006 / min.
    "openrouter:microsoft/mai-voice-2": {"per_1k_chars": 0.015, "per_minute": None},
    "openrouter:openai/whisper-1": {"per_1k_chars": None, "per_minute": 0.006},
}


def _audio_rates() -> dict[str, dict]:
    """Shipped defaults merged with any `config.pricing.audio` overrides."""
    merged: dict[str, dict] = dict(_DEFAULT_AUDIO_PRICING)
    try:
        from ..config import get_config_manager
        overrides = get_config_manager().get("pricing.audio") or {}
        if isinstance(overrides, dict):
            merged.update(overrides)
    except Exception:  # noqa: BLE001 — pricing is best-effort
        pass
    return merged


def estimate_audio_cost(
    *, model: str, chars: int | None = None, seconds: float | None = None
) -> dict | None:
    """Estimate TTS/STT cost from the configurable per-model audio rate table.

    Returns a normalized dict (``cost_total``/``currency``/``pricing_snapshot``) —
    deliberately NOT the token ``CostEstimate`` (audio has no input/output split) —
    or ``None`` when the model has no configured rate or no measurable units (e.g.
    the provider didn't report audio duration).
    """
    rate = _audio_rates().get(model)
    if not rate:
        return None

    per_1k_chars = rate.get("per_1k_chars")
    per_minute = rate.get("per_minute")
    cost = 0.0
    snapshot: dict[str, float] = {}
    measured = False
    if chars is not None and per_1k_chars:
        cost += (chars / 1000.0) * per_1k_chars
        snapshot["per_1k_chars"] = per_1k_chars
        measured = True
    if seconds is not None and per_minute:
        cost += (seconds / 60.0) * per_minute
        snapshot["per_minute"] = per_minute
        measured = True

    if not measured:
        return None
    return {"cost_total": cost, "currency": "USD", "pricing_snapshot": snapshot}


# Per-image rates (USD). A shipped *estimate* — the authoritative rate for an install
# belongs in `config.pricing.images.{provider:model}` (the override map below), since an
# added default wouldn't reach installs with a pre-existing config.json. flux klein is a
# small/cheap model; treat 0.01/image as a rough placeholder.
_DEFAULT_IMAGE_PRICING: dict[str, dict[str, float]] = {
    "openrouter:black-forest-labs/flux.2-klein-4b": {"per_image": 0.01},
}


def _image_rates() -> dict[str, dict]:
    """Shipped defaults merged with any `config.pricing.images` overrides."""
    merged: dict[str, dict] = dict(_DEFAULT_IMAGE_PRICING)
    try:
        from ..config import get_config_manager
        overrides = get_config_manager().get("pricing.images") or {}
        if isinstance(overrides, dict):
            merged.update(overrides)
    except Exception:  # noqa: BLE001 — pricing is best-effort
        pass
    return merged


def estimate_image_cost(*, model: str, images: int = 1) -> dict | None:
    """Estimate image-generation cost from the configurable per-model rate table.

    Returns ``{cost_total, currency, pricing_snapshot}`` (mirrors ``estimate_audio_cost`` —
    not the token ``CostEstimate``) or ``None`` when the model has no configured rate."""
    rate = _image_rates().get(model)
    if not rate or not rate.get("per_image"):
        return None
    per_image = rate["per_image"]
    return {
        "cost_total": images * per_image,
        "currency": "USD",
        "pricing_snapshot": {"per_image": per_image},
    }

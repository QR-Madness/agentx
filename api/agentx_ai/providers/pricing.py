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

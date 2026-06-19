"""Unified usage/cost ledger (Foundation #5).

A single content-free spend record per metered model call, across every surface
(main chat, Alloy specialists, the Ambassador's LLM calls, and voice TTS/STT).
Rows carry only metering — model, units, cost — never message text, so the
Ambassador can record spend without violating its "never pollute the transcript"
invariant. ``/metrics/usage`` aggregates this table.

Writes are **best-effort**: a ledger failure must never break a turn, so every
path swallows-and-logs (mirroring the transcript persistence in ``views.py``).

The ``ref`` key (``conversation_id:turn_index`` for chat/alloy) lets the live
writer and the history backfill UPSERT the same turn instead of double-counting;
ambassador/voice rows pass ``ref=None`` (always insert — NULLs are distinct under
the UNIQUE constraint).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The known spend sources (kept here so callers and the metrics endpoint agree).
SOURCES = (
    "chat",
    "alloy",
    "ambassador_llm",
    "ambassador_tts",
    "ambassador_stt",
)


def turn_ref(conversation_id: str | None, turn_index: int | None) -> str | None:
    """Build the dedupe ref for a transcript-backed turn (chat/alloy)."""
    if conversation_id is None or turn_index is None:
        return None
    return f"{conversation_id}:{turn_index}"


def record_usage(
    *,
    source: str,
    model: str | None,
    provider: str | None = None,
    conversation_id: str | None = None,
    agent_id: str | None = None,
    units: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
    ref: str | None = None,
) -> None:
    """Append one usage event. Best-effort: never raises.

    ``cost`` is either a token ``CostEstimate`` or the audio-cost dict (both carry
    ``cost_total``/``currency``/``pricing_snapshot``) or ``None`` (model has no
    pricing). ``units`` is free-form metering (``{tokens_in,tokens_out}`` |
    ``{chars}`` | ``{audio_seconds,bytes}``).
    """
    if source not in SOURCES:
        # Not fatal — record it, but surface the typo in logs.
        logger.warning(f"usage_ledger: unknown source '{source}'")

    cost_total = cost.get("cost_total") if cost else None
    currency = (cost.get("currency") if cost else None) or "USD"
    snapshot = cost.get("pricing_snapshot") if cost else None

    try:
        from sqlalchemy import text
        from ..kit.agent_memory.connections import get_postgres_session

        with get_postgres_session() as session:
            session.execute(
                text("""
                    INSERT INTO usage_events
                        (source, conversation_id, agent_id, provider, model,
                         units, cost_total, currency, pricing_snapshot, ref)
                    VALUES
                        (:source, :conversation_id, :agent_id, :provider, :model,
                         CAST(:units AS JSONB), :cost_total, :currency,
                         CAST(:snapshot AS JSONB), :ref)
                    ON CONFLICT (ref) DO UPDATE SET
                        ts = NOW(),
                        units = EXCLUDED.units,
                        cost_total = EXCLUDED.cost_total,
                        currency = EXCLUDED.currency,
                        pricing_snapshot = EXCLUDED.pricing_snapshot,
                        model = EXCLUDED.model,
                        provider = EXCLUDED.provider,
                        agent_id = EXCLUDED.agent_id
                """),
                {
                    "source": source,
                    "conversation_id": conversation_id,
                    "agent_id": agent_id,
                    "provider": provider,
                    "model": model,
                    "units": json.dumps(units or {}),
                    "cost_total": cost_total,
                    "currency": currency,
                    "snapshot": json.dumps(snapshot) if snapshot is not None else None,
                    "ref": ref,
                },
            )
            session.commit()
    except Exception as e:  # noqa: BLE001 — metering is best-effort, never break a turn
        logger.debug(f"usage_ledger record failed (source={source}, model={model}): {e}")

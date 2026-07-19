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
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# The known spend sources (kept here so callers and the metrics endpoint agree).
SOURCES = (
    "chat",
    "alloy",
    "ambassador_llm",
    "ambassador_tts",
    "ambassador_stt",
    "aide",
    "search",
    "image",
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
    units: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
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


def _tok_row(r: Any, key: str) -> dict[str, Any]:  # RowMapping | Mapping
    return {
        key: r[key],
        "turns": int(r["turns"] or 0),
        "tokens_input": int(r["tokens_input"] or 0),
        "tokens_output": int(r["tokens_output"] or 0),
        "tokens_total": int(r["tokens_total"] or 0),
        "cost_total": round(float(r["cost_total"] or 0.0), 6),
    }


def aggregate_usage(days: int, *, conversation_id: str | None = None) -> dict[str, Any]:
    """Aggregate the ledger over the last ``days`` days (clamped 1-90), optionally
    scoped to one conversation. Returns ``{totals, by_model, by_agent, by_source,
    daily, days}`` — the shared read behind ``/metrics/usage`` and the Ambassador's
    ``usage_report`` belt tool. Raises on DB failure (callers own their degrade).

    The SQL stays fully static (this module is not S608-exempt): the token
    expressions are inlined per query and the optional conversation scope binds a
    nullable param.
    """
    days = max(1, min(int(days), 90))
    from sqlalchemy import text

    from ..kit.agent_memory.connections import get_postgres_session

    params = {"days": days, "cid": conversation_id}
    with get_postgres_session() as session:
        totals_row = session.execute(text("""
            SELECT
                COUNT(*) AS turns,
                SUM(COALESCE((units->>'tokens_in')::int, 0)) AS tokens_input,
                SUM(COALESCE((units->>'tokens_out')::int, 0)) AS tokens_output,
                SUM(COALESCE((units->>'tokens_total')::int, 0)) AS tokens_total,
                SUM(COALESCE(cost_total, 0)) AS cost_total,
                MAX(currency) AS cost_currency
            FROM usage_events
            WHERE ts >= NOW() - (:days || ' days')::interval
              AND (CAST(:cid AS TEXT) IS NULL OR conversation_id = :cid)
        """), params).mappings().first() or {}

        by_model_rows = session.execute(text("""
            SELECT
                COALESCE(model, 'unknown') AS model,
                COUNT(*) AS turns,
                SUM(COALESCE((units->>'tokens_in')::int, 0)) AS tokens_input,
                SUM(COALESCE((units->>'tokens_out')::int, 0)) AS tokens_output,
                SUM(COALESCE((units->>'tokens_total')::int, 0)) AS tokens_total,
                SUM(COALESCE(cost_total, 0)) AS cost_total
            FROM usage_events
            WHERE ts >= NOW() - (:days || ' days')::interval
              AND (CAST(:cid AS TEXT) IS NULL OR conversation_id = :cid)
            GROUP BY COALESCE(model, 'unknown')
            ORDER BY cost_total DESC, tokens_total DESC
        """), params).mappings().all()

        by_agent_rows = session.execute(text("""
            SELECT
                COALESCE(agent_id, '_default') AS agent_id,
                COUNT(*) AS turns,
                SUM(COALESCE((units->>'tokens_in')::int, 0)) AS tokens_input,
                SUM(COALESCE((units->>'tokens_out')::int, 0)) AS tokens_output,
                SUM(COALESCE((units->>'tokens_total')::int, 0)) AS tokens_total,
                SUM(COALESCE(cost_total, 0)) AS cost_total
            FROM usage_events
            WHERE ts >= NOW() - (:days || ' days')::interval
              AND (CAST(:cid AS TEXT) IS NULL OR conversation_id = :cid)
            GROUP BY COALESCE(agent_id, '_default')
            ORDER BY cost_total DESC, tokens_total DESC
        """), params).mappings().all()

        by_source_rows = session.execute(text("""
            SELECT
                source,
                COUNT(*) AS turns,
                SUM(COALESCE((units->>'tokens_in')::int, 0)) AS tokens_input,
                SUM(COALESCE((units->>'tokens_out')::int, 0)) AS tokens_output,
                SUM(COALESCE((units->>'tokens_total')::int, 0)) AS tokens_total,
                SUM(COALESCE(cost_total, 0)) AS cost_total
            FROM usage_events
            WHERE ts >= NOW() - (:days || ' days')::interval
              AND (CAST(:cid AS TEXT) IS NULL OR conversation_id = :cid)
            GROUP BY source
            ORDER BY cost_total DESC, turns DESC
        """), params).mappings().all()

        daily_rows = session.execute(text("""
            SELECT
                to_char(date_trunc('day', ts), 'YYYY-MM-DD') AS date,
                COUNT(*) AS turns,
                SUM(COALESCE((units->>'tokens_total')::int, 0)) AS tokens_total,
                SUM(COALESCE(cost_total, 0)) AS cost_total
            FROM usage_events
            WHERE ts >= NOW() - (:days || ' days')::interval
              AND (CAST(:cid AS TEXT) IS NULL OR conversation_id = :cid)
            GROUP BY date_trunc('day', ts)
            ORDER BY date_trunc('day', ts)
        """), params).mappings().all()

    return {
        "totals": {
            "turns": int(totals_row["turns"] or 0) if totals_row else 0,
            "tokens_input": int(totals_row["tokens_input"] or 0) if totals_row else 0,
            "tokens_output": int(totals_row["tokens_output"] or 0) if totals_row else 0,
            "tokens_total": int(totals_row["tokens_total"] or 0) if totals_row else 0,
            "cost_total": round(float(totals_row["cost_total"] or 0.0), 6) if totals_row else 0.0,
            "cost_currency": (totals_row.get("cost_currency") or "USD") if totals_row else "USD",
        },
        "by_model": [_tok_row(r, "model") for r in by_model_rows],
        "by_agent": [_tok_row(r, "agent_id") for r in by_agent_rows],
        "by_source": [_tok_row(r, "source") for r in by_source_rows],
        "daily": [
            {
                "date": r["date"],
                "turns": int(r["turns"] or 0),
                "tokens_total": int(r["tokens_total"] or 0),
                "cost_total": round(float(r["cost_total"] or 0.0), 6),
            }
            for r in daily_rows
        ],
        "days": days,
    }

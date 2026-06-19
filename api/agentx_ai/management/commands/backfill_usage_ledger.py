# pyright: reportAttributeAccessIssue=false
"""Backfill the usage ledger from historical conversation_logs (Foundation #5).

The cost-tracking ledger (`usage_events`) becomes the single source for
`/metrics/usage`. Assistant turns recorded before the ledger existed (and any
turns logged between the ledger landing and the live-writer wiring) live only in
`conversation_logs.metadata`. This command upserts those into `usage_events`
keyed by `conversation_id:turn_index`, so the metrics switch loses no history and
re-running it never double-counts (idempotent UPSERT on `ref`).

Usage:
    python manage.py backfill_usage_ledger          # backfill everything
    python manage.py backfill_usage_ledger --days 30
    python manage.py backfill_usage_ledger --dry-run
"""

from __future__ import annotations

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Backfill usage_events from historical conversation_logs assistant turns."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--days", type=int, default=None,
                            help="Only backfill turns from the last N days (default: all).")
        parser.add_argument("--dry-run", action="store_true",
                            help="Count eligible turns without writing.")

    def handle(self, *args, **opts) -> None:
        import json

        from sqlalchemy import text
        from agentx_ai.kit.agent_memory.connections import get_postgres_session

        days = opts.get("days")
        dry_run = opts.get("dry_run", False)

        # An assistant turn is "alloy" when its metadata carries a delegation id,
        # else "chat". tokens_total preserves the legacy token_count column for
        # turns that predate the in/out split. The optional day window is a bound
        # parameter (make_interval) — no value is interpolated into the SQL.
        day_clause = ""
        params: dict[str, int] = {}
        if days:
            day_clause = " AND timestamp >= NOW() - make_interval(days => :days)"
            params["days"] = int(days)

        select_sql = text(
            "SELECT "
            "conversation_id::text AS conversation_id, "
            "turn_index, agent_id, model, "
            "metadata->>'provider' AS provider, "
            "COALESCE((metadata->>'tokens_input')::int, 0) AS tokens_in, "
            "COALESCE((metadata->>'tokens_output')::int, 0) AS tokens_out, "
            "CASE WHEN metadata ? 'tokens_input' "
            "     THEN COALESCE((metadata->>'tokens_input')::int, 0) "
            "        + COALESCE((metadata->>'tokens_output')::int, 0) "
            "     ELSE COALESCE(token_count, 0) END AS tokens_total, "
            "(metadata->>'cost_estimate')::float AS cost_total, "
            "COALESCE(metadata->>'cost_currency', 'USD') AS currency, "
            "metadata->'pricing_snapshot' AS pricing_snapshot, "
            "(metadata ? 'delegation_id') AS is_alloy "
            "FROM conversation_logs "
            "WHERE role = 'assistant'" + day_clause
        )

        upsert_sql = text("""
            INSERT INTO usage_events
                (source, conversation_id, agent_id, provider, model,
                 units, cost_total, currency, pricing_snapshot, ref)
            VALUES
                (:source, :conversation_id, :agent_id, :provider, :model,
                 CAST(:units AS JSONB), :cost_total, :currency,
                 CAST(:snapshot AS JSONB), :ref)
            ON CONFLICT (ref) DO UPDATE SET
                ts = usage_events.ts,
                units = EXCLUDED.units,
                cost_total = EXCLUDED.cost_total,
                currency = EXCLUDED.currency,
                pricing_snapshot = EXCLUDED.pricing_snapshot,
                model = EXCLUDED.model,
                provider = EXCLUDED.provider,
                agent_id = EXCLUDED.agent_id
        """)

        written = 0
        scanned = 0
        try:
            with get_postgres_session() as session:
                rows = session.execute(select_sql, params).fetchall()
                for r in rows:
                    scanned += 1
                    ref = f"{r.conversation_id}:{r.turn_index}"
                    units = {
                        "tokens_in": int(r.tokens_in or 0),
                        "tokens_out": int(r.tokens_out or 0),
                        "tokens_total": int(r.tokens_total or 0),
                    }
                    if dry_run:
                        continue
                    session.execute(upsert_sql, {
                        "source": "alloy" if r.is_alloy else "chat",
                        "conversation_id": r.conversation_id,
                        "agent_id": r.agent_id,
                        "provider": r.provider,
                        "model": r.model,
                        "units": json.dumps(units),
                        "cost_total": r.cost_total,
                        "currency": r.currency or "USD",
                        "snapshot": json.dumps(r.pricing_snapshot) if r.pricing_snapshot else None,
                        "ref": ref,
                    })
                    written += 1
                if not dry_run:
                    session.commit()
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Backfill failed: {e}"))
            return

        if dry_run:
            self.stdout.write(self.style.SUCCESS(f"[dry-run] {scanned} assistant turns eligible."))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Backfilled {written}/{scanned} assistant turns into usage_events."
            ))

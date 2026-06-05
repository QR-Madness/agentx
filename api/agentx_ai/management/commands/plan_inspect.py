"""
Django management command to inspect a plan's execution state — the observability
we were missing when plan conversations failed to restore/resume.

It reads two durable sources for a session and prints them side by side:

  1. Redis plan state (``PlanStateStore``): overall status, per-subtask status,
     completed/total counts, the cancel flag, remaining TTL, and a ``resumable``
     verdict. This is what the resume endpoint + the in-conversation resume nudge
     read.
  2. PostgreSQL ``conversation_logs`` rows for the session: turn_index, role,
     content length, and whether the row carries a ``plan`` metadata card. This
     is what conversation *restore* reads — if these rows are missing, the
     conversation 404s on open.

Read-only; no LLM, no writes.

Usage:
    python manage.py plan_inspect <session_id> [plan_id]
    python manage.py plan_inspect <session_id> --json
"""

from __future__ import annotations

import json
from typing import Any

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Inspect a plan's Redis state and the session's conversation_logs rows "
        "(observability for restore/resume debugging). Read-only."
    )

    def add_arguments(self, parser):
        parser.add_argument("session_id", help="Conversation / session id")
        parser.add_argument(
            "plan_id",
            nargs="?",
            default=None,
            help="Plan id (optional — omit to only dump conversation_logs)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit a machine-readable JSON object instead of a table.",
        )

    def handle(self, *args, **opts):
        session_id: str = opts["session_id"]
        plan_id: str | None = opts["plan_id"]
        as_json: bool = opts["json"]

        plan_state = self._read_plan_state(session_id, plan_id) if plan_id else None
        rows = self._read_conversation_logs(session_id)

        if as_json:
            self.stdout.write(json.dumps({
                "session_id": session_id,
                "plan_id": plan_id,
                "plan_state": plan_state,
                "conversation_logs": rows,
            }, indent=2, default=str))
            return

        self._print_plan_state(plan_id, plan_state)
        self._print_conversation_logs(session_id, rows)

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def _read_plan_state(self, session_id: str, plan_id: str) -> dict | None:
        from agentx_ai.agent.plan_state import PlanStateStore

        store = PlanStateStore(session_id)
        data = store.get_status(plan_id)
        if not data:
            return None
        # `data` is a Redis hash with flat subtask:* fields; surface the digest
        # plus the derived resumable/ttl verdicts the resume path relies on.
        subtasks = {}
        for key, value in data.items():
            k = key.decode() if isinstance(key, bytes) else key
            v = value.decode() if isinstance(value, bytes) else value
            if k.startswith("subtask:") and k.endswith(":status"):
                sid = k.split(":", 2)[1]
                subtasks[sid] = v
        return {
            "status": _d(data.get("status") or data.get(b"status")),
            "completed_count": _d(data.get("completed_count") or data.get(b"completed_count")),
            "subtask_count": _d(data.get("subtask_count") or data.get(b"subtask_count")),
            "cancel_requested": store.is_cancel_requested(plan_id),
            "ttl_seconds": store.get_ttl(plan_id),
            "resumable": store.is_resumable(plan_id),
            "subtasks": dict(sorted(subtasks.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)),
        }

    def _read_conversation_logs(self, session_id: str) -> list[dict]:
        from agentx_ai.kit.agent_memory.connections import PostgresConnection

        conn: Any = PostgresConnection.get_engine().raw_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT turn_index, role, content, metadata
                    FROM conversation_logs
                    WHERE conversation_id = %s
                    ORDER BY turn_index ASC
                    """,
                    (session_id,),
                )
                out = []
                for turn_index, role, content, metadata in cur.fetchall():
                    meta = metadata if isinstance(metadata, dict) else {}
                    out.append({
                        "turn_index": turn_index,
                        "role": role,
                        "content_len": len(content or ""),
                        "has_plan": bool(meta.get("plan")),
                        "plan_status": (meta.get("plan") or {}).get("status"),
                        "interrupted": bool(meta.get("interrupted")),
                        "steered": bool(meta.get("steered")),
                    })
                return out
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Printers
    # ------------------------------------------------------------------

    def _print_plan_state(self, plan_id: str | None, state: dict | None):
        self.stdout.write(self.style.MIGRATE_HEADING("Plan state (Redis)"))  # type: ignore[attr-defined]
        if not plan_id:
            self.stdout.write("  (no plan_id given — skipped)")
            return
        if state is None:
            self.stdout.write(self.style.WARNING(f"  No Redis state for plan {plan_id} (expired or never created)."))  # type: ignore[attr-defined]
            return
        verdict = self.style.SUCCESS("resumable") if state["resumable"] else "not resumable"  # type: ignore[attr-defined]
        self.stdout.write(
            f"  plan={plan_id} status={state['status']} "
            f"{state['completed_count']}/{state['subtask_count']} done  "
            f"cancel={state['cancel_requested']} ttl={state['ttl_seconds']}s  [{verdict}]"
        )
        for sid, status in state["subtasks"].items():
            self.stdout.write(f"    subtask {sid}: {status}")

    def _print_conversation_logs(self, session_id: str, rows: list[dict]):
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("conversation_logs (PostgreSQL — what restore reads)"))  # type: ignore[attr-defined]
        if not rows:
            self.stdout.write(self.style.ERROR(  # type: ignore[attr-defined]
                f"  ZERO rows for {session_id} — restore will 404 / show 'Failed to restore'."
            ))
            return
        for r in rows:
            flags = []
            if r["has_plan"]:
                flags.append(f"plan:{r['plan_status']}")
            if r["interrupted"]:
                flags.append("interrupted")
            if r["steered"]:
                flags.append("steered")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            self.stdout.write(
                f"  #{r['turn_index']:<3} {r['role']:<12} {r['content_len']:>6} chars{suffix}"
            )
        self.stdout.write(f"  ({len(rows)} rows)")


def _d(value):
    """Decode a Redis bytes value to str (passthrough for non-bytes/None)."""
    if isinstance(value, bytes):
        return value.decode()
    return value

"""
Django management command to export a user's agent memory to a JSON envelope.

Serializes the full memory graph (conversations/turns, facts/entities,
strategies/tool-invocations, goals) plus the PostgreSQL audit mirror into a
single round-trippable file. Re-import with ``import_memory``.

Exports are text-only — embeddings are regenerated from text on import, so the
files are small, deterministic and diffable (and portable across embedders).

Usage:
    python manage.py export_memory
    python manage.py export_memory --channel _global --output snapshot.json
    python manage.py export_memory --user-id alice --channel _all
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)

# Mirror views.DEFAULT_USER_ID (single-user until auth lands).
DEFAULT_USER_ID = "default"


class Command(BaseCommand):
    help = "Export a user's agent memory to a round-trippable JSON envelope."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user-id",
            type=str,
            default=DEFAULT_USER_ID,
            help=f"User to export. Default: '{DEFAULT_USER_ID}'.",
        )
        parser.add_argument(
            "--channel",
            type=str,
            default="_all",
            help=(
                "Channel to export, '_all' for every channel (default), or a "
                "comma-separated set (e.g. '_self_x,_project_y')."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help=(
                "Output file path. Default: "
                "data/memory_exports/<ts>_<channel>.json"
            ),
        )

    def handle(self, *args, **options):
        from agentx_ai.kit.agent_memory.portability import MemoryExporter

        user_id = options["user_id"]
        channel = options["channel"]
        channel_set = _parse_channels(channel)

        self.stdout.write(
            f"Exporting memory for user='{user_id}' channel='{channel}'…"
        )

        try:
            export = MemoryExporter(
                user_id=user_id,
                channel=None if channel_set else channel,
                channels=channel_set,
            ).export()
        except Exception as e:  # noqa: BLE001 — surface a clean CLI error
            raise CommandError(f"Export failed: {e}") from e

        out_path = self._resolve_output(options["output"], channel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(export.model_dump_json(indent=2), encoding="utf-8")

        counts = export.counts()
        self.stdout.write(self.style.SUCCESS(f"Wrote {out_path}"))  # type: ignore[attr-defined]
        for name, n in counts.items():
            if n:
                self.stdout.write(f"  {name}: {n}")

    @staticmethod
    def _resolve_output(output: str | None, channel: str) -> Path:
        if output:
            return Path(output)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe_channel = (
            (channel or "_all").replace(",", "+").lstrip("_") or "all"
        )
        # Repo root is two levels up from api/agentx_ai/management/commands/… —
        # use cwd-relative data/ so it matches the rest of the bind-mounted data.
        return Path("data") / "memory_exports" / f"{ts}_{safe_channel}.json"


def _parse_channels(channel: str | None) -> list[str] | None:
    """A comma in --channel selects a channel set; otherwise None (single/all)."""
    if not channel or "," not in channel:
        return None
    return [c.strip() for c in channel.split(",") if c.strip()]

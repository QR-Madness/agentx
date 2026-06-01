"""
Django management command to import a memory export produced by ``export_memory``.

Idempotently MERGEs every node on its stable id. ``--mode merge`` (default)
upserts and leaves other data untouched; ``--mode replace`` wipes the target
channel(s) for the user first, so the channel ends up matching the file exactly.

Exports are text-only, so every embedding is regenerated from the node's
canonical text with this instance's model on import.

Usage:
    python manage.py import_memory --input snapshot.json
    python manage.py import_memory --input snapshot.json --mode replace --channel _global
    python manage.py import_memory --input snapshot.json --dry-run
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default"


class Command(BaseCommand):
    help = "Import a memory export (idempotent MERGE-on-id). Default mode: merge."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            required=True,
            help="Path to the JSON export file.",
        )
        parser.add_argument(
            "--user-id",
            type=str,
            default=DEFAULT_USER_ID,
            help=f"User to import into. Default: '{DEFAULT_USER_ID}'.",
        )
        parser.add_argument(
            "--mode",
            choices=["merge", "replace"],
            default="merge",
            help="merge = upsert (default); replace = wipe target channel first.",
        )
        parser.add_argument(
            "--channel",
            type=str,
            default=None,
            help="Override the wipe scope for replace mode (default: the file's channel).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse + summarize the file without writing anything.",
        )

    def handle(self, *args, **options):
        from agentx_ai.kit.agent_memory.portability import MemoryExport, MemoryImporter

        in_path = Path(options["input"])
        if not in_path.exists():
            raise CommandError(f"Input file not found: {in_path}")

        try:
            payload = json.loads(in_path.read_text(encoding="utf-8"))
            export = MemoryExport.model_validate(payload)
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"Could not parse export: {e}") from e

        self.stdout.write(
            f"Loaded {in_path} — schema v{export.schema_version}, "
            f"user='{export.user_id}', channel='{export.channel}' "
            f"(source embedder: {export.embedder.provider_model})"
        )
        for name, n in export.counts().items():
            if n:
                self.stdout.write(f"  {name}: {n}")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry-run: nothing written."))  # type: ignore[attr-defined]
            return

        try:
            summary = MemoryImporter(user_id=options["user_id"]).import_export(
                export, mode=options["mode"], channel=options["channel"]
            )
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"Import failed: {e}") from e

        self.stdout.write(self.style.SUCCESS(  # type: ignore[attr-defined]
            f"Imported (mode={summary['mode']}, channel={summary['channel']}, "
            f"recomputed_embeddings={summary['recomputed_embeddings']})"
        ))
        for name, c in summary["imported"].items():
            self.stdout.write(f"  {name}: +{c['created']} new / {c['total']} total")
        self.stdout.write(
            f"  pg_conversation_logs: {summary['pg_conversation_logs']}, "
            f"pg_tool_invocations: {summary['pg_tool_invocations']}"
        )

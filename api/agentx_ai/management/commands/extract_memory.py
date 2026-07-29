"""
Django management command for Extract — export with verified deletion.

Serializes the named channel(s) into a vault artifact, verifies the written
file against the live stores, and only then wipes those channels (Neo4j + the
PostgreSQL mirror). The artifact and a ``.receipt.json`` sidecar stay in
``data/vault/`` (nothing hard-deletes them automatically).

Usage:
    python manage.py extract_memory --channel _self_myagent --dry-run
    python manage.py extract_memory --channel _self_myagent,_project_ws_x --yes
"""

from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "default"


class Command(BaseCommand):
    help = (
        "Extract channel(s): export → verify → wipe → receipt. "
        "Real runs require --yes."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--channel",
            type=str,
            required=True,
            help="Channel(s) to extract — comma-separate for a set. '_all' is refused.",
        )
        parser.add_argument(
            "--user-id",
            type=str,
            default=DEFAULT_USER_ID,
            help=f"User to extract from. Default: '{DEFAULT_USER_ID}'.",
        )
        parser.add_argument(
            "--vault-dir",
            type=str,
            default=None,
            help="Override the vault directory (default: data/vault).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report counts + prospective vault path; write and wipe nothing.",
        )
        parser.add_argument(
            "--yes",
            action="store_true",
            help="Confirm the wipe (a real extract deletes the channel from the live system).",
        )

    def handle(self, *args, **options):
        from agentx_ai.kit.agent_memory.portability import ExtractError, extract_memory

        channels = [c.strip() for c in options["channel"].split(",") if c.strip()]
        dry_run = options["dry_run"]

        if not dry_run and not options["yes"]:
            raise CommandError(
                "A real extract WIPES the channel(s) from the live system after "
                "verifying the artifact. Re-run with --yes to confirm, or use "
                "--dry-run to preview."
            )

        try:
            receipt = extract_memory(
                user_id=options["user_id"],
                channels=channels,
                vault_dir=options["vault_dir"],
                dry_run=dry_run,
            )
        except ExtractError as e:
            raise CommandError(str(e)) from e
        except Exception as e:  # noqa: BLE001 — surface a clean CLI error
            raise CommandError(f"Extract failed: {e}") from e

        if dry_run:
            self.stdout.write(self.style.WARNING(  # type: ignore[attr-defined]
                f"Dry-run — nothing written or wiped. Would write: {receipt.file}"
            ))
            for name, n in receipt.counts.items():
                if n:
                    self.stdout.write(f"  {name}: {n}")
            return

        if not receipt.verified:
            raise CommandError(
                f"Verify FAILED — nothing deleted; artifact kept at {receipt.file}. "
                f"{receipt.error}"
            )

        self.stdout.write(self.style.SUCCESS(  # type: ignore[attr-defined]
            f"Extracted {', '.join(receipt.channels)} → {receipt.file}"
        ))
        self.stdout.write(f"  sha256: {receipt.sha256}")
        for name, n in receipt.counts.items():
            if n:
                self.stdout.write(f"  {name}: {n}")
        self.stdout.write("Wiped from live stores:")
        for name, n in receipt.wiped_counts.items():
            if n:
                self.stdout.write(f"  {name}: {n}")

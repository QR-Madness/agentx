# pyright: reportAttributeAccessIssue=false
# Django's BaseCommand.style exposes color methods (SUCCESS, ERROR, WARNING, ...)
# that django-stubs does not type; suppress those false positives here.
"""
Manage encryption of the durable log archive (see logging_kit.archive_crypto).

Usage:
    python manage.py rotate_log_keys --status        # show keyring + segment state
    python manage.py rotate_log_keys --seal          # seal pending plaintext days
    python manage.py rotate_log_keys                  # re-wrap DEK under a new password
    python manage.py rotate_log_keys --reencrypt      # deep rotation (new DEK + rewrite)

``--seal`` and the default re-wrap prompt for the password(s) when not passed via
``--password`` / ``--old-password`` / ``--new-password``.
"""

import getpass

from django.core.management.base import BaseCommand, CommandError

from agentx_ai.logging_kit import archive_crypto


class Command(BaseCommand):
    help = "Seal / rotate / re-encrypt the durable log archive keyring"

    def add_arguments(self, parser):
        parser.add_argument("--status", action="store_true", help="Show keyring + segment status")
        parser.add_argument("--seal", action="store_true", help="Seal pending plaintext day segments")
        parser.add_argument("--reencrypt", action="store_true", help="Deep rotation: new DEK + rewrite all segments")
        parser.add_argument("--password", type=str, help="Current password (for --seal)")
        parser.add_argument("--old-password", type=str, help="Old password (for rotation)")
        parser.add_argument("--new-password", type=str, help="New password (for rotation)")

    def handle(self, *args, **options):
        if options["status"]:
            self._status()
            return

        if not archive_crypto.is_encryption_active():
            raise CommandError(
                "No log-archive keyring exists yet. Set up authentication first "
                "(`task auth:setup`); the keyring is created on setup/first login."
            )

        if options["seal"]:
            self._seal(options.get("password"))
            return

        if options["reencrypt"]:
            self._reencrypt(options.get("old_password"), options.get("new_password"))
            return

        self._rewrap(options.get("old_password"), options.get("new_password"))

    # ------------------------------------------------------------------ #
    def _status(self):
        info = archive_crypto.keyring_status()
        self.stdout.write(self.style.SUCCESS("Log-archive encryption status"))
        for key in ("keyring_present", "unlocked", "sealed_segments", "pending_segments", "created_at", "rotated_at"):
            if key in info:
                self.stdout.write(f"  {key}: {info[key]}")

    def _seal(self, password):
        password = password or getpass.getpass("Password: ")
        try:
            dek = archive_crypto.unwrap_dek(password)
        except archive_crypto.BadPassword:
            raise CommandError("Incorrect password")
        count = archive_crypto.seal_pending(dek)
        archive_crypto.set_cached_dek(dek)
        self.stdout.write(self.style.SUCCESS(f"✓ Sealed {count} pending segment(s)"))

    def _rewrap(self, old_password, new_password):
        old_password = old_password or getpass.getpass("Current password: ")
        new_password = new_password or self._prompt_new()
        try:
            archive_crypto.rewrap_dek(new_password, old_password=old_password)
        except archive_crypto.BadPassword:
            raise CommandError("Incorrect current password")
        self.stdout.write(self.style.SUCCESS("✓ Keyring re-wrapped under the new password (no archives rewritten)"))

    def _reencrypt(self, old_password, new_password):
        old_password = old_password or getpass.getpass("Current password: ")
        new_password = new_password or self._prompt_new()
        try:
            count = archive_crypto.reencrypt_all(old_password, new_password)
        except archive_crypto.BadPassword:
            raise CommandError("Incorrect current password")
        self.stdout.write(self.style.SUCCESS(f"✓ Re-encrypted {count} segment(s) under a brand-new DEK"))

    def _prompt_new(self) -> str:
        while True:
            new = getpass.getpass("New password (min 8 chars): ")
            if len(new) < 8:
                self.stdout.write(self.style.ERROR("Password must be at least 8 characters. Try again."))
                continue
            if new != getpass.getpass("Confirm new password: "):
                self.stdout.write(self.style.ERROR("Passwords do not match. Try again."))
                continue
            return new

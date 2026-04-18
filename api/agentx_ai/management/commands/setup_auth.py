"""
Management command to set up AgentX authentication.

Usage:
    python manage.py setup_auth                    # Interactive password setup
    python manage.py setup_auth --password=...     # Non-interactive setup
    python manage.py setup_auth --check            # Check if setup is required
"""

import getpass
from django.core.management.base import BaseCommand, CommandError

from agentx_ai.auth.service import get_auth_service


class Command(BaseCommand):
    help = "Set up AgentX root user authentication"

    def add_arguments(self, parser):
        parser.add_argument(
            "--password",
            type=str,
            help="Password to set (non-interactive mode). Min 8 characters.",
        )
        parser.add_argument(
            "--check",
            action="store_true",
            help="Check if setup is required without making changes",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force password reset even if already configured (dangerous!)",
        )

    def handle(self, *args, **options):
        auth_service = get_auth_service()

        # Check mode
        if options["check"]:
            if auth_service.is_setup_required():
                self.stdout.write(self.style.WARNING("Setup required: No root password configured"))
                return
            else:
                self.stdout.write(self.style.SUCCESS("Setup complete: Root password is configured"))
                return

        # Check if already set up
        if not auth_service.is_setup_required():
            if not options["force"]:
                self.stdout.write(self.style.WARNING(
                    "Root password is already configured.\n"
                    "Use --force to reset it (this will invalidate all sessions)."
                ))
                return
            else:
                self.stdout.write(self.style.WARNING("Force mode: Resetting existing password..."))

        # Get password
        password = options.get("password")

        if password:
            # Non-interactive mode
            if len(password) < 8:
                raise CommandError("Password must be at least 8 characters")
        else:
            # Interactive mode
            self.stdout.write("\n" + "=" * 50)
            self.stdout.write("AgentX Authentication Setup")
            self.stdout.write("=" * 50 + "\n")
            self.stdout.write("Please enter a password for the root user.")
            self.stdout.write("This password will be required to access the API.\n")

            while True:
                password = getpass.getpass("Enter password (min 8 chars): ")

                if len(password) < 8:
                    self.stdout.write(self.style.ERROR("Password must be at least 8 characters. Try again."))
                    continue

                confirm = getpass.getpass("Confirm password: ")

                if password != confirm:
                    self.stdout.write(self.style.ERROR("Passwords do not match. Try again."))
                    continue

                break

        # Set up the password
        try:
            # If forcing, we need to directly update the password
            if options["force"] and not auth_service.is_setup_required():
                # Use internal method to update password
                import bcrypt
                from agentx_ai.kit.agent_memory.connections import get_postgres_session

                password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12))

                with get_postgres_session() as session:
                    session.execute(
                        "UPDATE agentx_auth SET password_hash = %s, updated_at = NOW() WHERE username = 'root'",
                        (password_hash.decode('utf-8'),)
                    )

                # Invalidate all sessions
                auth_service.invalidate_all_sessions()
                self.stdout.write(self.style.SUCCESS("\n✓ Root password reset successfully"))
                self.stdout.write("All existing sessions have been invalidated.")
            else:
                success = auth_service.setup_root_password(password)

                if success:
                    self.stdout.write(self.style.SUCCESS("\n✓ Root password configured successfully"))
                else:
                    raise CommandError("Failed to set password (already configured?)")

            self.stdout.write("\nYou can now log in with:")
            self.stdout.write("  Username: root")
            self.stdout.write("  Password: <your password>\n")

            # Reminder about enabling auth
            self.stdout.write(self.style.NOTICE(
                "Remember to enable authentication by setting:\n"
                "  AGENTX_AUTH_ENABLED=true\n"
                "in your environment or .env file."
            ))

        except ValueError as e:
            raise CommandError(str(e))
        except Exception as e:
            raise CommandError(f"Failed to set up authentication: {e}")

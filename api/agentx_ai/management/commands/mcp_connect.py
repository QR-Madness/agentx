"""
Management command to connect to configured MCP servers.

Usage:
    python manage.py mcp_connect              # Connect all servers
    python manage.py mcp_connect filesystem   # Connect specific server
    python manage.py mcp_connect --list       # List configured servers
"""

from django.core.management.base import BaseCommand

from agentx_ai.mcp import get_mcp_manager


class Command(BaseCommand):
    help = "Connect to configured MCP servers"

    def add_arguments(self, parser):
        parser.add_argument(
            "servers",
            nargs="*",
            help="Server names to connect (default: all configured servers)",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List configured servers without connecting",
        )

    def handle(self, *args, **options):
        manager = get_mcp_manager()
        configured = manager.registry.list()

        if options["list"]:
            if not configured:
                self.stdout.write("No MCP servers configured.")
                return
            self.stdout.write(f"Configured MCP servers ({len(configured)}):")
            for config in configured:
                status = "connected" if manager.is_connected(config.name) else "disconnected"
                self.stdout.write(f"  {config.name} ({config.transport.value}) - {status}")
            return

        server_names = options["servers"]
        if not server_names:
            # Connect all
            self.stdout.write(f"Connecting to {len(configured)} configured servers...")
            results = manager.connect_all()
            for name, result in results.items():
                if result["status"] == "connected":
                    self.stdout.write(self.style.SUCCESS(f"  ✓ {name}: connected"))
                else:
                    self.stdout.write(self.style.ERROR(f"  ✗ {name}: {result.get('error', 'unknown error')}"))
        else:
            for name in server_names:
                try:
                    connection = manager.connect(name)
                    self.stdout.write(self.style.SUCCESS(
                        f"  ✓ {name}: {len(connection.tools)} tools, {len(connection.resources)} resources"
                    ))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  ✗ {name}: {e}"))

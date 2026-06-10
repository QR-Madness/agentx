"""Startup banner — an ASCII logo + a live status table.

Rendered once from ``AppConfig.ready()`` when running under a live server. Every
row is best-effort and bounded so a down dependency degrades the row rather than
stalling boot. Gated by ``AGENTX_LOG_BANNER`` (defaults to follow decorations).
"""

from __future__ import annotations

import os
import socket

from .flags import LogFlags, read_flags

_LOGO = r"""
   _                    _    __  __
  /_\   __ _  ___ _ _  | |_  \ \/ /
 / _ \ / _` |/ -_) ' \ |  _|  >  <
/_/ \_\\__, |\___|_||_| \__| /_/\_\
       |___/   AI Agent Platform
"""


def _probe(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ok(up: bool) -> str:
    return "[green]● up[/green]" if up else "[red]● down[/red]"


def _providers() -> str:
    try:
        from agentx_ai.providers.registry import get_registry

        names = get_registry().list_providers()
        return ", ".join(sorted(names)) if names else "[yellow]none configured[/yellow]"
    except Exception:  # noqa: BLE001
        return "[grey50]?[/grey50]"


def _db_status() -> str:
    rh = os.environ.get("REDIS_HOST", "localhost")
    ph = os.environ.get("POSTGRES_HOST", "localhost")
    nh = os.environ.get("NEO4J_HOST", "localhost")
    rp = int(os.environ.get("REDIS_PORT", "6379") or 6379)
    pp = int(os.environ.get("POSTGRES_PORT", "5432") or 5432)
    np_ = int(os.environ.get("NEO4J_BOLT_PORT", "7687") or 7687)
    return (
        f"Neo4j {_ok(_probe(nh, np_))}   "
        f"Postgres {_ok(_probe(ph, pp))}   "
        f"Redis {_ok(_probe(rh, rp))}"
    )


def _mcp_count() -> str:
    try:
        from agentx_ai.mcp import get_mcp_manager

        return f"{len(get_mcp_manager().list_connections())} connected"
    except Exception:  # noqa: BLE001
        return "[grey50]?[/grey50]"


def render_startup_banner(flags: LogFlags | None = None) -> None:
    flags = flags or read_flags()
    if not flags.banner:
        return
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        from agentx_ai import VERSION

        console = Console(
            stderr=True, force_terminal=True if flags.force_color else None
        )

        port = os.environ.get("AGENTX_API_PORT", "12319")
        debug = os.environ.get("DJANGO_DEBUG", "true").lower() in ("true", "1", "yes")

        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold bright_black")
        table.add_column()
        table.add_row("Version", f"[bold]{VERSION}[/bold]")
        table.add_row("API", f"http://localhost:{port}/api")
        table.add_row("Mode", "[yellow]DEBUG[/yellow]" if debug else "[green]production[/green]")
        table.add_row("Providers", _providers())
        table.add_row("Datastores", _db_status())
        table.add_row("MCP servers", _mcp_count())
        decor = "on" if flags.decorations else "off"
        table.add_row(
            "Logging",
            f"decorations [cyan]{decor}[/cyan] · llm [cyan]{flags.llm_level}[/cyan] · "
            f"api [cyan]{'on' if flags.api_enabled else 'off'}[/cyan] · "
            f"archive [cyan]{'on' if flags.archive_enabled else 'off'}[/cyan]",
        )

        logo = Text(_LOGO, style="bold cyan")
        console.print(Panel.fit(logo, border_style="cyan"))
        console.print(table)
        console.print()
    except Exception:  # noqa: BLE001 — a banner must never break boot
        pass

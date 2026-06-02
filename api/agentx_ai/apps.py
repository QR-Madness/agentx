import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgentxAiConfig(AppConfig):
    # Standard Django idiom; pyright sees the base as a cached_property descriptor.
    default_auto_field = 'django.db.models.BigAutoField'  # pyright: ignore[reportAssignmentType]
    name = 'agentx_ai'

    def ready(self) -> None:
        # Avoid starting the worker during management commands like migrate,
        # test, or makemigrations — only run under the live server.
        #
        # Detection has to handle `uv run uvicorn agentx_api.asgi:application`,
        # where argv[0] is the *full path* to the uvicorn binary (so a bare
        # "uvicorn" string never matches). Check the basename of argv[0] and
        # any asgi/wsgi target, not just exact-string membership.
        argv = list(getattr(os, "sys").argv)  # type: ignore[attr-defined]
        prog = os.path.basename(argv[0]) if argv else ""
        run_main = os.environ.get("RUN_MAIN") == "true"
        is_server = (
            prog in ("uvicorn", "daphne", "gunicorn", "hypercorn")
            or any(arg in ("runserver", "uvicorn", "daphne") for arg in argv)
            or any(("asgi" in arg or "wsgi" in arg) for arg in argv[1:])
            or os.environ.get("AGENTX_ENABLE_BG_WORKER") == "1"
        )

        if not (run_main or is_server):
            return

        try:
            from .background import start_worker
            start_worker()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Background chat worker failed to start: {exc}")

        # Restore MCP servers that were connected when the API last shut down
        # (auto_connect=True). Best-effort on a daemon thread so a slow or dead
        # server can't block startup; connect_persisted logs and skips failures.
        try:
            import threading

            def _reconnect_mcp() -> None:
                try:
                    from .mcp import get_mcp_manager
                    get_mcp_manager().connect_persisted()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"MCP auto-connect on startup failed: {exc}")

            threading.Thread(
                target=_reconnect_mcp, name="mcp-auto-connect", daemon=True
            ).start()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not launch MCP auto-connect thread: {exc}")

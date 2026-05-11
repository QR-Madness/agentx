import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AgentxAiConfig(AppConfig):
    default_auto_field: str = 'django.db.models.BigAutoField'
    name: str = 'agentx_ai'

    def ready(self) -> None:
        # Avoid starting the worker during management commands like migrate,
        # test, or makemigrations — only run under the live server.
        run_main = os.environ.get("RUN_MAIN") == "true"
        is_server = any(
            arg in ("runserver", "uvicorn", "daphne") for arg in os.sys.argv  # type: ignore[attr-defined]
        ) or os.environ.get("AGENTX_ENABLE_BG_WORKER") == "1"

        if not (run_main or is_server):
            return

        try:
            from .background import start_worker
            start_worker()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Background chat worker failed to start: {exc}")

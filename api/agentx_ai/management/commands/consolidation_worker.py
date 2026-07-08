"""
Django management command that runs the background memory consolidation worker.

Runs the same ``ConsolidationWorker`` loop (consolidate, patterns, promote,
decay, cleanup, audit partitions, distill_procedures) as a long-lived process —
but under ``django.setup()``, so it inherits the project ``.env``, the shared
``ConfigManager``, and the ``logging_kit`` pipeline (its logs land in
``agentx.log`` / ``/api/logs`` / archives instead of vanishing to stdout).

Replaces the old bare ``python -m …consolidation.worker`` entrypoint, which
skipped Django bootstrap and therefore never saw env-only provider keys (e.g.
OpenRouter) and never persisted its logs.

Usage:
    python manage.py consolidation_worker
"""

from __future__ import annotations

import logging

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Run the background memory consolidation worker (long-lived process)."

    def handle(self, *args, **options):
        # Settings (config.json) is the source of truth for provider keys; seed
        # it from `.env` once (this process may boot before the API has). Best
        # effort — the seed logs and swallows its own failures.
        from agentx_ai.config import get_config_manager

        get_config_manager().seed_provider_keys_from_env()

        from agentx_ai.kit.agent_memory.consolidation.worker import ConsolidationWorker

        ConsolidationWorker().run()

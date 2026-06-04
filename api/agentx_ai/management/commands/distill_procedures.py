"""
Django management command to run the procedural distillation job on demand.

Distills pending ``procedure_candidates`` (corrections/steers + explicit user
rules staged by the encode loop) into durable, scoped Procedures via the
async-aware consolidation registry — the same path the autonomous worker and the
manual /api/jobs trigger use.

Usage:
    python manage.py distill_procedures
"""

from __future__ import annotations

import asyncio
import json
import logging

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Distill pending procedure candidates into scoped Procedures (Slice 1)."

    def handle(self, *args, **options):
        from agentx_ai.kit.agent_memory.consolidation import JobRegistry

        registry = JobRegistry.get_instance()
        result = asyncio.run(registry.run_job("distill_procedures"))
        self.stdout.write(json.dumps(result, indent=2, default=str))

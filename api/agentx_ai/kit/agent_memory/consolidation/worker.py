"""Background worker for memory consolidation jobs."""

import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Optional

from ..connections import RedisConnection
from ..config import get_settings
from ..audit import MemoryAuditLogger
from .jobs import (
    consolidate_episodic_to_semantic,
    detect_patterns,
    apply_memory_decay,
    cleanup_old_memories,
    manage_audit_partitions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


class ConsolidationWorker:
    """Background worker that runs memory consolidation jobs."""

    def __init__(self):
        self.redis = RedisConnection.get_client()
        self.running = True
        self._audit_logger = MemoryAuditLogger(settings)
        self.jobs = {
            "consolidate": {
                "func": consolidate_episodic_to_semantic,
                "interval_minutes": 15
            },
            "patterns": {
                "func": detect_patterns,
                "interval_minutes": 60
            },
            "decay": {
                "func": apply_memory_decay,
                "interval_minutes": 60 * 24  # Daily
            },
            "cleanup": {
                "func": cleanup_old_memories,
                "interval_minutes": 60 * 24  # Daily
            },
            "audit_partitions": {
                "func": manage_audit_partitions,
                "interval_minutes": 60 * 24  # Daily
            }
        }

        # Register signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False

    def _should_run_job(self, job_name: str, interval_minutes: int) -> bool:
        """
        Check if job should run based on last run time.

        Args:
            job_name: Name of the job
            interval_minutes: Interval in minutes

        Returns:
            True if job should run
        """
        last_run_key = f"consolidation:last_run:{job_name}"
        last_run = self.redis.get(last_run_key)

        if not last_run:
            return True

        last_run_time = datetime.fromisoformat(last_run)
        return datetime.utcnow() - last_run_time > timedelta(minutes=interval_minutes)

    def _mark_job_run(self, job_name: str):
        """
        Mark job as run.

        Args:
            job_name: Name of the job
        """
        last_run_key = f"consolidation:last_run:{job_name}"
        self.redis.set(last_run_key, datetime.utcnow().isoformat())

    def run(self):
        """Main worker loop."""
        logger.info("Consolidation worker started")

        while self.running:
            for job_name, job_config in self.jobs.items():
                if not self.running:
                    break

                if self._should_run_job(job_name, job_config["interval_minutes"]):
                    logger.info(f"Running job: {job_name}")
                    start_time = time.perf_counter()
                    success = True
                    error_msg = None
                    items_processed = 0

                    try:
                        result = job_config["func"]()
                        # Extract items processed if the job returns a count
                        if isinstance(result, int):
                            items_processed = result
                        elif isinstance(result, dict) and "items_processed" in result:
                            items_processed = result["items_processed"]

                        self._mark_job_run(job_name)
                        logger.info(f"Job completed: {job_name}")
                    except Exception as e:
                        success = False
                        error_msg = str(e)
                        logger.error(f"Job failed: {job_name} - {e}")
                    finally:
                        latency_ms = int((time.perf_counter() - start_time) * 1000)
                        self._audit_logger.log_job(
                            job_name=job_name,
                            items_processed=items_processed,
                            latency_ms=latency_ms,
                            success=success,
                            error_message=error_msg,
                        )

            # Sleep between checks
            time.sleep(60)

        logger.info("Consolidation worker stopped")


if __name__ == "__main__":
    worker = ConsolidationWorker()
    worker.run()

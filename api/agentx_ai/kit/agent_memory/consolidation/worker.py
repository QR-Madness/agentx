"""Background worker for memory consolidation jobs."""

import time
import signal
import logging
import json
from datetime import datetime, timedelta
from uuid import uuid4

from ..connections import RedisConnection
from ..config import get_settings
from ..audit import MemoryAuditLogger
from .jobs import (
    consolidate_episodic_to_semantic,
    detect_patterns,
    apply_memory_decay,
    cleanup_old_memories,
    manage_audit_partitions,
    promote_to_global
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
        self.worker_id = str(uuid4())[:8]
        self.started_at = datetime.utcnow()
        self.jobs_run_count = 0
        self._last_heartbeat = datetime.utcnow()

        # Jobs with configurable intervals from settings
        self.jobs = {
            "consolidate": {
                "func": consolidate_episodic_to_semantic,
                "interval_minutes": settings.job_consolidate_interval
            },
            "patterns": {
                "func": detect_patterns,
                "interval_minutes": settings.job_patterns_interval
            },
            "promote": {
                "func": promote_to_global,
                "interval_minutes": settings.job_promote_interval
            },
            "decay": {
                "func": apply_memory_decay,
                "interval_minutes": settings.job_decay_interval
            },
            "cleanup": {
                "func": cleanup_old_memories,
                "interval_minutes": settings.job_cleanup_interval
            },
            "audit_partitions": {
                "func": manage_audit_partitions,
                "interval_minutes": settings.job_audit_partitions_interval
            }
        }

        # Register signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
        self._clear_heartbeat()

    def _send_heartbeat(self):
        """Send worker heartbeat to Redis."""
        key = f"worker:heartbeat:{self.worker_id}"
        data = {
            "worker_id": self.worker_id,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "jobs_run": self.jobs_run_count,
            "status": "running"
        }
        self.redis.setex(key, settings.worker_heartbeat_ttl, json.dumps(data))
        self._last_heartbeat = datetime.utcnow()

    def _clear_heartbeat(self):
        """Clear worker heartbeat on shutdown."""
        key = f"worker:heartbeat:{self.worker_id}"
        self.redis.delete(key)

    def _cleanup_stale_workers(self):
        """Remove stale heartbeats from crashed workers."""
        pattern = "worker:heartbeat:*"
        stale_threshold = timedelta(seconds=settings.worker_heartbeat_ttl * 2)

        for key in self.redis.scan_iter(match=pattern):
            try:
                data = self.redis.get(key)
                if not data:
                    continue

                worker_data = json.loads(data)
                last_heartbeat = datetime.fromisoformat(worker_data["last_heartbeat"])

                if datetime.utcnow() - last_heartbeat > stale_threshold:
                    worker_id = worker_data.get("worker_id", "unknown")
                    self.redis.delete(key)
                    logger.warning(f"Cleaned up stale worker: {worker_id}")
            except Exception as e:
                logger.debug(f"Error checking worker heartbeat {key}: {e}")

    def _maybe_send_heartbeat(self):
        """Send heartbeat if interval has elapsed."""
        if datetime.utcnow() - self._last_heartbeat >= timedelta(seconds=settings.worker_heartbeat_interval):
            self._send_heartbeat()

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
        logger.info(f"Consolidation worker started (id={self.worker_id})")

        # Cleanup stale workers on startup
        self._cleanup_stale_workers()

        # Send initial heartbeat
        self._send_heartbeat()

        while self.running:
            # Send heartbeat if interval has elapsed
            self._maybe_send_heartbeat()

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
                        self.jobs_run_count += 1
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

        logger.info(f"Consolidation worker stopped (id={self.worker_id}, jobs_run={self.jobs_run_count})")


if __name__ == "__main__":
    worker = ConsolidationWorker()
    worker.run()

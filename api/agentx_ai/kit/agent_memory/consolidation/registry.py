"""Central registry for consolidation jobs with status tracking."""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone

from ..connections import RedisConnection
from ..config import get_settings
from ..audit import MemoryAuditLogger
from .jobs import (
    consolidate_episodic_to_semantic,
    detect_patterns,
    apply_memory_decay,
    cleanup_old_memories,
    manage_audit_partitions,
    promote_to_global,
    link_facts_to_entities,
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class JobDefinition:
    """Definition of a consolidation job."""

    name: str
    func: Callable[[], Dict[str, Any]]
    interval_minutes: int
    description: str


@dataclass
class JobStatus:
    """Current status of a job."""

    name: str
    description: str
    interval_minutes: int
    status: str  # 'idle' | 'running' | 'disabled'
    last_run: Optional[str]
    last_success: Optional[str]
    last_error: Optional[str]
    run_count: int
    success_count: int
    failure_count: int
    avg_duration_ms: int
    success_rate: float


@dataclass
class JobHistoryEntry:
    """Single execution record for a job."""

    timestamp: str
    duration_ms: int
    success: bool
    items_processed: int
    metrics: Dict[str, Any]
    error: Optional[str] = None


class JobRegistry:
    """
    Central registry for all consolidation jobs with status tracking.

    Provides:
    - Job definitions with metadata
    - Status tracking via Redis
    - Manual job execution
    - Enable/disable functionality
    - Execution history
    """

    _instance: Optional["JobRegistry"] = None

    # Redis key patterns
    KEY_STATUS = "job:status:{name}"
    KEY_METRICS = "job:metrics:{name}"
    KEY_HISTORY = "job:history:{name}"
    KEY_DISABLED = "job:disabled:{name}"
    MAX_HISTORY = 20

    def __init__(self):
        self.redis = RedisConnection.get_client()
        self._audit_logger = MemoryAuditLogger(settings)

        # Register all jobs with descriptions
        self._jobs: Dict[str, JobDefinition] = {
            "consolidate": JobDefinition(
                name="consolidate",
                func=consolidate_episodic_to_semantic,
                interval_minutes=settings.job_consolidate_interval,
                description="Extract entities, facts, and relationships from recent conversations",
            ),
            "patterns": JobDefinition(
                name="patterns",
                func=detect_patterns,
                interval_minutes=settings.job_patterns_interval,
                description="Analyze successful conversations to learn procedural patterns",
            ),
            "promote": JobDefinition(
                name="promote",
                func=promote_to_global,
                interval_minutes=settings.job_promote_interval,
                description="Promote high-quality facts and entities to global channel",
            ),
            "decay": JobDefinition(
                name="decay",
                func=apply_memory_decay,
                interval_minutes=settings.job_decay_interval,
                description="Apply time-based decay to memory salience scores",
            ),
            "cleanup": JobDefinition(
                name="cleanup",
                func=cleanup_old_memories,
                interval_minutes=settings.job_cleanup_interval,
                description="Archive or delete old, low-salience memories",
            ),
            "audit_partitions": JobDefinition(
                name="audit_partitions",
                func=manage_audit_partitions,
                interval_minutes=settings.job_audit_partitions_interval,
                description="Manage PostgreSQL audit log partitions",
            ),
            "entity_linking": JobDefinition(
                name="entity_linking",
                func=link_facts_to_entities,
                interval_minutes=settings.job_entity_linking_interval,
                description="Link facts to existing entities via embedding similarity",
            ),
        }

    @classmethod
    def get_instance(cls) -> "JobRegistry":
        """Get singleton instance of JobRegistry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_metrics(self, name: str) -> Dict[str, Any]:
        """Get metrics for a job from Redis."""
        key = self.KEY_METRICS.format(name=name)
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return {
            "run_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_duration_ms": 0,
            "last_success": None,
            "last_error": None,
        }

    def _update_metrics(
        self,
        name: str,
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
    ) -> None:
        """Update metrics after job execution."""
        metrics = self._get_metrics(name)
        metrics["run_count"] += 1
        metrics["total_duration_ms"] += duration_ms

        if success:
            metrics["success_count"] += 1
            metrics["last_success"] = datetime.now(timezone.utc).isoformat()
        else:
            metrics["failure_count"] += 1
            metrics["last_error"] = error

        key = self.KEY_METRICS.format(name=name)
        self.redis.set(key, json.dumps(metrics))

    def _add_history(
        self,
        name: str,
        entry: JobHistoryEntry,
    ) -> None:
        """Add execution record to job history."""
        key = self.KEY_HISTORY.format(name=name)
        self.redis.lpush(key, json.dumps(asdict(entry)))
        self.redis.ltrim(key, 0, self.MAX_HISTORY - 1)

    def _set_status(self, name: str, status: str) -> None:
        """Set current status of a job."""
        key = self.KEY_STATUS.format(name=name)
        self.redis.set(key, status)

    def _get_status(self, name: str) -> str:
        """Get current status of a job."""
        # Check if disabled first
        disabled_key = self.KEY_DISABLED.format(name=name)
        if self.redis.exists(disabled_key):
            return "disabled"

        # Check if running
        status_key = self.KEY_STATUS.format(name=name)
        status = self.redis.get(status_key)
        return status if status else "idle"

    def _get_last_run(self, name: str) -> Optional[str]:
        """Get last run timestamp from consolidation keys."""
        key = f"consolidation:last_run:{name}"
        return self.redis.get(key)

    def list_jobs(self) -> List[JobStatus]:
        """List all jobs with current status."""
        result = []

        for name, job_def in self._jobs.items():
            metrics = self._get_metrics(name)
            status = self._get_status(name)
            last_run = self._get_last_run(name)

            run_count = metrics.get("run_count", 0)
            success_count = metrics.get("success_count", 0)
            avg_duration = 0
            if run_count > 0:
                avg_duration = metrics.get("total_duration_ms", 0) // run_count

            success_rate = 0.0
            if run_count > 0:
                success_rate = success_count / run_count

            result.append(
                JobStatus(
                    name=name,
                    description=job_def.description,
                    interval_minutes=job_def.interval_minutes,
                    status=status,
                    last_run=last_run,
                    last_success=metrics.get("last_success"),
                    last_error=metrics.get("last_error"),
                    run_count=run_count,
                    success_count=success_count,
                    failure_count=metrics.get("failure_count", 0),
                    avg_duration_ms=avg_duration,
                    success_rate=round(success_rate, 3),
                )
            )

        return result

    def get_job(self, name: str) -> Optional[JobStatus]:
        """Get status for a specific job."""
        if name not in self._jobs:
            return None

        jobs = self.list_jobs()
        for job in jobs:
            if job.name == name:
                return job
        return None

    def get_job_history(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history for a job."""
        if name not in self._jobs:
            return []

        key = self.KEY_HISTORY.format(name=name)
        entries = self.redis.lrange(key, 0, limit - 1)
        return [json.loads(entry) for entry in entries]

    def run_job(self, name: str) -> Dict[str, Any]:
        """
        Manually run a job.

        Returns:
            Dict with success, duration_ms, and result
        """
        if name not in self._jobs:
            return {"success": False, "error": f"Unknown job: {name}"}

        job_def = self._jobs[name]

        # Check if already running
        if self._get_status(name) == "running":
            return {"success": False, "error": "Job is already running"}

        # Mark as running
        self._set_status(name, "running")

        start_time = time.perf_counter()
        success = True
        error_msg = None
        result: Dict[str, Any] = {}
        items_processed = 0

        try:
            result = job_def.func() or {}
            if isinstance(result, dict):
                items_processed = result.get("items_processed", 0)
        except Exception as e:
            success = False
            error_msg = str(e)
            logger.error(f"Manual job run failed: {name} - {e}")
        finally:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Update metrics
            self._update_metrics(name, success, duration_ms, error_msg)

            # Add to history
            self._add_history(
                name,
                JobHistoryEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    duration_ms=duration_ms,
                    success=success,
                    items_processed=items_processed,
                    metrics=result if isinstance(result, dict) else {},
                    error=error_msg,
                ),
            )

            # Update last run timestamp
            last_run_key = f"consolidation:last_run:{name}"
            self.redis.set(last_run_key, datetime.now(timezone.utc).isoformat())

            # Clear running status
            self._set_status(name, "idle")

            # Log to audit
            self._audit_logger.log_job(
                job_name=name,
                items_processed=items_processed,
                latency_ms=duration_ms,
                success=success,
                error_message=error_msg,
                metadata={"manual": True},
            )

        return {
            "success": success,
            "duration_ms": duration_ms,
            "result": result if isinstance(result, dict) else {},
            "error": error_msg,
        }

    def disable_job(self, name: str) -> bool:
        """Disable a job from scheduled execution."""
        if name not in self._jobs:
            return False

        key = self.KEY_DISABLED.format(name=name)
        self.redis.set(key, "1")
        return True

    def enable_job(self, name: str) -> bool:
        """Enable a previously disabled job."""
        if name not in self._jobs:
            return False

        key = self.KEY_DISABLED.format(name=name)
        self.redis.delete(key)
        return True

    def is_disabled(self, name: str) -> bool:
        """Check if a job is disabled."""
        key = self.KEY_DISABLED.format(name=name)
        return bool(self.redis.exists(key))

    def run_consolidation_pipeline(
        self, jobs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the consolidation pipeline (multiple jobs).

        Args:
            jobs: List of job names to run. If None, runs consolidate, patterns, promote.

        Returns:
            Combined results from all jobs
        """
        if jobs is None:
            jobs = ["consolidate", "patterns", "promote"]

        results: Dict[str, Any] = {}
        errors: List[str] = []
        total_start = time.perf_counter()

        for job_name in jobs:
            if job_name not in self._jobs:
                errors.append(f"Unknown job: {job_name}")
                continue

            result = self.run_job(job_name)
            if result.get("success"):
                results[job_name] = result.get("result", {})
            else:
                error = result.get("error", "Unknown error")
                errors.append(f"{job_name}: {error}")
                results[job_name] = {"error": error}

        total_duration = int((time.perf_counter() - total_start) * 1000)

        return {
            "success": len(errors) == 0,
            "duration_ms": total_duration,
            "results": results,
            "errors": errors,
        }

    def get_worker_status(self) -> Optional[Dict[str, Any]]:
        """Get status of the background worker if running."""
        pattern = "worker:heartbeat:*"
        for key in self.redis.scan_iter(match=pattern):
            data = self.redis.get(key)
            if data:
                worker_data = json.loads(data)
                # Calculate uptime
                started_at = datetime.fromisoformat(worker_data["started_at"])
                uptime = (datetime.now(timezone.utc) - started_at).total_seconds()
                return {
                    "id": worker_data.get("worker_id"),
                    "status": worker_data.get("status", "unknown"),
                    "uptime_seconds": int(uptime),
                    "jobs_run": worker_data.get("jobs_run", 0),
                    "last_heartbeat": worker_data.get("last_heartbeat"),
                }
        return None

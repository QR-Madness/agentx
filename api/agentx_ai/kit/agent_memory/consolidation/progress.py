"""Consolidation progress publisher via Redis pub/sub.

Provides a ConsolidationProgress helper that:
- Publishes progress events to Redis channel 'consolidation:progress'
- Manages 'consolidation:active' state key with TTL for durability
- Allows any SSE endpoint to subscribe and relay events to clients
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from ..connections import RedisConnection

logger = logging.getLogger(__name__)

# Redis keys and channels
CHANNEL_PROGRESS = "consolidation:progress"
KEY_ACTIVE = "consolidation:active"
ACTIVE_TTL_SECONDS = 300  # 5 minutes, refreshed during run


class ConsolidationProgress:
    """
    Manages consolidation progress broadcasting via Redis pub/sub.

    Usage:
        progress = ConsolidationProgress(jobs=["consolidate", "patterns"])
        progress.start()
        progress.job_start("consolidate", 1, 3)
        progress.emit("consolidate", "processing", {"conversation": "1 of 5"})
        progress.job_done("consolidate", True, 1200, {"entities": 5})
        progress.complete({"success": True, ...})
    """

    def __init__(
        self,
        jobs: List[str],
        triggered_by: str = "manual",
        run_id: Optional[str] = None,
    ):
        self.run_id = run_id or str(uuid4())[:8]
        self.jobs = jobs
        self.triggered_by = triggered_by
        self.redis = RedisConnection.get_client()
        self._started_at = datetime.now(timezone.utc)

    def _publish(self, event: str, data: Dict[str, Any]) -> None:
        """Publish a progress event to Redis pub/sub."""
        payload = {
            "event": event,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        try:
            self.redis.publish(CHANNEL_PROGRESS, json.dumps(payload))
        except Exception as e:
            logger.warning(f"Failed to publish consolidation progress: {e}")

    def _refresh_active(self) -> None:
        """Refresh the consolidation:active key TTL."""
        try:
            active_data = json.dumps({
                "run_id": self.run_id,
                "started_at": self._started_at.isoformat(),
                "jobs": self.jobs,
                "triggered_by": self.triggered_by,
            })
            self.redis.set(KEY_ACTIVE, active_data, ex=ACTIVE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to refresh consolidation:active key: {e}")

    def start(self) -> None:
        """Signal pipeline start."""
        self._refresh_active()
        self._publish("start", {
            "jobs": self.jobs,
            "total_jobs": len(self.jobs),
            "triggered_by": self.triggered_by,
        })

    def job_start(self, job_name: str, index: int, total: int) -> None:
        """Signal a job is starting."""
        self._refresh_active()
        self._publish("job_start", {
            "job": job_name,
            "index": index,
            "total": total,
        })

    def emit(self, job_name: str, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Emit a granular progress event during processing."""
        self._refresh_active()
        data: Dict[str, Any] = {"job": job_name, "stage": stage}
        if details:
            data.update(details)
        self._publish("progress", data)

    def job_done(
        self,
        job_name: str,
        success: bool,
        duration_ms: int,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Signal a job completed."""
        self._publish("job_done", {
            "job": job_name,
            "success": success,
            "duration_ms": duration_ms,
            "result": result or {},
        })

    def complete(self, result: Dict[str, Any]) -> None:
        """Signal the entire pipeline completed. Clears active key."""
        self._publish("done", result)
        try:
            self.redis.delete(KEY_ACTIVE)
        except Exception as e:
            logger.warning(f"Failed to clear consolidation:active key: {e}")

    def error(self, message: str) -> None:
        """Signal an error."""
        self._publish("error", {"error": message})
        try:
            self.redis.delete(KEY_ACTIVE)
        except Exception as e:
            logger.warning(f"Failed to clear consolidation:active key on error: {e}")

    def as_callback(self) -> Callable[..., None]:
        """
        Return a callback function suitable for injection into job functions.

        The callback signature: callback(stage: str, details: dict, job: str)
        """
        def callback(stage: str, details: Optional[Dict[str, Any]] = None, job: str = "") -> None:
            self.emit(job, stage, details)
        return callback


# --- Helper functions ---

def get_active_consolidation() -> Optional[Dict[str, Any]]:
    """
    Check if a consolidation run is currently active.

    Returns:
        Dict with run_id, started_at, jobs, triggered_by if active, else None
    """
    try:
        redis = RedisConnection.get_client()
        data = redis.get(KEY_ACTIVE)
        if data and isinstance(data, (str, bytes)):
            return json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to check consolidation:active: {e}")
    return None

"""
Redis-backed plan execution state store.

Tracks active plan progress using Redis Hashes with automatic TTL expiration.
State tracking is best-effort — execution continues even if Redis is unavailable.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PLAN_KEY_PREFIX = "plan"
PLAN_TTL_SECONDS = 3600  # 1 hour, matching working memory


def _get_redis_client():
    """Get Redis client from memory connections (lazy import)."""
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


class PlanStateStore:
    """
    Redis-backed storage for active plan execution state.

    Each plan is stored as a Redis Hash with fields for overall status
    and per-subtask status/results. Key pattern: plan:{session_id}:{plan_id}
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    def _key(self, plan_id: str) -> str:
        return f"{PLAN_KEY_PREFIX}:{self.session_id}:{plan_id}"

    def create(self, plan_id: str, plan) -> None:
        """Initialize plan state in Redis."""
        try:
            client = _get_redis_client()
            key = self._key(plan_id)
            now = datetime.now(timezone.utc).isoformat()

            fields = {
                "status": "active",
                "task": plan.task[:500],
                "complexity": plan.complexity.value,
                "subtask_count": str(len(plan.steps)),
                "completed_count": "0",
                "created_at": now,
                "updated_at": now,
            }

            for step in plan.steps:
                fields[f"subtask:{step.id}:status"] = "pending"
                fields[f"subtask:{step.id}:description"] = step.description[:500]

            client.hset(key, mapping=fields)
            client.expire(key, PLAN_TTL_SECONDS)
            logger.debug(f"Created plan state: {key} ({len(plan.steps)} subtasks)")
        except Exception as e:
            logger.warning(f"Failed to create plan state in Redis: {e}")

    def update_subtask(
        self,
        plan_id: str,
        subtask_id: int,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Atomically update a subtask's status and optional result/error."""
        try:
            client = _get_redis_client()
            key = self._key(plan_id)
            now = datetime.now(timezone.utc).isoformat()

            fields = {
                f"subtask:{subtask_id}:status": status,
                "updated_at": now,
            }
            if result is not None:
                fields[f"subtask:{subtask_id}:result"] = result[:2000]
            if error is not None:
                fields[f"subtask:{subtask_id}:error"] = error[:1000]

            client.hset(key, mapping=fields)

            if status == "complete":
                client.hincrby(key, "completed_count", 1)
        except Exception as e:
            logger.warning(f"Failed to update subtask {subtask_id} state: {e}")

    def get_status(self, plan_id: str) -> Optional[dict]:
        """Read full plan state."""
        try:
            client = _get_redis_client()
            data = client.hgetall(self._key(plan_id))
            return data if data else None
        except Exception as e:
            logger.warning(f"Failed to read plan state: {e}")
            return None

    def mark_complete(self, plan_id: str, status: str = "complete") -> None:
        """Mark plan as complete or failed."""
        try:
            client = _get_redis_client()
            key = self._key(plan_id)
            now = datetime.now(timezone.utc).isoformat()
            client.hset(key, mapping={"status": status, "updated_at": now})
        except Exception as e:
            logger.warning(f"Failed to mark plan {status}: {e}")

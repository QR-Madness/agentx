"""
Redis-backed plan execution state store.

Tracks active plan progress using Redis Hashes with automatic TTL expiration.
State tracking is best-effort — execution continues even if Redis is unavailable.
"""

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from .planner import TaskPlan

logger = logging.getLogger(__name__)

PLAN_KEY_PREFIX = "plan"
PLAN_TTL_SECONDS = 3600  # 1 hour, matching working memory

# Subtask statuses that count as terminal — a subtask in any of these is done
# (for good or ill) and is not re-executed on resume. "running"/"pending" are
# non-terminal, so a process that died mid-subtask re-runs that subtask.
_TERMINAL_STATUSES = {"complete", "failed", "skipped", "abandoned"}

# Plan statuses from which a plan can be resumed: "active" (in-flight) and
# "interrupted" (hard-stopped — Stop/GeneratorExit — with non-terminal work left).
_RESUMABLE_STATUSES = {"active", "interrupted"}


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
                # Full structural snapshot so the plan can be rebuilt and
                # resumed after a process death (the per-subtask flat fields
                # below stay the live source of truth for status/result/UI).
                "plan_json": json.dumps(plan.to_dict()),
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
            logger.info("plan=%s subtask=%s status=%s", plan_id, subtask_id, status)
        except Exception as e:
            logger.warning(f"Failed to update subtask {subtask_id} state: {e}")

    def get_status(self, plan_id: str) -> Optional[dict]:
        """Read full plan state."""
        try:
            client = _get_redis_client()
            data = cast(Optional[dict], client.hgetall(self._key(plan_id)))
            return data if data else None
        except Exception as e:
            logger.warning(f"Failed to read plan state: {e}")
            return None

    @staticmethod
    def _overlay_result(status: str, stored_result: Optional[str], error: Optional[str]) -> Optional[str]:
        """Reconstruct a subtask's in-memory ``result`` from its persisted status.

        Only "complete" stores a real result; the other terminal states are
        re-derived to the same sentinel strings the executor uses, so the
        rebuilt plan's dependency-skip and synthesis logic behave identically.
        """
        if status == "complete":
            return stored_result
        if status == "failed":
            return f"[FAILED: {error}]" if error else "[FAILED]"
        if status == "skipped":
            return "[SKIPPED: all dependencies failed]"
        if status == "abandoned":
            return "[ABANDONED: plan cancelled]"
        return None  # pending / running → not yet produced

    def load_plan(self, plan_id: str) -> Optional["TaskPlan"]:
        """Rebuild a ``TaskPlan`` from Redis for resumption.

        Reconstructs the structural skeleton from the ``plan_json`` snapshot,
        then overlays each subtask's *live* status/result so completed work is
        preserved and only non-terminal subtasks re-execute. Returns ``None``
        when there's no resumable state (missing, expired, or already finished).
        """
        from .planner import TaskPlan

        data = self.get_status(plan_id)
        if not data:
            return None
        # `active` (in-flight) and `interrupted` (hard-stopped with work left) are
        # resumable; a complete/failed/cancelled plan is done.
        if data.get("status") not in _RESUMABLE_STATUSES:
            return None
        raw = data.get("plan_json")
        if not raw:
            # Pre-B1 snapshot without the structural blob — can't safely resume.
            return None

        try:
            plan = TaskPlan.from_dict(json.loads(raw))
        except Exception as e:
            logger.warning(f"Failed to rebuild plan {plan_id} from snapshot: {e}")
            return None

        for step in plan.steps:
            status = data.get(f"subtask:{step.id}:status", "pending")
            if status in _TERMINAL_STATUSES:
                step.completed = True
                step.result = self._overlay_result(
                    status,
                    data.get(f"subtask:{step.id}:result"),
                    data.get(f"subtask:{step.id}:error"),
                )
            else:
                # running/pending → reset so it re-executes cleanly on resume.
                step.completed = False
                step.result = None

        return plan

    def is_resumable(self, plan_id: str) -> bool:
        """True when a plan still has executable (non-terminal) work to resume."""
        plan = self.load_plan(plan_id)
        return plan is not None and not plan.is_complete()

    def get_ttl(self, plan_id: str) -> Optional[int]:
        """Remaining time-to-live (seconds) for the plan's Redis key.

        Returns the seconds until the snapshot expires (how long it stays
        resumable), or ``None`` when the key is missing or has no expiry.
        """
        try:
            client = _get_redis_client()
            ttl = client.ttl(self._key(plan_id))
            return int(ttl) if ttl is not None and ttl >= 0 else None
        except Exception as e:
            logger.warning(f"Failed to read TTL for plan {plan_id}: {e}")
            return None

    def mark_complete(self, plan_id: str, status: str = "complete") -> None:
        """Mark the plan's terminal/transition status (complete|failed|cancelled|interrupted)."""
        try:
            client = _get_redis_client()
            key = self._key(plan_id)
            now = datetime.now(timezone.utc).isoformat()
            client.hset(key, mapping={"status": status, "updated_at": now})
            logger.info("plan=%s status=%s", plan_id, status)
        except Exception as e:
            logger.warning(f"Failed to mark plan {status}: {e}")

    def mark_interrupted(self, plan_id: str) -> None:
        """Mark a plan as hard-stopped with work left — resumable (see _RESUMABLE_STATUSES)."""
        self.mark_complete(plan_id, status="interrupted")

    def clear_cancel(self, plan_id: str) -> None:
        """Best-effort clear the cancel flag (call only AFTER it's been observed,
        so a still-running executor can't miss the cancel request)."""
        try:
            client = _get_redis_client()
            client.hdel(self._key(plan_id), "cancel_requested")
        except Exception as e:
            logger.warning(f"Failed to clear cancel flag for plan {plan_id}: {e}")

    def request_cancel(self, plan_id: str) -> bool:
        """Flag a plan for cancellation. Returns True if the flag was written."""
        try:
            client = _get_redis_client()
            key = self._key(plan_id)
            if not client.exists(key):
                return False
            client.hset(key, "cancel_requested", "1")
            client.expire(key, PLAN_TTL_SECONDS)
            logger.info("plan=%s CANCEL requested", plan_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to request cancel for plan {plan_id}: {e}")
            return False

    def is_cancel_requested(self, plan_id: str) -> bool:
        """Check whether cancellation has been requested for this plan."""
        try:
            client = _get_redis_client()
            value = client.hget(self._key(plan_id), "cancel_requested")
            if value is None:
                return False
            if isinstance(value, bytes):
                value = value.decode()
            return value == "1"
        except Exception as e:
            logger.warning(f"Failed to read cancel flag for plan {plan_id}: {e}")
            return False

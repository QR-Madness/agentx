"""Agent lifecycle hooks — a seam between the agent and side-effecting consumers.

The agent emits lifecycle events (task complete/error, turn stored, tool used)
and subscribers react. Today the only subscriber is ``MemoryRecorder``, which
performs the memory writes that used to be scattered (with their own try/except)
through ``Agent.run()``/``Agent.chat()``. Moving them here lets a second agent —
or a future telemetry/cost consumer — subscribe without editing the agent core.

Note: retrieval (``memory.remember(...)``) is an active query whose result the
agent uses, not a fire-and-forget event, so it stays a direct call in the agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..kit.agent_memory.models import Turn

logger = logging.getLogger(__name__)


@dataclass
class TaskOutcome:
    """Everything the write-side hooks need to record a finished task."""

    task_id: str
    task: str
    status: str  # 'complete' | 'failed' | 'cancelled'
    answer: Optional[str] = None
    total_tokens: int = 0
    total_time_ms: float = 0.0
    reasoning_steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    goal_id: Optional[str] = None
    error: Optional[str] = None


class AgentHooks:
    """Base class of agent lifecycle subscribers. All methods are no-ops;
    subscribers override only the events they care about."""

    def on_task_complete(self, outcome: TaskOutcome) -> None:
        return None

    def on_task_error(self, outcome: TaskOutcome) -> None:
        return None

    def on_turn(self, turn: "Turn") -> None:
        return None

    def on_goal_complete(
        self, goal_id: str, status: str, result: Optional[str] = None
    ) -> None:
        return None

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int,
        error_message: Optional[str],
    ) -> None:
        return None


class MemoryRecorder(AgentHooks):
    """Records agent lifecycle events into the memory subsystem.

    Each memory call keeps its own best-effort guard (preserving the prior
    per-operation fault isolation — e.g. a failed reflection must not block goal
    completion); the agent's dispatcher adds a second guard against a broken
    subscriber.
    """

    def __init__(self, memory: Any):
        self._memory = memory

    def on_task_complete(self, outcome: TaskOutcome) -> None:
        try:
            self._memory.reflect({
                "task_id": outcome.task_id,
                "task": outcome.task[:200],
                "status": "complete",
                "total_tokens": outcome.total_tokens,
                "total_time_ms": outcome.total_time_ms,
                "reasoning_steps": outcome.reasoning_steps,
                "tools_used": outcome.tools_used,
            })
        except Exception as e:
            logger.warning(f"Failed to trigger memory reflection: {e}")

        if outcome.goal_id:
            try:
                goal_status = "abandoned" if outcome.answer == "[CANCELLED]" else "completed"
                self._memory.complete_goal(
                    outcome.goal_id,
                    status=goal_status,
                    result=outcome.answer[:500] if outcome.answer else None,
                )
            except Exception as e:
                logger.warning(f"Failed to complete goal: {e}")

    def on_task_error(self, outcome: TaskOutcome) -> None:
        try:
            self._memory.reflect({
                "task_id": outcome.task_id,
                "task": outcome.task[:200],
                "status": "failed",
                "error": outcome.error,
            })
        except Exception as e:
            logger.warning(f"Failed to trigger memory reflection: {e}")

        if outcome.goal_id:
            try:
                self._memory.complete_goal(
                    outcome.goal_id,
                    status="abandoned",
                    result=f"Task failed: {(outcome.error or '')[:400]}",
                )
            except Exception as e:
                logger.warning(f"Failed to update goal status: {e}")

    def on_turn(self, turn: "Turn") -> None:
        try:
            self._memory.store_turn(turn)
        except Exception as e:
            logger.warning(f"Failed to store turn in memory: {e}")

    def on_goal_complete(
        self, goal_id: str, status: str, result: Optional[str] = None
    ) -> None:
        try:
            self._memory.complete_goal(goal_id, status=status, result=result)
        except Exception as e:
            logger.warning(f"Failed to complete subgoal {goal_id}: {e}")

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int,
        error_message: Optional[str],
    ) -> None:
        try:
            self._memory.record_tool_usage(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                success=success,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to record tool usage in memory: {e}")

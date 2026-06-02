"""
AlloyExecutor — runs one in-flight delegation for a workflow.

The executor is constructed per chat request when ``workflow_id`` is set and
stashed on the supervising Agent as ``_active_alloy_executor``. The streaming
tool loop hands every ``delegate_to`` tool call to ``executor.delegate(...)``,
which spins up a specialist Agent, streams its tokens back as
``delegation_chunk`` SSE events, and returns the final accumulated content as
the supervisor's tool result.
"""

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, AsyncGenerator, Optional

if TYPE_CHECKING:
    from ..providers.pricing import CostEstimate
from uuid import uuid4

from ..agent.profiles import get_profile_manager
from ..agent.session import Session
from ..providers.base import Message, MessageRole
from .models import MemberRole, Workflow

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class AlloyExecutor:
    """Owns one in-flight workflow execution for a single supervising agent."""

    def __init__(
        self,
        supervisor_agent,
        session: Session,
        *,
        workflow: Optional[Workflow] = None,
        channel: Optional[str] = None,
        delegator_agent_id: Optional[str] = None,
        max_delegation_depth: int = 3,
        max_parallel_delegations: int = 3,
    ):
        """Owns one delegating agent's in-flight delegations.

        Two modes:
        - **Workflow** (Agent Alloy): pass ``workflow``; the shared channel and
          delegator id are derived from it and targets must be workflow
          specialists.
        - **Ad-hoc** (Phase 16.4): pass ``channel`` + ``delegator_agent_id``;
          any agent profile (except the delegator) is a valid target.
        """
        self.workflow = workflow
        self.supervisor = supervisor_agent
        self.session = session
        self.max_delegation_depth = max_delegation_depth
        self.max_parallel_delegations = max_parallel_delegations
        # `history` is appended from concurrent delegate() branches (fan-out);
        # guard slot allocation + append with a lock so Turn indices stay unique.
        self.history: list[dict] = []  # {target_agent_id, status, result_preview}
        self._history_lock = asyncio.Lock()

        if workflow is not None:
            self.channel: str = workflow.shared_channel
            self.delegator_agent_id: str = workflow.supervisor_agent_id
        else:
            self.channel = channel or "_global"
            self.delegator_agent_id = delegator_agent_id or ""

    def _validate_target(self, target_agent_id: str) -> Optional[str]:
        """Return an error string if ``target_agent_id`` is not delegable, else None."""
        # No self-delegation (both modes) — satisfies the Phase 16.4 safeguard
        # and is harmless for workflow supervisors (they never list themselves).
        if target_agent_id == self.delegator_agent_id:
            return "an agent cannot delegate to itself"
        if self.workflow is not None:
            member = self.workflow.get_member(target_agent_id)
            if member is None or member.role != MemberRole.SPECIALIST:
                return f"agent_id {target_agent_id!r} is not a specialist in this workflow"
        else:
            if get_profile_manager().get_profile_by_agent_id(target_agent_id) is None:
                return f"no agent profile for agent_id {target_agent_id!r}"
        return None

    async def delegate(
        self,
        target_agent_id: str,
        task: str,
        *,
        tool_call_id: str,
        depth: int = 0,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Run one specialist for one task.

        Yields ``(sse_event_string, partial_text)`` tuples. The final
        ``partial_text`` value is the full specialist content and becomes the
        tool result returned to the supervisor.

        ``depth`` is the caller's delegation depth (0 at the top level). It is a
        parameter rather than instance state so concurrent fan-out branches do
        not race a shared counter; this delegation runs at ``depth + 1``.
        """
        # ------- validate target -------
        err = self._validate_target(target_agent_id)
        if err is not None:
            yield _sse("delegation_complete", {
                "target_agent_id": target_agent_id,
                "tool_call_id": tool_call_id,
                "status": "failed",
                "error": err,
                "result_preview": "",
            }), f"[delegation rejected: {err}]"
            return

        if depth >= self.max_delegation_depth:
            err = f"max delegation depth ({self.max_delegation_depth}) reached"
            yield _sse("delegation_complete", {
                "target_agent_id": target_agent_id,
                "tool_call_id": tool_call_id,
                "status": "failed",
                "error": err,
                "result_preview": "",
            }), f"[delegation rejected: {err}]"
            return

        # ------- resolve specialist profile -------
        pm = get_profile_manager()
        profile = next(
            (p for p in pm.list_profiles() if p.agent_id == target_agent_id),
            None,
        )
        if profile is None:
            err = f"no profile found for agent_id {target_agent_id!r}"
            yield _sse("delegation_complete", {
                "target_agent_id": target_agent_id,
                "tool_call_id": tool_call_id,
                "status": "failed",
                "error": err,
                "result_preview": "",
            }), f"[delegation rejected: {err}]"
            return

        # ------- announce -------
        delegation_id = uuid4().hex[:8]
        yield _sse("delegation_start", {
            "delegation_id": delegation_id,
            "target_agent_id": target_agent_id,
            "tool_call_id": tool_call_id,
            "task": task[:500],
            "depth": depth + 1,
            "supervisor_agent_id": self.delegator_agent_id,
            "shared_channel": self.channel,
        }), ""

        # ------- create child goal (best effort) -------
        child_goal_id: Optional[str] = None
        memory_for_goal = getattr(self.supervisor, "memory", None)
        if memory_for_goal is not None:
            try:
                from ..kit.agent_memory.models import Goal
                goal = Goal(
                    id=str(uuid4()),
                    description=f"[delegation→{target_agent_id}] {task[:400]}",
                    status="active",
                    priority=3,
                    channel=self.channel,
                )
                memory_for_goal.add_goal(goal)
                child_goal_id = goal.id
            except Exception as e:
                logger.warning(f"Failed to create delegation goal: {e}")

        # ------- build specialist agent -------
        from ..agent.core import Agent, AgentConfig
        from ..streaming.tool_loop import streaming_tool_loop, ToolLoopResult

        specialist_config = AgentConfig(
            name=profile.name,
            user_id=self.supervisor.config.user_id,
            default_model=profile.default_model or self.supervisor.config.default_model,
            agent_id=profile.agent_id,
            prompt_profile_id=profile.prompt_profile_id,
            memory_channel=self.channel,
            enable_memory=profile.enable_memory,
            enable_tools=profile.enable_tools,
            max_tool_rounds=self.supervisor.config.max_tool_rounds,
        )
        specialist = Agent(specialist_config)
        # Specialists do not re-delegate. Only the supervisor receives the
        # `delegate_to` tool; giving it to leaf specialists creates a
        # self-referential descriptor (the only listed target is the
        # specialist themselves) and routinely confuses smaller models into
        # emitting fake delegation JSON as their assistant text.

        # ------- compose specialist messages -------
        messages = self._build_specialist_messages(profile, task, specialist)

        # ------- pick provider + tool list -------
        provider, model_id = specialist.registry.get_provider_for_model(
            specialist.config.default_model
        )
        tools = specialist._get_tools_for_provider() if profile.enable_tools else None

        # ------- stream specialist run -------
        accumulated = ""
        status = "success"
        error: Optional[str] = None
        loop_result = ToolLoopResult()
        t0 = time.perf_counter()
        try:
            async for event_str in streaming_tool_loop(
                provider, model_id, messages, tools, specialist,
                temperature=profile.temperature,
                max_tokens=4096,
                max_tool_rounds=specialist.config.max_tool_rounds,
                task_context=task,
                emit_trajectory_info=False,
                result=loop_result,
            ):
                # Re-wrap every nested event as a delegation-scoped event so
                # the specialist's activity stays inside the delegation card
                # rather than fragmenting the supervisor's chat flow.
                event_name, _, payload = event_str.partition("\n")
                event_name = event_name.removeprefix("event: ").strip()
                data_line = payload.split("data: ", 1)[1].rstrip() if "data: " in payload else "{}"
                try:
                    inner = json.loads(data_line)
                except json.JSONDecodeError:
                    inner = {}

                if event_name == "chunk":
                    yield _sse("delegation_chunk", {
                        "delegation_id": delegation_id,
                        "target_agent_id": target_agent_id,
                        "content": inner.get("content", ""),
                    }), accumulated
                elif event_name == "tool_call":
                    yield _sse("delegation_tool_call", {
                        "delegation_id": delegation_id,
                        "target_agent_id": target_agent_id,
                        "tool": inner.get("tool"),
                        "tool_call_id": inner.get("tool_call_id"),
                        "arguments": inner.get("arguments", {}),
                    }), accumulated
                elif event_name == "tool_result":
                    yield _sse("delegation_tool_result", {
                        "delegation_id": delegation_id,
                        "target_agent_id": target_agent_id,
                        "tool": inner.get("tool"),
                        "tool_call_id": inner.get("tool_call_id"),
                        "content": inner.get("content", ""),
                        "success": inner.get("success", True),
                        "duration_ms": inner.get("duration_ms"),
                    }), accumulated
                # Drop info/other events — they would create top-level cards.
                accumulated = loop_result.content
        except Exception as e:
            logger.exception(f"Specialist {target_agent_id} failed during delegation")
            status = "failed"
            error = str(e)
            accumulated = accumulated or f"[delegation failed: {error}]"
        duration_ms = (time.perf_counter() - t0) * 1000

        # Reserve a unique history slot + Turn index up front (fan-out safe).
        async with self._history_lock:
            turn_index = len(self.history)
            self.history.append({
                "target_agent_id": target_agent_id,
                "status": status,
                "result_preview": accumulated[:200],
            })

        # ------- cost estimate (reuses the supervisor done-event path) -------
        cost: Optional["CostEstimate"] = None
        try:
            from ..providers.pricing import estimate_cost
            caps = provider.get_capabilities(model_id)
            cost = estimate_cost(caps, loop_result.tokens_in, loop_result.tokens_out)
        except Exception as e:
            logger.warning(f"Failed to estimate delegation cost: {e}")

        # ------- record result in shared channel + close goal -------
        # Store the full accumulated specialist output (no pre-truncation):
        # recall is responsible for budgeting, not the storage layer.
        if memory_for_goal is not None:
            try:
                from ..kit.agent_memory.models import Turn
                delegation_metadata = {
                    "delegation_id": delegation_id,
                    "agent_id": target_agent_id,
                    "task": task[:500],
                }
                memory_for_goal.store_turn(Turn(
                    conversation_id=self.session.id,
                    index=turn_index,
                    role="assistant",
                    content=f"[{target_agent_id} → delegation] {accumulated}",
                    channel=self.channel,
                    metadata=delegation_metadata,
                ))
            except Exception as e:
                logger.warning(f"Failed to store delegation turn: {e}")
        if child_goal_id and memory_for_goal is not None:
            try:
                memory_for_goal.complete_goal(
                    child_goal_id,
                    status="completed" if status == "success" else "blocked",
                    result=accumulated[:500] if accumulated else None,
                )
            except Exception as e:
                logger.warning(f"Failed to close delegation goal: {e}")

        yield _sse("delegation_complete", {
            "delegation_id": delegation_id,
            "target_agent_id": target_agent_id,
            "tool_call_id": tool_call_id,
            "status": status,
            "error": error,
            "result_preview": accumulated[:500],
            "tokens_input": loop_result.tokens_in,
            "tokens_output": loop_result.tokens_out,
            "duration_ms": duration_ms,
            "cost_estimate": cost["cost_total"] if cost else None,
            "cost_currency": cost["currency"] if cost else None,
            "pricing_snapshot": cost["pricing_snapshot"] if cost else None,
        }), accumulated

    # ------------------------------------------------------------------

    def _build_specialist_messages(self, profile, task: str, specialist) -> list[Message]:
        """Compose a self-contained message list for the specialist."""
        from ..prompts import get_prompt_manager

        pm = get_prompt_manager()
        system_prompt = pm.get_system_prompt(
            profile_id=profile.prompt_profile_id,
            agent_name=profile.name,
            agent_system_prompt=profile.system_prompt,
        )

        messages: list[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    (system_prompt or "You are a specialist agent.")
                    + "\n\nYou are operating as part of a multi-agent conversation. "
                    "You were delegated this task by another agent and do not "
                    "have access to the user's original conversation. Focus on "
                    "completing the task and return a concise, complete answer."
                ),
            )
        ]

        # Inject relevant memories from the shared channel if memory is up.
        memory = specialist.memory if profile.enable_memory else None
        if memory is not None:
            try:
                bundle = memory.remember(
                    query=task,
                    top_k=specialist.config.memory_top_k,
                    time_window_hours=specialist.config.memory_time_window_hours,
                )
                if bundle:
                    ctx = bundle.to_context_string(
                        turn_char_limit=specialist.config.memory_recall_turn_chars,
                        max_turns=specialist.config.memory_recall_max_turns,
                    )
                    if ctx:
                        messages.append(Message(
                            role=MessageRole.SYSTEM,
                            content=f"Shared workflow memory:\n{ctx}",
                        ))
            except Exception as e:
                logger.warning(f"Specialist memory recall failed: {e}")

        messages.append(Message(role=MessageRole.USER, content=task))
        return messages

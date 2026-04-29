"""
AlloyExecutor — runs one in-flight delegation for a workflow.

The executor is constructed per chat request when ``workflow_id`` is set and
stashed on the supervising Agent as ``_active_alloy_executor``. The streaming
tool loop hands every ``delegate_to`` tool call to ``executor.delegate(...)``,
which spins up a specialist Agent, streams its tokens back as
``delegation_chunk`` SSE events, and returns the final accumulated content as
the supervisor's tool result.
"""

import json
import logging
from typing import AsyncGenerator, Optional
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
        workflow: Workflow,
        supervisor_agent,
        session: Session,
        *,
        max_delegation_depth: int = 3,
    ):
        self.workflow = workflow
        self.supervisor = supervisor_agent
        self.session = session
        self.max_delegation_depth = max_delegation_depth
        self.depth = 0
        self.history: list[dict] = []  # {target_agent_id, status, result_preview}

    async def delegate(
        self,
        target_agent_id: str,
        task: str,
        *,
        tool_call_id: str,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Run one specialist for one task.

        Yields ``(sse_event_string, partial_text)`` tuples. The final
        ``partial_text`` value is the full specialist content and becomes the
        tool result returned to the supervisor.
        """
        # ------- validate target -------
        member = self.workflow.get_member(target_agent_id)
        if member is None or member.role != MemberRole.SPECIALIST:
            err = (
                f"agent_id {target_agent_id!r} is not a specialist in this workflow"
            )
            yield _sse("delegation_complete", {
                "target_agent_id": target_agent_id,
                "tool_call_id": tool_call_id,
                "status": "failed",
                "error": err,
                "result_preview": "",
            }), f"[delegation rejected: {err}]"
            return

        if self.depth >= self.max_delegation_depth:
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
        self.depth += 1
        delegation_id = uuid4().hex[:8]
        yield _sse("delegation_start", {
            "delegation_id": delegation_id,
            "target_agent_id": target_agent_id,
            "tool_call_id": tool_call_id,
            "task": task[:500],
            "depth": self.depth,
            "supervisor_agent_id": self.workflow.supervisor_agent_id,
            "shared_channel": self.workflow.shared_channel,
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
                    channel=self.workflow.shared_channel,
                )
                memory_for_goal.add_goal(goal)
                child_goal_id = goal.id
            except Exception as e:
                logger.warning(f"Failed to create delegation goal: {e}")

        # ------- build specialist agent -------
        from ..agent.core import Agent, AgentConfig
        from ..streaming.tool_loop import streaming_tool_loop

        specialist_config = AgentConfig(
            name=profile.name,
            user_id=self.supervisor.config.user_id,
            default_model=profile.default_model or self.supervisor.config.default_model,
            agent_id=profile.agent_id,
            prompt_profile_id=profile.prompt_profile_id,
            memory_channel=self.workflow.shared_channel,
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
        try:
            async for event_str, loop_result in streaming_tool_loop(
                provider, model_id, messages, tools, specialist,
                temperature=profile.temperature,
                max_tokens=4096,
                max_tool_rounds=specialist.config.max_tool_rounds,
                task_context=task,
                emit_trajectory_info=False,
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
        finally:
            self.depth -= 1

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
                    index=len(self.history),
                    role="assistant",
                    content=f"[{target_agent_id} → delegation] {accumulated}",
                    channel=self.workflow.shared_channel,
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

        self.history.append({
            "target_agent_id": target_agent_id,
            "status": status,
            "result_preview": accumulated[:200],
        })

        yield _sse("delegation_complete", {
            "delegation_id": delegation_id,
            "target_agent_id": target_agent_id,
            "tool_call_id": tool_call_id,
            "status": status,
            "error": error,
            "result_preview": accumulated[:500],
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
                    + "\n\nYou are operating as part of a multi-agent workflow. "
                    "You were delegated this task by a supervisor and do not "
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

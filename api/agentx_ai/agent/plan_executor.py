"""
Plan Execution Engine.

Iterates TaskPlan subtasks respecting dependency order, executing each
via the provider's streaming or synchronous completion with tool-use loops.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from ..providers.base import Message, MessageRole
from .planner import TaskPlan, Subtask

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class PlanExecutor:
    """
    Executes a TaskPlan by iterating subtasks with dependency ordering.

    For each subtask, builds a self-contained message list and runs
    through the provider's completion with tool-use support.
    """

    def __init__(self, agent, state_store):
        """
        Args:
            agent: Agent instance (provides provider registry, MCP client, tool execution)
            state_store: PlanStateStore for Redis progress tracking
        """
        self.agent = agent
        self.state = state_store
        # Conversation context inherited from the chat request. Set per-call
        # via execute()/execute_streaming(); subtasks reuse only the system
        # framing, synthesis reuses the full thread.
        self.conversation_context: list[Message] = []

    def _complete_subtask_goal(
        self,
        subtask: Subtask,
        status: str,
        result: Optional[str] = None,
    ) -> None:
        """Update the memory goal linked to a subtask, if one exists."""
        memory = getattr(self.agent, "memory", None)
        if not memory or not subtask.goal_id:
            return
        try:
            memory.complete_goal(subtask.goal_id, status=status, result=result)
        except Exception as e:
            logger.warning(f"Failed to complete subgoal {subtask.goal_id}: {e}")

    # ------------------------------------------------------------------
    # Synchronous execution (for Agent.run)
    # ------------------------------------------------------------------

    def execute(self, plan: TaskPlan, context: Optional[list[Message]] = None) -> str:
        """
        Execute a plan synchronously, returning the composed final answer.

        Args:
            plan: The task plan with subtasks
            context: Optional conversation context messages (system prompts +
                prior turns) inherited from the chat request. Subtasks see
                only the system slice; synthesis sees the full thread.

        Returns:
            Composed answer string from all subtask results
        """
        self.conversation_context = context or []
        plan_id = uuid4().hex[:8]
        self.plan_id = plan_id
        self.state.create(plan_id, plan)

        cancelled = False
        while not plan.is_complete():
            if self.state.is_cancel_requested(plan_id):
                cancelled = True
                break

            subtask = plan.get_next_subtask()
            if subtask is None:
                logger.warning("Plan deadlocked — no executable subtask found")
                break

            self.state.update_subtask(plan_id, subtask.id, "running")

            try:
                result = self._execute_subtask_sync(plan, subtask)
                plan.mark_complete(subtask.id, result)
                self.state.update_subtask(plan_id, subtask.id, "complete", result=result)
                self._complete_subtask_goal(
                    subtask, "completed", result[:500] if result else None
                )
                logger.info(f"Subtask {subtask.id} complete: {subtask.description[:60]}")
            except Exception as e:
                logger.error(f"Subtask {subtask.id} failed: {e}")
                self._handle_failure(plan, plan_id, subtask, e)

        if cancelled:
            self._abandon_pending_subtasks(plan, plan_id)
            self.state.mark_complete(plan_id, status="cancelled")
            return "[CANCELLED]"

        final_answer = self._compose_answer_sync(plan)
        self.state.mark_complete(plan_id)
        return final_answer

    # ------------------------------------------------------------------
    # Async streaming execution (for views.py SSE endpoint)
    # ------------------------------------------------------------------

    async def execute_streaming(
        self,
        plan: TaskPlan,
        provider,
        model_id: str,
        tools: Optional[list[dict[str, Any]]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_context_tokens: int = 100000,
        conversation_context: Optional[list[Message]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Execute a plan as an async generator, yielding SSE event strings.

        The caller (generate_sse) iterates with ``async for event in ...``.

        Yields:
            SSE-formatted event strings (plan_start, subtask_start, chunk,
            tool_call, tool_result, subtask_complete, subtask_failed,
            plan_complete)
        """
        self.conversation_context = conversation_context or []
        plan_id = uuid4().hex[:8]
        self.plan_id = plan_id
        start_time = time.time()
        self.state.create(plan_id, plan)

        # Stash metadata on self so the caller can read it after iteration
        self.full_content = ""
        self.tools_used = []
        self.total_tokens_in = 0
        self.total_tokens_out = 0

        yield _sse("plan_start", {
            "plan_id": plan_id,
            "task": plan.task[:200],
            "subtask_count": len(plan.steps),
            "complexity": plan.complexity.value,
        })

        cancelled = False
        while not plan.is_complete():
            if self.state.is_cancel_requested(plan_id):
                cancelled = True
                break

            subtask = plan.get_next_subtask()
            if subtask is None:
                logger.warning("Plan deadlocked — no executable subtask found")
                break

            self.state.update_subtask(plan_id, subtask.id, "running")
            yield _sse("subtask_start", {
                "plan_id": plan_id,
                "subtask_id": subtask.id,
                "description": subtask.description,
                "type": subtask.type.value,
                "progress": plan.get_progress(),
            })

            try:
                self._last_subtask_content = ""
                self._last_subtask_content_source = "final"
                async for event_str in self._execute_subtask_streaming(
                    plan, subtask, provider, model_id, tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_context_tokens=max_context_tokens,
                ):
                    yield event_str

                subtask_content = self._last_subtask_content
                plan.mark_complete(subtask.id, subtask_content)
                self.state.update_subtask(plan_id, subtask.id, "complete", result=subtask_content)
                self._complete_subtask_goal(
                    subtask, "completed", subtask_content[:500] if subtask_content else None
                )

                yield _sse("subtask_complete", {
                    "plan_id": plan_id,
                    "subtask_id": subtask.id,
                    "result_preview": subtask_content[:200] if subtask_content else "",
                    "progress": plan.get_progress(),
                })
                logger.info(
                    f"Subtask {subtask.id} complete [{self._last_subtask_content_source}]: "
                    f"{subtask.description[:60]}"
                )

            except Exception as e:
                logger.error(f"Subtask {subtask.id} failed: {e}")
                self._handle_failure(plan, plan_id, subtask, e)
                yield _sse("subtask_failed", {
                    "plan_id": plan_id,
                    "subtask_id": subtask.id,
                    "error": str(e)[:500],
                    "progress": plan.get_progress(),
                })

        if cancelled:
            self._abandon_pending_subtasks(plan, plan_id)
            total_time = (time.time() - start_time) * 1000
            self.state.mark_complete(plan_id, status="cancelled")
            yield _sse("plan_cancelled", {
                "plan_id": plan_id,
                "subtask_count": len(plan.steps),
                "completed_count": sum(
                    1 for s in plan.steps
                    if s.result and not s.result.startswith("[FAILED")
                    and not s.result.startswith("[ABANDONED")
                ),
                "total_time_ms": round(total_time, 1),
            })
            return

        # Synthesis: compose final answer from subtask results
        async for event_str in self._compose_answer_streaming(
            plan, provider, model_id,
            temperature=temperature, max_tokens=max_tokens,
        ):
            yield event_str

        total_time = (time.time() - start_time) * 1000
        self.state.mark_complete(plan_id)

        yield _sse("plan_complete", {
            "plan_id": plan_id,
            "subtask_count": len(plan.steps),
            "completed_count": sum(
                1 for s in plan.steps
                if s.result and not s.result.startswith("[FAILED")
            ),
            "total_time_ms": round(total_time, 1),
        })

    # ------------------------------------------------------------------
    # Internal: subtask execution
    # ------------------------------------------------------------------

    def _execute_subtask_sync(self, plan: TaskPlan, subtask: Subtask) -> str:
        """Execute a single subtask synchronously using _complete_with_tools."""
        messages = self._build_subtask_messages(plan, subtask)
        provider, model_id = self.agent.registry.get_provider_for_model(
            self.agent.config.default_model
        )
        use_tools = bool(subtask.tools_needed)
        tools = self.agent._get_tools_for_provider() if use_tools else None

        result, _ = self.agent._complete_with_tools(
            provider, model_id, messages, tools,
            temperature=0.7, max_tokens=2000,
        )
        return result.content

    async def _execute_subtask_streaming(
        self,
        plan: TaskPlan,
        subtask: Subtask,
        provider,
        model_id: str,
        tools: Optional[list[dict[str, Any]]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_context_tokens: int = 100000,
    ) -> AsyncGenerator[str, None]:
        """Stream a single subtask's execution, yielding SSE events."""
        from ..streaming.tool_loop import streaming_tool_loop

        messages = self._build_subtask_messages(plan, subtask)
        max_tool_rounds = getattr(self.agent.config, 'max_tool_rounds', 10)
        subtask_tools = tools if subtask.tools_needed else None

        # Always expose `delegate_to` to alloy subtasks. The planner has no
        # awareness of the per-workflow delegation tool, so `tools_needed`
        # never lists it; without this injection, supervisors running inside
        # a subtask can only narrate delegation in text rather than actually
        # invoking a specialist.
        alloy_executor = getattr(self.agent, "_active_alloy_executor", None)
        if alloy_executor is not None:
            from ..alloy.delegation_tool import DELEGATION_TOOL_NAME, build_delegation_tool

            delegation_descriptor = build_delegation_tool(alloy_executor.workflow)
            delegation_tool = {
                "type": "function",
                "function": {
                    "name": DELEGATION_TOOL_NAME,
                    "description": delegation_descriptor["description"],
                    "parameters": delegation_descriptor["input_schema"],
                },
            }
            existing = list(subtask_tools or [])
            already_present = any(
                t.get("function", {}).get("name") == DELEGATION_TOOL_NAME
                for t in existing
            )
            if not already_present:
                existing.append(delegation_tool)
            subtask_tools = existing

        async for event_str, loop_result in streaming_tool_loop(
            provider, model_id, messages, subtask_tools, self.agent,
            temperature=temperature,
            max_tokens=max_tokens,
            max_tool_rounds=max_tool_rounds,
            max_context_tokens=max_context_tokens,
            task_context=subtask.description,
            emit_trajectory_info=False,
        ):
            yield event_str

        # Accumulate metrics from this subtask
        self.tools_used.extend(loop_result.tools_used)
        self.total_tokens_in += loop_result.tokens_in
        self.total_tokens_out += loop_result.tokens_out

        # Pick the most useful representation of this subtask's output:
        # the supervisor's post-tool synthesis if it produced one; otherwise
        # the specialist's delegated output(s); otherwise the legacy
        # cross-round aggregate (used by non-alloy plans).
        content = (loop_result.final_content or "").strip()
        source = "final"
        if not content and loop_result.delegations:
            content = "\n\n---\n\n".join(loop_result.delegations).strip()
            source = "delegation"
        if not content:
            content = (loop_result.content or "").strip()
            source = "aggregate"
        self._last_subtask_content = content
        self._last_subtask_content_source = source

    # ------------------------------------------------------------------
    # Internal: message building
    # ------------------------------------------------------------------

    def _build_subtask_messages(self, plan: TaskPlan, subtask: Subtask) -> list[Message]:
        """Build a self-contained message list for a subtask.

        Inherits the *system* slice of the chat-request context (profile
        system prompt, Alloy supervisor framing, memory bundle, rolling
        summary) so each subtask has the same base framing as a normal turn.
        Prior user/assistant turns are intentionally not propagated — they
        explode token cost and pull focus away from the subtask.
        """
        messages = []

        # Carry over inherited system framing (skip non-system turns to keep
        # subtasks tightly scoped).
        for m in self.conversation_context:
            if m.role == MessageRole.SYSTEM:
                messages.append(m)

        # System prompt scoped to this subtask
        system_parts = [
            f"You are executing step {subtask.id + 1} of {len(plan.steps)} in a multi-step plan.",
            f"Overall task: {plan.task}",
            f"Your step: {subtask.description}",
            f"Step type: {subtask.type.value}",
            "Focus on completing this step thoroughly. Be concise but complete.",
        ]
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content="\n".join(system_parts),
        ))

        # Inject results from completed dependency subtasks
        dep_results = []
        for dep_id in subtask.dependencies:
            if 0 <= dep_id < len(plan.steps):
                dep = plan.steps[dep_id]
                if dep.completed and dep.result and not dep.result.startswith("[FAILED"):
                    dep_results.append(
                        f"Step {dep_id + 1} ({dep.description}):\n{dep.result}"
                    )

        if dep_results:
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content="Results from prerequisite steps:\n\n" + "\n\n---\n\n".join(dep_results),
            ))

        # User message is the subtask description
        messages.append(Message(
            role=MessageRole.USER,
            content=subtask.description,
        ))

        return messages

    # ------------------------------------------------------------------
    # Internal: failure handling
    # ------------------------------------------------------------------

    def _handle_failure(self, plan: TaskPlan, plan_id: str, subtask: Subtask, error: Exception):
        """Mark failed subtask and skip dependents whose dependencies all failed."""
        plan.steps[subtask.id].completed = True
        plan.steps[subtask.id].result = f"[FAILED: {error}]"
        self.state.update_subtask(plan_id, subtask.id, "failed", error=str(error))
        self._complete_subtask_goal(subtask, "blocked", f"[FAILED: {error}]"[:500])

        for step in plan.steps:
            if step.completed:
                continue
            if subtask.id not in step.dependencies:
                continue

            # Check if ALL dependencies of this step have failed
            all_deps_failed = all(
                plan.steps[d].completed
                and plan.steps[d].result
                and plan.steps[d].result.startswith("[FAILED")
                for d in step.dependencies
                if 0 <= d < len(plan.steps)
            )
            if all_deps_failed:
                step.completed = True
                step.result = "[SKIPPED: all dependencies failed]"
                self.state.update_subtask(plan_id, step.id, "skipped")
                self._complete_subtask_goal(step, "abandoned", "[SKIPPED: all dependencies failed]")
                logger.info(f"Skipped subtask {step.id}: all dependencies failed")

    def _abandon_pending_subtasks(self, plan: TaskPlan, plan_id: str) -> None:
        """Mark all not-yet-completed subtasks as abandoned (called on cancel)."""
        for step in plan.steps:
            if step.completed:
                continue
            step.completed = True
            step.result = "[ABANDONED: plan cancelled]"
            self.state.update_subtask(plan_id, step.id, "abandoned")
            self._complete_subtask_goal(step, "abandoned", "[ABANDONED: plan cancelled]")

    # ------------------------------------------------------------------
    # Internal: answer synthesis
    # ------------------------------------------------------------------

    def _compose_answer_sync(self, plan: TaskPlan) -> str:
        """Compose final answer from subtask results via a single LLM call."""
        messages = self._build_synthesis_messages(plan)
        provider, model_id = self.agent.registry.get_provider_for_model(
            self.agent.config.default_model
        )
        result = provider.complete(messages, model_id, temperature=0.5, max_tokens=4000)
        return result.content

    async def _compose_answer_streaming(
        self,
        plan: TaskPlan,
        provider,
        model_id: str,
        *,
        temperature: float = 0.5,
        max_tokens: int = 4000,
    ) -> AsyncGenerator[str, None]:
        """Stream the synthesis step, yielding chunk SSE events."""
        messages = self._build_synthesis_messages(plan)

        async for chunk in provider.stream(
            messages, model_id,
            temperature=temperature, max_tokens=max_tokens,
        ):
            if chunk.content:
                self.full_content += chunk.content
                yield _sse("chunk", {"content": chunk.content})

    def _build_synthesis_messages(self, plan: TaskPlan) -> list[Message]:
        """Build messages for the synthesis LLM call.

        Synthesis is the message the user actually reads, so it inherits the
        full conversation context (system framing + prior turns) and ends
        with the synthesis instruction + step results, ensuring the answer
        reads as a continuation of the chat rather than a standalone report.
        """
        step_summaries = []
        for step in plan.steps:
            if step.result and not step.result.startswith("[FAILED") and not step.result.startswith("[SKIPPED"):
                step_summaries.append(
                    f"Step {step.id + 1} ({step.description}):\n{step.result}"
                )
            else:
                status = step.result or "no result"
                step_summaries.append(
                    f"Step {step.id + 1} ({step.description}): {status}"
                )

        messages: list[Message] = list(self.conversation_context)
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content=(
                "You are composing a final response from the results of a multi-step plan. "
                "Synthesize the step results into a coherent, well-structured answer that "
                "continues the ongoing conversation above. Do not mention the steps or plan "
                "structure unless it adds clarity. If some steps failed, work with whatever "
                "results are available."
            ),
        ))
        messages.append(Message(
            role=MessageRole.USER,
            content=f"Original task: {plan.task}\n\nStep results:\n\n"
            + "\n\n---\n\n".join(step_summaries),
        ))
        return messages

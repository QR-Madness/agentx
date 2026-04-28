"""
Streaming tool-use loop.

Extracted from views.py and plan_executor.py to eliminate duplication.
Runs provider.stream() in a loop, executing tool calls between rounds,
and yielding SSE-formatted events for each chunk, tool call, and tool result.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from ..providers.base import Message, MessageRole
from .helpers import estimate_tokens
from .trajectory_compression import compress_trajectory

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@dataclass
class ToolLoopResult:
    """Accumulated state from a streaming tool loop run."""
    content: str = ""
    tools_used: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    tool_turns_data: list[dict] = field(default_factory=list)


async def streaming_tool_loop(
    provider,
    model_id: str,
    messages: list[Message],
    tools: Optional[list[dict[str, Any]]],
    agent,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_tool_rounds: int = 10,
    max_context_tokens: int = 100000,
    context_window: Optional[int] = None,
    context_warning_threshold: float = 0.85,
    task_context: str = "",
    emit_trajectory_info: bool = True,
    truncate_on_overflow: bool = True,
    capture_tool_turns: bool = False,
) -> AsyncGenerator[tuple[str, ToolLoopResult], None]:
    """
    Async generator running the streaming tool-use loop.

    Yields (sse_event_string, result) tuples. The result object is the
    same instance throughout and accumulates content, tools_used, token
    counts, and optionally tool_turns_data for DB persistence.

    Args:
        provider: Model provider instance with async stream() method
        model_id: Model identifier string
        messages: Conversation messages (modified in place with tool rounds)
        tools: MCP tool schemas for function calling, or None
        agent: Agent instance for executing tool calls via _execute_tool_calls
        temperature: Sampling temperature
        max_tokens: Max output tokens per completion call
        max_tool_rounds: Maximum number of tool-use rounds
        max_context_tokens: Hard context limit for trajectory compression
        context_window: Full context window size (for overflow warnings)
        context_warning_threshold: Warn when context usage exceeds this ratio
        task_context: Task description for trajectory compression focus
        emit_trajectory_info: Yield an info event when trajectory is compressed
        truncate_on_overflow: Apply hard-limit truncation if over max_context_tokens
        capture_tool_turns: Capture tool call/result data for DB persistence
    """
    from .helpers import truncate_tool_messages

    result = ToolLoopResult()

    for tool_round in range(max_tool_rounds + 1):
        round_tool_calls = []

        # Trajectory compression: consolidate older tool rounds
        if compress_trajectory(messages, max_context_tokens, task_context=task_context):
            estimated = estimate_tokens(messages)
            logger.info(f"Trajectory compressed, new estimate: ~{estimated:,} tokens")
            if emit_trajectory_info:
                yield _sse("info", {"type": "trajectory_compressed"}), result

        estimated_context_tokens = estimate_tokens(messages)
        logger.info(
            f"Tool round {tool_round + 1}: {len(messages)} messages, "
            f"~{estimated_context_tokens:,} tokens, limit={max_context_tokens}"
        )

        # Hard-limit truncation fallback
        if truncate_on_overflow and estimated_context_tokens > max_context_tokens:
            logger.warning("Context exceeds limit, truncating tool messages")
            truncate_tool_messages(messages, estimated_context_tokens, max_context_tokens)
            estimated_context_tokens = estimate_tokens(messages)

        if context_window and estimated_context_tokens > context_window * context_warning_threshold:
            logger.warning(
                f"Context usage high: {estimated_context_tokens:,} / {context_window:,} tokens "
                f"({100 * estimated_context_tokens / context_window:.1f}%)"
            )

        # Stream completion from provider
        async for chunk in provider.stream(
            messages, model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools if tool_round < max_tool_rounds else None,
            tool_choice="auto" if tools and tool_round < max_tool_rounds else None,
        ):
            if chunk.tool_calls:
                round_tool_calls.extend(chunk.tool_calls)
            if chunk.usage:
                result.tokens_in += chunk.usage.get("prompt_tokens", 0)
                result.tokens_out += chunk.usage.get("completion_tokens", 0)
            if chunk.content:
                result.content += chunk.content
                yield _sse("chunk", {"content": chunk.content}), result

        # No tool calls means we're done
        if not round_tool_calls:
            logger.debug(f"Stream loop complete after {tool_round + 1} round(s), no more tool calls")
            break

        # Split off Agent Alloy delegate_to calls — these run through the
        # active AlloyExecutor (async, streaming) instead of the sync
        # _execute_tool_calls path. Delegations emit their own
        # `delegation_start`/`delegation_chunk`/`delegation_complete` event
        # stream, so we suppress the generic `tool_call`/`tool_result` SSE
        # events for them to avoid a redundant ToolExecutionBlock card.
        alloy_executor = getattr(agent, "_active_alloy_executor", None)
        delegation_calls = []
        regular_calls = []
        for tc in round_tool_calls:
            if tc.name == "delegate_to" and alloy_executor is not None:
                delegation_calls.append(tc)
            else:
                regular_calls.append(tc)

        # Emit tool call events (only for non-delegation calls)
        for tc in round_tool_calls:
            result.tools_used.append(tc.name)
        for tc in regular_calls:
            yield _sse("tool_call", {
                "tool": tc.name,
                "tool_call_id": tc.id,
                "arguments": tc.arguments,
            }), result

        # Add assistant message with tool_calls to conversation
        messages.append(Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in round_tool_calls
            ],
        ))

        delegation_messages: list[Message] = []
        # Per-tool-call delegation metadata captured for persistence so
        # restored conversations can reconstruct the delegation card with
        # its full raw content (rather than the LLM-facing storage hint).
        delegation_raw: dict[str, dict[str, Any]] = {}
        for tc in delegation_calls:
            args = tc.arguments or {}
            target = args.get("agent_id", "")
            task = args.get("task", "")
            accumulated = ""
            async for event_str, partial in alloy_executor.delegate(
                target, task, tool_call_id=tc.id,
            ):
                yield event_str, result
                accumulated = partial
            delegation_raw[tc.id] = {
                "raw_content": accumulated,
                "target_agent_id": target,
                "task": task,
            }
            # Route the delegation output through the same oversize handling
            # as a regular tool call, so a long specialist response is stored
            # in Redis with a retrieval key instead of being hard-truncated
            # by `truncate_tool_messages` (which would leave the supervisor
            # unable to recover the full content).
            tool_content = accumulated or "[delegation produced no output]"
            if hasattr(agent, "handle_oversized_tool_output"):
                tool_content = agent.handle_oversized_tool_output(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=tool_content,
                    task_context=task,
                )
            delegation_messages.append(Message(
                role=MessageRole.TOOL,
                content=tool_content,
                tool_call_id=tc.id,
                name=tc.name,
            ))

        # Execute remaining tools (sync) and emit result events
        tool_start_time = time.perf_counter()
        tool_messages = (
            agent._execute_tool_calls(regular_calls, task_context=task_context)
            if regular_calls else []
        )
        tool_messages = delegation_messages + tool_messages
        tool_total_time = (time.perf_counter() - tool_start_time) * 1000
        tool_avg_time = tool_total_time / len(tool_messages) if tool_messages else 0

        delegation_tool_call_ids = {tc.id for tc in delegation_calls}
        for tm in tool_messages:
            is_error = tm.content.startswith('{"error"') or tm.content.startswith("Error:")
            # Skip emitting a generic tool_result for delegations; the
            # delegation_complete event already carries the result.
            if tm.tool_call_id not in delegation_tool_call_ids:
                yield _sse("tool_result", {
                    "tool": tm.name,
                    "tool_call_id": tm.tool_call_id,
                    "content": tm.content[:500],
                    "success": not is_error,
                    "duration_ms": round(tool_avg_time, 2),
                }), result

            if capture_tool_turns:
                result.tool_turns_data.append({
                    "type": "tool_call",
                    "tool": tm.name,
                    "tool_call_id": tm.tool_call_id,
                    "arguments": next(
                        (tc.arguments for tc in round_tool_calls if tc.id == tm.tool_call_id),
                        {},
                    ),
                })
                result_entry: dict[str, Any] = {
                    "type": "tool_result",
                    "tool": tm.name,
                    "tool_call_id": tm.tool_call_id,
                    "content": tm.content[:2000],
                    "success": not is_error,
                    "duration_ms": round(tool_avg_time, 2),
                }
                if tm.tool_call_id in delegation_raw:
                    # Persist the full specialist output + delegation context
                    # so the client can reconstruct a DelegationMessage card
                    # on conversation restore.
                    result_entry["delegation"] = delegation_raw[tm.tool_call_id]
                result.tool_turns_data.append(result_entry)

        messages.extend(tool_messages)

        logger.info(
            f"Tool round {tool_round + 1}: executed "
            f"{', '.join(tc.name for tc in round_tool_calls)}"
        )

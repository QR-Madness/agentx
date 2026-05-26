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
from .helpers import estimate_tokens, truncate_tool_messages
from .trajectory_compression import compress_trajectory

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@dataclass
class ToolLoopResult:
    """Accumulated state from a streaming tool loop run."""
    content: str = ""
    final_content: str = ""
    delegations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    tool_turns_data: list[dict] = field(default_factory=list)


def _prepare_round_context(
    messages: list[Message],
    tool_round: int,
    *,
    max_context_tokens: int,
    task_context: str,
    truncate_on_overflow: bool,
    context_window: Optional[int],
    context_warning_threshold: float,
) -> bool:
    """Pre-stream context management for one round.

    Compresses the trajectory when it grows too large, applies a hard-limit
    truncation fallback, and logs a high-usage warning. Returns whether
    trajectory compression fired so the caller can emit the `info` event
    (kept in the coordinator to preserve the yield there).
    """
    compressed = compress_trajectory(messages, max_context_tokens, task_context=task_context)
    if compressed:
        estimated = estimate_tokens(messages)
        logger.info(f"Trajectory compressed, new estimate: ~{estimated:,} tokens")

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

    return compressed


def _partition_tool_calls(round_tool_calls: list, agent) -> tuple[list, list]:
    """Split a round's tool calls into Agent Alloy delegations vs regular calls.

    `delegate_to` calls run through the active AlloyExecutor (async, streaming)
    and emit their own event stream, so they're separated from the sync
    `_execute_tool_calls` path. Delegations are only split out when an
    AlloyExecutor is active; otherwise every call is treated as regular.
    """
    alloy_executor = getattr(agent, "_active_alloy_executor", None)
    delegation_calls = []
    regular_calls = []
    for tc in round_tool_calls:
        if tc.name == "delegate_to" and alloy_executor is not None:
            delegation_calls.append(tc)
        else:
            regular_calls.append(tc)
    return delegation_calls, regular_calls


async def _run_delegations(
    delegation_calls: list,
    alloy_executor,
    agent,
    *,
    result: "ToolLoopResult",
    delegation_messages: list[Message],
    delegation_raw: dict[str, dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Run Agent Alloy delegations, yielding their event stream.

    Produced TOOL messages are appended to `delegation_messages` and per-call
    metadata to `delegation_raw` (so a restored conversation can rebuild the
    delegation card). Long specialist output is routed through the agent's
    oversize handling, same as a regular tool result.
    """
    for tc in delegation_calls:
        # delegation_calls is only populated when alloy_executor is set;
        # this guard narrows the Optional for the type checker.
        if alloy_executor is None:
            continue
        args = tc.arguments or {}
        target = args.get("agent_id", "")
        task = args.get("task", "")
        accumulated = ""
        async for event_str, partial in alloy_executor.delegate(
            target, task, tool_call_id=tc.id,
        ):
            yield event_str
            accumulated = partial
        if accumulated:
            result.delegations.append(accumulated)
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


async def _execute_and_emit_tools(
    regular_calls: list,
    delegation_messages: list[Message],
    round_tool_calls: list,
    delegation_tool_call_ids: set,
    agent,
    messages: list[Message],
    *,
    task_context: str,
    capture_tool_turns: bool,
    result: "ToolLoopResult",
    delegation_raw: dict[str, dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Execute the regular (sync) tool calls, emit `tool_result` events, and
    extend `messages` with the round's tool results.

    Delegation results (already streamed by `_run_delegations`) are prepended
    but suppressed from the generic `tool_result` events. When
    `capture_tool_turns` is set, tool call/result data is captured on `result`
    for DB persistence.
    """
    tool_start_time = time.perf_counter()
    tool_messages = (
        agent._execute_tool_calls(regular_calls, task_context=task_context)
        if regular_calls else []
    )
    tool_messages = delegation_messages + tool_messages
    tool_total_time = (time.perf_counter() - tool_start_time) * 1000
    tool_avg_time = tool_total_time / len(tool_messages) if tool_messages else 0

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
            })

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
    result: Optional[ToolLoopResult] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator running the streaming tool-use loop.

    Thin coordinator: each round prepares context (`_prepare_round_context`),
    streams a completion, then — if the model requested tools — partitions them
    (`_partition_tool_calls`), runs any Alloy delegations (`_run_delegations`),
    and executes the rest (`_execute_and_emit_tools`).

    Yields SSE event strings. State is accumulated on the `result` object,
    which the caller should pre-allocate and pass in so it remains
    readable even when the generator never yields (e.g. an empty
    completion with no tool calls). If `result` is None, a fresh
    instance is created internally but will not be observable by the
    caller.

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
    if result is None:
        result = ToolLoopResult()

    for tool_round in range(max_tool_rounds + 1):
        round_tool_calls = []
        round_content = ""

        # Trajectory compression + truncation + high-usage warning
        if _prepare_round_context(
            messages, tool_round,
            max_context_tokens=max_context_tokens,
            task_context=task_context,
            truncate_on_overflow=truncate_on_overflow,
            context_window=context_window,
            context_warning_threshold=context_warning_threshold,
        ) and emit_trajectory_info:
            yield _sse("info", {"type": "trajectory_compressed"})

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
                round_content += chunk.content
                yield _sse("chunk", {"content": chunk.content})

        # No tool calls means we're done
        if not round_tool_calls:
            result.final_content = round_content
            logger.debug(f"Stream loop complete after {tool_round + 1} round(s), no more tool calls")
            # Surface empty completions explicitly so downstream code
            # (session storage, done event, parsed.content) doesn't carry
            # silently empty content through to the UI.
            if tool_round == 0 and not result.content:
                logger.warning(
                    f"Empty completion from model={model_id} (no content, no tool calls)"
                )
                fallback = "[empty response from model]"
                result.content = fallback
                result.final_content = fallback
                yield _sse("chunk", {"content": fallback})
            break

        # Split off Agent Alloy delegate_to calls from regular tool calls.
        alloy_executor = getattr(agent, "_active_alloy_executor", None)
        delegation_calls, regular_calls = _partition_tool_calls(round_tool_calls, agent)

        # Emit tool call events (only for non-delegation calls)
        for tc in round_tool_calls:
            result.tools_used.append(tc.name)
        for tc in regular_calls:
            yield _sse("tool_call", {
                "tool": tc.name,
                "tool_call_id": tc.id,
                "arguments": tc.arguments,
            })

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

        # Run Alloy delegations (stream their own events); capture their
        # TOOL messages + raw metadata for the tool-execution step.
        delegation_messages: list[Message] = []
        delegation_raw: dict[str, dict[str, Any]] = {}
        async for event_str in _run_delegations(
            delegation_calls, alloy_executor, agent,
            result=result,
            delegation_messages=delegation_messages,
            delegation_raw=delegation_raw,
        ):
            yield event_str

        # Execute remaining tools (sync), emit result events, extend messages
        delegation_tool_call_ids = {tc.id for tc in delegation_calls}
        async for event_str in _execute_and_emit_tools(
            regular_calls, delegation_messages, round_tool_calls,
            delegation_tool_call_ids, agent, messages,
            task_context=task_context,
            capture_tool_turns=capture_tool_turns,
            result=result,
            delegation_raw=delegation_raw,
        ):
            yield event_str

        logger.info(
            f"Tool round {tool_round + 1}: executed "
            f"{', '.join(tc.name for tc in round_tool_calls)}"
        )

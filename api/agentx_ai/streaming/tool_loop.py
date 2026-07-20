"""
Streaming tool-use loop.

Extracted from views.py and plan_executor.py to eliminate duplication.
Runs provider.stream() in a loop, executing tool calls between rounds,
and yielding SSE-formatted events for each chunk, tool call, and tool result.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from collections.abc import AsyncGenerator, Callable

from ..providers.base import Message, MessageRole
from .helpers import estimate_tokens, truncate_tool_messages
from .status import current_run_id, emit_status
from .steering import drain_steer_messages
from .trajectory_compression import compress_trajectory

logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _ambient_cancel_check() -> bool:
    """Resolve the current detached run and report whether it's been cancelled.

    Used as the default ``cancel_check`` so every turn (not just plans) stops
    promptly when the user hits Stop/Cancel — checked at the round boundaries
    (existing await/yield points), so it's purely cooperative: no threads, no
    ``asyncio.shield`` (that off-thread approach deadlocked ``gen.aclose()`` and
    was reverted). No run in context ⇒ never cancelled.
    """
    try:
        from .status import current_run_id
        rid = current_run_id.get()
        if not rid:
            return False
        from .chat_run import store
        return store.is_cancel_requested(rid)
    except Exception:
        return False


@dataclass
class ToolLoopResult:
    """Accumulated state from a streaming tool loop run."""
    content: str = ""
    final_content: str = ""
    delegations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    # Provider-billed cost (USD) summed from usage-accounting chunks — 0.0 when
    # the provider doesn't report billing; callers prefer this over a
    # list-price estimate. reasoning_tokens = hidden thinking tokens (billed
    # as output; already inside tokens_out) for display/telemetry.
    provider_cost: float = 0.0
    reasoning_tokens: int = 0
    tool_turns_data: list[dict] = field(default_factory=list)
    # Mid-turn steers folded in (for persistence). Each: {content, round,
    # after_tools, phase} where phase is "tool_boundary" or "would_end".
    steers: list[dict] = field(default_factory=list)
    # Background work-order reports folded in (`delegate_start`, for
    # persistence). Each: {content, delegation_id, target_agent_id,
    # tool_call_id, status, round, phase} where phase is "tool_boundary",
    # "would_end", or "round_exhausted".
    work_order_reports: list[dict] = field(default_factory=list)
    # The final round's finish reason ("stop", "length", …). "length" means the
    # answer was truncated by max_tokens — surfaced in the done event.
    finish_reason: str | None = None
    # Successful document-write tool calls this turn (create/update/append/edit
    # _document). Drives the research finalize nudge: no writes ⇒ nudge fires.
    docs_written: int = 0


def _prepare_round_context(
    messages: list[Message],
    tool_round: int,
    *,
    max_context_tokens: int,
    task_context: str,
    truncate_on_overflow: bool,
    context_window: int | None,
    context_warning_threshold: float,
    active_model: str | None = None,
) -> bool:
    """Pre-stream context management for one round.

    Compresses the trajectory when it grows too large, applies a hard-limit
    truncation fallback, and logs a high-usage warning. Returns whether
    trajectory compression fired so the caller can emit the `info` event
    (kept in the coordinator to preserve the yield there).
    """
    compressed = compress_trajectory(
        messages, max_context_tokens, task_context=task_context, active_model=active_model
    )
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


def _partition_tool_calls(round_tool_calls: list, agent) -> tuple[list, list, list]:
    """Split a round's tool calls into blocking delegations, background work
    orders, and regular calls.

    `delegate_to` calls run through the active AlloyExecutor (async, streaming)
    and emit their own event stream, so they're separated from the sync
    `_execute_tool_calls` path. `delegate_start` calls dispatch as background
    work orders when the executor has non-blocking mode enabled — with the knob
    off they gracefully degrade to blocking delegations. Delegations are only
    split out when an AlloyExecutor is active; otherwise every call is treated
    as regular (and fails as an unknown tool, as today).
    """
    alloy_executor = getattr(agent, "_active_alloy_executor", None)
    non_blocking = bool(getattr(alloy_executor, "non_blocking_enabled", False))
    delegation_calls = []
    background_calls = []
    regular_calls = []
    for tc in round_tool_calls:
        if alloy_executor is not None and tc.name == "delegate_start":
            (background_calls if non_blocking else delegation_calls).append(tc)
        elif alloy_executor is not None and tc.name == "delegate_to":
            delegation_calls.append(tc)
        else:
            regular_calls.append(tc)
    return delegation_calls, background_calls, regular_calls


# Metric/identity keys re-parsed from a `delegation_complete` SSE payload —
# folded onto the persisted delegation card so restored conversations carry
# honest terminal state (status/error) and work-order identity.
_COMPLETE_METRIC_KEYS = (
    "tokens_input", "tokens_output", "duration_ms",
    "cost_estimate", "cost_currency", "pricing_snapshot",
    # Exhibit wires the specialist produced — consumed (and stripped) by the
    # persistence step so reload rebuilds the cards.
    "exhibits",
    "status", "error", "mode", "parent_delegation_id", "delegation_id",
)


def _parse_complete_metrics(event_str: str) -> dict[str, Any]:
    """Re-parse a `delegation_complete` SSE for its metrics (keeps delegate()'s
    (event_str, partial) yield contract unchanged for other consumers)."""
    _, _, payload = event_str.partition("\n")
    data_line = payload.split("data: ", 1)[1].rstrip() if "data: " in payload else "{}"
    try:
        done = json.loads(data_line)
    except json.JSONDecodeError:
        return {}
    return {
        k: done[k]
        for k in _COMPLETE_METRIC_KEYS
        if k in done and done[k] is not None
    }


def _complete_matches_tool_call(event_str: str, tool_call_id: str) -> bool:
    """True when a `delegation_complete` SSE belongs to THIS branch's tool call.

    Chain nesting (Agentic Orgs) passes a lead-specialist's own delegation_*
    events through the same generator; they carry their own tool_call_id —
    capturing their metrics would transiently clobber the branch's (and stick
    if the outer delegation then errors)."""
    _, _, payload = event_str.partition("\n")
    data_line = payload.split("data: ", 1)[1].rstrip() if "data: " in payload else "{}"
    try:
        return json.loads(data_line).get("tool_call_id") == tool_call_id
    except json.JSONDecodeError:
        return False


# Internal tool whose calls are surfaced as typed `exhibit` events (rendered
# by the client's element registry) rather than generic tool_call/tool_result
# cards. The tool body still runs so the model gets a tool_result.
EXHIBIT_TOOL_NAME = "present_exhibit"


def _emit_exhibit_event(tc) -> list[str]:
    """Build an `exhibit` SSE from a `present_exhibit` tool call's arguments.

    Returns a single-element list on success, or empty if the declared exhibit
    is malformed — in which case nothing is shown to the client and the tool
    body's validation error (returned to the model) drives a re-present.
    """
    from pydantic import ValidationError

    from .exhibits import exhibit_from_present_call

    try:
        exhibit = exhibit_from_present_call(tc.arguments or {})
    except ValidationError as e:
        logger.warning(f"present_exhibit args invalid; suppressing exhibit event: {e}")
        return []
    return [_sse("exhibit", exhibit.model_dump())]


# Tools whose successful results are auto-captured as a passive `citation` exhibit
# (so web sources surface in the UI without the model having to present them). Both
# return a normalized `results: [{title, url}]` list.
WEB_SEARCH_TOOL_NAME = "web_search"
_AUTO_CITATION_TOOLS: frozenset[str] = frozenset({"web_search", "web_research"})
_DOC_CITATION_TOOLS: frozenset[str] = frozenset({"document_query"})
# Document-write tools: a successful call counts as "the deliverable was saved"
# for the research finalize nudge (ToolLoopResult.docs_written).
_DOC_WRITE_TOOLS: frozenset[str] = frozenset(
    {"create_document", "update_document", "append_to_document", "edit_document"}
)


def _delegation_media_arg(args: dict) -> list[str] | None:
    """Sanitize a delegation call's `media` argument into a list of document-id
    strings (models sometimes emit a single string or junk entries). None when
    absent/empty so simpler `delegate()` signatures stay compatible."""
    raw = args.get("media")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    ids = [str(x).strip() for x in raw if x is not None and str(x).strip()]
    return ids or None


def _is_doc_write_success(tm) -> bool:
    """Whether a doc-write tool message reports a genuine write.

    Doc tools return JSON dicts; parse and require no error marker (the cheap
    `startswith('{"error"')` heuristic misses `{"success": false, ...}` shapes,
    and a false positive here would suppress a needed finalize nudge).
    """
    try:
        payload = json.loads(tm.content)
    except (ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("success") is not False and not payload.get("error")


def _emit_web_search_citation(tm) -> list[str]:
    """Auto-capture a `web_search` tool result as a passive `citation` exhibit.

    Reads the full (un-capped) tool message content — `web_search` keeps results
    compact (no raw page content) so they're never oversize-routed before we see
    them. Returns empty when disabled by config, unparseable, failed, or empty.
    """
    from ..config import get_config_manager

    if not get_config_manager().get("citations.auto_capture_web_search", True):
        return []
    try:
        data = json.loads(tm.content)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or not data.get("success"):
        return []
    results = data.get("results") or []
    if not results:
        return []

    from .exhibits import citation_exhibit_from_web_search

    exhibit = citation_exhibit_from_web_search(results, exhibit_id=f"exh_src_{tm.tool_call_id}")
    if exhibit is None:
        return []
    return [_sse("exhibit", exhibit.model_dump())]


def _emit_document_citation(tm) -> list[str]:
    """Auto-capture a `document_query` result as a passive `doc` citation exhibit."""
    from ..config import get_config_manager

    if not get_config_manager().get("citations.auto_capture_document_query", True):
        return []
    try:
        data = json.loads(tm.content)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or not data.get("success"):
        return []
    results = data.get("results") or []
    if not results:
        return []

    from .exhibits import citation_exhibit_from_document_query

    exhibit = citation_exhibit_from_document_query(results, exhibit_id=f"exh_doc_{tm.tool_call_id}")
    if exhibit is None:
        return []
    return [_sse("exhibit", exhibit.model_dump())]


def _media_exhibit_wires(tm) -> list[dict]:
    """Exhibit wires derivable from one tool message: a `generate_image`/
    `generate_speech` result, or MCP media-passthrough `stored_media` entries.

    One definition serves both the live SSE emit and the synthetic
    `present_exhibit` persistence turns (so a media exhibit reloads exactly as
    it rendered live). Best-effort: unparseable/failed results yield nothing.
    """
    try:
        data = json.loads(tm.content)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []

    from .exhibits import (
        audio_exhibit_from_generate,
        image_exhibit_from_generate,
        media_exhibit_from_stored,
    )

    wires: list[dict] = []
    if tm.name == "generate_image":
        if data.get("success") and data.get("url"):
            ex = image_exhibit_from_generate(
                data["url"], exhibit_id=f"exh_img_{tm.tool_call_id}", alt=data.get("prompt"),
            )
            if ex is not None:
                wires.append(ex.model_dump())
    elif tm.name == "generate_speech":
        if data.get("success") and data.get("url"):
            ex = audio_exhibit_from_generate(
                data["url"], exhibit_id=f"exh_aud_{tm.tool_call_id}", caption=data.get("text_preview"),
            )
            if ex is not None:
                wires.append(ex.model_dump())
    else:
        for i, stored in enumerate(data.get("stored_media") or []):
            if not isinstance(stored, dict):
                continue
            ex = media_exhibit_from_stored(stored, exhibit_id=f"exh_mcp_{tm.tool_call_id}_{i}")
            if ex is not None:
                wires.append(ex.model_dump())
    return wires


# Cheap content pre-gate for `_media_exhibit_wires` — avoids a json.loads on
# every ordinary tool message.
def _may_carry_media(tm) -> bool:
    if tm.name in ("generate_image", "generate_speech"):
        return True
    return '"stored_media"' in (tm.content or "")


def _emit_model_media(media_blocks, result, capture_tool_turns: bool, model_id: str) -> list[str]:
    """Render non-text payloads a chat completion itself carried (`StreamChunk.media`
    — an image/audio-output model answering *as* the agent, no tool call involved).

    Each block is stored as workspace media (same caps as MCP passthrough) and
    surfaced as an image/audio exhibit; with `capture_tool_turns` the synthetic
    `present_exhibit` persistence turn rides `result.tool_turns_data` so the
    exhibit survives reload. Best-effort — a bad payload never breaks the turn."""
    import uuid as _uuid

    from ..mcp.media_passthrough import store_media_block
    from .exhibits import media_exhibit_from_stored

    events: list[str] = []
    for blk in media_blocks or []:
        if not isinstance(blk, dict) or blk.get("type") not in ("image", "audio"):
            continue
        stored = store_media_block(
            blk.get("data") or "", blk.get("mimeType") or "",
            tool_name="model-output", server_name=model_id,
        )
        if stored is None:
            continue
        exhibit_id = f"exh_out_{_uuid.uuid4().hex[:10]}"
        ex = media_exhibit_from_stored(stored, exhibit_id=exhibit_id)
        if ex is None:
            continue
        wire = ex.model_dump()
        events.append(_sse("exhibit", wire))
        if capture_tool_turns:
            result.tool_turns_data.append({
                "type": "tool_call",
                "tool": "present_exhibit",
                "tool_call_id": exhibit_id,
                "arguments": wire,
            })
    return events


def _emit_media_exhibits(tm) -> list[str]:
    """Auto-render a tool message's media as exhibits (the user-facing artifact):
    a generated image/speech clip, or media an external MCP tool returned
    (stored by `mcp.media_passthrough`).

    For generation tools, also emits a lightweight ``workspace_attached`` signal
    carrying the workspace the media landed in — so a conversation that had no
    workspace (and thus fell back to the personal Home store) can durably attach
    to it client-side."""
    if not _may_carry_media(tm):
        return []
    events = [_sse("exhibit", wire) for wire in _media_exhibit_wires(tm)]
    if not events:
        return []
    if tm.name in ("generate_image", "generate_speech"):
        try:
            ws_id = json.loads(tm.content).get("workspace_id")
        except (ValueError, TypeError, AttributeError):
            ws_id = None
        if ws_id:
            events.append(_sse("workspace_attached", {"workspace_id": ws_id}))
    return events


def _view_image_messages(tm, *, vision_capable: bool) -> list[Message]:
    """Turn a successful `view_image` tool result into the message(s) to feed the model.

    On a vision model: a user-role message carrying the image block (the agent now
    actually sees it next round). On a non-vision model: a short note instead — the
    request can't be honored. Returns [] for a failed/unparseable result (its error
    text already went back as the tool result)."""
    try:
        data = json.loads(tm.content)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict) or not data.get("success"):
        return []

    doc_id = data.get("document_id") or data.get("doc_id")
    ws_id = data.get("workspace_id")
    media_type = data.get("media_type")
    filename = data.get("filename") or doc_id
    if not (doc_id and ws_id and media_type):
        return []

    if not vision_capable:
        return [Message(
            role=MessageRole.USER,
            content=f"[Couldn't show '{filename}': this model has no vision capability.]",
        )]

    from ..providers.base import ImageRef

    return [Message(
        role=MessageRole.USER,
        content=f"[Image '{filename}', shown for you to view:]",
        images=[ImageRef(workspace_id=ws_id, doc_id=doc_id, media_type=media_type)],
    )]


async def _run_delegations(
    delegation_calls: list,
    alloy_executor,
    agent,
    *,
    result: ToolLoopResult,
    delegation_messages: list[Message],
    delegation_raw: dict[str, dict[str, Any]],
) -> AsyncGenerator[str]:
    """Run Agent Alloy delegations **concurrently**, yielding their interleaved
    event stream.

    When the supervisor emits several `delegate_to` calls in one turn, each runs
    as its own task (bounded by `alloy.max_parallel_delegations`) feeding a single
    queue, so their `delegation_*` events interleave in real time. Produced TOOL
    messages are appended to `delegation_messages` and per-call metadata to
    `delegation_raw` (so a restored conversation can rebuild the delegation card),
    **in original `delegation_calls` order** keyed by `tool_call_id` — out-of-order
    completion is fine because the provider matches tool results by id. Long
    specialist output is routed through the agent's oversize handling, same as a
    regular tool result.
    """
    if alloy_executor is None or not delegation_calls:
        return

    max_parallel = getattr(alloy_executor, "max_parallel_delegations", 3) or 3
    sem = asyncio.Semaphore(max_parallel)
    queue: asyncio.Queue = asyncio.Queue()
    done_sentinel = object()
    final_partials: dict[str, str] = {}
    final_metrics: dict[str, dict[str, Any]] = {}

    async def _drive(tc) -> None:
        """One delegation branch: stream its events into the shared queue."""
        args = tc.arguments or {}
        target = args.get("agent_id", "")
        task = args.get("task", "")
        media = _delegation_media_arg(args)
        accumulated = ""
        # Per-delegation wall-clock cap (`alloy.delegation_timeout_seconds`).
        # Applied to the consumption, not the semaphore wait, so queued
        # branches don't burn their budget waiting for a slot.
        timeout = getattr(alloy_executor, "delegation_timeout_seconds", 300) or None
        # Top-level fan-out is always depth 0; delegate() defaults depth=0.
        # Optional kwargs (depth, media) are OMITTED when unused so simpler
        # delegate() signatures (test fakes) stay compatible.
        extra = {"media": media} if media else {}
        gen = alloy_executor.delegate(target, task, tool_call_id=tc.id, **extra)

        async def _consume() -> None:
            nonlocal accumulated
            async for event_str, partial in gen:
                await queue.put(event_str)
                accumulated = partial
                if event_str.startswith("event: delegation_complete") and \
                        _complete_matches_tool_call(event_str, tc.id):
                    final_metrics[tc.id] = _parse_complete_metrics(event_str)

        try:
            async with sem:
                await asyncio.wait_for(_consume(), timeout)
        except asyncio.CancelledError:
            # Client disconnect / sibling teardown — propagate, don't swallow.
            raise
        except TimeoutError:
            try:
                await gen.aclose()
            except Exception:  # noqa: BLE001 - best-effort generator cleanup
                pass
            err = f"delegation timed out after {timeout}s"
            logger.warning(f"Delegation branch for {target!r}: {err}")
            await queue.put(_sse("delegation_complete", {
                "target_agent_id": target,
                "tool_call_id": tc.id,
                "status": "failed",
                "error": err,
                "result_preview": "",
            }))
            final_metrics[tc.id] = {"status": "failed", "error": err}
            accumulated = accumulated or f"[delegation failed: {err}]"
        except Exception as e:  # noqa: BLE001 - isolate this branch from siblings
            logger.exception(f"Delegation branch for {target!r} failed")
            await queue.put(_sse("delegation_complete", {
                "target_agent_id": target,
                "tool_call_id": tc.id,
                "status": "failed",
                "error": str(e),
                "result_preview": "",
            }))
            final_metrics[tc.id] = {"status": "failed", "error": str(e)}
            accumulated = accumulated or f"[delegation failed: {e}]"
        finally:
            final_partials[tc.id] = accumulated
            await queue.put(done_sentinel)

    tasks = [asyncio.create_task(_drive(tc)) for tc in delegation_calls]
    remaining = len(tasks)
    try:
        while remaining:
            item = await queue.get()
            if item is done_sentinel:
                remaining -= 1
                continue
            yield item
    finally:
        # GeneratorExit (client aclose) or normal exit — ensure no orphan tasks.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Post-gather aggregation, in original order, keyed by tool_call_id. Every
    # call must produce a TOOL message (provider contract) even if it errored,
    # was cancelled, or produced nothing.
    for tc in delegation_calls:
        args = tc.arguments or {}
        target = args.get("agent_id", "")
        task = args.get("task", "")
        accumulated = final_partials.get(tc.id, "")
        if accumulated:
            result.delegations.append(accumulated)
        delegation_raw[tc.id] = {
            "raw_content": accumulated,
            "target_agent_id": target,
            "task": task,
            **final_metrics.get(tc.id, {}),
        }
        # Route delegation output through the same oversize handling as a regular
        # tool call (sync; run serially here to avoid concurrent Redis writes), so
        # a long specialist response is stored with a retrieval key instead of
        # being hard-truncated.
        tool_content = accumulated or "[delegation produced no output]"
        if hasattr(agent, "handle_oversized_tool_output"):
            tool_content = agent.handle_oversized_tool_output(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=tool_content,
                task_context=task,
            )
        # Supervisor-facing exhibit note (appended AFTER oversize handling so it
        # always survives): the supervisor only sees the specialist's text —
        # without this it re-invents image markdown/URLs for artifacts it never
        # saw (guessing ws_home → client 404s). Persistence/previews stay clean;
        # this rides only the TOOL message.
        n_exhibits = len(final_metrics.get(tc.id, {}).get("exhibits") or [])
        if n_exhibits:
            tool_content = (
                f"{tool_content}\n\n[note: {n_exhibits} exhibit(s) (e.g. a "
                "generated image) were produced by this agent and are ALREADY "
                "displayed to the user. Do not write image markdown or URLs "
                "for them; refer to them in prose.]"
            )
        delegation_messages.append(Message(
            role=MessageRole.TOOL,
            content=tool_content,
            tool_call_id=tc.id,
            name=tc.name,
        ))


# ---------------------------------------------------------------------------
# Non-blocking work orders (`delegate_start`)
#
# A work order dispatches immediately: its TOOL message is a dispatch receipt
# (satisfying the provider's tool_result contract), the specialist runs as a
# detached task streaming its events straight onto the run's Redis bus, and
# the report folds back into the transcript as a user message at the same safe
# boundaries steering uses. Strictly within-turn: the would-end barrier waits
# for stragglers, and `streaming_tool_loop`'s finally cancels anything left.
# ---------------------------------------------------------------------------


@dataclass
class _WorkOrder:
    """One in-flight background delegation (a `delegate_start` work order)."""
    tc: Any
    target: str
    task_text: str
    delegation_id: str
    round_dispatched: int
    task: asyncio.Task
    final_partial: str = ""
    metrics: dict = field(default_factory=dict)
    folded: bool = False


def _emit_background_event(run_id: str | None, sse_event: str) -> None:
    """Append a background delegation event to the run's Redis event bus.

    Background drivers can't yield through the tool-loop generator (they run
    while the loop is doing other work), so they append straight to the bus —
    the same stream `_drive_run` copies generator yields into, hence identical
    for live and re-attached clients. No run in context (unit tests, non-run
    callers) ⇒ drop with a debug log.
    """
    if not run_id:
        logger.debug("background delegation event dropped (no run in context)")
        return
    try:
        from .chat_run import store
        store.append_event(run_id, sse_event)
    except Exception:  # noqa: BLE001 - the bus is best-effort telemetry here
        logger.warning("failed to append background delegation event", exc_info=True)


def _dispatch_background_delegations(
    background_calls: list,
    alloy_executor,
    agent,
    *,
    registry: dict[str, _WorkOrder],
    delegation_messages: list[Message],
    delegation_raw: dict[str, dict[str, Any]],
    sem: asyncio.Semaphore,
    tool_round: int,
) -> None:
    """Dispatch `delegate_start` calls as detached work-order tasks.

    Each call immediately gets a dispatch-receipt TOOL message (provider
    contract) and a seeded `delegation_raw` entry (so the persisted card shows
    a dispatched state even if the turn is interrupted); the specialist events
    stream to the run bus from the detached task. The caller owns `registry`
    cleanup — see `_cancel_pending_work_orders`.
    """
    run_id = current_run_id.get()
    timeout = getattr(alloy_executor, "delegation_timeout_seconds", 300) or None

    for tc in background_calls:
        args = tc.arguments or {}
        target = args.get("agent_id", "")
        task_text = args.get("task", "")
        media_arg = _delegation_media_arg(args)
        delegation_id = uuid4().hex[:8]

        async def _drive_background(
            tc=tc, target=target, task_text=task_text, delegation_id=delegation_id,
            media_arg=media_arg,
        ) -> tuple[str, dict[str, Any]]:
            accumulated = ""
            metrics: dict[str, Any] = {}
            extra = {"media": media_arg} if media_arg else {}
            gen = alloy_executor.delegate(
                target, task_text, tool_call_id=tc.id,
                mode="background", delegation_id=delegation_id,
                **extra,
            )

            async def _consume() -> None:
                nonlocal accumulated, metrics
                async for event_str, partial in gen:
                    _emit_background_event(run_id, event_str)
                    accumulated = partial
                    if event_str.startswith("event: delegation_complete") and \
                            _complete_matches_tool_call(event_str, tc.id):
                        metrics = _parse_complete_metrics(event_str)

            def _fail(err: str) -> None:
                nonlocal metrics, accumulated
                metrics = {
                    "status": "failed", "error": err,
                    "mode": "background", "delegation_id": delegation_id,
                }
                _emit_background_event(run_id, _sse("delegation_complete", {
                    "delegation_id": delegation_id,
                    "target_agent_id": target,
                    "tool_call_id": tc.id,
                    "status": "failed",
                    "error": err,
                    "result_preview": "",
                    "mode": "background",
                    "parent_delegation_id": None,
                }))
                accumulated = accumulated or f"[work order failed: {err}]"

            try:
                async with sem:
                    await asyncio.wait_for(_consume(), timeout)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                try:
                    await gen.aclose()
                except Exception:  # noqa: BLE001 - best-effort generator cleanup
                    pass
                logger.warning(
                    f"Work order {delegation_id} ({target!r}) timed out after {timeout}s"
                )
                _fail(f"work order timed out after {timeout}s")
            except Exception as e:  # noqa: BLE001 - isolate from the main loop
                logger.exception(f"Work order {delegation_id} ({target!r}) failed")
                _fail(str(e))
            return accumulated, metrics

        registry[tc.id] = _WorkOrder(
            tc=tc,
            target=target,
            task_text=task_text,
            delegation_id=delegation_id,
            round_dispatched=tool_round,
            task=asyncio.create_task(_drive_background()),
        )
        delegation_messages.append(Message(
            role=MessageRole.TOOL,
            content=(
                f"[dispatch receipt] Work Order {delegation_id} dispatched to "
                f"{target}. Its report will be delivered to you automatically "
                "later this turn — continue with other work; do not wait or "
                "poll for it."
            ),
            tool_call_id=tc.id,
            name=tc.name,
        ))
        delegation_raw[tc.id] = {
            "raw_content": "",
            "target_agent_id": target,
            "task": task_text,
            "mode": "background",
            "status": "dispatched",
            "delegation_id": delegation_id,
            "parent_delegation_id": None,
        }


def _settle_work_order_entry(
    wo: _WorkOrder, result: ToolLoopResult, *, status: str,
) -> None:
    """Update the persisted delegation card entry for a settled work order."""
    for entry in result.tool_turns_data:
        if entry.get("type") == "tool_result" and entry.get("tool_call_id") == wo.tc.id:
            dr = entry.get("delegation")
            if not isinstance(dr, dict):
                dr = {}
                entry["delegation"] = dr
            metrics = dict(wo.metrics)
            exhibits = metrics.pop("exhibits", None) or []
            dr.update(metrics)
            dr["status"] = status
            dr["raw_content"] = wo.final_partial
            # Synthetic present_exhibit turns for exhibits produced inside the
            # work order — mirror of the blocking-path strip in
            # `_execute_and_emit_tools` so reload rebuilds the cards.
            for i, wire in enumerate(exhibits):
                if isinstance(wire, dict):
                    result.tool_turns_data.append({
                        "type": "tool_call",
                        "tool": "present_exhibit",
                        "tool_call_id": wire.get("id") or f"exh_wo_{wo.tc.id}_{i}",
                        "arguments": wire,
                    })
            return


def _fold_completed_work_orders(
    registry: dict[str, _WorkOrder],
    messages: list[Message],
    agent,
    *,
    result: ToolLoopResult,
    tool_round: int,
    phase: str,
) -> list[str]:
    """Fold finished background work orders into the transcript.

    For each completed, not-yet-folded order: append its report as a USER
    message (the dispatch receipt already satisfied the provider's tool_result
    contract — the deferred result may not backfill it), record it on
    ``result.work_order_reports`` (persisted like steers), settle the persisted
    delegation card entry, and return `work_order_report` SSE strings for the
    caller to yield (yielding keeps bus ordering aligned with the fold).
    """
    events: list[str] = []
    for wo in registry.values():
        if wo.folded or not wo.task.done() or wo.task.cancelled():
            continue
        try:
            partial, metrics = wo.task.result()
        except Exception:  # noqa: BLE001 - driver isolates; belt and braces
            partial, metrics = "", {"status": "failed", "error": "work order crashed"}
        wo.final_partial, wo.metrics = partial, metrics
        raw_status = metrics.get("status") or "success"
        status = "completed" if raw_status == "success" else raw_status
        body = partial or metrics.get("error") or "[work order produced no output]"
        report = (
            f"[Work Order Report — {wo.target}, wo {wo.delegation_id}] "
            f"status: {status}\n\n{body}"
        )
        # Same oversize handling as a blocking delegation result (sync; serial).
        if hasattr(agent, "handle_oversized_tool_output"):
            report = agent.handle_oversized_tool_output(
                tool_call_id=wo.tc.id,
                tool_name="delegate_start",
                content=report,
                task_context=wo.task_text,
            )
        messages.append(Message(role=MessageRole.USER, content=report))
        if partial:
            result.delegations.append(partial)
        result.work_order_reports.append({
            "content": report,
            "delegation_id": wo.delegation_id,
            "target_agent_id": wo.target,
            "tool_call_id": wo.tc.id,
            "status": status,
            "round": tool_round,
            "phase": phase,
        })
        _settle_work_order_entry(wo, result, status=status)
        events.append(_sse("work_order_report", {
            "tool_call_id": wo.tc.id,
            "delegation_id": wo.delegation_id,
            "target_agent_id": wo.target,
            "status": status,
            "round": tool_round,
            "phase": phase,
        }))
        wo.folded = True
    return events


async def _await_work_orders(
    registry: dict[str, _WorkOrder],
    check_cancel: Callable[[], bool],
    *,
    all_of_them: bool = False,
) -> bool:
    """Wait until at least one (or all) pending work orders complete.

    Polls in short slices so user cancellation stays responsive during an
    otherwise-silent barrier (background events bypass the generator, so the
    driver's per-event cancel check never fires here). Returns False when
    cancellation was observed — the caller should stop.
    """
    while True:
        pending = [wo.task for wo in registry.values() if not wo.task.done()]
        if not pending:
            return True
        if not all_of_them and any(
            wo.task.done() and not wo.folded for wo in registry.values()
        ):
            return True
        if check_cancel():
            return False
        await asyncio.wait(
            pending,
            timeout=2,
            return_when=(
                asyncio.ALL_COMPLETED if all_of_them else asyncio.FIRST_COMPLETED
            ),
        )


async def _cancel_pending_work_orders(
    registry: dict[str, _WorkOrder],
    result: ToolLoopResult,
    run_id: str | None,
) -> None:
    """Cancel still-running work orders (the turn is ending or was cancelled)
    and settle their persisted cards.

    Finished-but-unfolded orders keep their real terminal state — the work
    completed even if its report never folded; only genuinely interrupted
    orders read "cancelled".
    """
    to_cancel = [wo for wo in registry.values() if not wo.task.done()]
    for wo in to_cancel:
        wo.task.cancel()
    if to_cancel:
        await asyncio.gather(
            *(wo.task for wo in to_cancel), return_exceptions=True,
        )
    for wo in registry.values():
        if wo.folded:
            continue
        if wo.task.cancelled():
            _settle_work_order_entry(wo, result, status="cancelled")
            _emit_background_event(run_id, _sse("delegation_complete", {
                "delegation_id": wo.delegation_id,
                "target_agent_id": wo.target,
                "tool_call_id": wo.tc.id,
                "status": "cancelled",
                "error": "work order cancelled with the run",
                "result_preview": "",
                "mode": "background",
                "parent_delegation_id": None,
            }))
        else:
            try:
                partial, metrics = wo.task.result()
            except Exception:  # noqa: BLE001
                partial, metrics = "", {"status": "failed", "error": "work order crashed"}
            wo.final_partial, wo.metrics = partial, metrics
            raw_status = metrics.get("status") or "success"
            _settle_work_order_entry(
                wo, result,
                status="completed" if raw_status == "success" else raw_status,
            )
        wo.folded = True


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
    result: ToolLoopResult,
    delegation_raw: dict[str, dict[str, Any]],
    suppress_result_ids: set | None = None,
    vision_capable: bool = False,
) -> AsyncGenerator[str]:
    """Execute the regular (sync) tool calls, emit `tool_result` events, and
    extend `messages` with the round's tool results.

    Delegation results (already streamed by `_run_delegations`) are prepended
    but suppressed from the generic `tool_result` events. Calls in
    `suppress_result_ids` (e.g. `present_exhibit`, whose output is the typed
    `exhibit` event) are likewise executed but not echoed as a `tool_result`.
    When `capture_tool_turns` is set, tool call/result data is captured on
    `result` for DB persistence.
    """
    suppress_result_ids = suppress_result_ids or set()
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
        # Count genuine document writes (drives the research finalize nudge).
        if tm.name in _DOC_WRITE_TOOLS and _is_doc_write_success(tm):
            result.docs_written += 1
        # Skip emitting a generic tool_result for delegations (the
        # delegation_complete event carries it) and for exhibits (the
        # `exhibit` event is the user-facing artifact).
        if (
            tm.tool_call_id not in delegation_tool_call_ids
            and tm.tool_call_id not in suppress_result_ids
        ):
            yield _sse("tool_result", {
                "tool": tm.name,
                "tool_call_id": tm.tool_call_id,
                "content": tm.content[:500],
                "success": not is_error,
                "duration_ms": round(tool_avg_time, 2),
            })
            # Auto-capture web sources as a passive `citation` exhibit, right
            # after the search's tool_result (so it reads as "searched, then
            # these are the sources"). Fires for web_search and web_research.
            if tm.name in _AUTO_CITATION_TOOLS and not is_error:
                for event_str in _emit_web_search_citation(tm):
                    yield event_str
            # Workspace document hits become passive `doc` citations (Bibliography).
            elif tm.name in _DOC_CITATION_TOOLS and not is_error:
                for event_str in _emit_document_citation(tm):
                    yield event_str
            # Generated media (image/speech) and MCP media passthrough render
            # inline as image/audio exhibits.
            elif not is_error:
                for event_str in _emit_media_exhibits(tm):
                    yield event_str

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
            delegated_exhibits: list[dict] = []
            if tm.tool_call_id in delegation_raw:
                # Persist the full specialist output + delegation context
                # so the client can reconstruct a DelegationMessage card
                # on conversation restore. Exhibit wires are stripped from the
                # delegation metadata (kept lean) and persisted as their own
                # synthetic present_exhibit turns below.
                dr = dict(delegation_raw[tm.tool_call_id])
                delegated_exhibits = dr.pop("exhibits", None) or []
                result_entry["delegation"] = dr
            result.tool_turns_data.append(result_entry)
            # Synthetic present_exhibit turns for exhibits produced INSIDE the
            # delegation: the existing reload path rebuilds an exhibit card from
            # a present_exhibit tool_call turn (same seam the direct image flow
            # uses), restoring the card right after the delegation card.
            for i, wire in enumerate(delegated_exhibits):
                if not isinstance(wire, dict):
                    continue
                result.tool_turns_data.append({
                    "type": "tool_call",
                    "tool": "present_exhibit",
                    "tool_call_id": wire.get("id") or f"exh_dlg_{tm.tool_call_id}_{i}",
                    "arguments": wire,
                })
            # Same synthetic-turn persistence for auto-rendered media exhibits
            # (generate_image / generate_speech / MCP media passthrough) — the
            # live `exhibit` SSE has no storage of its own, so without this the
            # media card was lost on reload.
            if not is_error and _may_carry_media(tm):
                for i, wire in enumerate(_media_exhibit_wires(tm)):
                    result.tool_turns_data.append({
                        "type": "tool_call",
                        "tool": "present_exhibit",
                        "tool_call_id": wire.get("id") or f"exh_med_{tm.tool_call_id}_{i}",
                        "arguments": wire,
                    })

    messages.extend(tool_messages)

    # `view_image` injection (on-demand vision): a successful call resolves an image
    # ref but returns no pixels — the model only "sees" it when we add a user-role image
    # block for the next round. Never auto-shoved; only when the agent asked. Skipped
    # (with a clarifying note) for a model that can't see images.
    for tm in tool_messages:
        if tm.name != "view_image":
            continue
        for msg in _view_image_messages(tm, vision_capable=vision_capable):
            messages.append(msg)


async def streaming_tool_loop(
    provider,
    model_id: str,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    agent,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_tool_rounds: int = 10,
    max_context_tokens: int = 100000,
    context_window: int | None = None,
    context_warning_threshold: float = 0.85,
    task_context: str = "",
    emit_trajectory_info: bool = True,
    truncate_on_overflow: bool = True,
    capture_tool_turns: bool = False,
    result: ToolLoopResult | None = None,
    cancel_check: Callable[[], bool] | None = None,
    vision_capable: bool = False,
    search_limit_override: int | None = None,
    finalize_nudge: str | None = None,
) -> AsyncGenerator[str]:
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
        search_limit_override: Per-turn search-budget cap to use instead of
            `search.per_turn_limit` (Research Mode passes the elevated cap; 0 = unlimited)
        finalize_nudge: Optional one-shot user message injected when the turn is
            about to end without a document write (Research Mode's delivery guard:
            fires near round exhaustion or at a natural stop, at most once)
    """
    if result is None:
        result = ToolLoopResult()

    # Cancellation seam: the default checks the ambient run's cancel flag (Stop);
    # the plan executor passes a check that also ORs the plan-cancel flag. Purely
    # cooperative — checked at round boundaries (existing await/yield points).
    check_cancel = cancel_check or _ambient_cancel_check

    # Open a per-turn web-search budget window (Foundation #5). web_search /
    # web_research run synchronously inside this loop, so the window is ambiently
    # resolvable. No window ⇒ unlimited, so this is the only gating site for an
    # interactive turn; background callers (alloy/planner) stay unbounded.
    from ..config import get_config_manager
    from ..agent.search_budget import search_budget_window

    # Research Mode passes an elevated cap via `search_limit_override`; otherwise
    # use the standard per-turn limit. 0 = unlimited either way.
    if search_limit_override is not None:
        _search_limit = int(search_limit_override or 0)
    else:
        _search_limit = int(get_config_manager().get("search.per_turn_limit", 8) or 0)
    _conv_id = getattr(getattr(agent, "session", None), "id", None)
    _agent_id = getattr(agent, "agent_id", None)
    # Background work-order registry (`delegate_start`): owned HERE so exactly
    # one finally cancels leftovers on every exit path — cooperative cancel
    # returns, client disconnect (GeneratorExit via aclose — awaiting in an
    # async-gen finally is legal during aclose), and exceptions. Within-turn
    # contract: nothing in this registry ever outlives this generator.
    work_orders: dict[str, _WorkOrder] = {}
    try:
        with search_budget_window(_search_limit, conversation_id=_conv_id, agent_id=_agent_id):
            async for _event in _run_tool_loop(
                provider, model_id, messages, tools, agent,
                temperature=temperature, max_tokens=max_tokens,
                max_tool_rounds=max_tool_rounds, max_context_tokens=max_context_tokens,
                context_window=context_window, context_warning_threshold=context_warning_threshold,
                task_context=task_context, emit_trajectory_info=emit_trajectory_info,
                truncate_on_overflow=truncate_on_overflow, capture_tool_turns=capture_tool_turns,
                result=result, check_cancel=check_cancel, work_orders=work_orders,
                vision_capable=vision_capable,
                finalize_nudge=finalize_nudge,
            ):
                yield _event
    finally:
        if work_orders:
            await _cancel_pending_work_orders(work_orders, result, current_run_id.get())


async def _run_tool_loop(
    provider,
    model_id: str,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    agent,
    *,
    temperature: float,
    max_tokens: int,
    max_tool_rounds: int,
    max_context_tokens: int,
    context_window: int | None,
    context_warning_threshold: float,
    task_context: str,
    emit_trajectory_info: bool,
    truncate_on_overflow: bool,
    capture_tool_turns: bool,
    result: ToolLoopResult,
    check_cancel: Callable[[], bool],
    work_orders: dict[str, _WorkOrder],
    vision_capable: bool = False,
    finalize_nudge: str | None = None,
) -> AsyncGenerator[str]:
    """Inner loop body — runs inside the per-turn search-budget window.

    `work_orders` is the turn's background-delegation registry, owned (and
    cancelled on exit) by `streaming_tool_loop`'s finally.
    """
    from ..agent.output_parser import parse_output
    from ..config import get_config_manager as _get_cfg

    # One-shot recovery when the final answer is cut off by max_tokens
    # (finish_reason == "length"): fold the partial answer in and ask the model
    # to continue. Reasoning models are the usual victims — thinking burns the
    # output budget before the visible answer completes.
    auto_continue = bool(_get_cfg().get("chat.auto_continue_on_length", True))
    continues_used = 0
    # Delivery guard (Research Mode): `finalize_nudge` fires at most once, when
    # the turn is close to ending with no document written — proactively near
    # round exhaustion, or reactively at a natural stop.
    nudge_used = False
    # Lazy semaphore bounding concurrent background work orders (its own pool —
    # a same-round blocking batch has its own, so worst-case 2× max_parallel
    # briefly; accepted).
    bg_sem: asyncio.Semaphore | None = None

    for tool_round in range(max_tool_rounds + 1):
        round_tool_calls = []
        round_content = ""
        round_finish: str | None = None

        # Stop promptly between rounds (before spending another model call) when
        # the run/plan was cancelled. `return` — not `break` — so callers skip the
        # post-loop "no tool calls" synthesis; partial content is already on
        # `result`.
        if check_cancel():
            logger.info("tool loop: cancellation observed at round %d; stopping", tool_round)
            result.final_content = round_content
            return

        # Trajectory compression + truncation + high-usage warning
        if _prepare_round_context(
            messages, tool_round,
            max_context_tokens=max_context_tokens,
            task_context=task_context,
            truncate_on_overflow=truncate_on_overflow,
            context_window=context_window,
            context_warning_threshold=context_warning_threshold,
            active_model=f"{getattr(provider, 'name', '')}:{model_id}",
        ) and emit_trajectory_info:
            yield _sse("info", {"type": "trajectory_compressed"})

        # Stream completion from provider. Announce the wait so the client shows
        # "Thinking…" until the first token arrives (it clears `status` on `chunk`).
        emit_status("thinking")
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
                result.provider_cost += float(chunk.usage.get("cost") or 0.0)
                result.reasoning_tokens += int(chunk.usage.get("reasoning_tokens") or 0)
            if chunk.finish_reason:
                round_finish = chunk.finish_reason
            if chunk.content:
                result.content += chunk.content
                round_content += chunk.content
                yield _sse("chunk", {"content": chunk.content})
            # Non-text payloads the completion itself carried (image/audio-output
            # models) — stored + rendered as exhibits instead of silently dropped.
            if chunk.media:
                for event_str in _emit_model_media(
                    chunk.media, result, capture_tool_turns, model_id,
                ):
                    yield event_str

        result.finish_reason = round_finish
        if round_finish == "length":
            logger.warning(
                "Tool round %d hit max_tokens (%d) for model=%s — output truncated%s",
                tool_round + 1, max_tokens, model_id,
                " mid-tool-call" if round_tool_calls else "",
            )

        # No tool calls means the model is ready to finish — unless the user
        # steered mid-turn, in which case fold the steer in and keep going so
        # the agent course-corrects within the same turn.
        if not round_tool_calls:
            steer_msgs = drain_steer_messages()
            if steer_msgs:
                # Keep the transcript coherent: the answer-so-far becomes an
                # assistant turn, then the steer is a fresh user turn, then we
                # loop for the agent's response. The chunks already streamed; the
                # client flushes its live bubble on the `steer` event. Thinking
                # is stripped so reasoning never feeds back as context.
                folded = parse_output(round_content).content.strip() if round_content else ""
                if folded:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=folded))
                for sm in steer_msgs:
                    messages.append(Message(role=MessageRole.USER, content=sm))
                    result.steers.append({
                        "content": sm, "round": tool_round,
                        "after_tools": [], "phase": "would_end",
                    })
                emit_status("thinking")
                continue

            if round_finish == "length":
                if auto_continue and continues_used < 1:
                    # One-shot recovery: fold the partial answer in (thinking
                    # stripped) and ask the model to pick up where it stopped.
                    continues_used += 1
                    logger.info("Auto-continuing after max_tokens truncation (1/1)")
                    folded = parse_output(round_content).content.strip() if round_content else ""
                    if folded:
                        messages.append(Message(role=MessageRole.ASSISTANT, content=folded))
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=(
                            "Continue your previous response exactly where it "
                            "stopped. Do not repeat anything."
                        ),
                    ))
                    emit_status("thinking")
                    continue
                # Still truncated (or auto-continue disabled) — surface it.
                emit_status("truncated")

            # Would-end barrier: background work orders are strictly
            # within-turn, so the turn must not end while any report is
            # unfolded. Fold whatever is done; if none are, wait
            # (cancel-responsive) for the first to land — then give the model
            # another round to react to the report(s).
            if any(not wo.folded for wo in work_orders.values()):
                folded_text = (
                    parse_output(round_content).content.strip() if round_content else ""
                )
                if folded_text:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=folded_text))
                if not any(
                    wo.task.done() and not wo.folded for wo in work_orders.values()
                ):
                    pending_n = sum(
                        1 for wo in work_orders.values() if not wo.task.done()
                    )
                    emit_status(
                        "thinking", label=f"Waiting for {pending_n} work order(s)…",
                    )
                    if not await _await_work_orders(work_orders, check_cancel):
                        logger.info(
                            "tool loop: cancellation observed during work-order barrier"
                        )
                        result.final_content = round_content
                        return
                for event_str in _fold_completed_work_orders(
                    work_orders, messages, agent,
                    result=result, tool_round=tool_round, phase="would_end",
                ):
                    yield event_str
                emit_status("thinking")
                continue

            # Delivery guard, reactive trigger: the model is stopping without
            # having written the deliverable document. Fold the partial in
            # (thinking stripped — same shape as steer folding) and nudge once.
            if finalize_nudge and not nudge_used and result.docs_written == 0:
                nudge_used = True
                folded = parse_output(round_content).content.strip() if round_content else ""
                if folded:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=folded))
                messages.append(Message(role=MessageRole.USER, content=finalize_nudge))
                logger.info("finalize nudge: fired at natural stop (round %d, no doc written)",
                            tool_round + 1)
                emit_status("thinking", label="Finalizing report…")
                continue

            result.final_content = round_content
            logger.debug(f"Stream loop complete after {tool_round + 1} round(s), no more tool calls")
            # Surface empty completions explicitly so downstream code
            # (session storage, done event, parsed.content) doesn't carry
            # silently empty content through to the UI. Covers ANY empty final
            # (not just round 0): after a delegation round the model sometimes
            # stops without synthesizing — fall back to the last specialist's
            # output so the bubble reads sensibly instead of rendering empty.
            # Checked on PARSED content: a think-only final is visibly empty
            # even though the raw accumulation isn't (reasoning models); the
            # fallback is appended (not overwritten) so thinking still persists.
            if not parse_output(result.content or "").content.strip():
                logger.warning(
                    f"Empty visible completion from model={model_id} "
                    f"(round {tool_round + 1}, delegations={len(result.delegations)})"
                )
                if result.delegations:
                    fallback = result.delegations[-1][:500].strip() or "[delegation completed]"
                else:
                    fallback = "[empty response from model]"
                result.content = (
                    f"{result.content}\n\n{fallback}" if result.content.strip() else fallback
                )
                result.final_content = fallback
                yield _sse("chunk", {"content": fallback})
            break

        # Split off Agent Alloy delegation calls (blocking `delegate_to` +
        # background `delegate_start`) from regular tool calls.
        alloy_executor = getattr(agent, "_active_alloy_executor", None)
        delegation_calls, background_calls, regular_calls = _partition_tool_calls(
            round_tool_calls, agent
        )

        # Emit tool call events (only for non-delegation calls). `present_exhibit`
        # calls are surfaced as typed `exhibit` events instead of a tool card.
        for tc in round_tool_calls:
            result.tools_used.append(tc.name)
        exhibit_tool_call_ids: set = set()
        for tc in regular_calls:
            if tc.name == EXHIBIT_TOOL_NAME:
                exhibit_tool_call_ids.add(tc.id)
                emit_status("running_tool", label="Presenting…")
                for event_str in _emit_exhibit_event(tc):
                    yield event_str
            else:
                emit_status("running_tool", label=f"Running {tc.name}…")
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

        # Dispatch background work orders FIRST (their receipts must exist for
        # this round's tool messages while blocking branches stream below; the
        # detached tasks stream their own events to the run bus).
        if background_calls:
            if bg_sem is None:
                bg_sem = asyncio.Semaphore(
                    getattr(alloy_executor, "max_parallel_delegations", 3) or 3
                )
            _dispatch_background_delegations(
                background_calls, alloy_executor, agent,
                registry=work_orders,
                delegation_messages=delegation_messages,
                delegation_raw=delegation_raw,
                sem=bg_sem,
                tool_round=tool_round,
            )

        async for event_str in _run_delegations(
            delegation_calls, alloy_executor, agent,
            result=result,
            delegation_messages=delegation_messages,
            delegation_raw=delegation_raw,
        ):
            yield event_str

        # Bail before running tools if cancellation arrived during the model
        # stream / delegations — the model produced tool calls, but a cancelled
        # run shouldn't spend (potentially slow) tool time. The provider's
        # call→result contract is unsent here (we never emitted these as a
        # completed round), so returning is safe.
        if check_cancel():
            logger.info("tool loop: cancellation observed before tools (round %d); stopping", tool_round)
            result.final_content = round_content
            return

        # Execute remaining tools (sync), emit result events, extend messages.
        # Background ids ride along so their receipt TOOL messages don't echo
        # as generic tool_result events (the delegation_* stream carries them).
        delegation_tool_call_ids = (
            {tc.id for tc in delegation_calls} | {tc.id for tc in background_calls}
        )
        async for event_str in _execute_and_emit_tools(
            regular_calls, delegation_messages, round_tool_calls,
            delegation_tool_call_ids, agent, messages,
            task_context=task_context,
            capture_tool_turns=capture_tool_turns,
            result=result,
            delegation_raw=delegation_raw,
            suppress_result_ids=exhibit_tool_call_ids,
            vision_capable=vision_capable,
        ):
            yield event_str

        logger.info(
            f"Tool round {tool_round + 1}: executed "
            f"{', '.join(tc.name for tc in round_tool_calls)}"
        )

        # Tools done; the next round digests their output (and may compress the
        # trajectory first). Surface that gap before the round-top "Thinking…".
        emit_status("reading")

        # Fold any mid-turn steer messages in as a fresh user turn at this safe
        # boundary so the next round responds to them (queued live-steering).
        round_tool_names = [tc.name for tc in round_tool_calls]
        for sm in drain_steer_messages():
            messages.append(Message(role=MessageRole.USER, content=sm))
            result.steers.append({
                "content": sm, "round": tool_round,
                "after_tools": round_tool_names, "phase": "tool_boundary",
            })

        # Fold reports from background work orders that finished during this
        # round — same safe boundary as steering.
        if work_orders:
            for event_str in _fold_completed_work_orders(
                work_orders, messages, agent,
                result=result, tool_round=tool_round, phase="tool_boundary",
            ):
                yield event_str

        # Delivery guard, proactive trigger: rounds are nearly exhausted and no
        # document has been written — nudge now, while tools are still on offer,
        # so the model can save the deliverable before the budget closes.
        if (
            finalize_nudge and not nudge_used and result.docs_written == 0
            and tool_round >= max(0, max_tool_rounds - 3)
        ):
            nudge_used = True
            messages.append(Message(role=MessageRole.USER, content=finalize_nudge))
            logger.info("finalize nudge: fired near round exhaustion (round %d/%d, no doc written)",
                        tool_round + 1, max_tool_rounds + 1)
    else:
        # Round budget exhausted with the model still requesting tools on the
        # final (tools-withheld) round — some models emit tool-call tokens
        # anyway; those calls were executed above, but no final completion ever
        # streamed, so without this floor the turn would end in silence
        # (observed live: think-only output, finish_reason=tool_calls). Run one
        # guaranteed text-only synthesis pass. An explicit instruction beats a
        # silently-missing `tools` param, which this model class ignores.
        logger.warning(
            "Tool rounds exhausted (%d) with the model still calling tools "
            "(model=%s) — running a forced synthesis pass",
            max_tool_rounds + 1, model_id,
        )
        # Await + fold every outstanding work order first so the forced
        # synthesis pass sees all the reports (within-turn contract).
        if any(not wo.folded for wo in work_orders.values()):
            emit_status("thinking", label="Waiting for work orders…")
            if not await _await_work_orders(work_orders, check_cancel, all_of_them=True):
                logger.info(
                    "tool loop: cancellation observed during exhaustion barrier"
                )
                return
            for event_str in _fold_completed_work_orders(
                work_orders, messages, agent,
                result=result, tool_round=max_tool_rounds, phase="round_exhausted",
            ):
                yield event_str
        messages.append(Message(
            role=MessageRole.USER,
            content=(
                "Tool budget is exhausted. Provide your complete final answer now, "
                "based on everything gathered. Do not call tools."
            ),
        ))
        emit_status("finalizing", label="Finalizing…")
        synthesis_content = ""
        async for chunk in provider.stream(
            messages, model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
            tool_choice=None,
        ):
            # Text-only by construction: any tool calls here are ignored, never
            # executed — this pass exists to produce the answer, nothing else.
            if chunk.usage:
                result.tokens_in += chunk.usage.get("prompt_tokens", 0)
                result.tokens_out += chunk.usage.get("completion_tokens", 0)
                result.provider_cost += float(chunk.usage.get("cost") or 0.0)
                result.reasoning_tokens += int(chunk.usage.get("reasoning_tokens") or 0)
            if chunk.finish_reason:
                result.finish_reason = chunk.finish_reason
            if chunk.content:
                result.content += chunk.content
                synthesis_content += chunk.content
                yield _sse("chunk", {"content": chunk.content})
        result.final_content = synthesis_content
        # Mirror the natural-stop empty-final fallback so downstream never
        # carries silently empty content (same shape as the in-loop guard).
        # Parsed check: a think-only synthesis is visibly empty; append the
        # fallback so thinking still persists.
        if not parse_output(synthesis_content or "").content.strip():
            logger.warning(
                f"Empty visible synthesis from model={model_id} after round exhaustion "
                f"(delegations={len(result.delegations)})"
            )
            if result.delegations:
                fallback = result.delegations[-1][:500].strip() or "[delegation completed]"
            else:
                fallback = "[tool budget exhausted before a final answer was produced]"
            result.content = (
                f"{result.content}\n\n{fallback}" if result.content.strip() else fallback
            )
            result.final_content = fallback
            yield _sse("chunk", {"content": fallback})

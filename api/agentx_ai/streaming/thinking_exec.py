"""
Multi-pass thinking execution for the streaming chat turn.

Two patterns run as *wrapped streams* around the ordinary tool loop:

- ``deep_reflection`` — a hidden DRAFT completion and a hidden CRITIQUE
  completion are surfaced live inside a synthetic ``<think>`` block (the client
  ThinkingBubble renders it as thinking), then the FINAL pass streams normally
  with tools via ``streaming_tool_loop``.
- ``self_consistency`` — k parallel short samples (no tools) surface inside the
  synthetic ``<think>`` block, then a judged final pass streams normally.

Invariants:
- Pre-passes are TOOL-LESS (plain completions) — no tool_call/exhibit events
  can fire inside an open thinking bubble.
- Everything a pre-pass emits goes through :class:`ThinkTagSanitizer` — a
  native reasoner's own ``<think>`` tags nested inside the synthetic wrapper
  would corrupt ``parse_output``'s non-greedy regex and the live bubble.
- The synthetic ``</think>`` close is emitted in a ``finally`` so a mid-pass
  failure can't leave the client bubble open.
- The caller's ``ToolLoopResult`` is handed to the final pass, then its
  ``content`` is prefixed with the synthetic think block and its token counts
  absorb the pre-pass usage — done-event thinking, cost, and hard-stop
  persistence all keep working unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from ..providers.base import Message, MessageRole
from .status import emit_status
from .tool_loop import ToolLoopResult, _sse, streaming_tool_loop

logger = logging.getLogger(__name__)

# Longest tag we must hold back across chunk boundaries ("</thinking>").
_TAGS = ("<think>", "</think>", "<thinking>", "</thinking>")
_MAX_TAG_LEN = max(len(t) for t in _TAGS)

# Pre-pass output budgets (answer-shaped, deliberately smaller than the final).
_DRAFT_MAX_TOKENS = 1500
_CRITIQUE_MAX_TOKENS = 600
_SAMPLE_MAX_TOKENS = 800
# Cap each surfaced sample/draft excerpt so the bubble stays readable.
_BUBBLE_EXCERPT_CHARS = 1200


class ThinkTagSanitizer:
    """Streaming filter that strips think tags from embedded pass output.

    Stateful: a tag split across chunk boundaries ("…<thi" + "nk>…") is held
    back until it can be judged. ``flush()`` returns any held tail (a partial
    that never became a tag passes through verbatim).
    """

    def __init__(self) -> None:
        self._buf = ""

    @staticmethod
    def _strip(text: str) -> str:
        for tag in _TAGS:
            text = text.replace(tag, "")
        return text

    def feed(self, chunk: str) -> str:
        text = self._buf + chunk
        # Hold back the shortest tail that could still become a tag.
        keep = 0
        limit = min(len(text), _MAX_TAG_LEN - 1)
        for i in range(1, limit + 1):
            tail = text[len(text) - i:]
            if any(t.startswith(tail) for t in _TAGS):
                keep = i
        cut = len(text) - keep
        self._buf = text[cut:]
        return self._strip(text[:cut])

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return self._strip(out)


def _chunk(text: str) -> str:
    """A synthetic text-delta SSE event (same wire shape as the tool loop's)."""
    return _sse("chunk", {"content": text})


def _prepend_think(result: ToolLoopResult, think_text: str) -> None:
    """Fold the synthetic think block into the caller's result so parse_output
    extracts it as this turn's thinking (persistence/cost paths unchanged)."""
    result.content = f"<think>{think_text}</think>{result.content}"


def _usage_tokens(completion) -> tuple[int, int]:
    usage = getattr(completion, "usage", None) or {}
    return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)


async def _complete(model: str, messages: list[Message], *, preferred_fallback: str | None,
                    temperature: float, max_tokens: int):
    """One hidden pre-pass completion via the hardened fallback path."""
    from ..providers.registry import get_registry

    return await get_registry().complete_with_fallback(
        model or "",
        messages,
        preferred_fallback=preferred_fallback,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _excerpt(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _BUBBLE_EXCERPT_CHARS:
        return text
    return text[:_BUBBLE_EXCERPT_CHARS].rstrip() + " …[trimmed]"


async def reflection_multipass_stream(
    provider, model_id: str, messages: list[Message], tools, agent,
    *, result: ToolLoopResult, active_model: str | None, **loop_kwargs,
) -> AsyncGenerator[str]:
    """deep_reflection: draft → critique surfaced as live thinking, then the
    improved final answer streams normally (tools intact)."""
    sanitizer = ThinkTagSanitizer()
    think_parts: list[str] = []
    pre_in = pre_out = 0
    draft_text = ""
    critique_text = ""

    yield _chunk("<think>")
    try:
        emit_status("reasoning_step", "Drafting…")
        yield _chunk("Draft:\n")
        try:
            draft = await _complete(
                active_model or "", messages,
                preferred_fallback=active_model,
                temperature=float(loop_kwargs.get("temperature", 0.7) or 0.7),
                max_tokens=_DRAFT_MAX_TOKENS,
            )
            t_in, t_out = _usage_tokens(draft)
            pre_in, pre_out = pre_in + t_in, pre_out + t_out
            draft_text = sanitizer.feed(draft.content or "") + sanitizer.flush()
        except Exception as e:  # noqa: BLE001 — degrade to a plain turn
            logger.warning(f"deep_reflection draft pass failed: {e}")
        shown_draft = _excerpt(draft_text) or "(draft unavailable)"
        yield _chunk(shown_draft)
        think_parts.append(f"Draft:\n{shown_draft}")

        if draft_text:
            emit_status("reasoning_step", "Critiquing the draft…")
            yield _chunk("\n\nCritique:\n")
            try:
                from ..prompts import get_prompt
                critique_instruction = get_prompt("reasoning.chat.multipass_critique")
                critique = await _complete(
                    active_model or "",
                    [*messages,
                     Message(role=MessageRole.ASSISTANT, content=draft_text),
                     Message(role=MessageRole.USER, content=critique_instruction)],
                    preferred_fallback=active_model,
                    temperature=0.3,
                    max_tokens=_CRITIQUE_MAX_TOKENS,
                )
                t_in, t_out = _usage_tokens(critique)
                pre_in, pre_out = pre_in + t_in, pre_out + t_out
                critique_text = sanitizer.feed(critique.content or "") + sanitizer.flush()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"deep_reflection critique pass failed: {e}")
            shown_critique = _excerpt(critique_text) or "(critique unavailable)"
            yield _chunk(shown_critique)
            think_parts.append(f"Critique:\n{shown_critique}")
    except GeneratorExit:
        # Hard stop mid-pre-pass: an async generator must not yield again
        # while closing — persistence tolerates an unclosed think tag.
        raise
    finally:
        # Never leave the client's thinking bubble open, whatever happened
        # above — except on GeneratorExit, where yielding is illegal.
        import sys
        if not isinstance(sys.exception(), GeneratorExit):
            yield _chunk("</think>")

    # Final pass: normal streamed tool loop, briefed with the draft + critique.
    final_messages = list(messages)
    if draft_text:
        from ..prompts import get_prompt
        brief = get_prompt(
            "reasoning.chat.multipass_final",
            draft=draft_text, critique=critique_text or "(no critique available)",
        )
        final_messages.append(Message(role=MessageRole.SYSTEM, content=brief))

    emit_status("reasoning_step", "Finalizing…")
    async for event in streaming_tool_loop(
        provider, model_id, final_messages, tools, agent, result=result, **loop_kwargs,
    ):
        yield event

    _prepend_think(result, "\n\n".join(think_parts) or "(reflection passes unavailable)")
    result.tokens_in += pre_in
    result.tokens_out += pre_out


async def self_consistency_stream(
    provider, model_id: str, messages: list[Message], tools, agent,
    *, result: ToolLoopResult, active_model: str | None, **loop_kwargs,
) -> AsyncGenerator[str]:
    """self_consistency: k parallel tool-less samples surfaced as live
    thinking, then a judged final answer streams normally."""
    from ..config import get_config_manager
    from ..model_roles import resolve_member_model

    cfg = get_config_manager()
    k = max(2, min(5, int(cfg.get("reasoning.sc_k", 3) or 3)))
    explicit = cfg.get("reasoning.sc_model", "")
    sample_model = resolve_member_model("thinking_sc", explicit) or explicit or (active_model or "")

    think_parts: list[str] = []
    pre_in = pre_out = 0
    samples: list[str] = []

    yield _chunk("<think>")
    try:
        emit_status("reasoning_step", f"Sampling {k} independent solutions…")
        yield _chunk(f"Sampling {k} independent solutions…\n")
        tasks = [
            _complete(
                sample_model, messages,
                preferred_fallback=active_model,
                temperature=0.8,  # diversity is the point of sampling
                max_tokens=_SAMPLE_MAX_TOKENS,
            )
            for _ in range(k)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for i, outcome in enumerate(outcomes, 1):
            if isinstance(outcome, BaseException):
                logger.warning(f"self_consistency sample {i} failed: {outcome}")
                continue
            t_in, t_out = _usage_tokens(outcome)
            pre_in, pre_out = pre_in + t_in, pre_out + t_out
            sanitizer = ThinkTagSanitizer()
            text = sanitizer.feed(outcome.content or "") + sanitizer.flush()
            if not text.strip():
                continue
            samples.append(text)
            shown = _excerpt(text)
            yield _chunk(f"\nSolution {i}:\n{shown}\n")
            think_parts.append(f"Solution {i}:\n{shown}")
    except GeneratorExit:
        raise  # hard stop: no further yields allowed while closing
    finally:
        import sys
        if not isinstance(sys.exception(), GeneratorExit):
            yield _chunk("</think>")

    # Judged final pass (streams; tools intact). With zero usable samples the
    # turn degrades to a plain streamed answer.
    final_messages = list(messages)
    if samples:
        from ..prompts import get_prompt
        brief = get_prompt(
            "reasoning.chat.sc_judge",
            samples="\n\n".join(
                f"Solution {i}:\n{s}" for i, s in enumerate(samples, 1)
            ),
        )
        final_messages.append(Message(role=MessageRole.SYSTEM, content=brief))

    emit_status("reasoning_step", "Judging the consensus…")
    async for event in streaming_tool_loop(
        provider, model_id, final_messages, tools, agent, result=result, **loop_kwargs,
    ):
        yield event

    _prepend_think(
        result,
        "\n\n".join(think_parts) or "(sampling unavailable — answered directly)",
    )
    result.tokens_in += pre_in
    result.tokens_out += pre_out


def thinking_stream(
    pattern: str | None,
    provider, model_id: str, messages: list[Message], tools, agent,
    *, result: ToolLoopResult, active_model: str | None = None, **loop_kwargs,
) -> AsyncGenerator[str]:
    """The single chooser the chat path iterates: multi-pass wrapper for
    ``deep_reflection``/``self_consistency``, the plain tool loop otherwise."""
    if pattern == "deep_reflection":
        return reflection_multipass_stream(
            provider, model_id, messages, tools, agent,
            result=result, active_model=active_model, **loop_kwargs,
        )
    if pattern == "self_consistency":
        return self_consistency_stream(
            provider, model_id, messages, tools, agent,
            result=result, active_model=active_model, **loop_kwargs,
        )
    return streaming_tool_loop(
        provider, model_id, messages, tools, agent, result=result, **loop_kwargs,
    )

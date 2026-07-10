"""
Chat thinking patterns — reasoning compiled INTO the streaming chat turn.

The offline kit (CoT/ToT/ReAct/Reflection classes) is blocking and lives on
``/api/agent/run``; chat instead treats a pattern as a *plan* applied to the
one execution engine every turn already uses (the streaming tool loop):

- a system-layer :class:`LedgerBlock` directive (``cot``, ``reflection``),
- an optional hidden pre-call (``step_back`` — principles extracted first),
- an output-budget floor (thinking spends output tokens before the answer),
- and, for the multi-pass patterns (``deep_reflection``, ``self_consistency``),
  a wrapped stream executed by ``streaming/thinking_exec.py``.

Resolution precedence: per-turn override > profile ``reasoning_strategy``
(≠auto) > ``preferences.default_reasoning_strategy`` (≠auto) > auto-selection
(``reasoning/selection.py``). Every source passes through the degradation map
(legacy ``tot``/``react`` values keep working) and per-pattern config enables.
All best-effort: a failure anywhere yields the empty plan, never a dead turn.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..agent.context_ledger import LedgerBlock

logger = logging.getLogger(__name__)

# Patterns whose whole point is model thinking — they raise the output floor
# so the thinking doesn't starve the visible answer.
_THINKING_PATTERNS: frozenset[str] = frozenset(
    {"native", "cot", "step_back", "reflection", "deep_reflection"}
)

# Pattern → config enable key (reasoning.<key>). `native` has no toggle — it
# is the absence of a scaffold.
_ENABLE_KEYS: dict[str, str] = {
    "cot": "cot_enabled",
    "step_back": "step_back_enabled",
    "reflection": "reflection_enabled",
    "deep_reflection": "reflection_enabled",
    "self_consistency": "self_consistency_enabled",
}


@dataclass(frozen=True)
class ThinkingPlan:
    """The per-turn compilation of a thinking pattern.

    ``pattern`` None ⇒ plain turn (no scaffold, no floor beyond the model's own
    native-reasoning floor). ``blocks`` are ready-to-unpack ledger blocks.
    ``min_output`` is the thinking output floor (None ⇒ no floor). ``note`` is
    a one-line user-facing status when a degradation/adjustment was applied.
    """

    pattern: str | None = None
    blocks: list[LedgerBlock] = field(default_factory=list)
    min_output: int | None = None
    note: str | None = None
    auto_selected: bool = False


EMPTY_PLAN = ThinkingPlan()


def _cfg():
    from ..config import get_config_manager

    return get_config_manager()


def _pattern_enabled(pattern: str, cfg) -> bool:
    key = _ENABLE_KEYS.get(pattern)
    if key is None:
        return True
    return bool(cfg.get(f"reasoning.{key}", True))


def _directive_block(pattern: str) -> list[LedgerBlock]:
    """The system-layer directive for prompt-only patterns ("" ⇒ no block).

    Non-mandatory at priority 92 — thinking guidance should survive ordinary
    pressure (above project blocks) but never crowd out the base prompt.
    """
    from ..prompts import get_prompt

    key = {
        "cot": "reasoning.chat.cot_directive",
        "reflection": "reasoning.chat.reflection_directive",
        "native": "",  # native = no scaffold
    }.get(pattern, "")
    if not key:
        return []
    try:
        content = get_prompt(key)
    except Exception as e:  # noqa: BLE001 — a missing template never kills a turn
        logger.warning(f"thinking directive template {key} failed: {e}")
        return []
    if not content:
        return []
    return [LedgerBlock(key=f"thinking_{pattern}", priority=92, content=content)]


def _react_nudge_block() -> list[LedgerBlock]:
    """The tool-narration nudge used when a legacy `react` selection degrades
    to native (the tool loop IS ReAct; the nudge keeps the visible narration)."""
    from ..prompts import get_prompt

    try:
        content = get_prompt("reasoning.chat.react_narration_nudge")
    except Exception:  # noqa: BLE001
        return []
    return (
        [LedgerBlock(key="thinking_react_nudge", priority=92, content=content)]
        if content
        else []
    )


async def _step_back_block(
    message: str, active_model: str | None
) -> list[LedgerBlock]:
    """The step_back pre-call: extract governing principles BEFORE the turn.

    One bounded hidden completion (≤300 output tokens, hard timeout). Emits the
    ``reasoning_step`` status while it runs — status rides the run event bus,
    so it reaches the client even though the chat generator is blocked here.
    Failure/timeout returns [] and the caller degrades the pattern to ``cot``.
    """
    from ..model_roles import resolve_member_model
    from ..prompts import get_prompt
    from ..providers.base import Message, MessageRole
    from ..providers.registry import get_registry
    from ..streaming.status import emit_status

    cfg = _cfg()
    explicit = cfg.get("reasoning.step_back_model", "")
    # "" ⇒ the active turn model (bucket-b semantics; role: refs still expand).
    model = resolve_member_model("thinking_step_back", explicit) or explicit or ""
    timeout = float(cfg.get("reasoning.step_back_timeout_seconds", 20))

    emit_status("reasoning_step", "Distilling the underlying principles…")
    try:
        prompt = get_prompt("reasoning.chat.step_back_extract", question=message[:2000])
        result = await asyncio.wait_for(
            get_registry().complete_with_fallback(
                model,
                [Message(role=MessageRole.USER, content=prompt)],
                preferred_fallback=active_model,
                temperature=0.2,
                max_tokens=300,
            ),
            timeout=timeout,
        )
        principles = (result.content or "").strip()
    except Exception as e:  # noqa: BLE001 — pre-call is best-effort
        logger.warning(f"step_back pre-call failed: {e}")
        return []
    if not principles:
        return []
    return [LedgerBlock(
        key="thinking_step_back",
        priority=92,
        content=(
            "Governing principles distilled for this question (apply them "
            f"explicitly in your reasoning):\n{principles}"
        ),
    )]


async def supports_reasoning_hardened(provider, model_id: str, caps) -> bool:
    """Whether the model reasons natively — hardened against cold catalogs.

    A provider with a lazy model catalog (OpenRouter) reports
    ``supports_reasoning=False`` for an uncached model; warm it once and
    re-check before concluding (mirrors ``Agent._model_supports_tools``).
    Defaults to the cheap answer (False) on probe failure — the cost of a miss
    is only a scaffold that native thinking makes redundant, not a broken turn.
    """
    try:
        if caps.supports_reasoning:
            return True
        warm = getattr(provider, "fetch_models", None)
        if warm is not None:
            await asyncio.wait_for(warm(), timeout=15.0)
            caps = provider.get_capabilities(model_id)
        return bool(caps.supports_reasoning)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"reasoning capability probe failed: {e}")
        return False


def thinking_min_output(pattern: str | None, supports_reasoning: bool) -> int | None:
    """The thinking output floor for this turn (None ⇒ no floor).

    ``reasoning.min_output_tokens`` > 0 pins an explicit floor; 0 (auto) floors
    at ``REASONING_MIN_OUTPUT_TOKENS`` whenever a thinking pattern is active or
    the model reasons natively — thinking spends output tokens before the
    visible answer, so the bare chat default starves it.
    """
    from ..streaming.constants import REASONING_MIN_OUTPUT_TOKENS

    explicit = int(_cfg().get("reasoning.min_output_tokens", 0) or 0)
    active = (pattern in _THINKING_PATTERNS) or supports_reasoning
    if explicit > 0:
        return explicit if active or pattern else None
    return REASONING_MIN_OUTPUT_TOKENS if active else None


async def resolve_thinking_plan(
    message: str,
    *,
    turn_override: str | None,
    profile_strategy: str | None,
    provider,
    model_id: str,
    caps,
    active_model: str | None,
    research_active: bool,
    tools_likely: bool,
    direct_mode: bool = False,
    session_had_thinking: bool = False,
) -> ThinkingPlan:
    """Compile this turn's thinking pattern into a :class:`ThinkingPlan`.

    Research turns keep their own rigorous prompt (no pattern stacking); direct
    mode bypasses the ledger entirely. Never raises.
    """
    from .selection import normalize_chat_pattern, select_pattern

    cfg = _cfg()
    if direct_mode or research_active:
        return EMPTY_PLAN
    if not bool(cfg.get("reasoning.chat_patterns_enabled", True)):
        return EMPTY_PLAN

    supports_reasoning = session_had_thinking or await supports_reasoning_hardened(
        provider, model_id, caps
    )

    # --- Resolve the requested pattern through the precedence chain ---------
    pattern: str | None = None
    note: str | None = None
    auto_selected = False
    requested_react = False
    for source in (
        turn_override,
        profile_strategy,
        cfg.get("preferences.default_reasoning_strategy", ""),
    ):
        if not source or str(source).strip().lower() == "auto":
            continue
        requested_react = requested_react or str(source).strip().lower() == "react"
        p, n = normalize_chat_pattern(str(source))
        if p is not None and _pattern_enabled(p, cfg):
            pattern, note = p, n
            break
        # A disabled/unknown pattern falls through to the next source.

    if pattern is None and not requested_react:
        sc_ok = _pattern_enabled("self_consistency", cfg)
        pattern, confidence = select_pattern(
            message,
            supports_reasoning=supports_reasoning,
            tools_likely=tools_likely,
            sc_enabled=sc_ok,
        )
        auto_selected = pattern is not None
        if (
            pattern is None
            and confidence < 0.5
            and bool(cfg.get("reasoning.auto_classifier_enabled", True))
            and len(message) >= int(cfg.get("reasoning.classifier_min_chars", 240))
        ):
            from .selection import llm_tiebreak

            pattern = await llm_tiebreak(
                message,
                supports_reasoning=supports_reasoning,
                active_model=active_model,
            )
            auto_selected = pattern is not None
        if pattern is not None and not _pattern_enabled(pattern, cfg):
            pattern = None
        # Auto NEVER picks the expensive multi-pass reflection.
        if auto_selected and pattern == "deep_reflection":
            pattern = "reflection"

    # --- Compile the pattern -------------------------------------------------
    blocks: list[LedgerBlock] = []
    if pattern == "step_back":
        blocks = await _step_back_block(message, active_model)
        if not blocks:
            pattern, note = "cot", "Step-back unavailable — thinking step-by-step"
    if pattern in ("cot", "reflection"):
        blocks = _directive_block(pattern)
    elif pattern == "native" and requested_react:
        blocks = _react_nudge_block()

    if note:
        from ..streaming.status import emit_status

        emit_status("composing", note)

    return ThinkingPlan(
        pattern=pattern,
        blocks=blocks,
        min_output=thinking_min_output(pattern, supports_reasoning),
        note=note,
        auto_selected=auto_selected,
    )

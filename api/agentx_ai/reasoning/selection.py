"""
Shared strategy/pattern selection — ONE brain for "how should the agent think?".

Both consumers delegate here so their heuristics can't drift apart:
- the chat path (``reasoning/chat_patterns.py``) selecting a **thinking pattern**
  compiled into the streaming tool loop, and
- the offline kit's ``ReasoningOrchestrator._classify_task`` (``/api/agent/run``).

Selection is deliberately layered for latency: pure keyword heuristics first
(0ms, always), then an optional LLM tiebreak (``llm_tiebreak``, fast_utility
role, bounded) ONLY when the heuristics are unconfident, the message is
non-trivial, and ``reasoning.auto_classifier_enabled`` is on. The unconfident
default is cheap on purpose — patterns are progressive enhancements, so a wrong
"none" is invisible while a wrong scaffold wastes budget.

Import discipline: this module stays pure at import time (no provider/registry
imports at module scope) — ``reasoning/__init__`` already eagerly imports the
five strategy modules; selection must not deepen that graph.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks for strategy selection (moved from orchestrator.py —
    the orchestrator re-exports it for backward compatibility)."""
    SIMPLE = "simple"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    RESEARCH = "research"
    PLANNING = "planning"
    CODE = "code"
    MATH = "math"
    UNKNOWN = "unknown"


# Chat-side thinking patterns. `tot`/`react` are offline-kit strategies that
# DEGRADE honestly in chat (the tool loop already IS ReAct with real function
# calling; ToT is too costly for a chat default) — the map below is applied to
# every requested value so legacy profile enums keep working.
CHAT_PATTERNS: tuple[str, ...] = (
    "native", "cot", "step_back", "reflection", "deep_reflection", "self_consistency",
)

# Legacy/offline strategy value → chat pattern. Values not in this map and not
# in CHAT_PATTERNS resolve to None (no scaffold).
CHAT_DEGRADATIONS: dict[str, str] = {
    "tot": "cot",
    "react": "native",
}

# Human-readable notes surfaced (once, as a status line) when a degradation is
# applied, so the user isn't silently given something else than they picked.
DEGRADATION_NOTES: dict[str, str] = {
    "tot": "Tree-of-thought runs in offline tasks — using step-by-step thinking here",
    "react": "Tool use is built into chat turns — thinking natively",
}


def normalize_chat_pattern(value: str | None) -> tuple[str | None, str | None]:
    """Map a requested strategy value onto a chat pattern.

    Returns ``(pattern, degradation_note)``. ``auto``/empty/None → (None, None)
    — the caller decides whether to run auto-selection. Unknown values return
    (None, None) so a stale/foreign string can never raise mid-turn.
    """
    if not value:
        return None, None
    v = value.strip().lower()
    if not v or v == "auto":
        return None, None
    if v in CHAT_PATTERNS:
        return v, None
    if v in CHAT_DEGRADATIONS:
        return CHAT_DEGRADATIONS[v], DEGRADATION_NOTES.get(v)
    logger.debug(f"Unknown thinking pattern {value!r} — ignoring")
    return None, None


def classify_task_type(task: str) -> TaskType:
    """Classify the task type based on keywords and patterns.

    Moved verbatim from ``ReasoningOrchestrator._classify_task`` so the chat
    selector and the offline orchestrator share one definition.
    """
    task_lower = task.lower()

    # Math indicators
    math_words = ["calculate", "compute", "solve", "equation", "formula", "sum", "product", "divide", "multiply"]
    if any(w in task_lower for w in math_words):
        return TaskType.MATH

    # Code indicators
    code_words = ["code", "program", "function", "implement", "algorithm", "debug", "fix the bug", "script"]
    if any(w in task_lower for w in code_words):
        return TaskType.CODE

    # Research indicators
    research_words = ["search", "find information", "look up", "research", "what is the latest", "current"]
    if any(w in task_lower for w in research_words):
        return TaskType.RESEARCH

    # Planning indicators
    planning_words = ["plan", "design", "strategy", "approach", "organize", "schedule", "roadmap"]
    if any(w in task_lower for w in planning_words):
        return TaskType.PLANNING

    # Creative indicators
    creative_words = ["write", "create", "compose", "draft", "story", "essay", "poem", "creative"]
    if any(w in task_lower for w in creative_words):
        return TaskType.CREATIVE

    # Analytical indicators
    analytical_words = ["analyze", "compare", "evaluate", "assess", "review", "examine", "explain why"]
    if any(w in task_lower for w in analytical_words):
        return TaskType.ANALYTICAL

    # Simple task indicators
    simple_words = ["what is", "who is", "when did", "where is", "define", "list"]
    if any(w in task_lower for w in simple_words):
        return TaskType.SIMPLE

    return TaskType.UNKNOWN


# Auto-selection: task type → preferred chat pattern. Two tables — native
# reasoners never get a CoT scaffold stacked on top of their own thinking
# (double-spends the output budget); non-native models get the visible-steps
# scaffold instead. ``self_consistency`` is inserted by ``select_pattern``
# under its own stricter gate (math/logic AND no tools likely AND enabled).
_AUTO_NATIVE: dict[TaskType, str | None] = {
    TaskType.SIMPLE: None,
    TaskType.ANALYTICAL: "native",
    TaskType.CREATIVE: "reflection",
    TaskType.RESEARCH: "native",
    TaskType.PLANNING: "native",
    TaskType.CODE: "reflection",
    TaskType.MATH: "native",
    TaskType.UNKNOWN: None,
}
_AUTO_SCAFFOLD: dict[TaskType, str | None] = {
    TaskType.SIMPLE: None,
    TaskType.ANALYTICAL: "cot",
    TaskType.CREATIVE: "reflection",
    TaskType.RESEARCH: None,       # the tool loop does the work; no scaffold
    TaskType.PLANNING: "cot",
    TaskType.CODE: "reflection",
    TaskType.MATH: "cot",
    TaskType.UNKNOWN: None,
}

# Conceptual "why/how does X work" phrasing benefits from step-back (extract
# the governing principle first) — checked before the generic tables.
_STEP_BACK_MARKERS = (
    "why does", "why do", "why is", "how does", "how do", "what principle",
    "underlying", "fundamentally", "first principles", "conceptually",
)


def select_pattern(
    message: str,
    *,
    supports_reasoning: bool,
    tools_likely: bool,
    sc_enabled: bool = False,
) -> tuple[str | None, float]:
    """Heuristic auto-selection of a chat thinking pattern.

    Returns ``(pattern | None, confidence 0..1)``. Confidence < 0.5 means the
    keyword heuristics didn't recognize the task — callers may consult the LLM
    tiebreak, else fall back to the cheap default (``native`` for reasoning
    models, no scaffold otherwise).
    """
    task_type = classify_task_type(message)

    # Self-consistency: only clear math/logic with no tool involvement — k
    # samples on a tool-needing turn would sample without the tools.
    if (
        sc_enabled
        and task_type == TaskType.MATH
        and not tools_likely
    ):
        return "self_consistency", 0.8

    if task_type in (TaskType.ANALYTICAL, TaskType.UNKNOWN) and any(
        m in message.lower() for m in _STEP_BACK_MARKERS
    ):
        return "step_back", 0.7

    table = _AUTO_NATIVE if supports_reasoning else _AUTO_SCAFFOLD
    pattern = table.get(task_type)
    confidence = 0.3 if task_type == TaskType.UNKNOWN else 0.8
    if pattern is None and supports_reasoning and task_type == TaskType.UNKNOWN:
        # Cheap unconfident default: let a native reasoner think natively.
        return "native", confidence
    return pattern, confidence


async def llm_tiebreak(
    message: str,
    *,
    supports_reasoning: bool,
    active_model: str | None = None,
    timeout_seconds: float = 5.0,
) -> str | None:
    """One bounded LLM call to pick a pattern when heuristics are unconfident.

    fast_utility-role model (``reasoning.classifier_model``, "" ⇒ role), ≤150
    output tokens, hard timeout — the heuristic answer stands on any failure.
    Returns a pattern name or None. Deferred imports keep module import pure.
    """
    import asyncio

    from ..config import get_config_manager
    from ..model_roles import resolve_member_model
    from ..prompts.loader import get_prompt_loader
    from ..providers.base import NO_REASONING, Message, MessageRole
    from ..providers.registry import get_registry

    cfg = get_config_manager()
    explicit = cfg.get("reasoning.classifier_model", "")
    model = resolve_member_model("thinking_classifier", explicit) or explicit or ""

    try:
        prompt = get_prompt_loader().get(
            "reasoning.chat.classifier",
            message=message[:1500],
            native="yes" if supports_reasoning else "no",
        )
        # NO_REASONING: a thinking route burns the 150-token budget on hidden
        # reasoning and returns empty — the classifier must stay cheap + fast.
        result = await asyncio.wait_for(
            get_registry().complete_with_fallback(
                model,
                [Message(role=MessageRole.USER, content=prompt)],
                preferred_fallback=active_model,
                temperature=0.0,
                max_tokens=150,
                extra_body=NO_REASONING,
            ),
            timeout=timeout_seconds,
        )
        answer = (result.content or "").strip().lower()
        # The prompt asks for one bare word; scan defensively anyway.
        for candidate in (*CHAT_PATTERNS, "none"):
            if candidate in answer:
                return None if candidate == "none" else candidate
        return None
    except Exception as e:  # noqa: BLE001 — tiebreak is best-effort
        logger.debug(f"thinking classifier tiebreak skipped: {e}")
        return None

"""
Helper functions for streaming chat.

Extracted from views.py to reduce complexity and enable reuse.
"""

import json
import logging
from typing import Any, TypeVar

from ..tokens import estimate_messages
from .constants import (
    CHAR_TO_TOKEN_RATIO,
    CONTEXT_BUFFER_TOKENS,
    MAX_OUTPUT_TOKENS_CEILING,
    MIN_OUTPUT_TOKENS,
    MIN_TOOL_CONTENT_SIZE,
    REASONING_MIN_OUTPUT_TOKENS,
    TRUNCATION_MARKER,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def estimate_tokens(messages: list[Any]) -> int:
    """
    Estimate token count from messages.

    Delegates to the shared :func:`agentx_ai.tokens.estimate_messages` (tiktoken-backed,
    with a chars/4 fallback) so every budget consumer agrees on token sizing.

    Args:
        messages: List of Message objects with content attribute

    Returns:
        Estimated token count
    """
    return estimate_messages(messages)


def compute_adaptive_max_tokens(
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    *,
    context_window: int,
    max_output_tokens: int,
    max_output_override: int | None,
    supports_reasoning: bool = False,
) -> int:
    """Adaptive per-call output budget from context usage.

    Accounts for tool-schema tokens too: estimate_tokens() only sees message
    content, but the provider also counts the serialized tool definitions
    toward the prompt. Clamps the capability-reported output cap to a sane
    ceiling (a model advertising max_output == context_window would otherwise
    make us request ~the whole window as output → provider 400); an explicit
    override is trusted as-is. The floor is raised for reasoning models, whose
    thinking spends output budget before the visible answer.
    """
    estimated_input = estimate_tokens(messages)
    estimated_tool_tokens = (
        len(json.dumps(tools)) // CHAR_TO_TOKEN_RATIO if tools else 0
    )
    available_for_output = (
        context_window - estimated_input - estimated_tool_tokens - CONTEXT_BUFFER_TOKENS
    )
    output_cap = (
        max_output_tokens
        if max_output_override
        else min(max_output_tokens, MAX_OUTPUT_TOKENS_CEILING)
    )
    min_output = REASONING_MIN_OUTPUT_TOKENS if supports_reasoning else MIN_OUTPUT_TOKENS
    adaptive = max(min(output_cap, available_for_output), min_output)

    logger.debug(
        f"Adaptive max_tokens: {adaptive} "
        f"(context_window={context_window}, estimated_input={estimated_input}, "
        f"tool_tokens={estimated_tool_tokens}, output_cap={output_cap}, "
        f"max_output={max_output_tokens}, available={available_for_output})"
    )
    return adaptive


def truncate_tool_messages(
    messages: list[Any],
    current_tokens: int,
    limit_tokens: int,
) -> int:
    """
    Truncate tool message contents to fit within token limit.

    Iterates through messages in reverse (most recent first) and
    truncates TOOL role messages until under the limit.

    Args:
        messages: List of Message objects (modified in place)
        current_tokens: Current estimated token count
        limit_tokens: Maximum allowed tokens

    Returns:
        Number of messages truncated
    """
    from ..providers.base import MessageRole

    if current_tokens <= limit_tokens:
        return 0

    excess_chars = (current_tokens - limit_tokens) * CHAR_TO_TOKEN_RATIO
    truncated_count = 0

    for msg in reversed(messages):
        if msg.role != MessageRole.TOOL or not msg.content:
            continue
        if len(msg.content) <= MIN_TOOL_CONTENT_SIZE:
            continue

        old_len = len(msg.content)
        trim_amount = min(excess_chars, old_len - MIN_TOOL_CONTENT_SIZE)
        msg.content = msg.content[: old_len - trim_amount] + TRUNCATION_MARKER
        excess_chars -= trim_amount
        truncated_count += 1

        logger.debug(f"Trimmed tool message by {trim_amount} chars")

        if excess_chars <= 0:
            break

    return truncated_count


def resolve_with_priority[T](*sources: T | None) -> T | None:
    """
    Return first non-None value from sources.

    Useful for config resolution with priority chains like:
    request_value > profile_value > default_value

    Args:
        *sources: Values to check in priority order

    Returns:
        First non-None value, or None if all are None

    Example:
        temperature = resolve_with_priority(
            request_temp,           # Highest priority
            profile.temperature,    # Fallback
            0.7,                    # Default
        )
    """
    return next((s for s in sources if s is not None), None)

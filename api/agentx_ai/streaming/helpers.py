"""
Helper functions for streaming chat.

Extracted from views.py to reduce complexity and enable reuse.
"""

import logging
from typing import Any, Optional, TypeVar

from .constants import (
    CHAR_TO_TOKEN_RATIO,
    MIN_TOOL_CONTENT_SIZE,
    TRUNCATION_MARKER,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def estimate_tokens(messages: list[Any]) -> int:
    """
    Estimate token count from messages.

    Uses a conservative char-to-token ratio. Each message's content
    is summed and divided by the ratio.

    Args:
        messages: List of Message objects with content attribute

    Returns:
        Estimated token count
    """
    total_chars = sum(len(m.content or "") for m in messages)
    return total_chars // CHAR_TO_TOKEN_RATIO


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


def resolve_with_priority(*sources: Optional[T]) -> Optional[T]:
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

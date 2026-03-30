"""
Streaming utilities for agent chat.

This module contains helpers for SSE streaming, context management,
and token estimation extracted from views.py during complexity reduction.
"""

from .constants import (
    CHAR_TO_TOKEN_RATIO,
    CONTEXT_BUFFER_TOKENS,
    CONTEXT_WARNING_THRESHOLD,
    DEFAULT_MAX_TOOL_ROUNDS,
    MAX_INPUT_TOKENS,
    MIN_OUTPUT_TOKENS,
    MIN_TOOL_CONTENT_SIZE,
    STREAM_CLOSE_DELAY,
    TRUNCATION_MARKER,
)
from .helpers import (
    estimate_tokens,
    resolve_with_priority,
    truncate_tool_messages,
)
from .tool_loop import ToolLoopResult, streaming_tool_loop
from .trajectory_compression import compress_trajectory

__all__ = [
    # Constants
    "CHAR_TO_TOKEN_RATIO",
    "CONTEXT_BUFFER_TOKENS",
    "CONTEXT_WARNING_THRESHOLD",
    "DEFAULT_MAX_TOOL_ROUNDS",
    "MAX_INPUT_TOKENS",
    "MIN_OUTPUT_TOKENS",
    "MIN_TOOL_CONTENT_SIZE",
    "STREAM_CLOSE_DELAY",
    "TRUNCATION_MARKER",
    # Helpers
    "estimate_tokens",
    "resolve_with_priority",
    "truncate_tool_messages",
    # Tool loop
    "ToolLoopResult",
    "streaming_tool_loop",
    # Trajectory compression
    "compress_trajectory",
]

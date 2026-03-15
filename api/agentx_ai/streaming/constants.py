"""
Configuration constants for streaming chat.

These values were extracted from views.py during complexity reduction.
They can be moved to ConfigManager for hot-reloading if needed.
"""

# Token estimation
# Conservative estimate: 1 token ~ 4 characters
CHAR_TO_TOKEN_RATIO = 4

# Context management
CONTEXT_BUFFER_TOKENS = 2000  # Reserve for tool definitions & system overhead
MIN_OUTPUT_TOKENS = 2048  # Minimum output allocation to avoid truncation
MAX_INPUT_TOKENS = 32000  # Safety cap for input context
CONTEXT_WARNING_THRESHOLD = 0.8  # Warn when usage exceeds 80% of window

# Tool result truncation
MIN_TOOL_CONTENT_SIZE = 500  # Don't truncate tool results below this length
TRUNCATION_MARKER = "\n[TRUNCATED]"  # Appended to truncated tool results

# Streaming control
DEFAULT_MAX_TOOL_ROUNDS = 10  # Max tool call -> result round-trips
STREAM_CLOSE_DELAY = 0.05  # Seconds to wait before closing stream (flush buffer)

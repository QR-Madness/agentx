"""
Configuration constants for streaming chat.

These values were extracted from views.py during complexity reduction.
They can be moved to ConfigManager for hot-reloading if needed.
"""

# Token estimation
# Conservative estimate: 1 token ~ 4 characters
CHAR_TO_TOKEN_RATIO = 4

# Context management
CONTEXT_BUFFER_TOKENS = 2000  # Reserve for system overhead + token-estimate drift
MIN_OUTPUT_TOKENS = 2048  # Minimum output allocation to avoid truncation
MAX_INPUT_TOKENS = 32000  # Safety cap for input context
# Ceiling on a single response's output budget. Some providers (notably
# OpenRouter) report max_output_tokens == the full context window; honoring
# that verbatim makes us request ~the whole window as output, leaving no slack
# for input-estimate drift and triggering provider 400s ("requested N tokens").
# Capping keeps the adaptive budget sane. An explicit per-model override
# (context-limits config) still wins over this ceiling.
MAX_OUTPUT_TOKENS_CEILING = 32768
CONTEXT_WARNING_THRESHOLD = 0.8  # Warn when usage exceeds 80% of window

# Reasoning models burn output tokens on thinking before the visible answer,
# so the bare-4096 fallback starves them (observed: a 4-minute reasoning burn
# followed by a 39-token truncated answer). When a model reports
# supports_reasoning and its catalog gives no max_output_tokens, default and
# floor its output budget higher instead.
REASONING_DEFAULT_OUTPUT_TOKENS = 16384
REASONING_MIN_OUTPUT_TOKENS = 8192

# Tool result truncation
MIN_TOOL_CONTENT_SIZE = 500  # Don't truncate tool results below this length
TRUNCATION_MARKER = "\n[TRUNCATED]"  # Appended to truncated tool results

# Streaming control
DEFAULT_MAX_TOOL_ROUNDS = 10  # Max tool call -> result round-trips
STREAM_CLOSE_DELAY = 0.05  # Seconds to wait before closing stream (flush buffer)

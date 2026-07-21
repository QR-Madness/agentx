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
# Max tool call -> result round-trips. Raised 10 -> 30 (v0.21.247): a
# specialist doing real document work (read/edit/append, one call per round)
# burned 10 rounds mid-job and got force-wrapped — the informed wrap-up +
# narration cap + trajectory compression now bound runaway, so the cap can
# afford to be generous. Chat-path override: `chat.max_tool_rounds` config.
DEFAULT_MAX_TOOL_ROUNDS = 30
STREAM_CLOSE_DELAY = 0.05  # Seconds to wait before closing stream (flush buffer)

# Rate-limit wait-out: when a round's model call fails with a 429 BEFORE
# streaming anything, the tool loop waits these delays (seconds) between
# retries — visible via status events and cancellable between 1s steps,
# unlike the SDK's silent internal retries. A Retry-After header wins over
# the schedule slot. OpenRouter upstream throttling is the common case.
RATE_LIMIT_WAIT_SCHEDULE: tuple[int, ...] = (15, 30, 60, 120)

# Slow-start watchdog: while a round's FIRST chunk hasn't arrived, ping a
# status after this many seconds, then one every interval — the SDK's
# internal timeout+retry cycles are otherwise dead air (observed: 4+ silent
# minutes against a rate-limited route, indistinguishable from a hang).
SLOW_MODEL_FIRST_PING_SECONDS = 20
SLOW_MODEL_PING_INTERVAL_SECONDS = 30

# Narration-spin guard: rounds whose ONLY tool calls are update_conversation_state
# (a model narrating intentions into state instead of working — observed 9 solo
# rounds in a row burning the whole tool budget). Solo rounds past this cap get
# their state calls short-circuited with an error result. Mixed rounds (state
# write alongside real work) and multi-slot single rounds never count.
STATE_TOOL_SOLO_ROUND_CAP = 3

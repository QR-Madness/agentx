"""Single source of truth for token estimation.

Foundation #6 consolidated the handful of ad-hoc ``len(text) // 4`` estimators that
used to live in ``streaming/helpers.py``, ``agent/session.py``,
``agent/conversation_history.py`` and ``agent/context_ledger.py`` onto one
``tiktoken``-backed implementation. Every estimate flows through here so the
verbatim-budget assembler, the rolling-summary trigger, the tool-loop guard and the
Context Ledger all agree on "how big is this".

``tiktoken`` is an OpenAI tokenizer; for non-OpenAI models (Anthropic, local) it is
an approximation — but a far better one than chars/4, and consistent across the
codebase, which is what the budgeting math needs. If ``tiktoken`` is unavailable or
encoding fails, we fall back to the old chars/4 heuristic (logged once at debug).
"""

import logging

logger = logging.getLogger(__name__)

# Encoding used for estimation. ``o200k_base`` backs the GPT-4o / o-series models and
# is a reasonable generic estimator for the providers AgentX talks to.
_ENCODING_NAME = "o200k_base"

# Fallback heuristic: average chars per token when tiktoken can't be used. Also the
# basis for char-domain budget math elsewhere (e.g. ledger shrink helpers).
FALLBACK_CHARS_PER_TOKEN = 4

# Per-message overhead (role marker, formatting) added by ``estimate_messages``.
_PER_MESSAGE_OVERHEAD = 10

# Above this length, skip tiktoken and use the chars/4 heuristic. Hot callers
# (tool loop, trajectory compression) re-estimate a growing message list each
# iteration, and tool outputs can be large; exact counts don't change budgeting
# decisions at that size, so this caps worst-case tokenization cost.
_EXACT_MAX_CHARS = 20_000

_encoder = None
_encoder_failed = False


def _get_encoder():
    """Lazily build and cache the tiktoken encoder; ``None`` if unavailable."""
    global _encoder, _encoder_failed
    if _encoder is None and not _encoder_failed:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding(_ENCODING_NAME)
        except Exception as e:  # pragma: no cover - exercised via fallback test
            _encoder_failed = True
            logger.debug("tiktoken unavailable, using chars/%d heuristic: %s",
                         FALLBACK_CHARS_PER_TOKEN, e)
    return _encoder


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a raw string.

    Uses tiktoken when available (and the string is below ``_EXACT_MAX_CHARS``),
    otherwise the chars/4 heuristic. Returns 0 for empty input.
    """
    if not text:
        return 0
    if len(text) <= _EXACT_MAX_CHARS:
        enc = _get_encoder()
        if enc is not None:
            try:
                # ``disallowed_special=()`` so user text containing special-token
                # markers (e.g. "<|endoftext|>") is counted, not rejected.
                return len(enc.encode(text, disallowed_special=()))
            except Exception:  # pragma: no cover - defensive
                pass
    return len(text) // FALLBACK_CHARS_PER_TOKEN


def estimate_messages(messages) -> int:
    """Estimate the token count of a list of messages, including per-message overhead.

    Each message contributes ``estimate_tokens(content) + _PER_MESSAGE_OVERHEAD`` to
    cover the role marker and formatting.
    """
    return sum(estimate_tokens(m.content or "") + _PER_MESSAGE_OVERHEAD for m in messages)

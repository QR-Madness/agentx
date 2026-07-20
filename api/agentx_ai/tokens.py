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


# Media attachments materialize as modality tokens at provider-conversion time —
# AFTER budget fitting — so the ledger must weigh the refs, not the (absent) bytes.
# Conservative flat/derived estimates: providers charge images as ~85–1600 modality
# tokens depending on detail tiling, audio at roughly tens of tokens per second.
IMAGE_TOKEN_ESTIMATE = 1100
# Audio: derive seconds from blob size (~16KB/s for mp3-ish encodings — WAV
# overestimates, which is the safe direction) at ~32 tokens/sec ⇒ bytes/500.
_AUDIO_BYTES_PER_TOKEN = 500
AUDIO_TOKEN_MIN, AUDIO_TOKEN_MAX = 200, 8000


def estimate_media_tokens(message) -> int:
    """Conservative modality-token estimate for a message's attached media refs.

    Without this, a message carrying images/audio was budgeted as its text only
    and ballooned at the provider boundary — the ledger literally couldn't see
    media weight. Blob sizes are read best-effort (a missing doc counts the
    floor estimate; estimation must never touch the turn's failure modes).
    """
    images = getattr(message, "images", None) or []
    audio = getattr(message, "audio", None) or []
    if not images and not audio:
        return 0

    total = len(images) * IMAGE_TOKEN_ESTIMATE
    for ref in audio:
        size = _audio_blob_size(getattr(ref, "doc_id", None))
        est = size // _AUDIO_BYTES_PER_TOKEN if size else AUDIO_TOKEN_MIN
        total += max(AUDIO_TOKEN_MIN, min(est, AUDIO_TOKEN_MAX))
    return total


# doc_id → size_bytes memo. Blobs are content-addressed/immutable, and hot callers
# (tool loop, trajectory compression) re-estimate the same message list every
# round — without this each round would re-hit PG per audio ref.
_audio_size_cache: dict[str, int] = {}
_AUDIO_SIZE_CACHE_MAX = 256


def _audio_blob_size(doc_id) -> int:
    if not doc_id:
        return 0
    cached = _audio_size_cache.get(doc_id)
    if cached is not None:
        return cached
    size = 0
    try:
        from .kit.workspaces import repository

        doc = repository.get_document(doc_id)
        size = int((doc or {}).get("size_bytes") or 0)
    except Exception:  # noqa: BLE001 — estimation is best-effort
        return 0  # uncached: a transient DB hiccup shouldn't pin 0 forever
    if len(_audio_size_cache) >= _AUDIO_SIZE_CACHE_MAX:
        _audio_size_cache.clear()
    _audio_size_cache[doc_id] = size
    return size


def estimate_messages(messages) -> int:
    """Estimate the token count of a list of messages, including per-message overhead.

    Each message contributes ``estimate_tokens(content) + _PER_MESSAGE_OVERHEAD`` to
    cover the role marker and formatting, plus a conservative modality-token
    estimate for any attached media refs (see :func:`estimate_media_tokens`).
    """
    return sum(
        estimate_tokens(m.content or "") + _PER_MESSAGE_OVERHEAD + estimate_media_tokens(m)
        for m in messages
    )

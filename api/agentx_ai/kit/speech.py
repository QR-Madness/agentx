"""Neutral speech seam — TTS + STT as a capability, owned by no surface.

Chat's ``generate_speech`` tool, the chat audio-input STT fallback, and the
Ambassador's voice mode all speak through here (ADR-11: capabilities live in
neutral kit modules; surfaces consume, never own). The Ambassador keeps its
*profile-level* voice precedence in its own thin wrappers and passes the result
down as explicit args — this module resolves only explicit arg → global config
→ shipped default, and knows nothing about any surface.

Config keys remain in the historical ``ambassador.*`` namespace (they predate
the extraction; a ``speech.*`` namespace with ``ambassador.*`` fallback is the
someday-migration — tracked in todo/backlog/multimodal.md).

Resolution is **strict**: TTS/STT must never degrade to a text model. An
unconfigured/unsupported model raises :class:`SpeechUnavailable` with a stable
``code`` so callers can surface a clean, actionable message.
"""

from __future__ import annotations

import logging
import re

from ..providers.base import SpeechResult, TranscriptionResult

logger = logging.getLogger(__name__)


class SpeechUnavailable(Exception):
    """Raised when speech can't be produced/transcribed (no provider/model
    configured, or the resolved model doesn't support it). Carries a stable
    ``code`` so a view can return a structured 422 the client can act on."""

    def __init__(self, message: str, *, code: str = "voice_unconfigured") -> None:
        super().__init__(message)
        self.code = code


# Voice (TTS) defaults. The shipped speech model is OpenRouter-only, so spoken
# output needs an OpenRouter key — callers degrade gracefully when it's absent.
# MAI-Voice-2 voices use the Azure locale format.
DEFAULT_SPEECH_MODEL = "openrouter:microsoft/mai-voice-2"
DEFAULT_SPEECH_VOICE = "en-US-Harper:MAI-Voice-2"
# Speech-to-text. OpenRouter-only, like TTS.
DEFAULT_TRANSCRIPTION_MODEL = "openrouter:openai/whisper-1"
# Guard against pathological uploads (push-to-talk clips are tiny).
MAX_AUDIO_BYTES = 25 * 1024 * 1024
# Hard ceiling on synthesized text (TTS bills per character).
MAX_SPEECH_CHARS = 4000


def sanitize_speakable_text(text: str) -> str:
    """Speech hygiene shared by every TTS caller: never speak reasoning
    (strip ``<think>`` blocks — old persisted records may still carry them),
    flatten markdown glyphs that read as noise aloud, and cap length."""
    from ..streaming.thinking_exec import strip_think_blocks

    text = strip_think_blocks((text or "").strip())
    text = re.sub(r"[*#_`]+", "", text).strip()
    if len(text) > MAX_SPEECH_CHARS:
        text = text[:MAX_SPEECH_CHARS].rstrip()
    return text


async def synthesize_speech(
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    speed: float | None = None,
    usage_source: str = "tts",
    agent_id: str | None = None,
    registry=None,
) -> SpeechResult:
    """Synthesize spoken audio for ``text`` (sanitized + metered).

    Resolution precedence: explicit arg → global ``ambassador.speech_model`` /
    ``ambassador.voice`` config → shipped default. Raises
    :class:`SpeechUnavailable` on empty text, an unconfigured provider, or a
    non-TTS model. Usage is metered per input character under ``usage_source``.
    ``registry`` is injectable (the Ambassador passes its own; tests mock it).
    """
    from ..config import get_config_manager
    from ..providers.registry import get_registry

    text = sanitize_speakable_text(text)
    if not text:
        raise SpeechUnavailable("Nothing to speak.", code="empty_text")

    config = get_config_manager()
    speech_model = model or config.get("ambassador.speech_model") or DEFAULT_SPEECH_MODEL
    speech_voice = voice or config.get("ambassador.voice") or DEFAULT_SPEECH_VOICE

    try:
        provider, model_id = (registry or get_registry()).get_provider_for_model(speech_model)
    except Exception as e:  # noqa: BLE001 — unconfigured provider → clean 422
        raise SpeechUnavailable(
            f"No speech provider is configured for '{speech_model}'. "
            "Add an OpenRouter API key to enable voice.",
            code="voice_unconfigured",
        ) from e

    try:
        result = await provider.synthesize_speech(
            text,
            model=model_id,
            voice=speech_voice,
            response_format="mp3",
            speed=speed,
        )
    except NotImplementedError as e:
        raise SpeechUnavailable(
            f"'{speech_model}' does not support speech synthesis. "
            "Choose a text-to-speech model in the voice settings.",
            code="model_unsupported",
        ) from e
    except Exception as e:  # noqa: BLE001 — surface a clean failure
        logger.warning(f"Speech synthesis failed: {e}")
        raise SpeechUnavailable(str(e)[:300], code="synth_failed") from e

    try:  # TTS is billed per input character; metering never breaks voice
        from ..agent.usage_ledger import record_usage
        from ..providers.pricing import estimate_audio_cost

        record_usage(
            source=usage_source,
            model=speech_model,
            provider=getattr(provider, "name", None),
            agent_id=agent_id,
            units={"chars": len(text)},
            cost=estimate_audio_cost(model=speech_model, chars=len(text)),
        )
    except Exception as _uerr:  # noqa: BLE001
        logger.debug(f"TTS usage record skipped: {_uerr}")
    return result


async def transcribe_audio(
    audio: bytes,
    *,
    audio_format: str = "webm",
    model: str | None = None,
    language: str | None = None,
    usage_source: str = "stt",
    agent_id: str | None = None,
    registry=None,
) -> TranscriptionResult:
    """Transcribe spoken audio to text (STT), metered.

    Resolution precedence: explicit arg → global ``ambassador.transcription_model``
    config → shipped default. Raises :class:`SpeechUnavailable` on empty/oversized
    audio, an unconfigured provider, or a non-STT model. ``registry`` is
    injectable (the Ambassador passes its own; tests mock it).
    """
    from ..config import get_config_manager
    from ..providers.registry import get_registry

    if not audio:
        raise SpeechUnavailable("No audio to transcribe.", code="empty_audio")
    if len(audio) > MAX_AUDIO_BYTES:
        raise SpeechUnavailable("Audio recording is too large.", code="audio_too_large")

    config = get_config_manager()
    stt_model = model or config.get("ambassador.transcription_model") or DEFAULT_TRANSCRIPTION_MODEL

    try:
        provider, model_id = (registry or get_registry()).get_provider_for_model(stt_model)
    except Exception as e:  # noqa: BLE001 — unconfigured provider → clean 422
        raise SpeechUnavailable(
            f"No transcription provider is configured for '{stt_model}'. "
            "Add an OpenRouter API key to enable voice input.",
            code="transcription_unconfigured",
        ) from e

    try:
        result = await provider.transcribe_speech(
            audio, model=model_id, audio_format=audio_format, language=language
        )
    except NotImplementedError as e:
        raise SpeechUnavailable(
            f"'{stt_model}' does not support transcription. "
            "Choose a speech-to-text model in the voice settings.",
            code="transcription_model_unsupported",
        ) from e
    except Exception as e:  # noqa: BLE001 — surface a clean failure
        logger.warning(f"Transcription failed: {e}")
        raise SpeechUnavailable(str(e)[:300], code="transcription_failed") from e

    # STT is billed per minute of audio. The provider may or may not report
    # duration — probe the usage block under a few likely key names; when absent
    # we still record the byte size (cost stays null rather than fabricated).
    try:
        from ..agent.usage_ledger import record_usage
        from ..providers.pricing import estimate_audio_cost

        seconds = None
        usage = (result.raw_response or {}).get("usage") if result.raw_response else None
        if isinstance(usage, dict):
            for k in ("audio_seconds", "seconds", "duration", "duration_seconds"):
                v = usage.get(k)
                if isinstance(v, (int, float)):
                    seconds = float(v)
                    break
        record_usage(
            source=usage_source,
            model=stt_model,
            provider=getattr(provider, "name", None),
            agent_id=agent_id,
            units={"audio_seconds": seconds, "bytes": len(audio)},
            cost=estimate_audio_cost(model=stt_model, seconds=seconds),
        )
    except Exception as _uerr:  # noqa: BLE001 — metering never breaks voice
        logger.debug(f"STT usage record skipped: {_uerr}")
    return result

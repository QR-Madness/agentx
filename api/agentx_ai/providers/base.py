"""
Abstract base classes for model providers.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import AsyncIterator

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def log_llm_request(provider_name: str, request_params: dict[str, Any]) -> None:
    """Log an LLM request via the logging_kit cards.

    Verbosity is governed by ``AGENTX_LLM_LOG_LEVEL`` (off|summary|full); the
    legacy ``DEBUG_LOG_LLM_REQUESTS`` switch still maps to ``full``.

    The console only ever sees the compact one-line **summary** (logged at INFO,
    so it shows inline in the API console). When the level is ``full`` the entire
    redacted payload rides the ``llm_detail`` record extra — captured by the ring
    buffer (the in-app Log panel) and the on-disk archive, but **never rendered on
    the console**, so a single request can't wipe the scrollback. Best-effort —
    never let logging break a request.
    """
    try:
        from agentx_ai.logging_kit.llm_cards import render_llm_log

        rendered = render_llm_log(provider_name, request_params)
        if not rendered:
            return
        summary, detail = rendered
        if detail:
            logger.info(summary, extra={"llm_detail": detail})
        else:
            logger.info(summary)
    except Exception:  # noqa: BLE001
        pass


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A message in a conversation."""
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolCall(BaseModel):
    """A tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


def accumulate_tool_call_delta(
    pending_calls: dict[int, dict[str, Any]],
    tc_delta: dict[str, Any],
) -> None:
    """
    Accumulate a streaming tool call delta fragment.

    Tool calls arrive incrementally across multiple chunks. This helper
    accumulates the fragments into complete tool calls.

    Args:
        pending_calls: Dict mapping tool index to accumulated call data
        tc_delta: Delta fragment from a streaming chunk
    """
    idx = tc_delta.get("index", 0)
    if idx not in pending_calls:
        pending_calls[idx] = {"id": "", "name": "", "arguments": ""}

    entry = pending_calls[idx]
    if tc_delta.get("id"):
        entry["id"] = tc_delta["id"]

    func = tc_delta.get("function", {})
    if func.get("name"):
        entry["name"] = func["name"]
    if func.get("arguments"):
        entry["arguments"] += func["arguments"]


def finalize_tool_calls(pending_calls: dict[int, dict[str, Any]]) -> list[ToolCall]:
    """
    Convert accumulated tool call fragments to ToolCall objects.

    Args:
        pending_calls: Dict of accumulated tool call data

    Returns:
        List of complete ToolCall objects
    """
    completed = []
    for tc_data in pending_calls.values():
        try:
            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
        except json.JSONDecodeError:
            args = {"raw": tc_data["arguments"]}
        completed.append(ToolCall(
            id=tc_data["id"],
            name=tc_data["name"],
            arguments=args,
        ))
    return completed


def convert_messages_to_openai_format(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message objects to the OpenAI chat format.

    Shared by the OpenAI-compatible providers (OpenAI, OpenRouter, Vercel).
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        m: dict[str, Any] = {
            "role": msg.role.value,
            "content": msg.content,
        }
        if msg.name:
            m["name"] = msg.name
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            m["tool_calls"] = msg.tool_calls
        result.append(m)
    return result


def parse_openai_tool_calls(tool_calls: Any) -> list[ToolCall]:
    """Parse OpenAI SDK tool calls into internal ToolCall objects.

    Shared by the OpenAI-compatible providers (OpenAI, OpenRouter, Vercel).
    """
    result: list[ToolCall] = []
    for tc in tool_calls:
        args = tc.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        result.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=args,
        ))
    return result


class StreamChunk(BaseModel):
    """A chunk of streaming response."""
    content: str = ""
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, int] | None = None  # Token usage (available on final chunk)


class CompletionResult(BaseModel):
    """Result of a completion request."""
    content: str
    finish_reason: str
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, int] | None = None
    model: str
    raw_response: dict[str, Any] | None = None


@dataclass
class SpeechResult:
    """Result of a text-to-speech synthesis request.

    ``audio`` is the raw encoded audio (e.g. MP3 bytes), ready to write to a file
    or stream to a browser ``<audio>`` element. ``content_type`` is the response
    MIME type (``audio/mpeg`` for MP3).
    """
    audio: bytes
    content_type: str = "audio/mpeg"
    model: str = ""
    voice: str = ""
    generation_id: str | None = None


@dataclass
class ImageResult:
    """Result of an image-generation request.

    ``image`` is the raw encoded image (e.g. PNG bytes), ready to store as a blob or
    serve to an ``<img>``. ``content_type`` is the MIME type (``image/png``).
    """
    image: bytes
    content_type: str = "image/png"
    model: str = ""
    generation_id: str | None = None


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text (transcription) request."""
    text: str
    model: str = ""
    language: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class ModelCapabilities:
    """Capabilities of a model."""
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_speech: bool = False
    supports_transcription: bool = False
    context_window: int = 4096
    max_output_tokens: int | None = None
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    description: str | None = None
    pricing_currency: str = "USD"


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    api_key: str | None = None
    base_url: str | None = None
    # None means "unset" — each provider applies its own default (cloud: 60s,
    # LM Studio: 300s). An explicit value is always honored as-is.
    timeout: float | None = None
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """
    Abstract base class for model providers.

    All LLM providers (LM Studio, Anthropic, OpenAI) implement this interface
    to provide a unified way to interact with different models.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider."""
    
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion for the given messages.

        Args:
            messages: The conversation messages
            model: The model to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: How to select tools ("auto", "none", or specific tool)
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResult with the generated content
        """
        ...
    
    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion for the given messages.

        Args:
            messages: The conversation messages
            model: The model to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: How to select tools
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects as they arrive
        """
        ...
    
    @abstractmethod
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get the capabilities of a specific model.
        
        Args:
            model: The model name
            
        Returns:
            ModelCapabilities describing what the model supports
        """
    
    @abstractmethod
    def list_models(self) -> list[str]:
        """
        List all available models for this provider.
        
        Returns:
            List of model names
        """
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check if the provider is healthy and reachable.

        Returns:
            Health status dict with 'status' and optional details
        """
        return {"status": "unknown", "message": "Health check not implemented"}

    async def synthesize_speech(
        self,
        text: str,
        *,
        model: str,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float | None = None,
        **kwargs: Any,
    ) -> SpeechResult:
        """Synthesize speech (text-to-speech) for ``text``.

        Default raises — only providers exposing a TTS backend (currently
        OpenRouter, via the OpenAI-compatible ``/audio/speech`` endpoint) override
        this. Callers that need graceful degradation should resolve a
        speech-capable model and surface a clear "voice unconfigured" message.

        Args:
            text: The text to speak.
            model: The TTS model id.
            voice: Voice id (provider/model specific; omit to use the model default).
            response_format: Output container — ``mp3`` (default) or ``pcm``.
            speed: Optional playback-speed multiplier (provider-dependent).

        Returns:
            SpeechResult with the encoded audio bytes.
        """
        raise NotImplementedError(
            f"{self.name} does not support speech synthesis"
        )

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image from ``prompt``.

        Default raises — only providers exposing an image backend (currently OpenRouter,
        via the chat-completions endpoint with ``modalities: ["image","text"]``) override
        this. Callers should resolve an image-capable model (``output_modalities`` includes
        ``image``) and surface a clear "image generation unconfigured" message otherwise.

        Returns:
            ImageResult with the encoded image bytes.
        """
        raise NotImplementedError(
            f"{self.name} does not support image generation"
        )

    async def transcribe_speech(
        self,
        audio: bytes,
        *,
        model: str,
        audio_format: str = "webm",
        language: str | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe spoken audio to text (speech-to-text).

        Default raises — only providers exposing an STT backend (currently
        OpenRouter, via the OpenAI-compatible ``/audio/transcriptions`` endpoint)
        override this.

        Args:
            audio: The raw encoded audio bytes to transcribe.
            model: The STT model id.
            audio_format: Container of ``audio`` (``webm``/``mp3``/``wav``/``m4a``/``ogg``/…).
            language: Optional ISO-639-1 hint to override auto-detection.

        Returns:
            TranscriptionResult with the transcript text.
        """
        raise NotImplementedError(
            f"{self.name} does not support speech transcription"
        )

    async def close(self) -> None:
        """
        Release any long-lived resources (HTTP/SDK clients).

        Default is a no-op: providers that create a fresh client per request
        close it themselves. Providers holding a cached client (OpenAI,
        Anthropic) override this to close and reset it.
        """
        return None

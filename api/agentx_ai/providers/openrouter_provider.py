"""
OpenRouter model provider implementation.

OpenRouter provides a unified API to 100+ LLM models through an
OpenAI-compatible interface with dynamic model catalog.
"""

import base64
import logging
import time
from typing import Any
from collections.abc import AsyncIterator

import httpx

from .base import (
    CompletionResult,
    ImageResult,
    Message,
    ModelCapabilities,
    ModelProvider,
    ProviderConfig,
    SpeechResult,
    StreamChunk,
    TranscriptionResult,
    accumulate_tool_call_delta,
    convert_messages_to_openai_format,
    finalize_tool_calls,
    log_llm_request,
    normalize_openai_usage,
    parse_openai_tool_calls,
    process_reasoning_delta,
)

logger = logging.getLogger(__name__)


# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Cache duration for model catalog (5 minutes)
MODEL_CACHE_TTL = 300

# Default capabilities for models not in cache
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_streaming=True,
    context_window=8192,
)


class OpenRouterProvider(ModelProvider):
    """
    OpenRouter provider using OpenAI-compatible API.

    OpenRouter aggregates 100+ models from multiple providers
    (Anthropic, OpenAI, Meta, Mistral, Google, etc.) through a
    unified API with dynamic model discovery.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Any | None = None
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamp: float = 0

        # Attribution headers for OpenRouter
        self._site_url = config.extra.get("site_url", "") if config.extra else ""
        self._app_name = config.extra.get("app_name", "AgentX") if config.extra else "AgentX"

    @property
    def name(self) -> str:
        return "openrouter"

    def _get_client(self) -> Any:
        """Create async OpenAI client configured for OpenRouter.

        Creates a fresh client per request to avoid event loop issues.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            ) from None

        headers = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name

        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or OPENROUTER_BASE_URL,
            timeout=self.config.timeout or 60.0,
            max_retries=self.config.max_retries,
            default_headers=headers if headers else None,
        )

    async def _fetch_models_if_stale(self) -> None:
        """Fetch model catalog from OpenRouter if cache is stale."""
        now = time.time()
        if self._model_cache and (now - self._cache_timestamp) < MODEL_CACHE_TTL:
            return

        try:
            async with httpx.AsyncClient(timeout=30) as http_client:
                response = await http_client.get(
                    f"{self.config.base_url or OPENROUTER_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )
                response.raise_for_status()
                data = response.json()

                # Build cache indexed by model ID
                self._model_cache = {
                    model["id"]: model
                    for model in data.get("data", [])
                }
                self._cache_timestamp = now
                logger.info(f"Fetched {len(self._model_cache)} models from OpenRouter")
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            # Keep stale cache if available

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
        """Generate a completion using OpenRouter API."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens
        if tools:
            request_params["tools"] = tools
        if tool_choice:
            request_params["tool_choice"] = tool_choice
        if stop:
            request_params["stop"] = stop

        request_params.update(kwargs)

        logger.debug(f"OpenRouter request: model={model}, messages={len(messages)}")
        log_llm_request("OpenRouter", request_params)

        client = self._get_client()
        try:
            response = await client.chat.completions.create(**request_params)
        finally:
            await client.close()

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = parse_openai_tool_calls(choice.message.tool_calls)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return CompletionResult(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            raw_response=response.model_dump(),
        )

    async def stream(
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
        """Stream a completion using OpenRouter API."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens
        if tools:
            request_params["tools"] = tools
        if tool_choice:
            request_params["tool_choice"] = tool_choice
        if stop:
            request_params["stop"] = stop

        request_params.update(kwargs)

        logger.debug(f"OpenRouter stream: model={model}, messages={len(messages)}")
        log_llm_request("OpenRouter (stream)", request_params)

        client = self._get_client()
        try:
            stream = await client.chat.completions.create(
                **request_params,
                # OpenRouter usage accounting: the stream's final chunk then
                # carries authoritative token counts (reasoning tokens
                # INCLUDED — hidden thinking is billed as output but invisible
                # to any text-side estimate; a gpt-5.6-sol-pro turn metered
                # 10x low without this) plus the actually-billed cost.
                extra_body={"usage": {"include": True}},
            )
            pending_tool_calls: dict[int, dict[str, Any]] = {}
            in_reasoning = False
            usage_payload: dict[str, Any] | None = None

            async for chunk in stream:
                # Usage rides a trailing chunk with EMPTY `choices` — read it
                # before the choices guard skips that chunk entirely.
                if getattr(chunk, "usage", None) is not None:
                    usage_payload = normalize_openai_usage(chunk.usage)
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                # Accumulate tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_dict = {
                            "index": tc_delta.index,
                            "id": tc_delta.id,
                            "function": {
                                "name": tc_delta.function.name if tc_delta.function else None,
                                "arguments": tc_delta.function.arguments if tc_delta.function else None,
                            } if tc_delta.function else {},
                        }
                        accumulate_tool_call_delta(pending_tool_calls, tc_dict)

                # Reasoning models stream thinking in a non-standard delta field
                # (`reasoning` on OpenRouter, `reasoning_content` on OpenAI-style
                # upstreams). Surface it as <think> content so it's visible
                # instead of silently burning the output-token budget.
                reasoning = (
                    getattr(delta, "reasoning", None)
                    or getattr(delta, "reasoning_content", None)
                    or ""
                )
                content, in_reasoning = process_reasoning_delta(
                    reasoning, delta.content or "", in_reasoning
                )
                if content:
                    yield StreamChunk(content=content, finish_reason=finish_reason)

                # Handle stream end
                if finish_reason:
                    if in_reasoning:
                        # Close a reasoning block left open at stream end
                        yield StreamChunk(content="</think>", finish_reason=None)
                        in_reasoning = False
                    if finish_reason == "tool_calls" and pending_tool_calls:
                        yield StreamChunk(
                            content="",
                            finish_reason="tool_calls",
                            tool_calls=finalize_tool_calls(pending_tool_calls),
                        )
                        pending_tool_calls.clear()
                    elif finish_reason != "tool_calls":
                        yield StreamChunk(content="", finish_reason=finish_reason)

            if usage_payload is not None:
                yield StreamChunk(content="", usage=usage_payload)
        finally:
            await client.close()

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
        """Synthesize speech via OpenRouter's OpenAI-compatible ``/audio/speech``.

        Returns the raw audio bytes (MP3 by default) — OpenRouter streams the
        audio as the response body, not JSON. Raises on a non-2xx with the
        response body so the caller can surface a clear error.
        """
        body: dict[str, Any] = {
            "model": model,
            "input": text,
            "response_format": response_format,
        }
        # `voice` is required by some models and rejected (defaulted) by others —
        # send it only when explicitly set.
        if voice:
            body["voice"] = voice
        if speed is not None:
            body["speed"] = speed
        # Allow provider-specific pass-through (e.g. MAI-Voice-2 expressive style).
        provider_options = kwargs.get("provider")
        if provider_options:
            body["provider"] = provider_options

        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name

        base = self.config.base_url or OPENROUTER_BASE_URL
        logger.debug(f"OpenRouter TTS: model={model}, voice={voice}, chars={len(text)}")

        async with httpx.AsyncClient(timeout=self.config.timeout or 60.0) as http_client:
            response = await http_client.post(
                f"{base}/audio/speech", headers=headers, json=body
            )
            if response.status_code >= 400:
                detail = response.text[:500]
                raise RuntimeError(
                    f"OpenRouter speech synthesis failed ({response.status_code}): {detail}"
                )
            audio = response.content
            content_type = response.headers.get("content-type", "audio/mpeg")
            generation_id = response.headers.get("x-generation-id")

        return SpeechResult(
            audio=audio,
            content_type=content_type,
            model=model,
            voice=voice or "",
            generation_id=generation_id,
        )

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str,
        **kwargs: Any,
    ) -> ImageResult:
        """Generate an image via OpenRouter's chat-completions endpoint with
        ``modalities: ["image","text"]``. The image comes back as a base64 data URL on
        ``choices[0].message.images[0].image_url.url`` (per OpenRouter's image-gen API),
        which we decode to raw bytes. Raises on a non-2xx or a response with no image."""
        # Request image output only. Image-only models (e.g. flux) reject ["image","text"]
        # ("no endpoints support the requested output modalities"), and we never use the
        # text part anyway — ["image"] is accepted by image-capable models either way.
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image"],
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name

        base = self.config.base_url or OPENROUTER_BASE_URL
        logger.debug(f"OpenRouter image gen: model={model}, prompt_chars={len(prompt)}")

        async with httpx.AsyncClient(timeout=self.config.timeout or 120.0) as http_client:
            response = await http_client.post(
                f"{base}/chat/completions", headers=headers, json=body
            )
            if response.status_code >= 400:
                detail = response.text[:500]
                raise RuntimeError(
                    f"OpenRouter image generation failed ({response.status_code}): {detail}"
                )
            data = response.json()

        try:
            message = data["choices"][0]["message"]
            images = message.get("images") or []
            data_url = images[0]["image_url"]["url"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"OpenRouter returned no image (model '{model}' may not support image output)"
            ) from e

        # data_url is "data:{content_type};base64,{b64}"
        if not data_url.startswith("data:") or "," not in data_url:
            raise RuntimeError("OpenRouter image url was not a base64 data URL")
        header, b64 = data_url.split(",", 1)
        content_type = header[len("data:"):].split(";", 1)[0] or "image/png"
        image_bytes = base64.b64decode(b64)

        return ImageResult(
            image=image_bytes,
            content_type=content_type,
            model=model,
            generation_id=data.get("id"),
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
        """Transcribe audio via OpenRouter's OpenAI-compatible ``/audio/transcriptions``.

        OpenRouter takes the audio as **base64** in a JSON body (not multipart) and
        returns ``{text, usage}``. Raises on a non-2xx with the response body.
        """
        body: dict[str, Any] = {
            "model": model,
            "input_audio": {
                "data": base64.b64encode(audio).decode("ascii"),
                "format": audio_format,
            },
        }
        if language:
            body["language"] = language

        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name

        base = self.config.base_url or OPENROUTER_BASE_URL
        logger.debug(
            f"OpenRouter STT: model={model}, format={audio_format}, bytes={len(audio)}"
        )

        async with httpx.AsyncClient(timeout=self.config.timeout or 60.0) as http_client:
            response = await http_client.post(
                f"{base}/audio/transcriptions", headers=headers, json=body
            )
            if response.status_code >= 400:
                detail = response.text[:500]
                raise RuntimeError(
                    f"OpenRouter transcription failed ({response.status_code}): {detail}"
                )
            data = response.json()

        return TranscriptionResult(
            text=(data.get("text") or "").strip(),
            model=model,
            language=language,
            raw_response=data,
        )

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for an OpenRouter model from cached catalog."""
        if model in self._model_cache:
            info = self._model_cache[model]
            supported_params = info.get("supported_parameters", []) or []

            context_window = info.get("context_length", 8192)

            max_output = None
            top_provider = info.get("top_provider", {}) or {}
            if top_provider:
                max_output = top_provider.get("max_completion_tokens")

            pricing = info.get("pricing", {}) or {}
            cost_input = float(pricing["prompt"]) * 1000 if pricing.get("prompt") else None
            cost_output = float(pricing["completion"]) * 1000 if pricing.get("completion") else None

            architecture = info.get("architecture", {}) or {}
            input_modalities = list(architecture.get("input_modalities") or ["text"])
            output_modalities = list(architecture.get("output_modalities") or ["text"])

            supports_tools = (
                "tools" in supported_params
                or "function_calling" in supported_params
                or "tool_choice" in supported_params
            )
            supports_json_mode = (
                "response_format" in supported_params
                or "json_mode" in supported_params
                or "json_object" in supported_params
            )
            supports_reasoning = (
                "reasoning" in supported_params
                or "include_reasoning" in supported_params
            )

            return ModelCapabilities(
                supports_tools=supports_tools,
                supports_vision="image" in input_modalities or "vision" in supported_params,
                supports_streaming=True,
                supports_json_mode=supports_json_mode,
                supports_reasoning=supports_reasoning,
                supports_speech="audio" in output_modalities or "speech" in output_modalities,
                supports_transcription="transcription" in output_modalities,
                context_window=context_window,
                max_output_tokens=max_output,
                cost_per_1k_input=cost_input,
                cost_per_1k_output=cost_output,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                description=info.get("description"),
                pricing_currency="USD",
            )

        logger.warning(f"Unknown OpenRouter model: {model}, using default capabilities")
        return DEFAULT_CAPABILITIES

    def list_models(self) -> list[str]:
        """List available models from cache."""
        return list(self._model_cache.keys())

    async def fetch_models(self) -> list[dict[str, Any]]:
        """Fetch and return full model catalog."""
        await self._fetch_models_if_stale()
        return list(self._model_cache.values())

    async def health_check(self) -> dict[str, Any]:
        """Check if OpenRouter API is reachable."""
        if not self.config.api_key:
            return {
                "status": "not_configured",
                "error": "OPENROUTER_API_KEY not set",
            }

        try:
            await self._fetch_models_if_stale()
            return {
                "status": "healthy",
                "models_available": len(self._model_cache),
                "models": list(self._model_cache.keys())[:20],  # First 20
            }
        except Exception as e:
            logger.error(f"OpenRouter health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

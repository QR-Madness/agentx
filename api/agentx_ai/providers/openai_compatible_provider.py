"""Generic OpenAI-compatible provider — the backend behind custom endpoints.

Most of the ecosystem speaks the OpenAI chat-completions dialect: Groq, Together,
DeepSeek, Fireworks, xAI, Mistral, Cerebras, Ollama, vLLM, llama.cpp, and any
private gateway in front of them. Rather than shipping a class per vendor, a
custom catalog entry (``providers/catalog.py``) is instantiated here with its own
``base_url``, key, and headers.

Deliberately *not* a refactor of the existing providers. ``LMStudioProvider``
keeps its 300s timeout, raw-httpx streaming and local defaults; ``OpenAIProvider``
keeps its cached client. Both are load-bearing and stay untouched — this class
only serves entries the user registered.

Capabilities are the honest problem: a bare ``/v1/models`` listing says nothing
about context windows or tool support, and guessing high causes the exact failure
mode this overhaul exists to kill (a silent fallback to a model that can't hold
the context). So the defaults here are **conservative**, and the per-model
override in ``context_limits.models`` is the documented way to raise them.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from collections.abc import AsyncIterator

from .base import (
    CompletionResult,
    Message,
    ModelCapabilities,
    ModelProvider,
    ProviderConfig,
    StreamChunk,
    accumulate_tool_call_delta,
    convert_messages_to_openai_format,
    finalize_tool_calls,
    log_llm_request,
    normalize_openai_usage,
    parse_openai_tool_calls,
    process_reasoning_delta,
)

logger = logging.getLogger(__name__)

#: Model-list cache lifetime, matching the other cloud providers.
MODEL_CACHE_TTL = 300

#: Conservative until proven otherwise. An unknown endpoint gets a context window
#: we're confident almost every current model meets; raising it is a per-model
#: override, never a guess. Tools are assumed on because the OpenAI dialect
#: carries them and a server that lacks them errors clearly.
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_vision=False,
    supports_streaming=True,
    supports_json_mode=True,
    context_window=32768,
    max_output_tokens=4096,
)


class OpenAICompatibleProvider(ModelProvider):
    """A user-registered endpoint speaking the OpenAI chat-completions dialect."""

    DEFAULT_TIMEOUT = 120.0

    def __init__(self, config: ProviderConfig, *, name: str = "custom"):
        super().__init__(config)
        self._name = name
        self._model_cache: list[str] = []
        self._cache_timestamp: float = 0.0
        # Extra headers from the catalog entry — private gateways often want an
        # org id or a second auth header alongside the bearer token.
        extra = config.extra or {}
        headers = extra.get("headers")
        self._headers: dict[str, str] = dict(headers) if isinstance(headers, dict) else {}

    @property
    def name(self) -> str:
        return self._name

    def _get_client(self) -> Any:
        """A fresh AsyncOpenAI per request.

        Per-request construction (as LM Studio does) keeps the httpx pool bound to
        whichever event loop Django's async view is running on — a cached client
        raises "Connection error" the moment those differ.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            ) from None

        return AsyncOpenAI(
            # Endpoints that don't authenticate (a local vLLM) still need a
            # non-empty value or the SDK refuses to construct.
            api_key=self.config.api_key or "not-required",
            base_url=self.config.base_url,
            timeout=self.config.timeout or self.DEFAULT_TIMEOUT,
            max_retries=self.config.max_retries,
            default_headers=self._headers or None,
        )

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
        """Generate a completion against the configured endpoint."""
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

        logger.debug(f"{self._name} request: model={model}, messages={len(messages)}")
        log_llm_request(self._name, request_params)

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
            model=response.model or model,
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
        """Stream a completion against the configured endpoint."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
            "stream": True,
            # Authoritative token counts on a trailing chunk. Servers honor this
            # inconsistently; a missing usage chunk falls back to the text-side
            # estimate rather than failing.
            "stream_options": {"include_usage": True},
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

        logger.debug(f"{self._name} stream: model={model}, messages={len(messages)}")
        log_llm_request(f"{self._name} (stream)", request_params)

        client = self._get_client()
        pending_tool_calls: dict[int, dict[str, Any]] = {}
        in_reasoning = False
        usage_payload: dict[str, Any] | None = None

        try:
            stream = await client.chat.completions.create(**request_params)
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

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        tc_dict = {
                            "index": tc_delta.index,
                            "id": tc_delta.id,
                            "function": {
                                "name": tc_delta.function.name if tc_delta.function else None,
                                "arguments": (
                                    tc_delta.function.arguments if tc_delta.function else None
                                ),
                            } if tc_delta.function else {},
                        }
                        accumulate_tool_call_delta(pending_tool_calls, tc_dict)

                # DeepSeek-style servers stream thinking in `reasoning_content`;
                # surface it as <think> like the other OpenAI-dialect providers.
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content, in_reasoning = process_reasoning_delta(
                    reasoning, delta.content or "", in_reasoning
                )
                if content:
                    yield StreamChunk(content=content, finish_reason=finish_reason)

                if finish_reason:
                    if in_reasoning:
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

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Conservative defaults — see the module docstring on why we don't guess."""
        return DEFAULT_CAPABILITIES

    def list_models(self) -> list[str]:
        """Models from the last successful fetch (empty before the first)."""
        return self._model_cache.copy()

    async def fetch_models(self) -> list[str]:
        """Refresh the model list from ``/models``, keeping a stale cache on error."""
        now = time.time()
        if self._model_cache and (now - self._cache_timestamp) < MODEL_CACHE_TTL:
            return self._model_cache

        client = self._get_client()
        try:
            listing = await client.models.list()
            self._model_cache = sorted(m.id for m in listing.data)
            self._cache_timestamp = now
            logger.info(f"Fetched {len(self._model_cache)} models from '{self._name}'")
        except Exception as e:  # noqa: BLE001 — a listing failure must not break a turn
            logger.warning(f"Failed to fetch models from '{self._name}': {e}")
        finally:
            await client.close()

        return self._model_cache

    async def health_check(self) -> dict[str, Any]:
        """Reachability probe — the same ``/models`` call the Test button makes."""
        if not self.config.base_url:
            return {"status": "not_configured", "error": "No base URL set"}

        client = self._get_client()
        try:
            listing = await client.models.list()
            models = sorted(m.id for m in listing.data)
            self._model_cache = models
            self._cache_timestamp = time.time()
            return {
                "status": "healthy",
                "base_url": self.config.base_url,
                "models_available": len(models),
                "models": models[:20],
            }
        except Exception as e:  # noqa: BLE001 — health never raises, it reports
            logger.warning(f"Health check failed for '{self._name}': {e}")
            return {
                "status": "unhealthy",
                "base_url": self.config.base_url,
                "error": str(e),
            }
        finally:
            await client.close()

    async def close(self) -> None:
        """No-op — clients are created and closed per request."""
        return None

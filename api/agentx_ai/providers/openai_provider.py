"""
OpenAI model provider implementation.

Model capabilities are fetched dynamically from the OpenAI API
and cached for efficiency.
"""

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

# Cache TTL for model list (5 minutes)
MODEL_CACHE_TTL = 300

# Default capabilities for models (OpenAI doesn't expose full capabilities via API)
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_vision=True,
    supports_streaming=True,
    supports_json_mode=True,
    context_window=128000,
    max_output_tokens=4096,
)


class OpenAIProvider(ModelProvider):
    """OpenAI API provider for GPT models."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Any | None = None
        self._model_cache: list[str] = []
        self._cache_timestamp: float = 0
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def client(self) -> Any:
        """Lazy-load the async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. "
                    "Install with: pip install openai"
                ) from None

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout or 60.0,
                max_retries=self.config.max_retries,
            )
        return self._client
    
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
        """Generate a completion using OpenAI API."""
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

        # Add any extra kwargs
        request_params.update(kwargs)

        logger.debug(f"OpenAI request: model={model}, messages={len(messages)}")
        log_llm_request("OpenAI", request_params)

        response = await self.client.chat.completions.create(**request_params)

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
        """Stream a completion using OpenAI API."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
            "stream": True,
            # Ask for authoritative token counts on a trailing chunk. Without
            # this, streamed turns fall back to a visible-text estimate — which
            # silently drops hidden reasoning tokens (o-series/gpt-5 bill them
            # at the output rate but never emit them as text).
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

        logger.debug(f"OpenAI stream: model={model}, messages={len(messages)}")
        log_llm_request("OpenAI (stream)", request_params)

        stream = await self.client.chat.completions.create(**request_params)
        pending_tool_calls: dict[int, dict[str, Any]] = {}
        in_reasoning = False
        usage_payload: dict[str, Any] | None = None

        async for chunk in stream:
            # Usage rides a trailing chunk with EMPTY `choices` — read it before
            # the choices guard skips that chunk entirely.
            if getattr(chunk, "usage", None) is not None:
                usage_payload = normalize_openai_usage(chunk.usage)
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Accumulate tool call deltas (using shared helper)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # Convert SDK object to dict for shared helper
                    tc_dict = {
                        "index": tc_delta.index,
                        "id": tc_delta.id,
                        "function": {
                            "name": tc_delta.function.name if tc_delta.function else None,
                            "arguments": tc_delta.function.arguments if tc_delta.function else None,
                        } if tc_delta.function else {},
                    }
                    accumulate_tool_call_delta(pending_tool_calls, tc_dict)

            # OpenAI-compatible servers configured via base_url (e.g. DeepSeek)
            # stream thinking in `reasoning_content`; surface it as <think>.
            reasoning = getattr(delta, "reasoning_content", None) or ""
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

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for an OpenAI model.

        Note: OpenAI API doesn't expose detailed capabilities per model,
        so we return sensible defaults for all models.
        """
        return DEFAULT_CAPABILITIES

    def list_models(self) -> list[str]:
        """List available OpenAI models from cache."""
        return self._model_cache.copy()

    async def fetch_models(self) -> list[str]:
        """Fetch available models from OpenAI API and update cache."""
        now = time.time()
        if self._model_cache and (now - self._cache_timestamp) < MODEL_CACHE_TTL:
            return self._model_cache

        try:
            models = await self.client.models.list()
            # Filter to chat models only (gpt-*, o1-*)
            chat_models = [
                m.id async for m in models
                if m.id.startswith(("gpt-", "o1-", "o3-"))
            ]
            self._model_cache = sorted(chat_models)
            self._cache_timestamp = now
            logger.info(f"Fetched {len(self._model_cache)} OpenAI models")
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            # Keep stale cache if available

        return self._model_cache
    
    async def health_check(self) -> dict[str, Any]:
        """Check if OpenAI API is reachable."""
        if not self.config.api_key:
            return {
                "status": "not_configured",
                "error": "OPENAI_API_KEY not set",
            }

        try:
            models = await self.fetch_models()
            return {
                "status": "healthy",
                "models_available": len(models),
                "models": models[:20],  # First 20
            }
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the cached AsyncOpenAI client and reset it for re-creation."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
            finally:
                self._client = None

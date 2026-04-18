"""
Vercel AI Gateway model provider implementation.

Vercel AI Gateway provides a unified API to 100+ LLM models through an
OpenAI-compatible interface with automatic fallbacks and high availability.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Optional

import httpx

from .base import (
    CompletionResult,
    Message,
    ModelCapabilities,
    ModelProvider,
    ProviderConfig,
    StreamChunk,
    ToolCall,
    accumulate_tool_call_delta,
    finalize_tool_calls,
    log_llm_request,
)

logger = logging.getLogger(__name__)

# Vercel AI Gateway base URL
VERCEL_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"

# Cache duration for model catalog (5 minutes)
MODEL_CACHE_TTL = 300

# Default capabilities for models not in cache
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_streaming=True,
    context_window=8192,
)


class VercelProvider(ModelProvider):
    """
    Vercel AI Gateway provider using OpenAI-compatible API.

    Vercel AI Gateway aggregates 100+ models from multiple providers
    (Anthropic, OpenAI, Google, xAI, Meta, etc.) through a unified API
    with automatic fallbacks and zero token markup.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Optional[Any] = None
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamp: float = 0

    @property
    def name(self) -> str:
        return "vercel"

    def _get_client(self) -> Any:
        """Create async OpenAI client configured for Vercel AI Gateway.

        Creates a fresh client per request to avoid event loop issues.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )

        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or VERCEL_GATEWAY_BASE_URL,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal Message objects to OpenAI format."""
        result = []
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

    def _parse_tool_calls(self, tool_calls: Any) -> list[ToolCall]:
        """Parse OpenAI tool calls into internal format."""
        result = []
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

    async def _fetch_models_if_stale(self) -> None:
        """Fetch model catalog from Vercel AI Gateway if cache is stale."""
        now = time.time()
        if self._model_cache and (now - self._cache_timestamp) < MODEL_CACHE_TTL:
            return

        try:
            # Vercel's /v1/models endpoint doesn't require auth
            async with httpx.AsyncClient(timeout=30) as http_client:
                response = await http_client.get(
                    f"{self.config.base_url or VERCEL_GATEWAY_BASE_URL}/models",
                )
                response.raise_for_status()
                data = response.json()

                # Build cache indexed by model ID
                self._model_cache = {
                    model["id"]: model
                    for model in data.get("data", [])
                }
                self._cache_timestamp = now
                logger.info(f"Fetched {len(self._model_cache)} models from Vercel AI Gateway")
        except Exception as e:
            logger.error(f"Failed to fetch Vercel AI Gateway models: {e}")
            # Keep stale cache if available

    async def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion using Vercel AI Gateway API."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
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

        logger.debug(f"Vercel AI Gateway request: model={model}, messages={len(messages)}")
        log_llm_request("Vercel", request_params)

        client = self._get_client()
        try:
            response = await client.chat.completions.create(**request_params)
        finally:
            await client.close()

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = self._parse_tool_calls(choice.message.tool_calls)

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
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion using Vercel AI Gateway API."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
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

        logger.debug(f"Vercel AI Gateway stream: model={model}, messages={len(messages)}")
        log_llm_request("Vercel (stream)", request_params)

        client = self._get_client()
        try:
            stream = await client.chat.completions.create(**request_params)
            pending_tool_calls: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
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

                content = delta.content or ""
                if content:
                    yield StreamChunk(content=content, finish_reason=finish_reason)

                # Handle stream end
                if finish_reason == "tool_calls" and pending_tool_calls:
                    yield StreamChunk(
                        content="",
                        finish_reason="tool_calls",
                        tool_calls=finalize_tool_calls(pending_tool_calls),
                    )
                    pending_tool_calls.clear()
                elif finish_reason and finish_reason != "tool_calls":
                    yield StreamChunk(content="", finish_reason=finish_reason)
        finally:
            await client.close()

    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a Vercel AI Gateway model from cached catalog."""
        if model in self._model_cache:
            info = self._model_cache[model]
            tags = info.get("tags", [])

            # Parse context window
            context_window = info.get("context_window", 8192)

            # Parse max output tokens
            max_output = info.get("max_tokens")

            # Parse pricing (Vercel uses per-token pricing, convert to per-1k)
            pricing = info.get("pricing", {})
            cost_input = None
            cost_output = None
            if pricing:
                input_cost = pricing.get("input")
                output_cost = pricing.get("output")
                if input_cost:
                    cost_input = float(input_cost) * 1000
                if output_cost:
                    cost_output = float(output_cost) * 1000

            # Determine capabilities from tags
            return ModelCapabilities(
                supports_tools="tool-use" in tags,
                supports_vision="vision" in tags,
                supports_streaming=info.get("type") == "language",
                supports_json_mode=True,  # Most language models support JSON mode
                context_window=context_window,
                max_output_tokens=max_output,
                cost_per_1k_input=cost_input,
                cost_per_1k_output=cost_output,
            )

        # Default capabilities for unknown models
        logger.warning(f"Unknown Vercel AI Gateway model: {model}, using default capabilities")
        return DEFAULT_CAPABILITIES

    def list_models(self) -> list[str]:
        """List available models from cache."""
        return list(self._model_cache.keys())

    async def fetch_models(self) -> list[dict[str, Any]]:
        """Fetch and return full model catalog."""
        await self._fetch_models_if_stale()
        return list(self._model_cache.values())

    async def health_check(self) -> dict[str, Any]:
        """Check if Vercel AI Gateway API is reachable."""
        if not self.config.api_key:
            return {
                "status": "not_configured",
                "error": "AI_GATEWAY_API_KEY not set",
            }

        try:
            await self._fetch_models_if_stale()
            return {
                "status": "healthy",
                "models_available": len(self._model_cache),
                "models": list(self._model_cache.keys())[:20],  # First 20
            }
        except Exception as e:
            logger.error(f"Vercel AI Gateway health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

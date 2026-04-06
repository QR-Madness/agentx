"""
OpenRouter model provider implementation.

OpenRouter provides a unified API to 100+ LLM models through an
OpenAI-compatible interface with dynamic model catalog.
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
        self._client: Optional[Any] = None
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
            )

        headers = {}
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name

        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or OPENROUTER_BASE_URL,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            default_headers=headers if headers else None,
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
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion using OpenRouter API."""
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
        """Stream a completion using OpenRouter API."""
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

        logger.debug(f"OpenRouter stream: model={model}, messages={len(messages)}")
        log_llm_request("OpenRouter (stream)", request_params)

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
        """Get capabilities for an OpenRouter model from cached catalog."""
        if model in self._model_cache:
            info = self._model_cache[model]
            supported_params = info.get("supported_parameters", [])

            # Parse context window
            context_window = info.get("context_length", 8192)

            # Parse max output tokens from top_provider
            max_output = None
            top_provider = info.get("top_provider", {})
            if top_provider:
                max_output = top_provider.get("max_completion_tokens")

            # Parse pricing (OpenRouter uses per-token pricing, convert to per-1k)
            pricing = info.get("pricing", {})
            cost_input = None
            cost_output = None
            if pricing:
                prompt_cost = pricing.get("prompt")
                completion_cost = pricing.get("completion")
                if prompt_cost:
                    cost_input = float(prompt_cost) * 1000
                if completion_cost:
                    cost_output = float(completion_cost) * 1000

            return ModelCapabilities(
                supports_tools="function_calling" in supported_params or "tools" in supported_params,
                supports_vision="vision" in supported_params,
                supports_streaming=True,
                supports_json_mode="json_mode" in supported_params or "json_object" in supported_params,
                context_window=context_window,
                max_output_tokens=max_output,
                cost_per_1k_input=cost_input,
                cost_per_1k_output=cost_output,
            )

        # Default capabilities for unknown models
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

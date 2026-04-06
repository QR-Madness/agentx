"""
OpenAI model provider implementation.

Model capabilities are fetched dynamically from the OpenAI API
and cached for efficiency.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Optional

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
        self._client: Optional[Any] = None
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
                )

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client
    
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
        """Generate a completion using OpenAI API."""
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

        # Add any extra kwargs
        request_params.update(kwargs)

        logger.debug(f"OpenAI request: model={model}, messages={len(messages)}")
        log_llm_request("OpenAI", request_params)

        response = await self.client.chat.completions.create(**request_params)

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
        """Stream a completion using OpenAI API."""
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

        logger.debug(f"OpenAI stream: model={model}, messages={len(messages)}")
        log_llm_request("OpenAI (stream)", request_params)

        stream = await self.client.chat.completions.create(**request_params)
        pending_tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
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

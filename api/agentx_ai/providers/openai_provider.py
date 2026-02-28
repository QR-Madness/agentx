"""
OpenAI model provider implementation.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional

from .base import (
    CompletionResult,
    Message,
    ModelCapabilities,
    ModelProvider,
    ProviderConfig,
    StreamChunk,
    ToolCall,
)

logger = logging.getLogger(__name__)

# Model capabilities registry
OPENAI_MODELS = {
    "gpt-4-turbo": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
    "gpt-4-turbo-preview": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
    "gpt-4": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=8192,
        max_output_tokens=4096,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
    ),
    "gpt-4o": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gpt-4o-mini": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "gpt-3.5-turbo": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=16385,
        max_output_tokens=4096,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
    ),
}


class OpenAIProvider(ModelProvider):
    """OpenAI API provider for GPT models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Optional[Any] = None
    
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

        stream = await self.client.chat.completions.create(**request_params)

        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            content = delta.content or ""
            finish_reason = choice.finish_reason

            yield StreamChunk(
                content=content,
                finish_reason=finish_reason,
            )
    
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for an OpenAI model."""
        if model in OPENAI_MODELS:
            return OPENAI_MODELS[model]
        
        # Default capabilities for unknown models
        logger.warning(f"Unknown OpenAI model: {model}, using default capabilities")
        return ModelCapabilities(
            supports_tools=True,
            supports_streaming=True,
            context_window=8192,
        )
    
    def list_models(self) -> list[str]:
        """List available OpenAI models."""
        return list(OPENAI_MODELS.keys())
    
    async def health_check(self) -> dict[str, Any]:
        """Check if OpenAI API is reachable."""
        if not self.config.api_key:
            return {
                "status": "not_configured",
                "error": "OPENAI_API_KEY not set",
            }

        try:
            # Make a minimal API call to check connectivity
            models = await self.client.models.list()
            model_list = [m async for m in models]
            return {
                "status": "healthy",
                "models_available": len(model_list),
                "models": [m.id for m in model_list[:10]],  # First 10
            }
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

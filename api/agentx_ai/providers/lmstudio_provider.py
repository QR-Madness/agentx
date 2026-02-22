"""
LM Studio provider implementation.

LM Studio provides an OpenAI-compatible API, so this is essentially
a wrapper around the OpenAI provider with different defaults.
"""

import logging
from typing import Any, Iterator, Optional

from openai import OpenAI

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

# Default capabilities for local models
DEFAULT_LOCAL_CAPABILITIES = ModelCapabilities(
    supports_tools=False,  # Most local models don't support function calling
    supports_vision=False,
    supports_streaming=True,
    supports_json_mode=True,
    context_window=8192,
    max_output_tokens=4096,
    cost_per_1k_input=0.0,  # Local = free
    cost_per_1k_output=0.0,
)


class LMStudioProvider(ModelProvider):
    """
    LM Studio provider using OpenAI-compatible API.
    
    LM Studio runs locally and exposes an OpenAI-compatible API,
    making it easy to use local models with the same interface.
    """
    
    DEFAULT_BASE_URL = "http://localhost:1234/v1"
    DEFAULT_TIMEOUT = 300.0  # 5 minutes for large models
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self._timeout = config.timeout if config.timeout != 60.0 else self.DEFAULT_TIMEOUT
        self._available_models: Optional[list[dict[str, Any]]] = None
        
        # LM Studio doesn't need an API key, but OpenAI client requires one
        self._client = OpenAI(
            api_key=config.api_key or "lm-studio",  # Dummy key
            base_url=self.base_url,
            timeout=self._timeout,
        )

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def client(self) -> OpenAI:
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal Message objects to OpenAI format."""
        result = []
        for msg in messages:
            message_dict: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                message_dict["name"] = msg.name
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            result.append(message_dict)
        return result
    
    def complete(
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
        """Generate a completion using LM Studio API."""
        logger.debug(f"LM Studio request: model={model}, messages={len(messages)}")

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        if stop:
            request_kwargs["stop"] = stop
        # Note: Most local models don't support tools, but include if provided
        if tools:
            request_kwargs["tools"] = tools
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**request_kwargs)
        
        message = response.choices[0].message
        content = message.content or ""
        
        # Parse tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))
        
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return CompletionResult(
            content=content,
            finish_reason=response.choices[0].finish_reason or "stop",
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            raw_response=response.model_dump(),
        )
    
    def stream(
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
    ) -> Iterator[StreamChunk]:
        """Stream a completion using LM Studio API."""
        logger.debug(f"LM Studio stream: model={model}, messages={len(messages)}")

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        if stop:
            request_kwargs["stop"] = stop

        stream = self.client.chat.completions.create(**request_kwargs)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                yield StreamChunk(
                    content=delta.content or "",
                    finish_reason=chunk.choices[0].finish_reason,
                )
    
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a local model."""
        # Most local models have similar capabilities
        # Could be enhanced with model-specific info
        return DEFAULT_LOCAL_CAPABILITIES
    
    def list_models(self) -> list[str]:
        """List available models (cached from last fetch)."""
        if self._available_models:
            return [m["id"] for m in self._available_models]
        return []
    
    def fetch_models(self) -> list[dict[str, Any]]:
        """Fetch available models from LM Studio server."""
        try:
            models = self.client.models.list()
            self._available_models = [
                {"id": m.id, "owned_by": m.owned_by}
                for m in models.data
            ]
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to fetch LM Studio models: {e}")
            return []

    def health_check(self) -> dict[str, Any]:
        """Check if LM Studio server is reachable and list models."""
        try:
            models = self.client.models.list()
            model_list = [
                {"id": m.id, "owned_by": m.owned_by}
                for m in models.data
            ]
            self._available_models = model_list

            return {
                "status": "healthy",
                "base_url": self.base_url,
                "models_available": len(model_list),
                "models": model_list,
                "timeout_seconds": self._timeout,
            }
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return {
                "status": "unhealthy",
                "base_url": self.base_url,
                "error": str(e),
            }

    def close(self) -> None:
        """Close the client."""
        self._client.close()

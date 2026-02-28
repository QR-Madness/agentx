"""
Anthropic model provider implementation.
"""

import logging
from typing import Any, AsyncIterator, Optional

from .base import (
    CompletionResult,
    Message,
    MessageRole,
    ModelCapabilities,
    ModelProvider,
    ProviderConfig,
    StreamChunk,
    ToolCall,
)

logger = logging.getLogger(__name__)

# Model capabilities registry
ANTHROPIC_MODELS = {
    "claude-3-opus-20240229": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "claude-3-sonnet-20240229": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-3-haiku-20240307": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
    ),
    "claude-3-5-sonnet-20241022": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-3-5-haiku-20241022": ModelCapabilities(
        supports_tools=True,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
    ),
}

# Aliases for easier reference
MODEL_ALIASES = {
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
}


class AnthropicProvider(ModelProvider):
    """Anthropic API provider for Claude models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Optional[Any] = None
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def client(self) -> Any:
        """Lazy-load the async Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. "
                    "Install with: pip install anthropic"
                )

            client_kwargs: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self._client = AsyncAnthropic(**client_kwargs)
        return self._client
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model aliases to full model names."""
        return MODEL_ALIASES.get(model, model)
    
    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        """
        Convert internal Message objects to Anthropic format.
        
        Anthropic separates system prompt from messages.
        Returns (system_prompt, messages).
        """
        system_prompt = None
        converted = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic handles system prompt separately
                system_prompt = msg.content
            elif msg.role == MessageRole.TOOL:
                # Tool results in Anthropic format
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })
            else:
                role = "user" if msg.role == MessageRole.USER else "assistant"
                converted.append({
                    "role": role,
                    "content": msg.content,
                })
        
        return system_prompt, converted
    
    def _convert_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)
        
        return anthropic_tools
    
    def _parse_tool_calls(self, content: list[Any]) -> list[ToolCall]:
        """Parse Anthropic tool use blocks into internal format."""
        result = []
        for block in content:
            if hasattr(block, "type") and block.type == "tool_use":
                result.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
        return result
    
    def _extract_text(self, content: list[Any]) -> str:
        """Extract text content from Anthropic response."""
        texts = []
        for block in content:
            if hasattr(block, "type") and block.type == "text":
                texts.append(block.text)
        return "".join(texts)
    
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
        """Generate a completion using Anthropic API."""
        model = self._resolve_model(model)
        system_prompt, converted_messages = self._convert_messages(messages)

        request_params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
        }

        if system_prompt:
            request_params["system"] = system_prompt
        if tools:
            request_params["tools"] = self._convert_tools(tools)
        if tool_choice:
            # Convert OpenAI tool_choice format to Anthropic
            if tool_choice == "auto":
                request_params["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                request_params["tool_choice"] = {"type": "none"}
            elif isinstance(tool_choice, dict):
                request_params["tool_choice"] = tool_choice
        if stop:
            request_params["stop_sequences"] = stop

        logger.debug(f"Anthropic request: model={model}, messages={len(messages)}")

        response = await self.client.messages.create(**request_params)

        content = self._extract_text(response.content)
        tool_calls = self._parse_tool_calls(response.content)

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return CompletionResult(
            content=content,
            finish_reason=response.stop_reason or "end_turn",
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model,
            raw_response={"id": response.id, "type": response.type},
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
        """Stream a completion using Anthropic API."""
        model = self._resolve_model(model)
        system_prompt, converted_messages = self._convert_messages(messages)

        request_params: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system_prompt:
            request_params["system"] = system_prompt
        if tools:
            request_params["tools"] = self._convert_tools(tools)
        if stop:
            request_params["stop_sequences"] = stop

        logger.debug(f"Anthropic stream: model={model}, messages={len(messages)}")

        async with self.client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content=text)

            # Final chunk with finish reason
            response = await stream.get_final_message()
            yield StreamChunk(
                content="",
                finish_reason=response.stop_reason,
            )
    
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for an Anthropic model."""
        model = self._resolve_model(model)
        if model in ANTHROPIC_MODELS:
            return ANTHROPIC_MODELS[model]
        
        logger.warning(f"Unknown Anthropic model: {model}, using default capabilities")
        return ModelCapabilities(
            supports_tools=True,
            supports_streaming=True,
            context_window=200000,
        )
    
    def list_models(self) -> list[str]:
        """List available Anthropic models."""
        return list(ANTHROPIC_MODELS.keys()) + list(MODEL_ALIASES.keys())
    
    async def health_check(self) -> dict[str, Any]:
        """Check if Anthropic API is reachable."""
        if not self.config.api_key:
            return {
                "status": "not_configured",
                "error": "ANTHROPIC_API_KEY not set",
            }

        try:
            # Make a minimal API call to check connectivity
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return {
                "status": "healthy",
                "model": response.model,
                "models": list(ANTHROPIC_MODELS.keys()),
            }
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

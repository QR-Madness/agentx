"""
Anthropic model provider implementation.

Model capabilities use sensible defaults since Anthropic doesn't expose
a public models API. Users should use full model IDs directly.
"""

import json
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
    log_llm_request,
)

logger = logging.getLogger(__name__)

# Default capabilities for Claude models (fallback for unknown models)
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_vision=True,
    supports_streaming=True,
    supports_json_mode=False,
    context_window=200000,
    max_output_tokens=4096,
)

# Per-model capability overrides (prefix-matched, most specific first)
_MODEL_CAPABILITIES: list[tuple[str, ModelCapabilities]] = [
    # Claude 3.0 family — 4096 max output (deprecated, but kept for backwards compat)
    ("claude-3-haiku", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=4096,
    )),
    ("claude-3-sonnet", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=4096,
    )),
    ("claude-3-opus", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=4096,
    )),
    # Claude 3.5 family — 8192 max output (deprecated)
    ("claude-3-5-sonnet", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=8192,
    )),
    ("claude-3-5-haiku", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=8192,
    )),
    # Claude 4+ family
    ("claude-haiku-4", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=8192,
    )),
    ("claude-sonnet-4", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=8192,
    )),
    ("claude-opus-4", ModelCapabilities(
        supports_tools=True, supports_vision=True, supports_streaming=True,
        supports_json_mode=False, context_window=200000, max_output_tokens=8192,
    )),
]

# Known model IDs for list_models()
# These are the current Anthropic model identifiers as of April 2026
KNOWN_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20250514",
    "claude-opus-4-6",
]


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
                # Assistant messages with tool_calls need structured content blocks
                if role == "assistant" and msg.tool_calls:
                    content_blocks: list[dict[str, Any]] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"raw": args}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "input": args,
                        })
                    converted.append({"role": "assistant", "content": content_blocks})
                else:
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
        log_llm_request("Anthropic", request_params)

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
        if tool_choice:
            if tool_choice == "auto":
                request_params["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                request_params["tool_choice"] = {"type": "none"}
            elif isinstance(tool_choice, dict):
                request_params["tool_choice"] = tool_choice
        if stop:
            request_params["stop_sequences"] = stop

        logger.debug(f"Anthropic stream: model={model}, messages={len(messages)}")
        log_llm_request("Anthropic (stream)", request_params)

        async with self.client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content=text)

            # Final chunk with finish reason, tool calls, and usage
            response = await stream.get_final_message()
            tool_calls = self._parse_tool_calls(response.content)
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            yield StreamChunk(
                content="",
                finish_reason=response.stop_reason,
                tool_calls=tool_calls if tool_calls else None,
                usage=usage,
            )
    
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for an Anthropic model.

        Uses prefix matching against known model families to return
        accurate context_window and max_output_tokens per model.
        Falls back to conservative defaults for unknown models.
        """
        for prefix, caps in _MODEL_CAPABILITIES:
            if model.startswith(prefix):
                return caps
        return DEFAULT_CAPABILITIES

    def list_models(self) -> list[str]:
        """List known Anthropic models."""
        return KNOWN_MODELS.copy()

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
                model="claude-3-5-haiku-latest",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return {
                "status": "healthy",
                "model": response.model,
                "models": KNOWN_MODELS,
            }
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

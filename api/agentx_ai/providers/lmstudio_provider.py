"""
LM Studio provider implementation.

LM Studio provides an OpenAI-compatible API, so this is essentially
a wrapper around the OpenAI provider with different defaults.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from .base import (
    CompletionResult,
    Message,
    ModelCapabilities,
    ModelProvider,
    ProviderConfig,
    StreamChunk,
    ToolCall,
    log_llm_request,
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
        self._api_key = config.api_key or "lm-studio"

    @property
    def name(self) -> str:
        return "lmstudio"

    def _get_client(self) -> AsyncOpenAI:
        """Create a fresh async client for each request.

        This ensures the httpx connection pool is created in the correct event loop,
        avoiding 'Connection error' issues when Django's async view runs in different loops.
        """
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self._timeout,
            max_retries=0,
        )
    
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

        client = self._get_client()
        log_llm_request("LM Studio", request_kwargs)
        try:
            response = await client.chat.completions.create(**request_kwargs)
        finally:
            await client.close()

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
        if tools:
            request_kwargs["tools"] = tools
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice

        client = self._get_client()
        log_llm_request("LM Studio (stream)", request_kwargs)
        try:
            logger.debug("Creating stream...")
            stream = await client.chat.completions.create(**request_kwargs)
            logger.debug("Stream created, starting iteration...")

            # Accumulate tool call fragments across chunks
            pending_tool_calls: dict[int, dict[str, Any]] = {}
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if not delta:
                    continue

                # Log finish reason when stream is ending
                if choice.finish_reason:
                    logger.debug(f"Stream chunk {chunk_count}: finish_reason={choice.finish_reason}")

                # Accumulate tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        entry = pending_tool_calls[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

                # Emit content chunks as they arrive
                if delta.content:
                    yield StreamChunk(
                        content=delta.content,
                        finish_reason=choice.finish_reason,
                    )

                # When stream ends with tool_calls, emit accumulated tool calls
                if choice.finish_reason == "tool_calls" and pending_tool_calls:
                    completed = []
                    for tc_data in pending_tool_calls.values():
                        try:
                            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {"raw": tc_data["arguments"]}
                        completed.append(ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=args,
                        ))
                    yield StreamChunk(
                        content="",
                        finish_reason="tool_calls",
                        tool_calls=completed,
                    )
                    pending_tool_calls.clear()
                elif choice.finish_reason and choice.finish_reason != "tool_calls":
                    yield StreamChunk(content="", finish_reason=choice.finish_reason)

            logger.debug(f"Stream iteration complete, processed {chunk_count} chunks")
        finally:
            logger.debug("Closing client...")
            await client.close()
            logger.debug("Client closed")
    
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
    
    async def fetch_models(self) -> list[dict[str, Any]]:
        """Fetch available models from LM Studio server."""
        client = self._get_client()
        try:
            models = await client.models.list()
            self._available_models = [
                {"id": m.id, "owned_by": m.owned_by}
                for m in models.data
            ]
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to fetch LM Studio models: {e}")
            return []
        finally:
            await client.close()

    async def health_check(self) -> dict[str, Any]:
        """Check if LM Studio server is reachable and list models."""
        client = self._get_client()
        try:
            models = await client.models.list()
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
        finally:
            await client.close()

    async def close(self) -> None:
        """Close the provider. No-op since clients are created per-request."""
        pass

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
    accumulate_tool_call_delta,
    finalize_tool_calls,
    log_llm_request,
)

logger = logging.getLogger(__name__)

# Default capabilities for local models
# Modern models typically have 128k+ context, so we use generous defaults
DEFAULT_LOCAL_CAPABILITIES = ModelCapabilities(
    supports_tools=True,  # Most modern local models support function calling
    supports_vision=False,
    supports_streaming=True,
    supports_json_mode=True,
    context_window=131072,  # 128k - common for modern models (Llama 3.x, Qwen, Nemotron, etc.)
    max_output_tokens=16384,  # 16k output - generous but reasonable
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
        import httpx
        # Use a custom transport with no HTTP/2 to avoid potential streaming issues
        # and no keep-alive to ensure fresh connections
        transport = httpx.AsyncHTTPTransport(
            retries=0,
            http2=False,  # Force HTTP/1.1 for better streaming compatibility
        )
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self._timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
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

    @staticmethod
    def _parse_sse_data(event_str: str) -> Optional[dict[str, Any]]:
        """
        Parse SSE event string and return JSON data.

        Handles both "data: " (with space) and "data:" (without) formats.
        Returns None for empty events or [DONE] marker.
        """
        data_content = ""
        for line in event_str.split("\n"):
            if line.startswith("data: "):
                data_content = line[6:]
                break
            elif line.startswith("data:"):
                data_content = line[5:]
                break

        if not data_content or data_content == "[DONE]":
            return None

        try:
            return json.loads(data_content)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _process_reasoning_content(
        reasoning: str,
        content: str,
        in_reasoning: bool,
    ) -> tuple[str, bool]:
        """
        Process reasoning_content and content fields from delta.

        Some models (like Nemotron) output thinking in 'reasoning_content'
        field. This method wraps reasoning in <think> tags for UI display.

        Args:
            reasoning: reasoning_content from delta
            content: content from delta
            in_reasoning: whether we're currently in a reasoning block

        Returns:
            (output_string, new_in_reasoning_state)
        """
        output = ""

        if reasoning:
            if not in_reasoning:
                # Start reasoning block
                output = f"<think>{reasoning}"
                in_reasoning = True
            else:
                # Continue reasoning block
                output = reasoning

        if content:
            if in_reasoning:
                # End reasoning block, then emit content
                output += f"</think>{content}"
                in_reasoning = False
            else:
                output += content

        return output, in_reasoning
    
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
        """Stream a completion using LM Studio API with raw httpx for better streaming."""
        import httpx

        logger.debug(f"LM Studio stream: model={model}, messages={len(messages)}")

        request_body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            request_body["max_tokens"] = max_tokens
        if stop:
            request_body["stop"] = stop
        if tools:
            request_body["tools"] = tools
            if tool_choice:
                request_body["tool_choice"] = tool_choice

        log_llm_request("LM Studio (stream)", request_body)

        # Use raw httpx for streaming (bypasses OpenAI client async iterator issues)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as http_client:
            logger.debug(f"Creating raw httpx stream to {url}...")
            async with http_client.stream(
                "POST",
                url,
                json=request_body,
                headers=headers,
            ) as response:
                logger.info(f"Raw httpx response: status={response.status_code}")

                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"HTTP error: {error_body}")
                    raise Exception(f"HTTP {response.status_code}: {error_body}")

                # State for streaming
                pending_tool_calls: dict[int, dict[str, Any]] = {}
                chunk_count = 0
                buffer = ""
                in_reasoning = False

                # Use aiter_bytes for unbuffered streaming
                async for raw_bytes in response.aiter_bytes():
                    buffer += raw_bytes.decode("utf-8", errors="replace")

                    # Parse SSE events from buffer (events separated by \n\n)
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        if not event_str.strip():
                            continue

                        # Parse SSE data
                        chunk_data = self._parse_sse_data(event_str)
                        if not chunk_data:
                            continue

                        chunk_count += 1
                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        # Process reasoning and content
                        content, in_reasoning = self._process_reasoning_content(
                            delta.get("reasoning_content", ""),
                            delta.get("content", ""),
                            in_reasoning,
                        )

                        # Accumulate tool call deltas (using shared helper)
                        if "tool_calls" in delta:
                            for tc_delta in delta["tool_calls"]:
                                accumulate_tool_call_delta(pending_tool_calls, tc_delta)

                        # Emit content chunks
                        if content:
                            yield StreamChunk(content=content, finish_reason=finish_reason)

                        # Handle stream end
                        if finish_reason:
                            # Close reasoning block if still open
                            if in_reasoning:
                                yield StreamChunk(content="</think>", finish_reason=None)
                                in_reasoning = False

                            # Emit tool calls or final chunk
                            if finish_reason == "tool_calls" and pending_tool_calls:
                                yield StreamChunk(
                                    content="",
                                    finish_reason="tool_calls",
                                    tool_calls=finalize_tool_calls(pending_tool_calls),
                                )
                                pending_tool_calls.clear()
                            elif finish_reason != "tool_calls":
                                yield StreamChunk(content="", finish_reason=finish_reason)

                logger.debug(f"Stream complete: {chunk_count} chunks")
    
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

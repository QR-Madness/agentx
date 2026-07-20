"""
Anthropic model provider implementation.

Model capabilities use sensible defaults since Anthropic doesn't expose
a public models API. Users should use full model IDs directly.
"""

import json
import logging
import time
from typing import Any
from collections.abc import AsyncIterator

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


def _anthropic_usage(usage: Any) -> dict[str, Any]:
    """Build the StreamChunk/CompletionResult usage dict from Anthropic's usage.

    `input_tokens`/`output_tokens` are authoritative (extended-thinking folds into
    output). Also surfaces prompt-cache activity **when present** — `cache_read`
    (billed ~0.1x) and `cache_creation` (~1.25x) — for metering visibility; both are
    absent today (AgentX sets no `cache_control`) and on older SDK versions, so they
    are guarded. Token totals stay unchanged; cache-aware cost is a later follow-up.
    """
    out: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if cache_read:
        out["cache_read_tokens"] = int(cache_read)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)
    if cache_creation:
        out["cache_creation_tokens"] = int(cache_creation)
    return out


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

# Fallback list used until `_fetch_models()` populates the dynamic cache (and
# whenever the live /v1/models call fails). These are the Anthropic model IDs
# as of April 2026; kept current enough to bootstrap, but the runtime list
# always comes from the API once we've talked to it.
KNOWN_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20250514",
    "claude-opus-4-6",
]

# Time-to-live for the dynamic model cache. Anthropic publishes new models
# infrequently; an hour is plenty.
MODEL_CACHE_TTL = 3600.0


class AnthropicProvider(ModelProvider):
    """Anthropic API provider for Claude models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client: Any | None = None
        # Dynamic model catalog populated from /v1/models on first use and
        # whenever the cache TTL expires. Mirrors the OpenRouter pattern so
        # the providers_models endpoint and the dashboard health pill stay
        # in sync with Anthropic's actual offerings.
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._cache_timestamp: float = 0.0
    
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
                ) from None

            client_kwargs: dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout or 60.0,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self._client = AsyncAnthropic(**client_kwargs)
        return self._client

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert internal Message objects to Anthropic format.
        
        Anthropic separates system prompt from messages.
        Returns (system_prompt, messages).
        """
        system_parts: list[str] = []
        converted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic takes ONE system param — collect and join every
                # system message. A turn carries several (base prompt, project
                # instructions, memory blocks, token-budget header); assigning
                # instead of appending silently dropped all but the last.
                if msg.content:
                    system_parts.append(msg.content)
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
                elif role == "user" and msg.images:
                    # Vision input: user message carries images → content becomes a
                    # block list (text + base64 image sources). Text-only stays a string.
                    # `msg.audio` is intentionally NOT converted — the Anthropic API has
                    # no audio input block; the STT-transcript fallback upstream covers it.
                    from .base import resolve_image_data

                    user_blocks: list[dict[str, Any]] = []
                    if msg.content:
                        user_blocks.append({"type": "text", "text": msg.content})
                    for ref in msg.images:
                        resolved = resolve_image_data(ref)
                        if resolved is None:
                            continue
                        media_type, b64 = resolved
                        user_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        })
                    converted.append({
                        "role": "user",
                        "content": user_blocks if user_blocks else msg.content,
                    })
                else:
                    converted.append({
                        "role": role,
                        "content": msg.content,
                    })

        return ("\n\n".join(system_parts) if system_parts else None), converted
    
    def _convert_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
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
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | None = None,
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

        usage = _anthropic_usage(response.usage)

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
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | None = None,
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
            usage = _anthropic_usage(response.usage)
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

    async def _fetch_models_if_stale(self) -> None:
        """Refresh the model cache from Anthropic's /v1/models if stale.

        Leaves any prior cache intact on failure so a transient blip doesn't
        blank the list mid-session.
        """
        now = time.time()
        if self._model_cache and (now - self._cache_timestamp) < MODEL_CACHE_TTL:
            return

        page = await self.client.models.list(limit=1000)
        # SDK returns a CursorPage with .data: list[ModelInfo]. We keep id +
        # display_name + type + created_at — enough for picker / dashboard
        # and forward-compatible if Anthropic adds new tiers.
        cache: dict[str, dict[str, Any]] = {}
        for m in getattr(page, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid:
                continue
            cache[mid] = {
                "id": mid,
                "display_name": getattr(m, "display_name", None),
                "type": getattr(m, "type", None),
                "created_at": getattr(m, "created_at", None),
            }
        if cache:
            self._model_cache = cache
            self._cache_timestamp = now
            logger.info(f"Fetched {len(cache)} models from Anthropic")

    def list_models(self) -> list[str]:
        """List Claude models.

        Returns the dynamic cache once we've talked to /v1/models; falls
        back to the hardcoded KNOWN_MODELS until then (so the UI has
        something before the first network call lands).
        """
        if self._model_cache:
            return list(self._model_cache.keys())
        return KNOWN_MODELS.copy()

    async def fetch_models(self) -> list[dict[str, Any]]:
        """Async hook for `/api/providers/models` — refreshes the cache if
        stale and returns the full per-model metadata."""
        await self._fetch_models_if_stale()
        return list(self._model_cache.values())

    async def health_check(self) -> dict[str, Any]:
        """Check if Anthropic API is reachable.

        Pings /v1/models — a free, no-token GET that validates auth and
        connectivity without invoking inference. The previous implementation
        pinged a hardcoded model ID (claude-3-5-haiku-latest) and broke
        silently once that model was retired; this approach has no such
        coupling.
        """
        if not self.config.api_key:
            return {
                "status": "not_configured",
                "error": "ANTHROPIC_API_KEY not set",
            }

        try:
            await self._fetch_models_if_stale()
            return {
                "status": "healthy",
                "models_available": len(self._model_cache),
                "models": list(self._model_cache.keys())[:20],
            }
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def close(self) -> None:
        """Close the cached AsyncAnthropic client and reset it for re-creation."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Anthropic client: {e}")
            finally:
                self._client = None

"""
Ollama model provider implementation for local models.
"""

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

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

# Common local model capabilities (can be overridden)
DEFAULT_LOCAL_CAPABILITIES = ModelCapabilities(
    supports_tools=False,
    supports_vision=False,
    supports_streaming=True,
    supports_json_mode=True,
    context_window=8192,
    max_output_tokens=4096,
    cost_per_1k_input=0.0,  # Local = free
    cost_per_1k_output=0.0,
)

# Known model capabilities
OLLAMA_MODELS = {
    "llama3.2": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "llama3.1": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "llama3": ModelCapabilities(
        supports_tools=False,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "mistral": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=32768,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "mixtral": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=32768,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "codellama": ModelCapabilities(
        supports_tools=False,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=16384,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "llava": ModelCapabilities(
        supports_tools=False,
        supports_vision=True,
        supports_streaming=True,
        supports_json_mode=False,
        context_window=4096,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "qwen2.5": ModelCapabilities(
        supports_tools=True,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
    "deepseek-coder-v2": ModelCapabilities(
        supports_tools=False,
        supports_vision=False,
        supports_streaming=True,
        supports_json_mode=True,
        context_window=128000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    ),
}


class OllamaProvider(ModelProvider):
    """Ollama provider for local models."""
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self._client: Optional[httpx.AsyncClient] = None
        self._available_models: Optional[list[str]] = None
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal Message objects to Ollama format."""
        result = []
        for msg in messages:
            role = msg.role.value
            # Ollama uses 'assistant' not 'tool' for tool responses
            if role == "tool":
                role = "assistant"
            
            result.append({
                "role": role,
                "content": msg.content,
            })
        return result
    
    def _convert_tools(
        self, tools: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Convert OpenAI-style tools to Ollama format."""
        if not tools:
            return None
        
        # Ollama uses similar format to OpenAI for tools
        return tools
    
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
        """Generate a completion using Ollama API."""
        request_data: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        if stop:
            request_data["options"]["stop"] = stop
        if tools:
            request_data["tools"] = self._convert_tools(tools)
        
        logger.debug(f"Ollama request: model={model}, messages={len(messages)}")
        
        response = await self.client.post("/api/chat", json=request_data)
        response.raise_for_status()
        data = response.json()
        
        message = data.get("message", {})
        content = message.get("content", "")
        
        # Parse tool calls if present
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))
        
        # Ollama provides token counts
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        
        return CompletionResult(
            content=content,
            finish_reason=data.get("done_reason", "stop"),
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            raw_response=data,
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
        """Stream a completion using Ollama API."""
        request_data: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        if stop:
            request_data["options"]["stop"] = stop
        
        logger.debug(f"Ollama stream: model={model}, messages={len(messages)}")
        
        async with self.client.stream(
            "POST", "/api/chat", json=request_data
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                message = data.get("message", {})
                content = message.get("content", "")
                
                finish_reason = None
                if data.get("done"):
                    finish_reason = data.get("done_reason", "stop")
                
                yield StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                )
    
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for an Ollama model."""
        # Check for exact match
        if model in OLLAMA_MODELS:
            return OLLAMA_MODELS[model]
        
        # Check for base model name (e.g., "llama3:70b" -> "llama3")
        base_model = model.split(":")[0]
        if base_model in OLLAMA_MODELS:
            return OLLAMA_MODELS[base_model]
        
        logger.warning(f"Unknown Ollama model: {model}, using default capabilities")
        return DEFAULT_LOCAL_CAPABILITIES
    
    def list_models(self) -> list[str]:
        """List available Ollama models (cached from last fetch)."""
        if self._available_models:
            return self._available_models
        return list(OLLAMA_MODELS.keys())
    
    async def fetch_models(self) -> list[str]:
        """Fetch available models from Ollama server."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = [m["name"] for m in data.get("models", [])]
            self._available_models = models
            return models
        except Exception as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
            return []
    
    async def health_check(self) -> dict[str, Any]:
        """Check if Ollama server is reachable."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = [m["name"] for m in data.get("models", [])]
            self._available_models = models
            
            return {
                "status": "healthy",
                "models_available": len(models),
                "models": models[:5],  # First 5 for brevity
            }
        except httpx.ConnectError:
            return {
                "status": "unhealthy",
                "error": f"Cannot connect to Ollama at {self.base_url}",
            }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

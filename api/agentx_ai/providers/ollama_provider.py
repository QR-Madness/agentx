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
    DEFAULT_TIMEOUT = 300.0  # 5 minutes for large local models
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self.DEFAULT_BASE_URL
        self._available_models: Optional[list[str]] = None
        
        # Use configured timeout or default to 5 minutes for local models
        self._timeout = config.timeout if config.timeout != 60.0 else self.DEFAULT_TIMEOUT
    
    @property
    def name(self) -> str:
        return "ollama"
    
    def _get_client(self) -> httpx.AsyncClient:
        """Create a fresh HTTP client for each request.
        
        This avoids 'Event loop is closed' errors when used with
        Django's async_to_sync which creates new event loops.
        
        Uses separate connect/read timeouts - connect should be fast,
        but read can be very slow for large models.
        """
        timeout = httpx.Timeout(
            connect=10.0,       # 10s to establish connection
            read=self._timeout, # Long timeout for model inference
            write=30.0,         # 30s to send request
            pool=10.0,          # 10s to acquire connection from pool
        )
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )
    
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
    
    def _messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert messages to a single prompt string for /api/generate."""
        parts = []
        for msg in messages:
            role = msg.role.value.upper()
            parts.append(f"{role}: {msg.content}")
        parts.append("ASSISTANT:")  # Prompt for response
        return "\n\n".join(parts)
    
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
        """Generate a completion using Ollama API.
        
        Tries /api/chat first, falls back to /api/generate for older Ollama versions.
        """
        logger.debug(f"Ollama request: model={model}, messages={len(messages)}")
        
        async with self._get_client() as client:
            # Try /api/chat first (newer Ollama versions)
            try:
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
                
                response = await client.post("/api/chat", json=request_data)
                
                if response.status_code == 404:
                    # Fall back to /api/generate for older Ollama versions
                    raise httpx.HTTPStatusError(
                        "Chat endpoint not available",
                        request=response.request,
                        response=response,
                    )
                
                response.raise_for_status()
                data = response.json()
                
                message = data.get("message", {})
                content = message.get("content", "")
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    raise
                
                # Fall back to /api/generate
                logger.debug(f"Falling back to /api/generate for model={model}")
                generate_data: dict[str, Any] = {
                    "model": model,
                    "prompt": self._messages_to_prompt(messages),
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                }
                
                if max_tokens:
                    generate_data["options"]["num_predict"] = max_tokens
                if stop:
                    generate_data["options"]["stop"] = stop
                
                response = await client.post("/api/generate", json=generate_data)
                response.raise_for_status()
                data = response.json()
                content = data.get("response", "")
        
        # Parse tool calls if present (only from /api/chat)
        tool_calls = None
        message = data.get("message", {})
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
        
        async with self._get_client() as client:
            async with client.stream(
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
        
        # Check for versioned models (e.g., "llama3.2" -> check "llama3")
        for known_model in OLLAMA_MODELS:
            if base_model.startswith(known_model):
                return OLLAMA_MODELS[known_model]
        
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
            async with self._get_client() as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                data = response.json()
            
            models = [m["name"] for m in data.get("models", [])]
            self._available_models = models
            return models
        except Exception as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
            return []
    
    async def health_check(self) -> dict[str, Any]:
        """Check if Ollama server is reachable and list available models."""
        try:
            async with self._get_client() as client:
                # Get server version
                version_resp = await client.get("/api/version")
                version = None
                if version_resp.status_code == 200:
                    version = version_resp.json().get("version")
                
                # Get available models
                response = await client.get("/api/tags")
                response.raise_for_status()
                data = response.json()
            
            models = []
            for m in data.get("models", []):
                model_info = {
                    "name": m.get("name"),
                    "size": m.get("size"),
                    "parameter_size": m.get("details", {}).get("parameter_size"),
                }
                models.append(model_info)
            
            self._available_models = [m["name"] for m in models]
            
            return {
                "status": "healthy",
                "base_url": self.base_url,
                "version": version,
                "models_available": len(models),
                "models": models,  # Full list with details
                "timeout_seconds": self._timeout,
            }
        except httpx.ConnectError:
            return {
                "status": "unhealthy",
                "base_url": self.base_url,
                "error": f"Cannot connect to Ollama at {self.base_url}",
            }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "unhealthy",
                "base_url": self.base_url,
                "error": str(e),
            }
    
    async def close(self) -> None:
        """No persistent client to close."""
        pass

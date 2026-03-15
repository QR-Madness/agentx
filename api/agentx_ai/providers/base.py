"""
Abstract base classes for model providers.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

def log_llm_request(provider_name: str, request_params: dict[str, Any]) -> None:
    """Log the full LLM request payload when DEBUG_LOG_LLM_REQUESTS is set."""
    if os.environ.get("DEBUG_LOG_LLM_REQUESTS", "").strip() in ("", "0", "false"):
        return
    try:
        dumped = json.dumps(request_params, indent=2, default=str)
    except Exception:
        dumped = str(request_params)
    logger.info(f"[DEBUG LLM REQUEST] {provider_name}:\n{dumped}")


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ToolCall(BaseModel):
    """A tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


def accumulate_tool_call_delta(
    pending_calls: dict[int, dict[str, Any]],
    tc_delta: dict[str, Any],
) -> None:
    """
    Accumulate a streaming tool call delta fragment.

    Tool calls arrive incrementally across multiple chunks. This helper
    accumulates the fragments into complete tool calls.

    Args:
        pending_calls: Dict mapping tool index to accumulated call data
        tc_delta: Delta fragment from a streaming chunk
    """
    idx = tc_delta.get("index", 0)
    if idx not in pending_calls:
        pending_calls[idx] = {"id": "", "name": "", "arguments": ""}

    entry = pending_calls[idx]
    if tc_delta.get("id"):
        entry["id"] = tc_delta["id"]

    func = tc_delta.get("function", {})
    if func.get("name"):
        entry["name"] = func["name"]
    if func.get("arguments"):
        entry["arguments"] += func["arguments"]


def finalize_tool_calls(pending_calls: dict[int, dict[str, Any]]) -> list[ToolCall]:
    """
    Convert accumulated tool call fragments to ToolCall objects.

    Args:
        pending_calls: Dict of accumulated tool call data

    Returns:
        List of complete ToolCall objects
    """
    completed = []
    for tc_data in pending_calls.values():
        try:
            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
        except json.JSONDecodeError:
            args = {"raw": tc_data["arguments"]}
        completed.append(ToolCall(
            id=tc_data["id"],
            name=tc_data["name"],
            arguments=args,
        ))
    return completed


class StreamChunk(BaseModel):
    """A chunk of streaming response."""
    content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    usage: Optional[dict[str, int]] = None  # Token usage (available on final chunk)


class CompletionResult(BaseModel):
    """Result of a completion request."""
    content: str
    finish_reason: str
    tool_calls: Optional[list[ToolCall]] = None
    usage: Optional[dict[str, int]] = None
    model: str
    raw_response: Optional[dict[str, Any]] = None


@dataclass
class ModelCapabilities:
    """Capabilities of a model."""
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_json_mode: bool = False
    context_window: int = 4096
    max_output_tokens: Optional[int] = None
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """
    Abstract base class for model providers.

    All LLM providers (LM Studio, Anthropic, OpenAI) implement this interface
    to provide a unified way to interact with different models.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @abstractmethod
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
        """
        Generate a completion for the given messages.

        Args:
            messages: The conversation messages
            model: The model to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: How to select tools ("auto", "none", or specific tool)
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResult with the generated content
        """
        ...
    
    @abstractmethod
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
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion for the given messages.

        Args:
            messages: The conversation messages
            model: The model to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: How to select tools
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects as they arrive
        """
        ...
    
    @abstractmethod
    def get_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get the capabilities of a specific model.
        
        Args:
            model: The model name
            
        Returns:
            ModelCapabilities describing what the model supports
        """
        pass
    
    @abstractmethod
    def list_models(self) -> list[str]:
        """
        List all available models for this provider.
        
        Returns:
            List of model names
        """
        pass
    
    def health_check(self) -> dict[str, Any]:
        """
        Check if the provider is healthy and reachable.

        Returns:
            Health status dict with 'status' and optional details
        """
        return {"status": "unknown", "message": "Health check not implemented"}

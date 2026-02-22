"""
Abstract base classes for model providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel


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


class StreamChunk(BaseModel):
    """A chunk of streaming response."""
    content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


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
        pass
    
    @abstractmethod
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
        pass
        # Subclasses must implement this as an async generator
        # yield StreamChunk(content="")  # type: ignore
    
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
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check if the provider is healthy and reachable.
        
        Returns:
            Health status dict with 'status' and optional details
        """
        return {"status": "unknown", "message": "Health check not implemented"}

"""
Model Provider Abstraction Layer for AgentX.

This module provides a unified interface to multiple LLM backends:
- OpenAI (GPT-4, GPT-4-turbo, GPT-3.5)
- Anthropic (Claude 3 Opus/Sonnet/Haiku)
- Ollama (local models: Llama, Mistral, etc.)
"""

from .base import ModelProvider, ModelCapabilities, CompletionResult, StreamChunk
from .registry import ProviderRegistry, get_provider, get_model_config, get_registry

__all__ = [
    "ModelProvider",
    "ModelCapabilities",
    "CompletionResult",
    "StreamChunk",
    "ProviderRegistry",
    "get_provider",
    "get_model_config",
    "get_registry",
]

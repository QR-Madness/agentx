"""
Provider registry for managing model providers and configurations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from .base import ModelCapabilities, ModelProvider, ProviderConfig

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: str
    model_id: Optional[str] = None  # If different from the key
    context_window: int = 8192
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    local: bool = False
    default_temperature: float = 0.7
    max_output_tokens: Optional[int] = None


class ProviderRegistry:
    """
    Registry for model providers.
    
    Manages provider instances and model configurations,
    providing a unified interface to get providers and models.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self._providers: dict[str, ModelProvider] = {}
        self._model_configs: dict[str, ModelConfig] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}
        
        # Load configuration if provided
        if config_path and config_path.exists():
            self.load_config(config_path)
        else:
            # Load from default location or environment
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load configuration from environment variables."""
        # OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            self._provider_configs["openai"] = ProviderConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        
        # Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            self._provider_configs["anthropic"] = ProviderConfig(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            )
        
        # Ollama (local, no API key needed)
        self._provider_configs["ollama"] = ProviderConfig(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from a YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Load provider configurations
        for provider_name, provider_data in config.get("providers", {}).items():
            self._provider_configs[provider_name] = ProviderConfig(
                api_key=provider_data.get("api_key") or os.environ.get(
                    provider_data.get("api_key_env", f"{provider_name.upper()}_API_KEY")
                ),
                base_url=provider_data.get("base_url"),
                timeout=provider_data.get("timeout", 60.0),
                max_retries=provider_data.get("max_retries", 3),
            )
        
        # Load model configurations
        for model_name, model_data in config.get("models", {}).items():
            self._model_configs[model_name] = ModelConfig(**model_data)
    
    def get_provider(self, name: str) -> ModelProvider:
        """
        Get or create a provider instance.
        
        Args:
            name: Provider name ("openai", "anthropic", "ollama")
            
        Returns:
            Configured ModelProvider instance
            
        Raises:
            ValueError: If provider is not supported or not configured
        """
        if name in self._providers:
            return self._providers[name]
        
        if name not in self._provider_configs:
            raise ValueError(
                f"Provider '{name}' not configured. "
                f"Set {name.upper()}_API_KEY environment variable or provide config."
            )
        
        config = self._provider_configs[name]
        
        if name == "openai":
            from .openai_provider import OpenAIProvider
            provider = OpenAIProvider(config)
        elif name == "anthropic":
            from .anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(config)
        elif name == "ollama":
            from .ollama_provider import OllamaProvider
            provider = OllamaProvider(config)
        else:
            raise ValueError(f"Unknown provider: {name}")
        
        self._providers[name] = provider
        return provider
    
    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self._model_configs.get(model)
    
    def get_provider_for_model(self, model: str) -> tuple[ModelProvider, str]:
        """
        Get the appropriate provider for a model.
        
        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus", "llama3")
            
        Returns:
            Tuple of (provider, model_id) where model_id may differ from model name
            
        Raises:
            ValueError: If provider cannot be determined
        """
        # Check if model is in our config
        if model in self._model_configs:
            config = self._model_configs[model]
            provider = self.get_provider(config.provider)
            model_id = config.model_id or model
            return provider, model_id
        
        # Infer provider from model name
        if model.startswith("gpt-") or model.startswith("o1"):
            return self.get_provider("openai"), model
        elif model.startswith("claude"):
            return self.get_provider("anthropic"), model
        elif ":" in model or model in ["llama3", "mistral", "mixtral", "codellama", "llava", "qwen2.5"]:
            # Ollama models often have tags like "llama3:70b"
            return self.get_provider("ollama"), model
        
        raise ValueError(
            f"Cannot determine provider for model '{model}'. "
            "Add it to models.yaml or use a recognized model name prefix."
        )
    
    def list_providers(self) -> list[str]:
        """List configured providers."""
        return list(self._provider_configs.keys())
    
    def list_models(self) -> list[str]:
        """List all configured models."""
        models = list(self._model_configs.keys())
        
        # Also include models from active providers
        for provider in self._providers.values():
            models.extend(provider.list_models())
        
        return list(set(models))
    
    async def health_check(self) -> dict[str, Any]:
        """Check health of all configured providers."""
        results = {}
        
        for name in self._provider_configs:
            try:
                provider = self.get_provider(name)
                results[name] = await provider.health_check()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        return results


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def get_provider(name: str) -> ModelProvider:
    """Get a provider by name from the global registry."""
    return get_registry().get_provider(name)


def get_model_config(model: str) -> Optional[ModelConfig]:
    """Get model configuration from the global registry."""
    return get_registry().get_model_config(model)

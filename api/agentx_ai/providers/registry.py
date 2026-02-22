"""
Provider registry for managing model providers and configurations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from .base import ModelProvider, ProviderConfig
from ..config import get_config_manager

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
        """Load configuration from ConfigManager with environment variable fallback."""
        config = get_config_manager()

        # LM Studio (local, OpenAI-compatible) - primary local provider
        lmstudio_url = config.get_provider_value(
            "lmstudio", "base_url", env_var="LMSTUDIO_BASE_URL"
        )
        if lmstudio_url:
            lmstudio_timeout = config.get_provider_value(
                "lmstudio", "timeout", env_var="LMSTUDIO_TIMEOUT", default=300
            )
            self._provider_configs["lmstudio"] = ProviderConfig(
                base_url=lmstudio_url,
                timeout=float(lmstudio_timeout),
            )

        # Anthropic (cloud) - primary cloud provider
        anthropic_key = config.get_provider_value(
            "anthropic", "api_key", env_var="ANTHROPIC_API_KEY"
        )
        if anthropic_key:
            anthropic_url = config.get_provider_value(
                "anthropic", "base_url", env_var="ANTHROPIC_BASE_URL"
            )
            self._provider_configs["anthropic"] = ProviderConfig(
                api_key=anthropic_key,
                base_url=anthropic_url,
            )

        # OpenAI (cloud) - experimental
        openai_key = config.get_provider_value(
            "openai", "api_key", env_var="OPENAI_API_KEY"
        )
        if openai_key:
            openai_url = config.get_provider_value(
                "openai", "base_url", env_var="OPENAI_BASE_URL"
            )
            self._provider_configs["openai"] = ProviderConfig(
                api_key=openai_key,
                base_url=openai_url,
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
            name: Provider name ("lmstudio", "anthropic", "openai")

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
                f"Configure it in Settings or set {name.upper()}_BASE_URL / {name.upper()}_API_KEY."
            )

        config = self._provider_configs[name]

        if name == "lmstudio":
            from .lmstudio_provider import LMStudioProvider
            provider = LMStudioProvider(config)
        elif name == "anthropic":
            from .anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(config)
        elif name == "openai":
            from .openai_provider import OpenAIProvider
            provider = OpenAIProvider(config)
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
            model: Model name (e.g., "gpt-4", "claude-3-opus", "llama3.2")
            
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
        
        # Local model prefixes - check these FIRST 
        # because some local models have names like "gpt-oss" or "openai/..."
        local_prefixes = (
            "llama", "mistral", "mixtral", "codellama", "llava", 
            "qwen", "deepseek", "phi", "gemma", "vicuna", "orca",
            "neural", "dolphin", "openchat", "starling", "yi",
            "command-r", "dbrx", "nous", "wizardlm", "zephyr",
            "gpt-oss", "ibm/", "google/", "meta/", "mistralai/",
            "openai-gpt-oss", "openai/gpt-oss",  # LM Studio naming
        )
        
        # Check if this looks like a local model
        is_local = (
            ":" in model or  # Ollama tags like "llama3.2:latest"
            "/" in model or  # LM Studio paths like "google/gemma-3-12b"
            model.startswith(local_prefixes)
        )
        
        if is_local:
            # Use LM Studio for local models
            if "lmstudio" in self._provider_configs:
                return self.get_provider("lmstudio"), model
            else:
                raise ValueError(
                    f"Local model '{model}' detected but LM Studio not configured. "
                    "Set the LM Studio URL in Settings or LMSTUDIO_BASE_URL environment variable."
                )
        
        # Infer provider from model name (cloud providers)
        if model.startswith("gpt-") or model.startswith("o1"):
            return self.get_provider("openai"), model
        elif model.startswith("claude"):
            return self.get_provider("anthropic"), model
        
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
    
    def health_check(self) -> dict[str, Any]:
        """Check health of all configured providers."""
        results = {}

        for name in self._provider_configs:
            try:
                provider = self.get_provider(name)
                results[name] = provider.health_check()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}

        return results

    def reload(self) -> None:
        """
        Reload configuration and clear cached providers.

        Call this after config changes to apply new settings.
        Running requests will continue using old providers;
        new requests will use the updated configuration.
        """
        # Clear cached provider instances
        self._providers.clear()
        self._provider_configs.clear()

        # Reload config from ConfigManager
        from ..config import reload_config
        reload_config()

        # Re-read provider configs
        self._load_default_config()

        logger.info("Provider registry reloaded")


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


def reload_providers() -> None:
    """Reload provider configuration from ConfigManager."""
    registry = get_registry()
    registry.reload()

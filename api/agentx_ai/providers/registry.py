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

        # OpenRouter (cloud aggregator - 100+ models)
        openrouter_key = config.get_provider_value(
            "openrouter", "api_key", env_var="OPENROUTER_API_KEY"
        )
        if openrouter_key:
            self._provider_configs["openrouter"] = ProviderConfig(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                extra={
                    "site_url": os.environ.get("OPENROUTER_SITE_URL", ""),
                    "app_name": os.environ.get("OPENROUTER_APP_NAME", "AgentX"),
                },
            )

        # Vercel AI Gateway (cloud aggregator - 100+ models, high availability)
        vercel_key = config.get_provider_value(
            "vercel", "api_key", env_var="AI_GATEWAY_API_KEY"
        )
        if vercel_key:
            vercel_url = config.get_provider_value(
                "vercel", "base_url", env_var="AI_GATEWAY_BASE_URL"
            )
            self._provider_configs["vercel"] = ProviderConfig(
                api_key=vercel_key,
                base_url=vercel_url or "https://ai-gateway.vercel.sh/v1",
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
            name: Provider name ("lmstudio", "anthropic", "openai", "openrouter")

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
        elif name == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(config)
        elif name == "vercel":
            from .vercel_provider import VercelProvider
            provider = VercelProvider(config)
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

        Models must use provider:model format (e.g., "anthropic:claude-3-opus").

        Args:
            model: Model in provider:model_id format

        Returns:
            Tuple of (provider, model_id)

        Raises:
            ValueError: If format is invalid or provider not configured
        """
        if ":" not in model:
            raise ValueError(
                f"Invalid model format '{model}'. "
                "Use provider:model format (e.g., 'anthropic:claude-3-5-sonnet-latest', 'lmstudio:llama3.2')."
            )

        provider_name, model_id = model.split(":", 1)

        if provider_name not in self._provider_configs:
            available = ", ".join(self._provider_configs.keys()) if self._provider_configs else "none"
            raise ValueError(
                f"Provider '{provider_name}' not configured. "
                f"Available providers: {available}"
            )

        return self.get_provider(provider_name), model_id
    
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

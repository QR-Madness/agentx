"""
Configuration manager for runtime settings.

Provides a centralized configuration system that:
- Persists to data/config.json
- Supports hot-reload without server restart
- Falls back to environment variables when config values are not set
"""

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default configuration structure
DEFAULT_CONFIG = {
    "providers": {
        "lmstudio": {
            "base_url": None,
            "timeout": 300,
        },
        "anthropic": {
            "api_key": None,
            "base_url": None,
        },
        "openai": {
            "api_key": None,
            "base_url": None,
        },
        "vercel": {
            "api_key": None,
            "base_url": None,
        },
    },
    "models": {
        "defaults": {
            "chat": None,
            "reasoning": None,
            "extraction": None,
        },
        "overrides": {},
    },
    "llm_settings": {
        "default_temperature": 0.7,
        "default_max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "context_limits": {
        # Provider-level defaults — only for local providers (LM Studio)
        # API providers (Anthropic, OpenAI, OpenRouter) use their own per-model capabilities
        "lmstudio": {
            "context_window": 32768,  # Conservative default for local models
            "max_output_tokens": 8192,
        },
        # Model-specific overrides (escape hatch for any provider)
        "models": {
            # Example: "claude-3-opus-20240229": {"context_window": 1000000, "max_output_tokens": 32000}
        },
    },
    "compression": {
        "enabled": True,
        "model": "anthropic:claude-haiku-4-5-20251001",
        "temperature": 0.2,
        "max_tokens": 1000,
        "max_summary_chars": 2000,
    },
    "trajectory_compression": {
        "enabled": True,
        "threshold_ratio": 0.75,       # Compress when context > 75% of limit
        "preserve_recent_rounds": 2,   # Keep last N tool-call rounds intact
        "model": "anthropic:claude-haiku-4-5-20251001",
        "temperature": 0.2,
        "max_tokens": 1500,
        "max_knowledge_chars": 3000,
    },
    "prompt_enhancement": {
        "enabled": True,
        "model": "anthropic:claude-haiku-4-5-20251001",
        "temperature": 0.7,
        "max_tokens": 1000,
        "system_prompt": "",  # Empty = use default hardcoded prompt
    },
    "preferences": {
        "default_model": None,
        "default_reasoning_strategy": "auto",
        "enable_memory_by_default": True,
    },
    "session": {
        "rolling_summary": {
            "enabled": True,
            "recent_window": 8,
            "model": "anthropic:claude-haiku-4-5-20251001",
        },
    },
}


class ConfigManager:
    """
    Manages runtime configuration with file persistence.

    Thread-safe singleton that loads config from data/config.json
    and supports partial updates with automatic persistence.
    """

    CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "config.json"

    def __init__(self):
        self._config: dict = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        """Load configuration from file or use defaults."""
        with self._lock:
            if self.CONFIG_PATH.exists():
                try:
                    with open(self.CONFIG_PATH) as f:
                        self._config = json.load(f)
                    logger.info(f"Loaded config from {self.CONFIG_PATH}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to load config: {e}, using defaults")
                    self._config = self._deep_copy(DEFAULT_CONFIG)
            else:
                logger.info("No config file found, using defaults")
                self._config = self._deep_copy(DEFAULT_CONFIG)

    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of a nested dict/list structure."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        return obj

    def _get_nested(self, data: dict, key: str, default: Any = None) -> Any:
        """Get a nested value using dot notation (e.g., 'providers.lmstudio.base_url')."""
        keys = key.split(".")
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def _set_nested(self, data: dict, key: str, value: Any) -> None:
        """Set a nested value using dot notation."""
        keys = key.split(".")
        current = data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.

        Examples:
            config.get("providers.lmstudio.base_url")
            config.get("preferences.default_model")
        """
        with self._lock:
            return self._get_nested(self._config, key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a config value by dot-notation key.

        Note: Call save() to persist changes to disk.
        """
        with self._lock:
            self._set_nested(self._config, key, value)

    def save(self) -> bool:
        """
        Persist current configuration to disk.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            try:
                # Ensure directory exists
                self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

                with open(self.CONFIG_PATH, "w") as f:
                    json.dump(self._config, f, indent=2)

                logger.info(f"Saved config to {self.CONFIG_PATH}")
                return True
            except IOError as e:
                logger.error(f"Failed to save config: {e}")
                return False

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load()

    def get_all(self) -> dict:
        """Get the entire configuration (for debugging only)."""
        with self._lock:
            return self._deep_copy(self._config)

    def get_provider_value(
        self,
        provider: str,
        key: str,
        env_var: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Get a provider config value with env var fallback.

        Args:
            provider: Provider name (lmstudio, anthropic, openai)
            key: Config key within provider (api_key, base_url, etc.)
            env_var: Environment variable to fall back to
            default: Default value if neither config nor env var is set

        Returns:
            Config value, env var value, or default (in that priority)
        """
        config_value = self.get(f"providers.{provider}.{key}")
        if config_value is not None:
            return config_value

        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                return env_value

        return default


# Global singleton instance
_config_manager: Optional[ConfigManager] = None
_config_lock = Lock()


def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager singleton."""
    global _config_manager
    with _config_lock:
        if _config_manager is None:
            _config_manager = ConfigManager()
        return _config_manager


def reload_config() -> None:
    """Reload the global config from disk."""
    manager = get_config_manager()
    manager.reload()


def get_context_limit_overrides(model_id: str, provider_name: str) -> dict[str, int]:
    """
    Get context limit overrides from config.

    This returns ONLY user-configured overrides. Provider capabilities
    should be used as the primary source, with these overrides applied on top.

    Provider-level overrides only apply to local providers (lmstudio) where
    hardware constraints may limit context. API providers (anthropic, openai,
    openrouter) already know their per-model capabilities.

    Priority:
    1. Model-specific override in context_limits.models.{model_id}
    2. Provider override in context_limits.{provider_name} (local providers only)

    Args:
        model_id: The model identifier
        provider_name: The provider name (lmstudio, anthropic, openai, etc.)

    Returns:
        Dict with overrides (may be empty, or have context_window and/or max_output_tokens)
    """
    config = get_config_manager()

    # Check for model-specific override first (works for all providers)
    model_override = config.get(f"context_limits.models.{model_id}")
    if model_override:
        return dict(model_override)

    # Provider-level overrides only apply to local providers
    LOCAL_PROVIDERS = {"lmstudio"}
    if provider_name in LOCAL_PROVIDERS:
        provider_config = config.get(f"context_limits.{provider_name}")
        if provider_config:
            return dict(provider_config)

    # No overrides — use provider's own per-model capabilities
    return {}


# Backwards compatibility alias
def get_context_limits(model_id: str, provider_name: str) -> dict[str, int]:
    """Deprecated: Use get_context_limit_overrides instead."""
    overrides = get_context_limit_overrides(model_id, provider_name)
    return {
        "context_window": overrides.get("context_window", 32768),
        "max_output_tokens": overrides.get("max_output_tokens", 4096),
    }

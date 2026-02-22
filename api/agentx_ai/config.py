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
    "preferences": {
        "default_model": None,
        "default_reasoning_strategy": "auto",
        "enable_memory_by_default": True,
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

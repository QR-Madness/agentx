"""
System Prompt Loader for AgentX.

Loads and caches system prompts from YAML configuration,
with support for variable substitution.
"""

from pathlib import Path
from typing import Any, Optional
import logging

import yaml

logger = logging.getLogger(__name__)


class SystemPromptLoader:
    """
    Loads system prompts from YAML configuration with variable substitution.

    Prompts are stored in system_prompts.yaml with a nested structure that
    maps to dot-notation keys (e.g., "extraction.system", "reasoning.cot.zero_shot").

    Usage:
        loader = get_prompt_loader()

        # Get a prompt with variable substitution
        prompt = loader.get("extraction.system",
            entity_types="Person, Organization",
            relationship_types="works_at, knows",
            condense_instruction=""
        )

        # Get a list (constants, examples)
        skip_patterns = loader.get_list("constants.skip_patterns")

        # Get examples for few-shot
        math_examples = loader.get_examples("examples.cot_math")
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            config_path: Path to YAML file. Defaults to system_prompts.yaml
                         in the same directory as this module.
        """
        self._config_path = config_path or (
            Path(__file__).parent / "system_prompts.yaml"
        )
        self._data: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        """Load prompts from YAML file (lazy loading)."""
        if self._loaded:
            return

        if not self._config_path.exists():
            logger.warning(f"System prompts file not found: {self._config_path}")
            self._loaded = True
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            logger.debug(f"Loaded system prompts from {self._config_path}")
        except Exception as e:
            logger.error(f"Failed to load system prompts: {e}")
            self._data = {}

        self._loaded = True

    def _get_nested(self, key: str) -> Any:
        """
        Get a value from nested dict using dot notation.

        Args:
            key: Dot-separated key (e.g., "extraction.system")

        Returns:
            The value at the key path, or None if not found
        """
        self._load()

        parts = key.split(".")
        current = self._data

        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None

        return current

    def get(self, key: str, default: str = "", **variables: Any) -> str:
        """
        Get a prompt by key with variable substitution.

        Variables in the prompt are in {variable_name} format and will be
        replaced with the provided keyword arguments.

        Args:
            key: Dot-notation key (e.g., "extraction.system")
            default: Default value if key not found
            **variables: Variables to substitute (e.g., entity_types="...")

        Returns:
            Rendered prompt string

        Example:
            prompt = loader.get("extraction.system",
                entity_types="Person, Organization",
                relationship_types="works_at, knows",
                condense_instruction=""
            )
        """
        template = self._get_nested(key)

        if template is None:
            logger.warning(f"Prompt not found: {key}")
            return default

        if not isinstance(template, str):
            logger.warning(f"Prompt at {key} is not a string: {type(template)}")
            return default

        # Substitute variables
        result = template
        for var_name, var_value in variables.items():
            result = result.replace(f"{{{var_name}}}", str(var_value))

        return result

    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """
        Get a list value by key.

        Useful for constants like skip_patterns, entity_types.

        Args:
            key: Dot-notation key (e.g., "constants.skip_patterns")
            default: Default value if key not found

        Returns:
            List value or default
        """
        value = self._get_nested(key)

        if value is None:
            return default if default is not None else []

        if isinstance(value, list):
            return value

        logger.warning(f"Value at {key} is not a list: {type(value)}")
        return default if default is not None else []

    def get_examples(self, key: str) -> list[dict[str, str]]:
        """
        Get few-shot examples by key.

        Args:
            key: Dot-notation key (e.g., "examples.cot_math")

        Returns:
            List of example dictionaries with question/reasoning/answer keys
        """
        return self.get_list(key, default=[])

    def get_dict(self, key: str, default: Optional[dict] = None) -> dict:
        """
        Get a dictionary value by key.

        Args:
            key: Dot-notation key
            default: Default value if key not found

        Returns:
            Dictionary value or default
        """
        value = self._get_nested(key)

        if value is None:
            return default if default is not None else {}

        if isinstance(value, dict):
            return value

        logger.warning(f"Value at {key} is not a dict: {type(value)}")
        return default if default is not None else {}

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return self._get_nested(key) is not None

    def list_keys(self, prefix: str = "") -> list[str]:
        """
        List all available keys under a prefix.

        Args:
            prefix: Key prefix to list under (empty for root)

        Returns:
            List of keys
        """
        self._load()

        if not prefix:
            return list(self._data.keys())

        parent = self._get_nested(prefix)
        if isinstance(parent, dict):
            return [f"{prefix}.{k}" for k in parent.keys()]

        return []

    def reload(self) -> None:
        """Force reload from YAML file."""
        self._loaded = False
        self._data = {}
        self._load()


# Singleton instance
_loader: Optional[SystemPromptLoader] = None


def get_prompt_loader() -> SystemPromptLoader:
    """
    Get the global SystemPromptLoader instance.

    The loader is created on first call and cached for subsequent calls.

    Returns:
        SystemPromptLoader singleton instance
    """
    global _loader
    if _loader is None:
        _loader = SystemPromptLoader()
    return _loader


def get_prompt(key: str, default: str = "", **variables: Any) -> str:
    """
    Convenience function to get a prompt directly.

    Equivalent to: get_prompt_loader().get(key, default, **variables)

    Args:
        key: Dot-notation key
        default: Default if not found
        **variables: Variables to substitute

    Returns:
        Rendered prompt string
    """
    return get_prompt_loader().get(key, default, **variables)


def get_prompt_list(key: str, default: Optional[list] = None) -> list:
    """
    Convenience function to get a list directly.

    Equivalent to: get_prompt_loader().get_list(key, default)

    Args:
        key: Dot-notation key
        default: Default if not found

    Returns:
        List value
    """
    return get_prompt_loader().get_list(key, default)

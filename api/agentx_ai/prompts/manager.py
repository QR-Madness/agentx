"""
Prompt Manager for AgentX.

Manages prompt composition, profile selection, and prompt storage.
"""

import logging
from pathlib import Path

import yaml

from .models import (
    GlobalPrompt,
    PromptConfig,
    PromptProfile,
    PromptSection,
    StructuredOutputConfig,
)
from .defaults import (
    DEFAULT_GLOBAL_PROMPT,
    DEFAULT_PROFILES,
    DEFAULT_SECTIONS,
)
from .mcp_prompt import generate_mcp_tools_prompt

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt composition and storage.
    
    Handles:
    - Loading/saving prompts from YAML configuration
    - Profile management (CRUD operations)
    - Composing final system prompts from components
    - MCP tools prompt generation
    """
    
    def __init__(self, config_path: Path | None = None):
        """
        Initialize the PromptManager.
        
        Args:
            config_path: Optional path to prompts.yaml for custom prompts
        """
        self.config_path = config_path
        
        # Initialize with defaults
        self._global_prompt = DEFAULT_GLOBAL_PROMPT
        self._profiles: dict[str, PromptProfile] = {p.id: p for p in DEFAULT_PROFILES}
        self._sections: dict[str, PromptSection] = {s.id: s for s in DEFAULT_SECTIONS}
        
        # Load custom configuration if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
    
    def _load_config(self, path: Path) -> None:
        """Load prompts configuration from YAML file."""
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            
            if not config:
                return
            
            # Load global prompt
            if "global_prompt" in config:
                self._global_prompt = GlobalPrompt(**config["global_prompt"])
            
            # Load custom sections
            if "sections" in config:
                for section_data in config["sections"]:
                    section = PromptSection(**section_data)
                    self._sections[section.id] = section
            
            # Load custom profiles
            if "profiles" in config:
                for profile_data in config["profiles"]:
                    # Resolve section references
                    section_ids = profile_data.pop("section_ids", [])
                    sections = [
                        self._sections[sid] 
                        for sid in section_ids 
                        if sid in self._sections
                    ]
                    profile_data["sections"] = sections
                    profile = PromptProfile(**profile_data)
                    self._profiles[profile.id] = profile
            
            logger.info(f"Loaded prompt configuration from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load prompt configuration: {e}")
    
    def save_config(self, path: Path | None = None) -> None:
        """Save current configuration to YAML file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")
        
        config = {
            "global_prompt": {
                "content": self._global_prompt.content,
                "enabled": self._global_prompt.enabled,
            },
            "sections": [
                {
                    "id": s.id,
                    "name": s.name,
                    "type": s.type,
                    "content": s.content,
                    "enabled": s.enabled,
                    "order": s.order,
                }
                for s in self._sections.values()
            ],
            "profiles": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "section_ids": [s.id for s in p.sections],
                    "is_default": p.is_default,
                }
                for p in self._profiles.values()
            ],
        }
        
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved prompt configuration to {save_path}")
    
    # =========================================================================
    # Global Prompt
    # =========================================================================
    
    def get_global_prompt(self) -> GlobalPrompt:
        """Get the current global prompt."""
        return self._global_prompt
    
    def set_global_prompt(self, content: str, enabled: bool = True) -> GlobalPrompt:
        """Set the global prompt content."""
        self._global_prompt = GlobalPrompt(content=content, enabled=enabled)
        return self._global_prompt
    
    # =========================================================================
    # Profiles
    # =========================================================================
    
    def list_profiles(self) -> list[PromptProfile]:
        """List all available profiles."""
        return list(self._profiles.values())
    
    def get_profile(self, profile_id: str) -> PromptProfile | None:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)
    
    def get_default_profile(self) -> PromptProfile | None:
        """Get the default profile."""
        for profile in self._profiles.values():
            if profile.is_default:
                return profile
        # Fall back to first profile
        return next(iter(self._profiles.values()), None)
    
    def create_profile(self, profile: PromptProfile) -> PromptProfile:
        """Create or update a profile."""
        self._profiles[profile.id] = profile
        return profile
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by ID."""
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            return True
        return False
    
    def set_default_profile(self, profile_id: str) -> bool:
        """Set a profile as the default."""
        if profile_id not in self._profiles:
            return False
        
        for pid, profile in self._profiles.items():
            # Create new profile with updated is_default flag
            self._profiles[pid] = PromptProfile(
                id=profile.id,
                name=profile.name,
                description=profile.description,
                sections=profile.sections,
                is_default=(pid == profile_id),
            )
        return True
    
    # =========================================================================
    # Sections
    # =========================================================================
    
    def list_sections(self) -> list[PromptSection]:
        """List all available sections."""
        return list(self._sections.values())
    
    def get_section(self, section_id: str) -> PromptSection | None:
        """Get a section by ID."""
        return self._sections.get(section_id)
    
    def create_section(self, section: PromptSection) -> PromptSection:
        """Create or update a section."""
        self._sections[section.id] = section
        return section
    
    def delete_section(self, section_id: str) -> bool:
        """Delete a section by ID."""
        if section_id in self._sections:
            del self._sections[section_id]
            return True
        return False
    
    # =========================================================================
    # Prompt Composition
    # =========================================================================

    def _ensure_layers_migrated(self) -> None:
        """One-time import of a legacy customized global prompt into the layer stack.

        First boot is normally a no-op (no legacy customization → the built-in
        layers already reproduce the default). But if a legacy ``prompts.yaml``
        gave us a global prompt that differs from the shipped default, preserve
        that user work as the reserved legacy-global custom layer rather than
        silently dropping it now that composition is stack-sourced. Guarded by a
        separate top-level config key so it runs exactly once and never collides
        with the per-layer delta dict under ``prompts.layers``.
        """
        from ..config import get_config_manager

        config = get_config_manager()
        if config.get("prompts.layers_migrated", False):
            return
        try:
            legacy = (self._global_prompt.content or "").strip()
            default = (DEFAULT_GLOBAL_PROMPT.content or "").strip()
            if legacy and legacy != default:
                from .layers import get_layer_store
                get_layer_store().set_singleton_override(legacy)
                logger.info("Migrated legacy global prompt into the layer stack")
            config.set("prompts.layers_migrated", True)
            config.save()
        except Exception as e:
            logger.warning(f"Prompt-layer migration skipped: {e}")

    def compose_prompt(
        self,
        profile_id: str | None = None,
        mcp_tools: list[dict] | None = None,
        additional_context: str | None = None,
        structured_output: StructuredOutputConfig | None = None,
        agent_name: str | None = None,
        agent_system_prompt: str | None = None,
    ) -> PromptConfig:
        """
        Compose a complete prompt configuration.

        Args:
            profile_id: Profile to use (None for default)
            mcp_tools: List of available MCP tools
            additional_context: Additional context to append
            structured_output: Structured output configuration
            agent_name: Agent name to inject as "Your name is {name}."
            agent_system_prompt: Agent-specific custom system prompt

        Returns:
            Complete PromptConfig ready for use
        """
        self._ensure_layers_migrated()

        # Global content now comes from the durable, layered prompt stack
        # (built-in defaults overlaid with the user's persisted overrides),
        # not the in-memory _global_prompt. See prompts/layers.py.
        from .layers import get_layer_store
        stack = get_layer_store().compose()
        global_prompt = GlobalPrompt(content=stack, enabled=bool(stack.strip()))

        # Prompt-profile sections only attach for an explicit, non-default
        # selection. The default ("General") profile's sections now live in the
        # stack, so auto-selecting it would double-inject them.
        profile = None
        if profile_id:
            candidate = self.get_profile(profile_id)
            if candidate is not None and not candidate.is_default:
                profile = candidate

        # Generate MCP tools prompt if tools provided
        mcp_tools_prompt = None
        if mcp_tools:
            mcp_tools_prompt = generate_mcp_tools_prompt(mcp_tools)

        return PromptConfig(
            global_prompt=global_prompt,
            profile=profile,
            mcp_tools_prompt=mcp_tools_prompt,
            additional_context=additional_context,
            structured_output=structured_output,
            agent_name=agent_name,
            agent_system_prompt=agent_system_prompt,
        )
    
    def get_system_prompt(
        self,
        profile_id: str | None = None,
        mcp_tools: list[dict] | None = None,
        additional_context: str | None = None,
        agent_name: str | None = None,
        agent_system_prompt: str | None = None,
    ) -> str:
        """
        Get the composed system prompt string.

        Convenience method that returns just the final string.

        Args:
            profile_id: Profile to use (None for default)
            mcp_tools: List of available MCP tools
            additional_context: Additional context to append
            agent_name: Agent name to inject as "Your name is {name}."
            agent_system_prompt: Agent-specific custom system prompt
        """
        config = self.compose_prompt(
            profile_id=profile_id,
            mcp_tools=mcp_tools,
            additional_context=additional_context,
            agent_name=agent_name,
            agent_system_prompt=agent_system_prompt,
        )
        return config.compose_system_prompt()


# =============================================================================
# Singleton instance
# =============================================================================

_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get or create the global PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        # Look for prompts.yaml in the project root
        config_path = Path(__file__).parent.parent.parent.parent / "prompts.yaml"
        _prompt_manager = PromptManager(config_path if config_path.exists() else None)
        logger.info("PromptManager initialized")
    return _prompt_manager

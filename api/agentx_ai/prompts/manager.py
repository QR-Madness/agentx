"""
Prompt Manager for AgentX.

Manages prompt composition, profile selection, and prompt storage.
"""

import logging
from pathlib import Path
from typing import Optional

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
    
    def __init__(self, config_path: Optional[Path] = None):
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
            with open(path, "r") as f:
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
    
    def save_config(self, path: Optional[Path] = None) -> None:
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
    
    def get_profile(self, profile_id: str) -> Optional[PromptProfile]:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)
    
    def get_default_profile(self) -> Optional[PromptProfile]:
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
    
    def get_section(self, section_id: str) -> Optional[PromptSection]:
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
    
    def compose_prompt(
        self,
        profile_id: Optional[str] = None,
        mcp_tools: Optional[list[dict]] = None,
        additional_context: Optional[str] = None,
        structured_output: Optional[StructuredOutputConfig] = None,
    ) -> PromptConfig:
        """
        Compose a complete prompt configuration.
        
        Args:
            profile_id: Profile to use (None for default)
            mcp_tools: List of available MCP tools
            additional_context: Additional context to append
            structured_output: Structured output configuration
            
        Returns:
            Complete PromptConfig ready for use
        """
        # Get profile
        profile = None
        if profile_id:
            profile = self.get_profile(profile_id)
        if not profile:
            profile = self.get_default_profile()
        
        # Generate MCP tools prompt if tools provided
        mcp_tools_prompt = None
        if mcp_tools:
            mcp_tools_prompt = generate_mcp_tools_prompt(mcp_tools)
        
        return PromptConfig(
            global_prompt=self._global_prompt,
            profile=profile,
            mcp_tools_prompt=mcp_tools_prompt,
            additional_context=additional_context,
            structured_output=structured_output,
        )
    
    def get_system_prompt(
        self,
        profile_id: Optional[str] = None,
        mcp_tools: Optional[list[dict]] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Get the composed system prompt string.
        
        Convenience method that returns just the final string.
        """
        config = self.compose_prompt(
            profile_id=profile_id,
            mcp_tools=mcp_tools,
            additional_context=additional_context,
        )
        return config.compose_system_prompt()


# =============================================================================
# Singleton instance
# =============================================================================

_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get or create the global PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        # Look for prompts.yaml in the project root
        config_path = Path(__file__).parent.parent.parent.parent / "prompts.yaml"
        _prompt_manager = PromptManager(config_path if config_path.exists() else None)
        logger.info("PromptManager initialized")
    return _prompt_manager

"""
Pydantic models for the prompt management system.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SectionType(str, Enum):
    """Types of prompt sections."""
    PERSONA = "persona"
    TASK = "task"
    FORMAT = "format"
    CONSTRAINTS = "constraints"
    EXAMPLES = "examples"
    CONTEXT = "context"
    CUSTOM = "custom"


class PromptSection(BaseModel):
    """
    A modular section of a prompt.
    
    Sections can be enabled/disabled and reordered within a profile.
    """
    id: str = Field(..., description="Unique identifier for the section")
    name: str = Field(..., description="Display name")
    type: SectionType = Field(default=SectionType.CUSTOM, description="Section type for categorization")
    content: str = Field(..., description="The prompt text content")
    enabled: bool = Field(default=True, description="Whether this section is active")
    order: int = Field(default=0, description="Sort order within the profile")
    
    class Config:
        use_enum_values = True


class PromptProfile(BaseModel):
    """
    A named collection of prompt sections.
    
    Profiles represent different use cases (e.g., "Developer", "Creative Writer").
    """
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Description of when to use this profile")
    sections: list[PromptSection] = Field(default_factory=list, description="Ordered list of sections")
    is_default: bool = Field(default=False, description="Whether this is the default profile")
    
    def get_enabled_sections(self) -> list[PromptSection]:
        """Get only enabled sections, sorted by order."""
        return sorted(
            [s for s in self.sections if s.enabled],
            key=lambda s: s.order
        )
    
    def compose(self) -> str:
        """Compose all enabled sections into a single prompt string."""
        sections = self.get_enabled_sections()
        if not sections:
            return ""
        return "\n\n".join(s.content for s in sections)


class GlobalPrompt(BaseModel):
    """
    The global prompt applied to all interactions.
    
    This defines the core persona and behavior that applies
    regardless of which profile is selected.
    """
    content: str = Field(..., description="The global prompt content")
    enabled: bool = Field(default=True, description="Whether the global prompt is active")


class StructuredOutputConfig(BaseModel):
    """Configuration for structured/constrained output."""
    enabled: bool = Field(default=False, description="Whether structured output is enabled")
    format: Optional[str] = Field(None, description="Output format: 'json', 'json_schema', etc.")
    output_schema: Optional[str] = Field(None, description="JSON schema for structured output")
    strict: bool = Field(default=False, description="Whether to enforce strict schema compliance")


class PromptConfig(BaseModel):
    """
    Complete prompt configuration for a request.
    
    Combines global prompt, profile, and any request-specific overrides.
    """
    global_prompt: Optional[GlobalPrompt] = None
    profile: Optional[PromptProfile] = None
    mcp_tools_prompt: Optional[str] = None
    structured_output: Optional[StructuredOutputConfig] = None
    
    # Request-specific overrides
    additional_context: Optional[str] = None
    system_override: Optional[str] = None  # Completely replace system prompt
    
    def compose_system_prompt(self) -> str:
        """
        Compose the final system prompt from all components.
        
        Order:
        1. Global prompt (persona, core behavior)
        2. MCP tools prompt (if tools are available)
        3. Profile sections (in defined order)
        4. Additional context (if provided)
        """
        if self.system_override:
            return self.system_override
        
        parts = []
        
        # Global prompt
        if self.global_prompt and self.global_prompt.enabled:
            parts.append(self.global_prompt.content)
        
        # MCP tools prompt
        if self.mcp_tools_prompt:
            parts.append(self.mcp_tools_prompt)
        
        # Profile sections
        if self.profile:
            profile_content = self.profile.compose()
            if profile_content:
                parts.append(profile_content)
        
        # Additional context
        if self.additional_context:
            parts.append(self.additional_context)
        
        return "\n\n".join(parts) if parts else ""

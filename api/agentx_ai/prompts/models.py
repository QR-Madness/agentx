"""
Pydantic models for the prompt management system.
"""

from datetime import datetime
from enum import Enum
from typing import Literal
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


class TemplateType(str, Enum):
    """Types of prompt templates."""
    SYSTEM = "system"      # Full system prompts
    USER = "user"          # User message templates
    SNIPPET = "snippet"    # Reusable text snippets


class PromptTemplate(BaseModel):
    """
    A mutable prompt template with rollback capability.

    Templates provide reusable prompt content that can be edited but always
    maintain a default_content for rollback. Used by the Prompt Library feature.
    """
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    content: str = Field(..., description="Current editable content")
    default_content: str = Field(..., description="Original content for rollback")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")
    placeholders: list[str] = Field(
        default_factory=list,
        description="Variable placeholders e.g., ['agent_name', 'context']"
    )
    type: TemplateType = Field(default=TemplateType.SNIPPET, description="Template type")
    is_builtin: bool = Field(default=False, description="Protected default template")
    description: str | None = Field(None, description="Usage description")
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default_factory=datetime.utcnow)

    def has_modifications(self) -> bool:
        """Check if content differs from default."""
        return self.content != self.default_content

    def reset_to_default(self) -> None:
        """Reset content to default_content."""
        self.content = self.default_content
        self.updated_at = datetime.utcnow()

    def render(self, **variables: str) -> str:
        """Render template content with variable substitution."""
        result = self.content
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    class Config:
        use_enum_values = True


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
    description: str | None = Field(None, description="Description of when to use this profile")
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


class PromptLayer(BaseModel):
    """One editable block in the layered system-prompt stack ("Prompt Stack").

    The composed system prompt is an ordered stack of these. Each layer carries
    two content fields:

      * ``default`` — shipped in code, owned by the app, updated by releases
        (the *sidecar*). Only built-in layers have one.
      * ``override`` — the user's edit. Optional.

    Effective content is the override if present, else the default. So untouched
    built-ins ride the default (release improvements flow in automatically), while
    edited layers are pinned to the user's text and never silently overwritten.
    ``base_version`` records the ``default_version`` the override was seeded from,
    so we can detect (and diff) when a release changes the default underneath an
    override.
    """

    id: str = Field(..., description="Stable slug (built-ins use well-known ids)")
    title: str = Field(..., description="Display name")
    kind: Literal["builtin", "custom"] = Field(
        default="custom", description="'builtin' = shipped (has a default sidecar); 'custom' = user-created"
    )
    default: str | None = Field(
        default=None, description="Shipped default content (built-ins only); the sidecar"
    )
    default_version: int = Field(
        default=1, description="Bumped when the shipped default changes (drives update detection)"
    )
    override: str | None = Field(
        default=None, description="User edit; None means 'use the default'"
    )
    base_version: int | None = Field(
        default=None, description="default_version the override was seeded/synced from"
    )
    enabled: bool = Field(default=True, description="Whether this layer is in the composed stack")
    order: int = Field(default=0, description="Position in the stack (ascending)")

    @property
    def effective(self) -> str:
        """The content that actually ships: override if set, else default."""
        if self.override is not None:
            return self.override
        return self.default or ""

    @property
    def modified(self) -> bool:
        """The user has an override that differs from the shipped default."""
        return self.override is not None and self.override != (self.default or "")

    @property
    def update_available(self) -> bool:
        """A release changed the default underneath the user's override — surface a
        diff rather than clobbering their work."""
        return (
            self.kind == "builtin"
            and self.override is not None
            and self.base_version is not None
            and self.default_version > self.base_version
        )


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
    format: str | None = Field(None, description="Output format: 'json', 'json_schema', etc.")
    output_schema: str | None = Field(None, description="JSON schema for structured output")
    strict: bool = Field(default=False, description="Whether to enforce strict schema compliance")


class PromptConfig(BaseModel):
    """
    Complete prompt configuration for a request.

    Combines global prompt, profile, and any request-specific overrides.
    """
    global_prompt: GlobalPrompt | None = None
    profile: PromptProfile | None = None
    mcp_tools_prompt: str | None = None
    structured_output: StructuredOutputConfig | None = None

    # Request-specific overrides
    additional_context: str | None = None
    system_override: str | None = None  # Completely replace system prompt

    # Agent identity
    agent_name: str | None = None  # Name to inject as "Your name is {name}."
    agent_system_prompt: str | None = None  # Agent-specific custom instructions

    def compose_system_prompt(self) -> str:
        """
        Compose the final system prompt from all components.

        Order:
        1. Agent name introduction (if provided)
        2. Agent-specific system prompt (if provided)
        3. Global prompt (persona, core behavior)
        4. MCP tools prompt (if tools are available)
        5. Profile sections (in defined order)
        6. Additional context (if provided)
        """
        if self.system_override:
            return self.system_override

        parts = []

        # Agent name introduction
        if self.agent_name:
            parts.append(f"Your name is {self.agent_name}.")

        # Agent-specific custom system prompt
        if self.agent_system_prompt:
            parts.append(self.agent_system_prompt)

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

        composed = "\n\n".join(parts) if parts else ""
        # Substitute the whitelisted `{token}` placeholders ({agent_name}/{date}/{time}).
        from .placeholders import substitute_placeholders
        return substitute_placeholders(composed, agent_name=self.agent_name or "")

"""
Pydantic models for agent profiles.

Agent profiles define customizable agent identities with names,
model settings, and behavior configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies for agent task execution."""
    AUTO = "auto"
    COT = "cot"  # Chain of Thought
    TOT = "tot"  # Tree of Thought
    REACT = "react"
    REFLECTION = "reflection"


class AgentProfile(BaseModel):
    """
    Configuration profile for an agent persona.

    Defines the agent's identity, default settings, and behavior configuration.
    The agent name is injected into system prompts as "Your name is {name}."
    """
    # Identity
    id: str = Field(..., description="Unique identifier for the profile")
    name: str = Field(..., description="Display name for the agent (used in prompts and UI)")
    avatar: Optional[str] = Field(None, description="Avatar icon name (e.g., 'sparkles', 'brain')")
    description: Optional[str] = Field(None, description="Description of this profile's purpose")

    # Model settings
    default_model: Optional[str] = Field(None, description="Default model to use for this profile")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature (0.0-2.0)")

    # Prompt configuration
    prompt_profile_id: Optional[str] = Field(
        None,
        description="ID of the PromptProfile to use for system prompt composition"
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt for this agent (prepended to conversations)"
    )

    # Behavior settings
    reasoning_strategy: ReasoningStrategy = Field(
        ReasoningStrategy.AUTO,
        description="Default reasoning strategy for task execution"
    )
    enable_memory: bool = Field(True, description="Whether to use the memory system")
    memory_channel: str = Field("_global", description="Memory channel to use for this profile")
    enable_tools: bool = Field(True, description="Whether to enable MCP tools")

    # Metadata
    is_default: bool = Field(False, description="Whether this is the default profile")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

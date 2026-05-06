"""
Pydantic models for agent profiles.

Agent profiles define customizable agent identities with names,
model settings, and behavior configuration.
"""

from datetime import datetime
from enum import Enum
import random
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Docker-style agent ID generation
# ---------------------------------------------------------------------------

_ADJECTIVES = [
    "bold", "brave", "bright", "calm", "clever", "cool", "cosmic", "crisp",
    "daring", "eager", "fast", "fierce", "gentle", "giddy", "gleaming",
    "golden", "grand", "happy", "hidden", "humble", "keen", "kind", "lively",
    "lucky", "merry", "mighty", "nimble", "noble", "patient", "plucky",
    "proud", "quiet", "radiant", "rising", "sharp", "silent", "smooth",
    "steady", "swift", "vivid", "warm", "wild", "witty", "zen",
]

_NOUNS = [
    "aurora", "beacon", "breeze", "cedar", "comet", "coral", "crane",
    "crystal", "dawn", "dune", "echo", "ember", "falcon", "fern", "flare",
    "frost", "gale", "grove", "hawk", "iris", "jade", "lark", "lotus",
    "maple", "meadow", "nebula", "oak", "olive", "osprey", "pearl",
    "phoenix", "pine", "quartz", "reef", "ridge", "sage", "spark",
    "spruce", "summit", "thistle", "tide", "vale", "wren",
]


def generate_agent_id() -> str:
    """Generate a Docker-style human-friendly agent ID (e.g., 'bold-cosmic-falcon')."""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


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
    agent_id: str = Field(default_factory=generate_agent_id, description="Unique human-friendly agent identifier (Docker-style)")
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
    # Phase 18.2: per-profile tool gating. Tool names are fully-qualified ("server.tool").
    allowed_tools: Optional[list[str]] = Field(
        None,
        description="If set, only these fully-qualified tools (server.tool) are exposed to this agent. None = all enabled.",
    )
    blocked_tools: list[str] = Field(
        default_factory=list,
        description="Fully-qualified tools (server.tool) explicitly hidden from this agent. Wins over allowed_tools.",
    )

    # Metadata
    is_default: bool = Field(False, description="Whether this is the default profile")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    @property
    def self_channel(self) -> str:
        """Memory channel for this agent's self-knowledge."""
        return f"_self_{self.agent_id}"

    class Config:
        use_enum_values = True

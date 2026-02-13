"""
Prompt Management System for AgentX.

Provides a hierarchical prompt composition system with:
- Global prompts (persona, core behavior)
- Prompt profiles (named collections of sections)
- Prompt sections (modular, reorderable prompt pieces)
- MCP tools prompts (auto-generated from available tools)
"""

from .models import (
    PromptSection,
    PromptProfile,
    GlobalPrompt,
    PromptConfig,
    StructuredOutputConfig,
)
from .manager import PromptManager, get_prompt_manager
from .mcp_prompt import generate_mcp_tools_prompt

__all__ = [
    "PromptSection",
    "PromptProfile", 
    "GlobalPrompt",
    "PromptConfig",
    "StructuredOutputConfig",
    "PromptManager",
    "get_prompt_manager",
    "generate_mcp_tools_prompt",
]

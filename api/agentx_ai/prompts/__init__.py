"""
Prompt Management System for AgentX.

Provides a hierarchical prompt composition system with:
- Global prompts (persona, core behavior)
- Prompt profiles (named collections of sections)
- Prompt sections (modular, reorderable prompt pieces)
- Prompt templates (mutable templates with rollback capability)
- MCP tools prompts (auto-generated from available tools)
"""

from .models import (
    PromptSection,
    PromptProfile,
    GlobalPrompt,
    PromptConfig,
    StructuredOutputConfig,
    PromptTemplate,
    TemplateType,
)
from .manager import PromptManager, get_prompt_manager
from .template_manager import PromptTemplateManager, get_template_manager
from .mcp_prompt import generate_mcp_tools_prompt
from .loader import (
    SystemPromptLoader,
    get_prompt_loader,
    get_prompt,
    get_prompt_list,
)

__all__ = [
    # Models
    "PromptSection",
    "PromptProfile",
    "GlobalPrompt",
    "PromptConfig",
    "StructuredOutputConfig",
    "PromptTemplate",
    "TemplateType",
    # Manager
    "PromptManager",
    "get_prompt_manager",
    "PromptTemplateManager",
    "get_template_manager",
    "generate_mcp_tools_prompt",
    # System prompt loader
    "SystemPromptLoader",
    "get_prompt_loader",
    "get_prompt",
    "get_prompt_list",
]

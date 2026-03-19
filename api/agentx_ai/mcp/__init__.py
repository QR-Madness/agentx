"""
MCP (Model Context Protocol) Client Module

This module provides the infrastructure for AgentX to consume tools from
external MCP servers, enabling integration with filesystem access, GitHub,
databases, and custom servers through a standardized protocol.

Also exposes internal tools (like read_stored_output) via the same interface.
"""

from .client import MCPClientManager, get_mcp_manager
from .server_registry import ServerRegistry, ServerConfig
from .tool_executor import ToolExecutor
from .internal_tools import (
    INTERNAL_SERVER_NAME,
    get_internal_tools,
    execute_internal_tool,
    register_tool,
)

__all__ = [
    "MCPClientManager",
    "ServerRegistry",
    "ServerConfig",
    "ToolExecutor",
    "get_mcp_manager",
    # Internal tools
    "INTERNAL_SERVER_NAME",
    "get_internal_tools",
    "execute_internal_tool",
    "register_tool",
]

"""
MCP (Model Context Protocol) Client Module

This module provides the infrastructure for AgentX to consume tools from
external MCP servers, enabling integration with filesystem access, GitHub,
databases, and custom servers through a standardized protocol.
"""

from .client import MCPClientManager
from .server_registry import ServerRegistry, ServerConfig
from .tool_executor import ToolExecutor

__all__ = [
    "MCPClientManager",
    "ServerRegistry",
    "ServerConfig",
    "ToolExecutor",
]

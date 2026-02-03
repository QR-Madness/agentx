"""
Tool Executor: Execute tools on connected MCP servers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from mcp import ClientSession
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

logger = logging.getLogger(__name__)

# Type for tool usage recording callback
ToolUsageRecorder = Callable[[str, dict, Any, bool, int, Optional[str]], None]


@dataclass
class ToolInfo:
    """Information about an available MCP tool."""
    
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str
    
    @classmethod
    def from_mcp_tool(cls, tool: Tool, server_name: str) -> "ToolInfo":
        """Create ToolInfo from MCP Tool object."""
        return cls(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
            server_name=server_name,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
        }


@dataclass
class ToolResult:
    """Result of executing an MCP tool."""
    
    success: bool
    content: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    is_error: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "is_error": self.is_error,
        }
    
    @property
    def text(self) -> str:
        """Get concatenated text content."""
        parts = []
        for item in self.content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)


class ToolExecutor:
    """Executes tools on MCP server sessions."""
    
    def __init__(self):
        self._tool_cache: dict[str, list[ToolInfo]] = {}  # server_name -> tools
        self._usage_recorder: Optional[ToolUsageRecorder] = None
    
    def set_usage_recorder(self, recorder: ToolUsageRecorder) -> None:
        """
        Set a callback function for recording tool usage.
        
        The callback receives: tool_name, input, output, success, latency_ms, error_message
        """
        self._usage_recorder = recorder
    
    async def discover_tools(self, session: ClientSession, server_name: str) -> list[ToolInfo]:
        """
        Discover available tools from an MCP server.
        
        Args:
            session: Active MCP client session
            server_name: Name of the server for tracking
            
        Returns:
            List of available tools
        """
        try:
            result = await session.list_tools()
            tools = [
                ToolInfo.from_mcp_tool(tool, server_name)
                for tool in result.tools
            ]
            
            # Cache the tools
            self._tool_cache[server_name] = tools
            
            logger.info(f"Discovered {len(tools)} tools from server '{server_name}'")
            for tool in tools:
                logger.debug(f"  - {tool.name}: {tool.description[:50]}...")
            
            return tools
        except Exception as e:
            logger.error(f"Failed to discover tools from '{server_name}': {e}")
            return []
    
    async def execute(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool on an MCP server.
        
        Args:
            session: Active MCP client session
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            ToolResult with execution results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing tool '{tool_name}' with args: {arguments}")
            
            result = await session.call_tool(tool_name, arguments)
            
            # Convert content to serializable format
            content = []
            for item in result.content:
                if isinstance(item, TextContent):
                    content.append({"type": "text", "text": item.text})
                elif isinstance(item, ImageContent):
                    content.append({
                        "type": "image",
                        "data": item.data,
                        "mimeType": item.mimeType,
                    })
                elif isinstance(item, EmbeddedResource):
                    content.append({
                        "type": "resource",
                        "resource": str(item.resource),
                    })
                else:
                    content.append({"type": "unknown", "data": str(item)})
            
            is_error = getattr(result, 'isError', False)
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Record tool usage if recorder is set
            if self._usage_recorder:
                try:
                    self._usage_recorder(
                        tool_name,
                        arguments,
                        content,
                        not is_error,
                        latency_ms,
                        None,
                    )
                except Exception as record_error:
                    logger.warning(f"Failed to record tool usage: {record_error}")
            
            return ToolResult(
                success=not is_error,
                content=content,
                is_error=is_error,
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Tool execution failed: {e}")
            
            # Record failed tool usage if recorder is set
            if self._usage_recorder:
                try:
                    self._usage_recorder(
                        tool_name,
                        arguments,
                        None,
                        False,
                        latency_ms,
                        str(e),
                    )
                except Exception as record_error:
                    logger.warning(f"Failed to record tool usage: {record_error}")
            
            return ToolResult(
                success=False,
                error=str(e),
                is_error=True,
            )
    
    def get_cached_tools(self, server_name: str | None = None) -> list[ToolInfo]:
        """
        Get cached tools.
        
        Args:
            server_name: If provided, get tools from specific server only.
                        If None, get tools from all servers.
        """
        if server_name:
            return self._tool_cache.get(server_name, [])
        
        all_tools = []
        for tools in self._tool_cache.values():
            all_tools.extend(tools)
        return all_tools
    
    def clear_cache(self, server_name: str | None = None) -> None:
        """Clear tool cache."""
        if server_name:
            self._tool_cache.pop(server_name, None)
        else:
            self._tool_cache.clear()
    
    def find_tool(self, tool_name: str) -> ToolInfo | None:
        """Find a tool by name across all cached servers."""
        for tools in self._tool_cache.values():
            for tool in tools:
                if tool.name == tool_name:
                    return tool
        return None

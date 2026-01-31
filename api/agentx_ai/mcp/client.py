"""
MCP Client Manager: Core class for managing MCP server connections.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from mcp import ClientSession
from mcp.types import Resource

from .server_registry import ServerConfig, ServerRegistry, TransportType
from .tool_executor import ToolExecutor, ToolInfo, ToolResult
from .transports.stdio import StdioTransport
from .transports.sse import SSETransport

logger = logging.getLogger(__name__)


@dataclass
class ResourceInfo:
    """Information about an MCP resource."""
    
    uri: str
    name: str
    description: str | None
    mime_type: str | None
    server_name: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "server_name": self.server_name,
        }


@dataclass
class ServerConnection:
    """Represents an active connection to an MCP server."""
    
    name: str
    session: ClientSession
    config: ServerConfig
    tools: list[ToolInfo] = field(default_factory=list)
    resources: list[ResourceInfo] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.config.transport.value,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
        }


class MCPClientManager:
    """
    Manager for MCP client connections.
    
    Handles connecting to multiple MCP servers via different transports
    (stdio, SSE) and provides unified access to their tools and resources.
    
    Example usage:
        manager = MCPClientManager()
        
        # Connect to a server
        async with manager.connect_server(config) as connection:
            # List available tools
            tools = connection.tools
            
            # Execute a tool
            result = await manager.call_tool(connection.name, "read_file", {"path": "/tmp/test.txt"})
    """
    
    def __init__(self, registry: ServerRegistry | None = None):
        """
        Initialize the MCP client manager.
        
        Args:
            registry: Server registry with pre-configured servers.
                     If None, a new empty registry is created.
        """
        self.registry = registry or ServerRegistry()
        self.tool_executor = ToolExecutor()
        self._stdio_transport = StdioTransport()
        self._sse_transport = SSETransport()
        self._active_connections: dict[str, ServerConnection] = {}
    
    @asynccontextmanager
    async def connect_server(
        self,
        config: ServerConfig | str,
    ) -> AsyncGenerator[ServerConnection, None]:
        """
        Connect to an MCP server.
        
        Args:
            config: ServerConfig object or name of a registered server
            
        Yields:
            ServerConnection with active session and discovered tools/resources
        """
        # Resolve config if string name provided
        if isinstance(config, str):
            resolved = self.registry.get(config)
            if not resolved:
                raise ValueError(f"Server '{config}' not found in registry")
            config = resolved
        
        config.validate()
        
        if config.transport == TransportType.STDIO:
            async with self._connect_stdio(config) as connection:
                yield connection
        elif config.transport == TransportType.SSE:
            async with self._connect_sse(config) as connection:
                yield connection
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")
    
    @asynccontextmanager
    async def _connect_stdio(self, config: ServerConfig) -> AsyncGenerator[ServerConnection, None]:
        """Connect via stdio transport."""
        async with self._stdio_transport.connect(
            name=config.name,
            command=config.command,
            args=config.args,
            env=config.resolve_env(),
        ) as session:
            connection = await self._setup_connection(session, config)
            try:
                yield connection
            finally:
                await self._cleanup_connection(connection)
    
    @asynccontextmanager
    async def _connect_sse(self, config: ServerConfig) -> AsyncGenerator[ServerConnection, None]:
        """Connect via SSE transport."""
        async with self._sse_transport.connect(
            name=config.name,
            url=config.url,
            headers=config.headers,
        ) as session:
            connection = await self._setup_connection(session, config)
            try:
                yield connection
            finally:
                await self._cleanup_connection(connection)
    
    async def _setup_connection(
        self,
        session: ClientSession,
        config: ServerConfig,
    ) -> ServerConnection:
        """Set up a new connection with tool/resource discovery."""
        # Discover tools
        tools = await self.tool_executor.discover_tools(session, config.name)
        
        # Discover resources
        resources = await self._discover_resources(session, config.name)
        
        connection = ServerConnection(
            name=config.name,
            session=session,
            config=config,
            tools=tools,
            resources=resources,
        )
        
        self._active_connections[config.name] = connection
        logger.info(f"Connected to '{config.name}': {len(tools)} tools, {len(resources)} resources")
        
        return connection
    
    async def _cleanup_connection(self, connection: ServerConnection) -> None:
        """Clean up a connection."""
        self._active_connections.pop(connection.name, None)
        self.tool_executor.clear_cache(connection.name)
    
    async def _discover_resources(
        self,
        session: ClientSession,
        server_name: str,
    ) -> list[ResourceInfo]:
        """Discover available resources from a server."""
        try:
            result = await session.list_resources()
            resources = [
                ResourceInfo(
                    uri=str(res.uri),
                    name=res.name,
                    description=getattr(res, 'description', None),
                    mime_type=getattr(res, 'mimeType', None),
                    server_name=server_name,
                )
                for res in result.resources
            ]
            logger.info(f"Discovered {len(resources)} resources from '{server_name}'")
            return resources
        except Exception as e:
            logger.debug(f"Failed to discover resources from '{server_name}': {e}")
            return []
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool on a connected server.
        
        Args:
            server_name: Name of the connected server
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            ToolResult with execution results
        """
        connection = self._active_connections.get(server_name)
        if not connection:
            return ToolResult(
                success=False,
                error=f"Server '{server_name}' is not connected",
                is_error=True,
            )
        
        return await self.tool_executor.execute(
            connection.session,
            tool_name,
            arguments,
        )
    
    async def read_resource(
        self,
        server_name: str,
        uri: str,
    ) -> dict[str, Any]:
        """
        Read a resource from a connected server.
        
        Args:
            server_name: Name of the connected server
            uri: Resource URI
            
        Returns:
            Resource content
        """
        connection = self._active_connections.get(server_name)
        if not connection:
            return {"error": f"Server '{server_name}' is not connected"}
        
        try:
            result = await connection.session.read_resource(uri)
            return {
                "uri": uri,
                "contents": [
                    {"text": content.text if hasattr(content, 'text') else str(content)}
                    for content in result.contents
                ],
            }
        except Exception as e:
            logger.error(f"Failed to read resource '{uri}': {e}")
            return {"error": str(e)}
    
    def list_tools(self, server_name: str | None = None) -> list[ToolInfo]:
        """
        List available tools.
        
        Args:
            server_name: If provided, list tools from specific server only.
            
        Returns:
            List of available tools
        """
        if server_name:
            connection = self._active_connections.get(server_name)
            return connection.tools if connection else []
        
        all_tools = []
        for connection in self._active_connections.values():
            all_tools.extend(connection.tools)
        return all_tools
    
    def list_resources(self, server_name: str | None = None) -> list[ResourceInfo]:
        """
        List available resources.
        
        Args:
            server_name: If provided, list resources from specific server only.
            
        Returns:
            List of available resources
        """
        if server_name:
            connection = self._active_connections.get(server_name)
            return connection.resources if connection else []
        
        all_resources = []
        for connection in self._active_connections.values():
            all_resources.extend(connection.resources)
        return all_resources
    
    def list_connections(self) -> list[ServerConnection]:
        """List all active connections."""
        return list(self._active_connections.values())
    
    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected."""
        return server_name in self._active_connections
    
    def get_connection(self, server_name: str) -> ServerConnection | None:
        """Get an active connection by name."""
        return self._active_connections.get(server_name)


# Global singleton for convenience
_global_manager: MCPClientManager | None = None


def get_mcp_manager() -> MCPClientManager:
    """Get the global MCP client manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = MCPClientManager()
    return _global_manager


def configure_mcp_manager(config_path: str | Path) -> MCPClientManager:
    """Configure the global MCP manager with a config file."""
    global _global_manager
    registry = ServerRegistry(config_path)
    _global_manager = MCPClientManager(registry)
    return _global_manager

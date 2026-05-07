"""
MCP Client Manager: Core class for managing MCP server connections.

Supports both scoped (context manager) and persistent connections.
Persistent connections stay alive across Django request cycles via a
background asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from mcp import ClientSession

from .server_registry import ServerConfig, ServerRegistry, TransportType
from .tool_executor import ToolExecutor, ToolInfo, ToolResult
from .internal_tools import (
    INTERNAL_SERVER_NAME,
    get_internal_tools,
    execute_internal_tool,
    is_internal_tool,
)
from .transports.stdio import StdioTransport
from .transports.sse import SSETransport
from .transports.streamable_http import StreamableHTTPTransport

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
    
    Supports two connection modes:
    
    1. **Scoped** (context manager) — connection lives within an `async with` block:
    
        async with manager.connect_server("filesystem") as conn:
            tools = conn.tools
    
    2. **Persistent** — connection stays alive until explicitly disconnected.
       Safe to call from synchronous Django views:
    
        manager.connect("filesystem")       # sync, blocks until connected
        tools = manager.list_tools()         # works across requests
        manager.disconnect("filesystem")     # sync, blocks until disconnected
    
    Persistent connections run on a dedicated background asyncio event loop.
    """
    
    def __init__(self, registry: ServerRegistry | None = None):
        self.registry = registry or ServerRegistry()
        self.tool_executor = ToolExecutor()
        self._stdio_transport = StdioTransport()
        self._sse_transport = SSETransport()
        self._streamable_http_transport = StreamableHTTPTransport()
        self._active_connections: dict[str, ServerConnection] = {}
        
        # Persistent connection state
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
    
    # ──────────────────────────────────────────────
    #  Background event loop (for persistent connections)
    # ──────────────────────────────────────────────
    
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure background event loop is running. Thread-safe."""
        if self._loop is not None and self._loop.is_running():
            return self._loop
        
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                daemon=True,
                name="mcp-event-loop",
            )
            self._loop_thread.start()
            logger.info("Started MCP background event loop")
            return self._loop
    
    def _run_async(self, coro, timeout: float = 60.0) -> Any:
        """Run an async coroutine on the background loop. Blocks until done."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)
    
    # ──────────────────────────────────────────────
    #  Persistent connection API (sync-safe)
    # ──────────────────────────────────────────────
    
    def connect(self, name: str) -> ServerConnection:
        """
        Connect to a named server persistently (sync-safe).
        
        The connection stays alive until `disconnect()` is called.
        
        Args:
            name: Name of a server registered in the registry.
            
        Returns:
            ServerConnection with discovered tools/resources.
        """
        if name in self._active_connections:
            logger.info(f"Server '{name}' is already connected")
            return self._active_connections[name]
        
        config = self.registry.get(name)
        if not config:
            raise ValueError(f"Server '{name}' not found in registry")
        
        return self._run_async(self._connect_persistent(config))
    
    def disconnect(self, name: str) -> bool:
        """
        Disconnect from a server (sync-safe).
        
        Returns:
            True if server was disconnected, False if not connected.
        """
        if name not in self._active_connections and name not in self._exit_stacks:
            return False
        
        self._run_async(self._disconnect_persistent(name))
        return True
    
    def connect_all(self) -> dict[str, dict[str, str]]:
        """
        Connect to all configured servers (sync-safe).
        
        Returns:
            Dict of server_name → {"status": "connected"} or {"status": "error", "error": "..."}
        """
        results = {}
        for config in self.registry.list():
            try:
                self.connect(config.name)
                results[config.name] = {"status": "connected"}
            except Exception as e:
                logger.warning(f"Failed to connect to '{config.name}': {e}")
                results[config.name] = {"status": "error", "error": str(e)}
        return results
    
    def disconnect_all(self) -> None:
        """Disconnect all active connections (sync-safe)."""
        names = list(self._active_connections.keys())
        for name in names:
            try:
                self.disconnect(name)
            except Exception as e:
                logger.warning(f"Failed to disconnect '{name}': {e}")
    
    async def _connect_persistent(self, config: ServerConfig) -> ServerConnection:
        """Connect to a server with a persistent (non-scoped) connection."""
        config.validate()
        
        stack = AsyncExitStack()
        await stack.__aenter__()
        
        try:
            if config.transport == TransportType.STDIO:
                session = await stack.enter_async_context(
                    self._stdio_transport.connect(
                        name=config.name,
                        command=config.command,
                        args=config.args,
                        env=config.resolve_env(),
                    )
                )
            elif config.transport == TransportType.SSE:
                session = await stack.enter_async_context(
                    self._sse_transport.connect(
                        name=config.name,
                        url=config.url,
                        headers=config.resolve_headers(),
                    )
                )
            elif config.transport == TransportType.STREAMABLE_HTTP:
                session = await stack.enter_async_context(
                    self._streamable_http_transport.connect(
                        name=config.name,
                        url=config.url,
                        headers=config.resolve_headers(),
                    )
                )
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")
            
            connection = await self._setup_connection(session, config)
            self._exit_stacks[config.name] = stack
            return connection
        except Exception:
            await stack.aclose()
            raise
    
    async def _disconnect_persistent(self, name: str) -> None:
        """Disconnect a persistent connection and clean up."""
        self._active_connections.pop(name, None)
        self.tool_executor.clear_cache(name)
        
        stack = self._exit_stacks.pop(name, None)
        if stack:
            try:
                await stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing connection to '{name}': {e}")
        
        logger.info(f"Disconnected persistent connection to '{name}'")
    
    # ──────────────────────────────────────────────
    #  Scoped connection API (async context manager)
    # ──────────────────────────────────────────────
    
    @asynccontextmanager
    async def connect_server(
        self,
        config: ServerConfig | str,
    ) -> AsyncGenerator[ServerConnection, None]:
        """
        Connect to an MCP server (scoped — connection dies when block exits).
        
        Args:
            config: ServerConfig object or name of a registered server
            
        Yields:
            ServerConnection with active session and discovered tools/resources
        """
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
        elif config.transport == TransportType.STREAMABLE_HTTP:
            async with self._connect_streamable_http(config) as connection:
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
            headers=config.resolve_headers(),
        ) as session:
            connection = await self._setup_connection(session, config)
            try:
                yield connection
            finally:
                await self._cleanup_connection(connection)
    
    @asynccontextmanager
    async def _connect_streamable_http(self, config: ServerConfig) -> AsyncGenerator[ServerConnection, None]:
        """Connect via Streamable HTTP transport."""
        async with self._streamable_http_transport.connect(
            name=config.name,
            url=config.url,
            headers=config.resolve_headers(),
        ) as session:
            connection = await self._setup_connection(session, config)
            try:
                yield connection
            finally:
                await self._cleanup_connection(connection)
    
    # ──────────────────────────────────────────────
    #  Shared internals
    # ──────────────────────────────────────────────
    
    async def _setup_connection(
        self,
        session: ClientSession,
        config: ServerConfig,
    ) -> ServerConnection:
        """Set up a new connection with tool/resource discovery."""
        tools = await self.tool_executor.discover_tools(session, config.name)
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
        """Execute a tool on a connected server."""
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
    
    def call_tool_sync(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool on a connected server (sync-safe).

        Internal tools (server_name="_internal") are executed directly without MCP.
        """
        # Handle internal tools directly (no MCP protocol needed)
        if server_name == INTERNAL_SERVER_NAME or is_internal_tool(tool_name):
            return execute_internal_tool(tool_name, arguments)

        return self._run_async(self.call_tool(server_name, tool_name, arguments))
    
    async def read_resource(
        self,
        server_name: str,
        uri: str,
    ) -> dict[str, Any]:
        """Read a resource from a connected server."""
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
    
    # ──────────────────────────────────────────────
    #  Query methods (always sync-safe)
    # ──────────────────────────────────────────────
    
    def list_tools(self, server_name: str | None = None) -> list[ToolInfo]:
        """
        List available tools from connected servers and internal tools.

        Args:
            server_name: Filter to specific server. Use "_internal" for internal tools only.
        """
        # Handle internal tools request
        if server_name == INTERNAL_SERVER_NAME:
            return get_internal_tools()

        if server_name:
            connection = self._active_connections.get(server_name)
            return connection.tools if connection else []

        # Return all tools: external + internal
        all_tools = []
        for connection in self._active_connections.values():
            all_tools.extend(connection.tools)

        # Always include internal tools
        all_tools.extend(get_internal_tools())

        return all_tools
    
    def list_resources(self, server_name: str | None = None) -> list[ResourceInfo]:
        """List available resources from connected servers."""
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
_global_lock = threading.Lock()


def get_mcp_manager() -> MCPClientManager:
    """
    Get the global MCP client manager instance.
    
    Auto-discovers mcp_servers.json from the project root on first call.
    Thread-safe via double-check locking.
    """
    global _global_manager
    if _global_manager is not None:
        return _global_manager
    
    with _global_lock:
        if _global_manager is not None:
            return _global_manager
        
        # Auto-discover config file from project root
        # TODO make this a bit smarter (search up the directory tree, support env var override, etc.)
        config_path = Path(__file__).parent.parent.parent.parent / "data" / "mcp_servers.json"
        if config_path.exists():
            registry = ServerRegistry(config_path)
            logger.info(f"MCP config loaded from {config_path}: {len(registry.list())} servers")
        else:
            registry = ServerRegistry()
            logger.info("No mcp_servers.json found, starting with empty registry")
        
        _global_manager = MCPClientManager(registry)
        return _global_manager


def configure_mcp_manager(config_path: str | Path) -> MCPClientManager:
    """Configure the global MCP manager with a config file."""
    global _global_manager
    with _global_lock:
        registry = ServerRegistry(config_path)
        _global_manager = MCPClientManager(registry)
        return _global_manager

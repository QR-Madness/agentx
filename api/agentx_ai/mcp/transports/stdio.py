"""
Stdio Transport: Subprocess-based transport for local MCP servers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Callable

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)


@dataclass
class StdioConnection:
    """Represents an active stdio connection to an MCP server."""
    
    session: ClientSession
    server_name: str
    process: asyncio.subprocess.Process | None = None
    _cleanup: Callable | None = None
    
    async def close(self) -> None:
        """Close the connection and cleanup resources."""
        if self._cleanup:
            await self._cleanup()
        logger.info(f"Closed stdio connection to {self.server_name}")


class StdioTransport:
    """Transport for connecting to MCP servers via subprocess stdio."""
    
    def __init__(self):
        self._active_connections: dict[str, StdioConnection] = {}
    
    @asynccontextmanager
    async def connect(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Connect to an MCP server via stdio.
        
        Args:
            name: Unique name for this connection
            command: Command to run (e.g., "npx", "python")
            args: Arguments for the command
            env: Environment variables to pass to the subprocess
            
        Yields:
            ClientSession: The connected MCP client session
        """
        # Merge environment
        full_env = {**os.environ, **(env or {})}
        
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=full_env,
        )
        
        logger.info(f"Connecting to MCP server '{name}' via stdio: {command} {' '.join(args or [])}")
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                connection = StdioConnection(
                    session=session,
                    server_name=name,
                )
                self._active_connections[name] = connection
                
                try:
                    logger.info(f"Connected to MCP server '{name}'")
                    yield session
                finally:
                    if name in self._active_connections:
                        del self._active_connections[name]
                    logger.info(f"Disconnected from MCP server '{name}'")
    
    def is_connected(self, name: str) -> bool:
        """Check if a server is connected."""
        return name in self._active_connections
    
    def get_connection(self, name: str) -> StdioConnection | None:
        """Get an active connection by name."""
        return self._active_connections.get(name)
    
    def list_connections(self) -> list[str]:
        """List all active connection names."""
        return list(self._active_connections.keys())

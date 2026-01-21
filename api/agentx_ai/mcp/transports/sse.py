"""
SSE Transport: Server-Sent Events transport for remote MCP servers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


@dataclass
class SSEConnection:
    """Represents an active SSE connection to an MCP server."""
    
    session: ClientSession
    server_name: str
    url: str
    
    async def close(self) -> None:
        """Close the connection."""
        logger.info(f"Closed SSE connection to {self.server_name}")


class SSETransport:
    """Transport for connecting to MCP servers via HTTP Server-Sent Events."""
    
    def __init__(self):
        self._active_connections: dict[str, SSEConnection] = {}
    
    @asynccontextmanager
    async def connect(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Connect to an MCP server via SSE.
        
        Args:
            name: Unique name for this connection
            url: SSE endpoint URL
            headers: HTTP headers to include in requests
            
        Yields:
            ClientSession: The connected MCP client session
        """
        logger.info(f"Connecting to MCP server '{name}' via SSE: {url}")
        
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                connection = SSEConnection(
                    session=session,
                    server_name=name,
                    url=url,
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
    
    def get_connection(self, name: str) -> SSEConnection | None:
        """Get an active connection by name."""
        return self._active_connections.get(name)
    
    def list_connections(self) -> list[str]:
        """List all active connection names."""
        return list(self._active_connections.keys())

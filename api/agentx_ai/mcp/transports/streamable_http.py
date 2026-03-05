"""
Streamable HTTP Transport: HTTP-based transport for remote MCP servers.

Uses the MCP Streamable HTTP protocol (replacing SSE for newer servers).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


@dataclass
class StreamableHTTPConnection:
    """Represents an active streamable HTTP connection to an MCP server."""

    session: ClientSession
    server_name: str
    url: str

    async def close(self) -> None:
        """Close the connection."""
        logger.info(f"Closed streamable HTTP connection to {self.server_name}")


class StreamableHTTPTransport:
    """Transport for connecting to MCP servers via Streamable HTTP."""

    def __init__(self):
        self._active_connections: dict[str, StreamableHTTPConnection] = {}

    @asynccontextmanager
    async def connect(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Connect to an MCP server via Streamable HTTP.

        Args:
            name: Unique name for this connection
            url: HTTP endpoint URL
            headers: HTTP headers to include in requests

        Yields:
            ClientSession: The connected MCP client session
        """
        logger.info(f"Connecting to MCP server '{name}' via streamable HTTP: {url}")

        async with streamablehttp_client(url, headers=headers) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()

                connection = StreamableHTTPConnection(
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

    def list_connections(self) -> list[str]:
        """List all active connection names."""
        return list(self._active_connections.keys())

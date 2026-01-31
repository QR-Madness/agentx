"""
MCP Transport Layer

Provides transport implementations for connecting to MCP servers:
- stdio: Subprocess-based transport for local servers
- sse: Server-Sent Events (HTTP) transport for remote servers
"""

from .stdio import StdioTransport
from .sse import SSETransport

__all__ = ["StdioTransport", "SSETransport"]

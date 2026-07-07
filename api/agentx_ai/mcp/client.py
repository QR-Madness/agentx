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
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import AsyncGenerator

from mcp import ClientSession
from pydantic import AnyUrl

from ..exceptions import MCPServerNotFoundError, MCPTransportError
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


def oauth_callback_url() -> str:
    """The OAuth redirect URI — a loopback endpoint on this API (RFC 8252).

    The API is the OAuth client (it holds the tokens and runs the MCP
    sessions), so the browser redirect must land here, not on the desktop
    client. Override for non-default ports / remote-API setups with
    ``AGENTX_OAUTH_REDIRECT_URL``. NOTE: `/api/mcp/oauth/callback` is exempted
    from auth middleware (the browser carries no token); it validates the
    OAuth `state` instead.
    """
    import os

    return (
        os.environ.get("AGENTX_OAUTH_REDIRECT_URL")
        or "http://localhost:12319/api/mcp/oauth/callback"
    )


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
        # server_name -> last resource-discovery error. Distinguishes "discovery
        # failed" from "server exposes no resources" (both yield an empty list).
        self._resource_discovery_errors: dict[str, str] = {}

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
            raise MCPServerNotFoundError(f"Server '{name}' not found in registry", server=name)
        
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
    
    def connect_persisted(self) -> dict[str, dict[str, str]]:
        """
        Connect to servers marked auto_connect=True (sync-safe, best-effort).

        Used at API startup to restore servers that were connected when the
        process last shut down. A failure on one server (e.g. unresolved env
        var, dead command) is logged and skipped — it never blocks the others.

        Returns:
            Dict of server_name → {"status": "connected"} or {"status": "error", "error": "..."}
        """
        results: dict[str, dict[str, str]] = {}
        for config in self.registry.list():
            if not getattr(config, "auto_connect", False):
                continue
            try:
                self.connect(config.name)
                results[config.name] = {"status": "connected"}
                logger.info(f"Auto-connected MCP server '{config.name}' on startup")
            except Exception as e:
                logger.warning(f"Auto-connect failed for '{config.name}': {e}")
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
    
    # ──────────────────────────────────────────────
    #  OAuth 2.1 (remote transports)
    # ──────────────────────────────────────────────

    def _build_oauth_provider(self, config: ServerConfig, interactive_flow=None):
        """Build the SDK's OAuthClientProvider for an `auth: {type: oauth}` server.

        The provider (an httpx.Auth) handles RFC 9728 resource-metadata
        discovery, RFC 7591 dynamic registration, PKCE, token refresh — backed
        by our per-server FileTokenStorage. `interactive_flow` (a PendingFlow)
        wires the browser round-trip; without one, a server that *needs* fresh
        consent fails fast with a clear "connect interactively" error instead
        of hanging a headless connect (startup restore, connect_all).
        """
        auth_cfg = config.resolve_auth()
        if not auth_cfg or auth_cfg.get("type") != "oauth":
            return None

        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        from .oauth_flow import publish_authorization_url
        from .oauth_storage import FileTokenStorage

        class _RestoringOAuthProvider(OAuthClientProvider):
            """OAuthClientProvider that restores token expiry on cold start.

            The SDK's ``_initialize`` loads stored tokens but NOT their expiry
            (``token_expiry_time`` stays ``None``), so a restarted process treats
            an expired access token as valid, sends it, gets a 401, and falls
            into the *full interactive* auth flow instead of a headless refresh —
            defeating the whole point of a stored refresh_token. We restore the
            persisted absolute expiry so ``is_token_valid()`` is accurate and the
            SDK refreshes proactively (RFC 6749 §6) via the refresh_token before
            ever sending a stale bearer.
            """

            async def _initialize(self) -> None:
                await super()._initialize()
                tok = self.context.current_tokens
                if not (tok and tok.access_token):
                    return
                storage = self.context.storage
                expires_at = storage.read_token_expiry() if isinstance(storage, FileTokenStorage) else None
                # Legacy token files (written before expiry was persisted) have
                # no absolute expiry — force a proactive refresh rather than risk
                # a stale bearer → 401 → interactive re-auth. A live refresh_token
                # reconnects headlessly; a dead one falls through to a re-auth
                # prompt, which is the correct outcome.
                self.context.token_expiry_time = (
                    expires_at if expires_at is not None else time.time() - 1
                )

        callback_url = oauth_callback_url()
        metadata = OAuthClientMetadata.model_validate({
            "client_name": "AgentX",
            "redirect_uris": [callback_url],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            **({"scope": auth_cfg["scope"]} if auth_cfg.get("scope") else {}),
        })
        storage = FileTokenStorage(config.name, preregistered=auth_cfg)

        if interactive_flow is not None:
            async def redirect_handler(url: str) -> None:
                publish_authorization_url(interactive_flow, url)
                logger.info(f"OAuth authorization required for '{config.name}' — awaiting user consent")

            async def callback_handler() -> tuple[str, str | None]:
                return await interactive_flow.future
        else:
            async def redirect_handler(url: str) -> None:
                raise MCPTransportError(
                    f"Server '{config.name}' requires OAuth authorization — "
                    "connect it from the Toolkit page to sign in."
                )

            async def callback_handler() -> tuple[str, str | None]:
                raise MCPTransportError(
                    f"Server '{config.name}' requires OAuth authorization — "
                    "connect it from the Toolkit page to sign in."
                )

        return _RestoringOAuthProvider(
            server_url=config.require_url(),
            client_metadata=metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )

    def connect_interactive(self, name: str, wait_s: float = 8.0) -> dict[str, Any]:
        """Connect with user-in-the-loop OAuth support (sync-safe).

        Returns one of:
          {"status": "connected", "connection": ServerConnection}
          {"status": "auth_required", "authorization_url": str}   — connect
            keeps running in the background; it completes when the user
            authorizes in the browser and the callback view resolves the flow.

        Non-OAuth servers (and OAuth servers with valid/refreshable stored
        tokens) connect synchronously, same as `connect()`.
        """
        from . import oauth_flow

        if name in self._active_connections:
            return {"status": "connected", "connection": self._active_connections[name]}

        config = self.registry.get(name)
        if not config:
            raise MCPServerNotFoundError(f"Server '{name}' not found in registry", server=name)

        auth_cfg = config.auth or {}
        if auth_cfg.get("type") != "oauth":
            return {"status": "connected", "connection": self.connect(name)}

        loop = self._ensure_loop()
        flow = oauth_flow.begin_flow(name, loop)
        task_future = asyncio.run_coroutine_threadsafe(
            self._connect_persistent(config, interactive_flow=flow), loop
        )

        def _on_done(fut) -> None:
            try:
                fut.result()
                oauth_flow.finish_flow(flow)
                # Mirror the connect endpoint's auto_connect persistence for
                # connections that complete after the interactive round-trip.
                try:
                    cfg = self.registry.get(name)
                    if cfg is not None and not cfg.auto_connect:
                        cfg.auto_connect = True
                        self.registry.save_to_file()
                except Exception as persist_err:  # noqa: BLE001 - best-effort
                    logger.warning(f"Could not persist auto_connect for '{name}': {persist_err}")
            except Exception as e:  # noqa: BLE001 - background terminal state
                # An explicit user cancel (cancel_flow) aborts the future; that
                # is not a failure — stay quiet so the card doesn't flip to
                # "sign-in failed". cancel_flow already dropped the bookkeeping.
                if flow.cancelled:
                    logger.info(f"OAuth connect for '{name}' cancelled by user")
                    return
                # A superseded attempt (future cancelled by a retry's begin_flow)
                # dies late — fail_flow is identity-guarded so it never disturbs
                # the newer pending flow.
                logger.warning(f"OAuth connect for '{name}' failed: {e}")
                oauth_flow.fail_flow(flow, str(e))

        task_future.add_done_callback(_on_done)

        # Either the connect finishes outright (stored tokens still good) or
        # the redirect handler publishes the consent URL — whichever is first.
        deadline = wait_s
        step = 0.05
        waited = 0.0
        while waited < deadline:
            if task_future.done():
                # Propagates connect errors (bad URL, refused, etc.).
                return {"status": "connected", "connection": task_future.result()}
            if flow.url_ready.is_set() and flow.authorization_url:
                return {"status": "auth_required", "authorization_url": flow.authorization_url}
            flow.url_ready.wait(step)
            waited += step
        raise MCPTransportError(
            f"Server '{name}' did not respond within {wait_s:.0f}s (no authorization URL either)"
        )

    async def _connect_persistent(
        self, config: ServerConfig, interactive_flow=None
    ) -> ServerConnection:
        """Connect to a server with a persistent (non-scoped) connection."""
        config.validate()
        auth = self._build_oauth_provider(config, interactive_flow=interactive_flow)

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if config.transport == TransportType.STDIO:
                session = await stack.enter_async_context(
                    self._stdio_transport.connect(
                        name=config.name,
                        command=config.require_command(),
                        args=config.args,
                        env=config.resolve_env(),
                    )
                )
            elif config.transport == TransportType.SSE:
                session = await stack.enter_async_context(
                    self._sse_transport.connect(
                        name=config.name,
                        url=config.require_url(),
                        headers=config.resolve_headers(),
                        auth=auth,
                    )
                )
            elif config.transport == TransportType.STREAMABLE_HTTP:
                session = await stack.enter_async_context(
                    self._streamable_http_transport.connect(
                        name=config.name,
                        url=config.require_url(),
                        headers=config.resolve_headers(),
                        auth=auth,
                    )
                )
            else:
                raise MCPTransportError(f"Unsupported transport: {config.transport}")

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
    ) -> AsyncGenerator[ServerConnection]:
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
                raise MCPServerNotFoundError(f"Server '{config}' not found in registry", server=config)
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
            raise MCPTransportError(f"Unsupported transport: {config.transport}")

    @asynccontextmanager
    async def _connect_stdio(self, config: ServerConfig) -> AsyncGenerator[ServerConnection]:
        """Connect via stdio transport."""
        async with self._stdio_transport.connect(
            name=config.name,
            command=config.require_command(),
            args=config.args,
            env=config.resolve_env(),
        ) as session:
            connection = await self._setup_connection(session, config)
            try:
                yield connection
            finally:
                await self._cleanup_connection(connection)
    
    @asynccontextmanager
    async def _connect_sse(self, config: ServerConfig) -> AsyncGenerator[ServerConnection]:
        """Connect via SSE transport."""
        async with self._sse_transport.connect(
            name=config.name,
            url=config.require_url(),
            headers=config.resolve_headers(),
            auth=self._build_oauth_provider(config),
        ) as session:
            connection = await self._setup_connection(session, config)
            try:
                yield connection
            finally:
                await self._cleanup_connection(connection)
    
    @asynccontextmanager
    async def _connect_streamable_http(self, config: ServerConfig) -> AsyncGenerator[ServerConnection]:
        """Connect via Streamable HTTP transport."""
        async with self._streamable_http_transport.connect(
            name=config.name,
            url=config.require_url(),
            headers=config.resolve_headers(),
            auth=self._build_oauth_provider(config),
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
            self._resource_discovery_errors.pop(server_name, None)
            logger.info(f"Discovered {len(resources)} resources from '{server_name}'")
            return resources
        except Exception as e:
            # Record the failure so it's distinguishable from "no resources".
            self._resource_discovery_errors[server_name] = str(e)
            logger.warning(f"Failed to discover resources from '{server_name}': {e}")
            return []

    def get_resource_discovery_error(self, server_name: str) -> str | None:
        """Return the last resource-discovery error for a server, or None if the
        most recent discovery succeeded (an empty resource list is a success)."""
        return self._resource_discovery_errors.get(server_name)
    
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
            result = await connection.session.read_resource(AnyUrl(uri))
            return {
                "uri": uri,
                "contents": [
                    # noqa kept on getattr: pyright can't narrow .text across the
                    # TextResourceContents|BlobResourceContents union via hasattr.
                    {"text": getattr(content, "text") if hasattr(content, "text") else str(content)}  # noqa: B009
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

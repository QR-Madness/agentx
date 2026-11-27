"""
AgentX domain exception hierarchy.

A small, deliberately shallow tree rooted at :class:`AgentXError`. It exists so
that the genuinely-distinct failure conditions at clean boundaries (provider
resolution, MCP server lookup) carry meaning and an HTTP mapping, instead of
surfacing as anonymous ``ValueError``s that the API layer can only flatten to a
400/500.

Scope note: this is intentionally *not* a sweep of the codebase's many
best-effort ``except Exception`` swallows (memory writes, telemetry,
consolidation per-item resilience) — those are correct as-is. Only boundary
raises adopt these types.

Back-compat: leaf errors that *replace* an existing ``raise ValueError`` also
inherit ``ValueError`` (e.g. :class:`ModelNotFoundError`). Every existing
``except ValueError`` therefore keeps catching them, so swapping a raise site is
behavior-preserving. The API boundary adds an ``except AgentXError`` clause
*before* its ``except ValueError`` to surface the richer ``http_status``.
"""

from typing import Any


class AgentXError(Exception):
    """Base for all AgentX domain errors.

    Carries an optional human-readable ``message`` (also used as ``str(self)``),
    arbitrary structured ``details``, and an ``http_status`` used by the API
    boundary mapper (``utils.responses.error_response``).
    """

    #: Default HTTP status used by the API boundary when this error escapes.
    http_status: int = 500

    #: Stable, machine-readable reason for the few errors a client acts on. The
    #: chat stream's ``error`` event carries it as ``code`` (see
    #: ``utils.responses.stream_error_payload``), so the client can offer the
    #: fix without parsing the message. ``None`` for everything else: a code is
    #: added only when a client has something to do with it, and once shipped it
    #: never changes meaning.
    code: str | None = None

    def __init__(self, message: str = "", **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details


class ConfigError(AgentXError):
    """Invalid or missing configuration."""

    http_status = 500


class ProviderError(AgentXError):
    """A model-provider-layer failure (resolution, request, or transport)."""

    http_status = 502


class ModelNotFoundError(ProviderError, ValueError):
    """No provider/model matches the requested identifier.

    Inherits ``ValueError`` for back-compat with the pre-existing
    ``raise ValueError("Unknown provider"/"Unknown model")`` call sites.
    """

    http_status = 404


class NoModelChosenError(ModelNotFoundError):
    """Nothing chose a model, and the floor model can't run on this install.

    Raised in place of the floor's provider error (see ``agent.core.FLOOR_MODEL``)
    so the user reads where to choose a model rather than which provider is
    missing for a model they never picked. A subclass, so every existing
    ``except ModelNotFoundError`` / ``except ValueError`` still catches it.

    The client offers to open Settings → Model Roles on the Global Default Model.
    """

    code = "no_model_chosen"


class NoProviderConnectedError(NoModelChosenError):
    """Nothing chose a model, and no provider is connected to choose one from.

    The fix starts in Settings → Model Providers rather than Model Roles, so it
    carries its own code. A subclass of ``NoModelChosenError``: anything that
    catches that still catches this.
    """

    code = "no_provider_connected"


class ProviderUnavailableError(ProviderError):
    """The upstream provider API is unreachable or returned a transport error."""

    http_status = 502


class MCPError(AgentXError):
    """An MCP client/server failure."""

    http_status = 503


class MCPServerNotFoundError(MCPError, ValueError):
    """The named MCP server is not present in the registry.

    Inherits ``ValueError`` for back-compat with the pre-existing
    ``raise ValueError("Server '…' not found in registry")`` call sites.
    """

    http_status = 404


class MCPTransportError(MCPError, ValueError):
    """An unsupported or misconfigured MCP transport.

    Inherits ``ValueError`` for back-compat with the pre-existing
    ``raise ValueError("Unsupported transport: …")`` call sites.
    """

    http_status = 400


class ToolExecutionError(AgentXError):
    """A tool raised while executing.

    Distinct from a tool *returning* an error result — that path is already
    modeled by ``mcp.tool_executor.ToolResult(success=False, error=…)`` and is
    left unchanged. This type is for callers that prefer to raise.
    """

    http_status = 500


class MemoryStoreError(AgentXError):
    """A memory-store (Neo4j/Postgres/Redis) operation failed at a boundary.

    Named ``MemoryStoreError`` rather than ``MemoryError`` so it does not shadow
    the Python builtin. Reserved for future memory-boundary adoption.
    """

    http_status = 503

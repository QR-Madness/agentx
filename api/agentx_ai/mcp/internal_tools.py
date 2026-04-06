"""
Internal Tools: Built-in tools exposed via the MCP interface.

These tools are registered directly in MCPClientManager and don't require
external MCP server connections. They provide access to internal AgentX
functionality like stored tool outputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from .tool_executor import ToolInfo, ToolResult

logger = logging.getLogger(__name__)

# Internal server name (virtual - not a real MCP connection)
INTERNAL_SERVER_NAME = "_internal"


@dataclass
class InternalTool:
    """Definition of an internal tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


# Registry of internal tools
_INTERNAL_TOOLS: dict[str, InternalTool] = {}

# Tools that retrieve already-stored content — their results must never be re-stored
RETRIEVAL_TOOL_NAMES: frozenset[str] = frozenset({
    "read_stored_output",
    "list_stored_outputs",
    "tool_output_query",
    "tool_output_section",
    "tool_output_path",
    "read_user_message",
})


def is_retrieval_tool(name: str) -> bool:
    """Check if a tool retrieves stored content (should bypass size gating)."""
    return name in RETRIEVAL_TOOL_NAMES


def register_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """
    Decorator to register an internal tool.

    Example:
        @register_tool(
            name="my_tool",
            description="Does something useful",
            input_schema={"type": "object", "properties": {...}}
        )
        def my_tool(arg1: str, arg2: int = 0) -> dict:
            return {"result": "..."}
    """
    def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        _INTERNAL_TOOLS[name] = InternalTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=func,
        )
        logger.debug(f"Registered internal tool: {name}")
        return func
    return decorator


# =============================================================================
# Tool Implementations
# =============================================================================

@register_tool(
    name="read_stored_output",
    description=(
        "Retrieve stored tool output content. Use this when you see "
        "'[OUTPUT STORED - key: xxx]' in tool results. The stored output "
        "contains the full content that was too large to include directly."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The storage key from the OUTPUT STORED message (e.g., 'read_file_143022_a7b3c8d1')",
            },
            "offset": {
                "type": "integer",
                "description": "Start position in content for pagination (default: 0)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum characters to return per page (default: 12000). Use offset to paginate.",
                "default": 12000,
            },
        },
        "required": ["key"],
    },
)
def read_stored_output(
    key: str,
    offset: int = 0,
    limit: int = 12000,
) -> dict[str, Any]:
    """Retrieve a stored tool output from Redis with automatic pagination."""
    from ..agent.tool_output_storage import get_tool_output

    data = get_tool_output(key)
    if not data:
        return {
            "error": f"Output not found or expired: {key}",
            "success": False,
        }

    full_content = data.get("content", "")
    total_size = len(full_content)

    # Apply pagination
    content = full_content[offset:offset + limit]
    has_more = (offset + len(content)) < total_size

    return {
        "content": content,
        "tool_name": data.get("tool_name"),
        "tool_call_id": data.get("tool_call_id"),
        "offset": offset,
        "limit": limit,
        "total_size": total_size,
        "returned_size": len(content),
        "has_more": has_more,
        "next_offset": offset + len(content) if has_more else None,
        "stored_at": data.get("stored_at"),
        "success": True,
    }


@register_tool(
    name="list_stored_outputs",
    description=(
        "List all stored tool outputs with metadata (not full content). "
        "Use this to see what outputs are available for retrieval."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Filter pattern for tool names (e.g., 'read_file_*'). Default: '*' for all.",
                "default": "*",
            },
        },
    },
)
def list_stored_outputs(pattern: str = "*") -> dict[str, Any]:
    """List available stored outputs with metadata."""
    from ..agent.tool_output_storage import list_tool_outputs

    outputs = list_tool_outputs(pattern)

    return {
        "outputs": outputs,
        "count": len(outputs),
        "success": True,
    }


@register_tool(
    name="tool_output_query",
    description=(
        "Semantic search over a stored tool output. Use this when you need to find "
        "specific information within a large stored output but don't know the exact "
        "section or location. Returns the most relevant chunks matching your query."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The storage key from the STORED/COMPRESSED SUMMARY message",
            },
            "query": {
                "type": "string",
                "description": "Natural language query describing what you're looking for",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of matching chunks to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["key", "query"],
    },
)
def tool_output_query(key: str, query: str, top_k: int = 5) -> dict[str, Any]:
    """Semantic search over chunks of a stored tool output."""
    from ..agent.tool_output_storage import get_tool_output
    from ..agent.tool_output_chunker import chunk_text, semantic_search_chunks

    data = get_tool_output(key)
    if not data:
        return {"error": f"Output not found or expired: {key}", "success": False}

    content = data.get("content", "")
    chunks = chunk_text(content)
    results = semantic_search_chunks(chunks, query, top_k)

    return {
        "results": results,
        "total_chunks": len(chunks),
        "query": query,
        "tool_name": data.get("tool_name"),
        "success": True,
    }


@register_tool(
    name="tool_output_section",
    description=(
        "Access a specific section of a stored tool output by name or heading. "
        "Use this when you know which section you need (from the Structure Index "
        "in the compressed summary). Omit section name to list available sections."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The storage key from the STORED/COMPRESSED SUMMARY message",
            },
            "section": {
                "type": "string",
                "description": "Section name/heading to retrieve. Omit to list available sections.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum characters to return from the section (default: 10000)",
                "default": 10000,
            },
        },
        "required": ["key"],
    },
)
def tool_output_section(key: str, section: str = "", limit: int = 10000) -> dict[str, Any]:
    """Access sections of a stored tool output."""
    from ..agent.tool_output_storage import get_tool_output
    from ..agent.tool_output_chunker import detect_sections, get_section_content

    data = get_tool_output(key)
    if not data:
        return {"error": f"Output not found or expired: {key}", "success": False}

    content = data.get("content", "")
    sections = detect_sections(content)

    if not section:
        # List available sections
        section_list = [
            {"name": s["name"], "size": s.get("size", s["end"] - s["start"])}
            for s in sections
        ]
        return {
            "sections": section_list,
            "count": len(section_list),
            "tool_name": data.get("tool_name"),
            "success": True,
        }

    # Retrieve specific section
    result = get_section_content(content, section, limit)
    if not result:
        section_names = [s["name"] for s in sections]
        return {
            "error": f"Section '{section}' not found",
            "available_sections": section_names,
            "success": False,
        }

    return {
        "section_name": result["name"],
        "content": result["content"],
        "size": len(result["content"]),
        "tool_name": data.get("tool_name"),
        "success": True,
    }


@register_tool(
    name="tool_output_path",
    description=(
        "Query a stored tool output using a JSON path expression. Use this when "
        "the stored output is JSON and you need a specific field or nested value. "
        "Supports dot notation (data.items), array indexing (items[0]), and "
        "wildcards (items[*])."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The storage key from the STORED/COMPRESSED SUMMARY message",
            },
            "jsonpath": {
                "type": "string",
                "description": "JSON path expression (e.g., 'data.results[0].name', 'items[*]')",
            },
        },
        "required": ["key", "jsonpath"],
    },
)
def tool_output_path(key: str, jsonpath: str) -> dict[str, Any]:
    """Query a stored tool output using a JSON path expression."""
    from ..agent.tool_output_storage import get_tool_output
    from ..agent.tool_output_chunker import resolve_json_path

    data = get_tool_output(key)
    if not data:
        return {"error": f"Output not found or expired: {key}", "success": False}

    content = data.get("content", "")
    return resolve_json_path(content, jsonpath)


@register_tool(
    name="read_user_message",
    description=(
        "Retrieve the full content of a cached user message. Use this when you see "
        "'[USER MESSAGE CACHED - key: xxx]' in the conversation. The cached message "
        "contains the full user input that was too large to include directly in context."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The storage key from the USER MESSAGE CACHED notice (e.g., 'msg_143022_a7b3c8d1')",
            },
            "offset": {
                "type": "integer",
                "description": "Start position in content for pagination (default: 0)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum characters to return per page (default: 12000). Use offset to paginate.",
                "default": 12000,
            },
        },
        "required": ["key"],
    },
)
def read_user_message(
    key: str,
    offset: int = 0,
    limit: int = 12000,
) -> dict[str, Any]:
    """Retrieve a cached user message from Redis with automatic pagination."""
    from ..agent.user_message_storage import get_user_message

    data = get_user_message(key)
    if not data:
        return {
            "error": f"User message not found or expired: {key}",
            "success": False,
        }

    full_content = data.get("content", "")
    total_size = len(full_content)

    # Apply pagination
    content = full_content[offset:offset + limit]
    has_more = (offset + len(content)) < total_size

    return {
        "content": content,
        "message_id": data.get("message_id"),
        "session_id": data.get("session_id"),
        "offset": offset,
        "limit": limit,
        "total_size": total_size,
        "returned_size": len(content),
        "has_more": has_more,
        "next_offset": offset + len(content) if has_more else None,
        "stored_at": data.get("stored_at"),
        "success": True,
    }


# =============================================================================
# Public API
# =============================================================================

def get_internal_tools() -> list[ToolInfo]:
    """
    Get all registered internal tools as ToolInfo objects.

    Returns:
        List of ToolInfo objects compatible with the MCP tool discovery system.
    """
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            server_name=INTERNAL_SERVER_NAME,
        )
        for tool in _INTERNAL_TOOLS.values()
    ]


def find_internal_tool(name: str) -> ToolInfo | None:
    """
    Find an internal tool by name.

    Args:
        name: Tool name to look up

    Returns:
        ToolInfo if found, None otherwise
    """
    tool = _INTERNAL_TOOLS.get(name)
    if tool:
        return ToolInfo(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            server_name=INTERNAL_SERVER_NAME,
        )
    return None


def execute_internal_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
    """
    Execute an internal tool by name.

    Args:
        name: Tool name to execute
        arguments: Tool arguments

    Returns:
        ToolResult with execution results
    """
    tool = _INTERNAL_TOOLS.get(name)
    if not tool:
        return ToolResult(
            success=False,
            error=f"Internal tool not found: {name}",
            is_error=True,
        )

    start_time = time.time()

    try:
        logger.info(f"Executing internal tool '{name}' with args: {arguments}")
        result = tool.handler(**arguments)

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Internal tool '{name}' completed in {latency_ms}ms")

        # Convert result to ToolResult format
        success = result.get("success", True) if isinstance(result, dict) else True

        # Format as text content
        import json
        content_text = json.dumps(result, indent=2, default=str)

        return ToolResult(
            success=success,
            content=[{"type": "text", "text": content_text}],
            is_error=not success,
        )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Internal tool '{name}' failed after {latency_ms}ms: {e}")

        return ToolResult(
            success=False,
            error=str(e),
            is_error=True,
        )


def is_internal_tool(name: str) -> bool:
    """Check if a tool name is an internal tool."""
    return name in _INTERNAL_TOOLS

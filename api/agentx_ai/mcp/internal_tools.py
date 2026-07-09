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
from typing import Any
from collections.abc import Callable

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
    "recall_user_history",
    "read_thread",
    "project_search",
    "workspace_search",  # legacy alias of project_search (pre-Projects rename)
    "document_query",
    "read_document",
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


@register_tool(
    name="recall_user_history",
    description=(
        "Recall what this user has said across past conversations. Returns a "
        "summary of user-authored turns matching an optional topic. Use when "
        "you need context about the user's prior questions, preferences, or "
        "ongoing work — not for the current conversation's transcript."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Optional focus query. Omit to get a general recap.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of past user turns to return (default 10, max 30).",
                "default": 10,
            },
        },
    },
)
def recall_user_history(
    topic: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Look up the calling user's prior turns via AgentMemory."""
    from .internal_context import current_context

    ctx = current_context()
    if ctx is None:
        return {
            "error": "No active agent context — recall_user_history can only be called during a chat turn.",
            "success": False,
        }

    try:
        from ..kit.memory_utils import get_agent_memory
    except Exception as e:
        return {"error": f"Memory system unavailable: {e}", "success": False}

    memory = get_agent_memory(
        user_id=ctx.user_id,
        channel=ctx.channel or "_default",
        agent_id=ctx.agent_id,
    )
    if memory is None:
        return {"error": "Memory system unavailable", "success": False}

    capped = max(1, min(int(limit or 10), 30))
    query = topic.strip() if topic else "user background, preferences, goals"

    try:
        bundle = memory.remember(query=query, top_k=capped * 2)
    except Exception as e:
        return {"error": f"Recall failed: {e}", "success": False}

    # Cached cross-conversation recap (refreshed during consolidation), used to
    # fill the summary field below.
    try:
        from ..kit.agent_memory.recap import get_cached_recap
        _recap = get_cached_recap(ctx.user_id, ctx.channel or "_default")
        recap_summary = _recap.get("summary", "") if _recap else ""
    except Exception:  # noqa: BLE001
        recap_summary = ""

    if bundle is None:
        return {"summary": recap_summary, "user_turns": [], "facts": [], "success": True}

    user_turns: list[dict[str, Any]] = []
    seen_conv_pairs: set[tuple[Any, Any]] = set()
    for t in bundle.relevant_turns or []:
        if t.get("role") != "user":
            continue
        if ctx.conversation_id and t.get("conversation_id") == ctx.conversation_id:
            continue
        key = (t.get("conversation_id"), t.get("timestamp"))
        if key in seen_conv_pairs:
            continue
        seen_conv_pairs.add(key)
        user_turns.append({
            "timestamp": str(t.get("timestamp")),
            "conversation_id": t.get("conversation_id"),
            "content": (t.get("content") or "")[:600],
        })
        if len(user_turns) >= capped:
            break

    facts = [
        {
            "claim": (f.get("claim") if isinstance(f, dict) else getattr(f, "claim", "")) or "",
            "confidence": (f.get("confidence") if isinstance(f, dict) else getattr(f, "confidence", 0.0)) or 0.0,
        }
        for f in (bundle.facts or [])[:10]
    ]

    return {
        "topic": topic,
        "summary": recap_summary,
        "user_turns": user_turns,
        "facts": facts,
        "turn_count": len(user_turns),
        "success": True,
    }


@register_tool(
    name="read_thread",
    description=(
        "Pull the verbatim turns of a PAST conversation — use it to see what was "
        "actually said in a thread surfaced under 'Threads you can pull'. Pass the "
        "`conversation_id` from a lead (and its `center_turn` to focus on the "
        "relevant part). Returns the real turns (both sides), not a summary. Use "
        "sparingly — at most a couple of pulls per turn; read the one-line lead "
        "first and pull only when the specifics matter."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "conversation_id": {
                "type": "string",
                "description": "The conversation to pull, from a 'Threads you can pull' lead.",
            },
            "center_turn": {
                "type": "integer",
                "description": "Optional turn index (from the lead) to center the window on.",
            },
        },
        "required": ["conversation_id"],
    },
)
def read_thread(conversation_id: str, center_turn: int | None = None) -> dict[str, Any]:
    """Load a past thread's verbatim turns on demand (episodic pull)."""
    if not conversation_id or not conversation_id.strip():
        return {"error": "conversation_id is required", "success": False}

    memory, err = _memory_for_ctx()
    if memory is None:
        return err or {"error": "Memory system unavailable", "success": False}

    try:
        turns = memory.read_thread(conversation_id.strip(), center_turn=center_turn)
    except Exception as e:  # noqa: BLE001 — a tool error never breaks the turn
        return {"error": f"read_thread failed: {e}", "success": False}

    return {
        "conversation_id": conversation_id,
        "turns": turns,
        "turn_count": len(turns),
        "success": True,
    }


@register_tool(
    name="checkpoint",
    description=(
        "Record a checkpoint of where you are in the current task. The "
        "checkpoint is re-injected into the system prompt every turn and "
        "survives automatic context compression. Use when you have made a "
        "non-trivial decision, completed a sub-step, or want to lock in "
        "intent before continuing. Also use it to keep an interesting or "
        "useful thought you don't want to lose — your raw reasoning isn't "
        "saved, so the scratchpad is how a thought survives the turn."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-sentence summary of the current state.",
            },
            "decisions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of decisions locked in.",
                "default": [],
            },
            "next_step": {
                "type": "string",
                "description": "Optional one-line description of what's next.",
                "default": "",
            },
            "replace": {
                "type": "boolean",
                "description": (
                    "Replace your prior checkpoints with this one (supersede the "
                    "running anchor set) instead of appending. Use when earlier "
                    "checkpoints are now stale so they don't pile up or contradict."
                ),
                "default": False,
            },
        },
        "required": ["summary"],
    },
)
def checkpoint(
    summary: str,
    decisions: list[str] | None = None,
    next_step: str = "",
    replace: bool = False,
) -> dict[str, Any]:
    """Persist a checkpoint anchor for the current conversation."""
    from .internal_context import current_context
    from ..agent.checkpoint_storage import add_checkpoint, list_checkpoints

    ctx = current_context()
    if ctx is None or not ctx.conversation_id:
        return {
            "error": "Checkpoint requires an active conversation context.",
            "success": False,
        }

    if not summary or not summary.strip():
        return {"error": "summary is required", "success": False}

    entry = add_checkpoint(
        conversation_id=ctx.conversation_id,
        summary=summary,
        decisions=decisions,
        next_step=next_step,
        replace=bool(replace),
    )
    total = len(list_checkpoints(ctx.conversation_id))

    return {
        "stored": entry,
        "checkpoint_count": total,
        "note": "Checkpoint will be re-injected into system context every turn.",
        "success": True,
    }


# --- File Workspaces & Document RAG (todo/backlog/workspaces.md) -------------
# Two-tier retrieval over the conversation's attached project: project_search
# (catalog → which file) then document_query (semantic → which passage), with
# read_document for full paginated text, and create_document/update_document for
# durable writes. All scope to the active workspace_id from the per-turn internal
# context; no project attached → a clear, non-fatal error.
# NOTE: project_search was `workspace_search` before the Projects rename —
# _TOOL_ALIASES keeps old conversations/procedural records executable.

def _active_workspace_id() -> str | None:
    from .internal_context import current_context

    ctx = current_context()
    return ctx.workspace_id if ctx else None


@register_tool(
    name="project_search",
    description=(
        "Search the attached project's documents by filename, tag, or topic "
        "(catalog search). Use this FIRST to find which file is relevant, then "
        "`document_query` for the exact passage. Returns documents with their "
        "id, filename, tags, and summary."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords, filename, or topic."},
            "limit": {"type": "integer", "description": "Max files (default 10).", "default": 10},
        },
        "required": ["query"],
    },
)
def project_search(query: str, limit: int = 10) -> dict[str, Any]:
    from ..kit.workspaces import retrieval

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {"error": "This conversation is not in a project — ask the user to attach or create one in the Projects hub.", "success": False}
    results = retrieval.search_manifest(workspace_id, query, limit=limit)
    return {"results": results, "count": len(results), "success": True}


@register_tool(
    name="document_query",
    description=(
        "Semantic search across the attached project's document contents — finds "
        "the passages most relevant to a natural-language question. Returns chunks "
        "with their document_id, filename, text, and similarity score. Use "
        "`read_document` to read more around a hit."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language question or topic."},
            "top_k": {"type": "integer", "description": "Max passages (default 5).", "default": 5},
        },
        "required": ["query"],
    },
)
def document_query(query: str, top_k: int = 5) -> dict[str, Any]:
    from ..kit.workspaces import retrieval

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {"error": "This conversation is not in a project — ask the user to attach or create one in the Projects hub.", "success": False}
    results = retrieval.query_chunks(workspace_id, query, top_k=top_k)
    return {"results": results, "count": len(results), "success": True}


@register_tool(
    name="read_document",
    description=(
        "Read a document's full text (paginated), scoped to the attached project. "
        "Use after `project_search`/`document_query` to read more context around a "
        "hit. Returns a slice with has_more/total_chars for further pagination."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "Document id from a search result."},
            "offset": {"type": "integer", "description": "Start character (default 0).", "default": 0},
            "limit": {"type": "integer", "description": "Max characters (default 12000).", "default": 12000},
        },
        "required": ["document_id"],
    },
)
def read_document(document_id: str, offset: int = 0, limit: int = 12000) -> dict[str, Any]:
    from ..kit.workspaces import retrieval

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {"error": "This conversation is not in a project — ask the user to attach or create one in the Projects hub.", "success": False}
    doc = retrieval.read_document(document_id, offset=offset, limit=limit, workspace_id=workspace_id)
    if doc is None:
        return {"error": f"Document {document_id} not found in this project.", "success": False}
    return {**doc, "success": True}


@register_tool(
    name="create_document",
    description=(
        "Create a durable document in the attached project. This is the RIGHT way to "
        "produce a lasting file (notes, plans, reports, drafts, living reference docs): "
        "it appears in the user's Projects hub, survives this conversation, and becomes "
        "searchable via `project_search`/`document_query` after a brief indexing pass. "
        "Markdown (`.md`) is preferred; `.txt` also works. Fails if the filename already "
        "exists — use `update_document` to change an existing file. Do NOT use shell "
        "`write_file` or external filesystem tools for project documents: those files "
        "are temporary and never become part of the project."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name for the new document, e.g. `notes.md` (optionally one folder level: `research/notes.md`). Allowed types: md, markdown, txt.",
            },
            "content": {"type": "string", "description": "The document's full content."},
        },
        "required": ["filename", "content"],
    },
)
def create_document_tool(filename: str, content: str) -> dict[str, Any]:
    from ..kit.workspaces.service import WorkspaceError, create_text_document

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {
            "error": "This conversation is not in a project — ask the user to attach or "
                     "create one in the Projects hub if they want a durable file.",
            "success": False,
        }
    try:
        doc = create_text_document(
            workspace_id=workspace_id, filename=filename, content=content
        )
    except WorkspaceError as e:
        out: dict[str, Any] = {"error": e.message, "success": False}
        if e.document_id:
            out["existing_document_id"] = e.document_id
        return out
    return {
        "success": True,
        "document": {
            "id": doc["id"],
            "filename": doc["filename"],
            "status": doc["status"],
            "size_bytes": doc["size_bytes"],
        },
        "note": "Indexing in the background — it will show as ready in the project shortly.",
    }


@register_tool(
    name="update_document",
    description=(
        "Replace the full content of an existing project document (get its `document_id` "
        "from the project file list or `project_search`). Read it first with "
        "`read_document` if you don't have the current content — this replaces the WHOLE "
        "file. Use this to keep living documents (notes, plans, memory files) current as "
        "the work evolves. The document is re-indexed automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The document's id (from the project file list or a search result).",
            },
            "content": {"type": "string", "description": "The new full content of the document."},
        },
        "required": ["document_id", "content"],
    },
)
def update_document_tool(document_id: str, content: str) -> dict[str, Any]:
    from ..kit.workspaces.service import WorkspaceError, update_text_document

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {
            "error": "This conversation is not in a project — ask the user to attach or "
                     "create one in the Projects hub if they want a durable file.",
            "success": False,
        }
    try:
        doc = update_text_document(
            workspace_id=workspace_id, document_id=document_id, content=content
        )
    except WorkspaceError as e:
        return {"error": e.message, "success": False}
    return {
        "success": True,
        "document": {
            "id": doc["id"],
            "filename": doc["filename"],
            "status": doc["status"],
            "size_bytes": doc["size_bytes"],
        },
        "note": "Re-indexing in the background.",
    }


def _read_modify_write_document(
    document_id: str, transform: Callable[[str], str | dict[str, Any]]
) -> dict[str, Any]:
    """Shared read→modify→write for the partial-edit tools (append/edit).

    Reads a text document's current content + sha256, applies ``transform`` (which
    returns the new text, or a tool-shaped error dict to abort), and writes it back
    with an optimistic-concurrency guard (``expected_sha256``): if another writer
    changed the file since we read it, the service raises a ``conflict`` and we
    return a soft "re-read and try again" — a lightweight write-lock without a queue.
    """
    from ..kit.workspaces import repository, storage
    from ..kit.workspaces.service import WorkspaceError, update_text_document

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {
            "error": "This conversation is not in a project — ask the user to attach or "
                     "create one in the Projects hub.",
            "success": False,
        }
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return {"error": f"Document {document_id} not found in this project.", "success": False}
    raw = storage.read_blob(doc["storage_key"])
    if raw is None:
        return {"error": "Couldn't read the document's current content.", "success": False}
    try:
        current = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "This file isn't editable text (only .md/.txt can be edited).", "success": False}

    result = transform(current)
    if isinstance(result, dict):  # transform aborted with a tool-shaped error
        return result

    try:
        updated = update_text_document(
            workspace_id=workspace_id,
            document_id=document_id,
            content=result,
            expected_sha256=doc["sha256"],
        )
    except WorkspaceError as e:
        if e.code == "conflict":
            return {
                "success": False,
                "error": "This file was just changed by someone else — re-read it "
                         "with read_document and try your edit again.",
            }
        return {"error": e.message, "success": False}
    return {
        "success": True,
        "document": {
            "id": updated["id"],
            "filename": updated["filename"],
            "status": updated["status"],
            "size_bytes": updated["size_bytes"],
        },
        "note": "Re-indexing in the background.",
    }


@register_tool(
    name="append_to_document",
    description=(
        "Append text to the END of an existing project document — without resending the "
        "whole file. Ideal for living logs, running notes, or changelogs. Get the "
        "`document_id` from the project file list or `project_search`. A newline is added "
        "between the existing content and your text if needed. Only `.md`/`.txt` files."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document's id."},
            "text": {"type": "string", "description": "Text to append at the end."},
        },
        "required": ["document_id", "text"],
    },
)
def append_to_document_tool(document_id: str, text: str) -> dict[str, Any]:
    def _t(current: str) -> str:
        sep = "" if (not current or current.endswith("\n")) else "\n"
        return current + sep + text
    return _read_modify_write_document(document_id, _t)


@register_tool(
    name="edit_document",
    description=(
        "Make a targeted edit to a project document by find-and-replace — change one "
        "passage without rewriting the whole file (get the `document_id` from the file "
        "list or `project_search`). `find` must match the current text exactly. By default "
        "it edits a single, unique match; pass `replace_all=true` to replace every "
        "occurrence. Only `.md`/`.txt` files. To replace the whole file use `update_document`."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document's id."},
            "find": {"type": "string", "description": "Exact text to find."},
            "replace": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence (default false — requires a unique match).",
                "default": False,
            },
        },
        "required": ["document_id", "find", "replace"],
    },
)
def edit_document_tool(
    document_id: str, find: str, replace: str, replace_all: bool = False
) -> dict[str, Any]:
    def _t(current: str) -> str | dict[str, Any]:
        if not find:
            return {"error": "`find` is required.", "success": False}
        count = current.count(find)
        if count == 0:
            return {"error": "`find` text was not found in the document.", "success": False}
        if count > 1 and not replace_all:
            return {
                "error": f"`find` matches {count} places — pass replace_all=true to replace "
                         "them all, or use a longer, unique `find`.",
                "success": False,
            }
        return current.replace(find, replace) if replace_all else current.replace(find, replace, 1)
    return _read_modify_write_document(document_id, _t)


@register_tool(
    name="list_project_files",
    description=(
        "List the files in the attached project — names, ids, types, sizes, tags, and a "
        "short summary — so you know what's there before reading or editing. Use this to "
        "enumerate the project; use `project_search`/`document_query` to find a specific one."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max files (default 50).", "default": 50},
        },
    },
)
def list_project_files_tool(limit: int = 50) -> dict[str, Any]:
    from ..kit.workspaces import repository

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {
            "error": "This conversation is not in a project — ask the user to attach or "
                     "create one in the Projects hub.",
            "success": False,
        }
    limit = max(1, min(int(limit or 50), 200))
    docs = [d for d in repository.list_documents(workspace_id) if d.get("status") == "ready"][:limit]
    files = [
        {
            "document_id": d["id"],
            "filename": d["filename"],
            "content_type": d.get("content_type"),
            "size_bytes": int(d.get("size_bytes") or 0),
            "tags": list(d.get("tags") or []),
            "summary": d.get("summary") or "",
        }
        for d in docs
    ]
    return {"files": files, "count": len(files), "success": True}


@register_tool(
    name="delete_document",
    description=(
        "Delete a file from the attached project (get its `document_id` from the file list "
        "or `project_search`). Permanent — remove obsolete or superseded files. Avatar/app "
        "icons can't be deleted here."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document's id to delete."},
        },
        "required": ["document_id"],
    },
)
def delete_document_tool(document_id: str) -> dict[str, Any]:
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import release_blob_if_unreferenced

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {
            "error": "This conversation is not in a project — ask the user to attach or "
                     "create one in the Projects hub.",
            "success": False,
        }
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return {"error": f"Document {document_id} not found in this project.", "success": False}
    if str(doc.get("filename") or "").startswith("avatars/"):
        return {"error": "Refusing to delete an avatar (an app icon, not project content).", "success": False}
    storage_key = repository.delete_document(document_id)
    if storage_key:
        release_blob_if_unreferenced(workspace_id, storage_key)
    return {"success": True, "deleted": True, "filename": doc.get("filename")}


@register_tool(
    name="rename_document",
    description=(
        "Rename a project file's name (get its `document_id` from the file list or "
        "`project_search`). Renames the base name only — the folder and extension are "
        "kept, and the file's identity/history are preserved. `name` is just the new "
        "base name (no path, no extension needed)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {"type": "string", "description": "The document's id."},
            "name": {"type": "string", "description": "The new base name (no folder/extension)."},
        },
        "required": ["document_id", "name"],
    },
)
def rename_document_tool(document_id: str, name: str) -> dict[str, Any]:
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import WorkspaceError, rename_document

    workspace_id = _active_workspace_id()
    if not workspace_id:
        return {
            "error": "This conversation is not in a project — ask the user to attach or "
                     "create one in the Projects hub.",
            "success": False,
        }
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return {"error": f"Document {document_id} not found in this project.", "success": False}
    if str(doc.get("filename") or "").startswith("avatars/"):
        return {"error": "Refusing to rename an avatar (an app icon, not project content).", "success": False}
    try:
        renamed = rename_document(workspace_id=workspace_id, document_id=document_id, new_base=name)
    except WorkspaceError as e:
        return {"error": e.message, "success": False}
    return {"success": True, "document": {"id": renamed["id"], "filename": renamed["filename"]}}


@register_tool(
    name="view_image",
    description=(
        "Look at an image — load it into your vision so you can actually see and describe it. "
        "Use this for image files in the attached project (e.g. a `.png`/`.jpg` from "
        "`project_search`/`list_files`) or an image generated earlier in this conversation. "
        "`read_document` only returns text, so use THIS to view a picture. After viewing, the "
        "image appears in the conversation and you can describe or reason about what it shows."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The image's document id (from a search/list result or an image catalog entry).",
            },
        },
        "required": ["document_id"],
    },
)
def view_image(document_id: str) -> dict[str, Any]:
    """Resolve a workspace image doc for vision. The actual image block is injected
    into the next model round by the streaming tool loop (which knows whether the model
    can see). Here we only validate access + that it's an image; resolution is scoped to
    the attached workspace or the user's Home (where generated images land)."""
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import MEDIA_CONTENT_TYPES
    from .internal_context import current_context

    document_id = (document_id or "").strip()
    if not document_id:
        return {"error": "document_id is required", "success": False}

    ctx = current_context()
    attached = ctx.workspace_id if ctx else None
    user_id = ctx.user_id if ctx else "default"
    home_id = repository.ensure_home_workspace(user_id)["id"]

    doc = repository.get_document(document_id)
    if doc is None:
        return {"error": f"Image {document_id} not found.", "success": False}
    if doc.get("workspace_id") not in {attached, home_id}:
        return {
            "error": "That image isn't in this project's files or your Home space.",
            "success": False,
        }
    media_type = doc.get("content_type")
    if media_type not in MEDIA_CONTENT_TYPES:
        return {
            "error": f"{doc.get('filename')} isn't a viewable image (type {media_type}). "
                     "Use read_document for text files.",
            "success": False,
        }
    # The loop reads these fields to build the image block + gate on vision capability.
    return {
        "success": True,
        "document_id": document_id,
        "workspace_id": doc["workspace_id"],
        "media_type": media_type,
        "filename": doc.get("filename"),
    }


# --- Image generation (multi-modal; the result renders as an image exhibit) --


@register_tool(
    name="generate_image",
    description=(
        "Generate an image from a text prompt and show it to the user in this conversation. "
        "Use when the user asks you to draw / create / generate an image, diagram, or picture. "
        "The image is rendered for the user automatically — you don't get the pixels back, so "
        "just confirm what you made; don't try to describe it pixel-by-pixel."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to depict — be specific and visual."},
        },
        "required": ["prompt"],
    },
)
def generate_image(prompt: str) -> dict[str, Any]:
    from ..config import DEFAULT_IMAGE_MODEL, get_config_manager
    from .internal_context import current_context

    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "prompt is required", "success": False}

    cfg = get_config_manager()
    if not cfg.get("images.enabled", True):
        return {"error": "Image generation is disabled in settings.", "success": False}

    ctx = current_context()
    user_id = ctx.user_id if ctx else "default"
    model = (cfg.get("images.default_model", DEFAULT_IMAGE_MODEL) or "").strip()

    try:
        from ..agent.image_gen import generate_and_store_image
        from ..providers.registry import get_registry
        from ..utils.async_bridge import run_coro_sync

        provider, model_id, _ = get_registry().resolve_with_fallback(model)
        # The shared helper does generate → store_media → meter; bridge it from this
        # sync tool. Same path the direct image-model conversation flow uses.
        info = run_coro_sync(
            generate_and_store_image(
                prompt,
                provider=provider,
                model=model_id,
                workspace_id=(ctx.workspace_id if ctx and ctx.workspace_id else None),
                user_id=user_id,
                conversation_id=(ctx.conversation_id if ctx else None),
                agent_id=(ctx.agent_id if ctx else None),
            ),
            timeout=120.0,
        )
    except NotImplementedError:
        return {"error": "The configured provider can't generate images.", "success": False}
    except Exception as e:  # noqa: BLE001 — a tool error never breaks the turn
        logger.warning(f"generate_image failed: {e}")
        return {"error": f"Image generation failed: {str(e)[:200]}", "success": False}

    return {
        "success": True,
        "doc_id": info["doc_id"],
        "workspace_id": info["workspace_id"],
        "url": info["url"],
        "prompt": prompt,
    }


# --- Agent shells (sandboxed command execution — kit/shell) ------------------
# Opt-in (shell.enabled). Commands run in a bubblewrap jail scoped to the
# conversation's work dir (materialized from the attached workspace); env scrubbed
# of secrets, network off. The file tools are the safe structured subset (path-jailed,
# no subprocess). All resolve workspace_id + conversation_id from the per-turn context.

def _shell_context() -> tuple[str | None, str | None]:
    from .internal_context import current_context

    ctx = current_context()
    if ctx is None:
        return None, None
    return ctx.workspace_id, ctx.conversation_id


def _shell_guard() -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Gate a shell tool: returns (workspace_id, conversation_id, error_dict|None)."""
    if not _shell_allowed():
        return None, None, {
            "error": "Shell is not enabled for this project. Turn on 'Allow shell' in the "
                     "project attached to this conversation.",
            "success": False,
        }
    ws, conv = _shell_context()
    if not conv:
        return None, None, {"error": "Shell requires an active conversation.", "success": False}
    return ws, conv, None


@register_tool(
    name="run_command",
    description=(
        "Run a shell command in the attached project (a sandboxed, TEMPORARY working copy "
        "of its files). It runs in the project's shell backend — a locked-down bubblewrap "
        "jail (no network) or a persistent container (installs + network), per the "
        "project's setting. Use for inspecting/transforming files (ls, grep, sed, python3, "
        "etc.). Files created here are NOT project documents — use `create_document` for "
        "durable files. Returns stdout, stderr, and exit_code."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command (run via sh -lc)."},
            "timeout": {"type": "integer", "description": "Max seconds (capped by config)."},
        },
        "required": ["command"],
    },
)
def run_command(command: str, timeout: int | None = None) -> dict[str, Any]:
    from ..config import get_config_manager
    from ..kit.shell import dispatch, policy

    ws, conv, err = _shell_guard()
    if err:
        return err
    assert conv is not None

    cfg = get_config_manager()
    deny = policy.DEFAULT_DENY_PATTERNS + list(cfg.get("shell.deny_patterns", []) or [])
    reason = policy.check_command(command, deny)
    if reason:
        return {"error": reason, "blocked": True, "success": False}

    try:
        result = dispatch.run(ws, conv, command, timeout=timeout)
    except dispatch.ShellUnavailable as e:
        return {"error": str(e), "success": False}

    max_chars = int(cfg.get("shell.max_output_chars", 20000))
    out, out_trunc = policy.cap_output(result.stdout, max_chars)
    errout, err_trunc = policy.cap_output(result.stderr, max_chars)
    logger.info(
        "🐚 SHELL run ws=%s conv=%s rc=%s timed_out=%s dur=%dms sandbox=%s cmd=%r",
        ws, conv, result.exit_code, result.timed_out, result.duration_ms, result.sandbox,
        command[:200],
    )
    return {
        "stdout": out, "stderr": errout, "exit_code": result.exit_code,
        "timed_out": result.timed_out, "duration_ms": result.duration_ms,
        "sandbox": result.sandbox, "truncated": out_trunc or err_trunc,
        "success": result.exit_code == 0 and not result.timed_out,
    }


@register_tool(
    name="write_file",
    description=(
        "Create or overwrite a text file in the attached project's shell working copy — a "
        "TEMPORARY sandbox (files here are NOT project documents and are eventually "
        "cleaned up; use `create_document` for durable project files). Safer/more "
        "reliable than echo/heredoc for scratch work the shell will process."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within the shell working directory."},
            "content": {"type": "string", "description": "File contents (UTF-8 text)."},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> dict[str, Any]:
    from ..kit.shell import dispatch

    ws, conv, err = _shell_guard()
    if err:
        return err
    assert conv is not None
    return dispatch.write_file(ws, conv, path, content)


@register_tool(
    name="read_file",
    description="Read a text file from the attached project's shell working copy (relative path).",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within the shell working directory."},
            "offset": {"type": "integer", "description": "Start character (default 0).", "default": 0},
            "limit": {"type": "integer", "description": "Max characters (default 12000).", "default": 12000},
        },
        "required": ["path"],
    },
)
def read_file(path: str, offset: int = 0, limit: int = 12000) -> dict[str, Any]:
    from ..kit.shell import dispatch

    ws, conv, err = _shell_guard()
    if err:
        return err
    assert conv is not None
    return dispatch.read_file(ws, conv, path, offset=offset, limit=limit)


@register_tool(
    name="list_files",
    description="List files in the attached project's shell working copy (relative paths; temporary sandbox, not the project's document list — that is in the Project files block or `project_search`).",
    input_schema={"type": "object", "properties": {}},
)
def list_files() -> dict[str, Any]:
    from ..kit.shell import dispatch

    ws, conv, err = _shell_guard()
    if err:
        return err
    assert conv is not None
    return dispatch.list_files(ws, conv)


@register_tool(
    name="scratchpad_note",
    description=(
        "Jot a short free-form working note for the current conversation. Notes "
        "are re-injected into the system prompt every turn and survive automatic "
        "context compression — use them for loose reminders, intermediate "
        "findings, or anything you want to keep in view that isn't a formal "
        "checkpoint. Call with no `note` (or `read=true`) to read back your "
        "current working state: scratchpad notes, active checkpoints, and active "
        "goals (this does NOT echo the conversation transcript). Set "
        "`replace=true` to clear existing notes before writing the new one."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "The note to append. Omit to read back current state.",
            },
            "replace": {
                "type": "boolean",
                "description": "Clear existing notes before adding this one.",
                "default": False,
            },
            "read": {
                "type": "boolean",
                "description": "Read back current working state without writing.",
                "default": False,
            },
        },
    },
)
def scratchpad_note(
    note: str | None = None,
    replace: bool = False,
    read: bool = False,
) -> dict[str, Any]:
    """Append a scratchpad note, or read back the agent's working state."""
    from .internal_context import current_context
    from ..agent.scratchpad_storage import add_note, clear_notes, list_notes

    ctx = current_context()
    if ctx is None or not ctx.conversation_id:
        return {
            "error": "scratchpad_note requires an active conversation context.",
            "success": False,
        }

    # Read-back mode: surface only the non-transcript working state.
    if read or not (note and note.strip()):
        from ..agent.checkpoint_storage import list_checkpoints

        active_goals: list[dict[str, Any]] = []
        try:
            from ..kit.memory_utils import get_agent_memory

            memory = get_agent_memory(
                user_id=ctx.user_id,
                channel=ctx.channel or "_default",
                agent_id=ctx.agent_id,
            )
            if memory is not None:
                for g in memory.get_active_goals() or []:
                    active_goals.append({
                        "description": getattr(g, "description", ""),
                        "status": getattr(g, "status", "active"),
                        "priority": getattr(g, "priority", 3),
                    })
        except Exception as e:  # noqa: BLE001 - memory optional
            logger.debug(f"scratchpad read-back goals skipped: {e}")

        return {
            "notes": list_notes(ctx.conversation_id),
            "checkpoints": list_checkpoints(ctx.conversation_id),
            "active_goals": active_goals,
            "success": True,
        }

    # Write mode.
    if replace:
        clear_notes(ctx.conversation_id)
    entry = add_note(conversation_id=ctx.conversation_id, note=note)
    total = len(list_notes(ctx.conversation_id))

    return {
        "stored": entry,
        "note_count": total,
        "note": "Scratchpad notes are re-injected into system context every turn.",
        "success": True,
    }


@register_tool(
    name="update_conversation_state",
    description=(
        "Update the structured state of THIS conversation — durable, slot-based "
        "working memory that is re-injected every turn and survives automatic "
        "context compression. Prefer this over free-form scratchpad notes for "
        "things worth keeping structured: `goals` (what we're trying to achieve), "
        "`decisions` (choices locked in), `open_threads` (unresolved questions / "
        "next steps), `artifacts` (documents, drafts, or outputs produced), and "
        "`narrative` (a freeform catch-all for anything that doesn't fit a named "
        "slot). Write in your own words — only record what you have authored or "
        "confirmed, never paste raw tool/web/file output into a slot. Entries "
        "append by default; set `replace=true` to supersede the whole slot when "
        "the prior set is stale."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "slot": {
                "type": "string",
                "enum": ["goals", "decisions", "open_threads", "artifacts", "narrative"],
                "description": "Which slot to write to.",
            },
            "entries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more short entries to record in the slot.",
            },
            "replace": {
                "type": "boolean",
                "description": "Replace the whole slot instead of appending (supersede a stale set).",
                "default": False,
            },
        },
        "required": ["slot", "entries"],
    },
)
def update_conversation_state(
    slot: str,
    entries: list[str],
    replace: bool = False,
) -> dict[str, Any]:
    """Write agent-authored entries to a conversation-state slot (single-writer).

    The agent is the sole writer via this tool (author=``agent``); user edits
    arrive through the client state API (author=``user``) in a later slice. We
    only ever store what the model explicitly passes as ``entries`` — tool/web
    output is never auto-ingested into a slot (poisoning defense).
    """
    from .internal_context import current_context
    from ..agent.conversation_state_storage import SLOTS, update_slot

    ctx = current_context()
    if ctx is None or not ctx.conversation_id:
        return {
            "error": "update_conversation_state requires an active conversation context.",
            "success": False,
        }

    if slot not in SLOTS:
        return {
            "error": f"Unknown slot {slot!r}. Valid slots: {', '.join(SLOTS)}.",
            "success": False,
        }

    clean = [e for e in (entries or []) if e and e.strip()]
    if not clean:
        return {"error": "entries must contain at least one non-empty string.", "success": False}

    try:
        state = update_slot(ctx.conversation_id, slot, clean, author="agent", replace=bool(replace))
    except ValueError as e:
        return {"error": str(e), "success": False}

    return {
        "slot": slot,
        "slot_count": len(state.entries(slot)),
        "note": "Conversation state re-injected into system context every turn.",
        "success": True,
    }


@register_tool(
    name="present_exhibit",
    description=(
        "Present a rendered exhibit to the user. Declare it as a list of typed "
        "`elements` (they render in order); you may include several or call this "
        "multiple times in a turn — each exhibit joins the conversation's gallery. "
        "Re-call with the same `id` to revise an exhibit in place. Element types:\n"
        "- `mermaid`: a diagram, when a picture beats prose (flows, sequences, "
        "state machines, hierarchies, ER/class). `content` is raw Mermaid source "
        "starting with a diagram keyword (graph, flowchart, sequenceDiagram, "
        "classDiagram, stateDiagram, erDiagram, gantt, pie, mindmap, ...).\n"
        "- `choice`: ask the user to pick from `options` (with an optional "
        "`prompt`). IMPORTANT: after presenting a choice, present the options and "
        "then STOP — do NOT fabricate the user's answer or call further tools to "
        "proceed past it. Their selection arrives as their next message.\n"
        "- `table`: genuinely tabular data — `columns` (a handful; wide or huge "
        "tables render poorly in a chat column, so summarize or split big data) "
        "and `rows` (each row aligns to the columns).\n"
        "- `citation`: sources you actually used — a list of `sources`, each with "
        "a `label` (and optional `url`, `quote`, `source_type`). Web results from "
        "`web_search` are recorded as sources automatically, so don't re-list them "
        "as inline links — instead spotlight a key web source as `kind:'active'` "
        "(with a short `quote`), and add any non-web sources (docs/memory) you used. "
        "Sources default to `passive` (record-keeping); `active` is for references "
        "you may point back to. Don't pad.\n"
        "For diagrams/tables, still describe them briefly in your normal reply — "
        "an exhibit complements your text, it doesn't replace it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "minItems": 1,
                "description": "Ordered list of elements to render in the exhibit.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["mermaid", "choice", "table", "citation"],
                            "description": "Element type: 'mermaid', 'choice', 'table', or 'citation'.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Raw Mermaid diagram source (required for 'mermaid').",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Selectable options (required for 'choice'; 1-10).",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optional question shown above a 'choice' element's buttons.",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Column headers (required for 'table'; keep it to a handful).",
                        },
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                            "description": "Table rows for 'table'; each row aligns to columns.",
                        },
                        "caption": {
                            "type": "string",
                            "description": "Optional caption shown under a 'table'.",
                        },
                        "sources": {
                            "type": "array",
                            "description": "Cited sources (required for 'citation').",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "description": "Source title/name."},
                                    "url": {"type": "string", "description": "Optional link."},
                                    "quote": {"type": "string", "description": "Short relevant excerpt (for active sources)."},
                                    "kind": {
                                        "type": "string",
                                        "enum": ["active", "passive"],
                                        "description": "'active' (key reference, folds out) or 'passive' (record-keeping). Default passive.",
                                    },
                                    "source_type": {
                                        "type": "string",
                                        "enum": ["web", "memory", "doc"],
                                        "description": "Optional provenance hint.",
                                    },
                                },
                                "required": ["label"],
                            },
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional caption/title for this element.",
                        },
                    },
                    "required": ["type"],
                },
            },
            "id": {
                "type": "string",
                "description": "Optional stable id. Reuse it to amend a prior exhibit; omit for a new one.",
            },
            "title": {
                "type": "string",
                "description": "Optional title for the whole exhibit.",
            },
            "layout": {
                "type": "string",
                "enum": ["stack"],
                "description": "How elements are arranged. Only 'stack' (vertical) for now.",
                "default": "stack",
            },
        },
        "required": ["elements"],
    },
)
def present_exhibit(
    elements: list[dict[str, Any]],
    id: str | None = None,
    title: str | None = None,
    layout: str = "stack",
) -> dict[str, Any]:
    """Validate a declared exhibit and confirm it to the model.

    The exhibit itself is streamed to the client by the tool loop (derived from
    these same arguments via :func:`exhibits.exhibit_from_present_call`); this
    handler's job is to validate the declaration and return a result so the
    model gets feedback (and can re-present on error).
    """
    from pydantic import ValidationError

    from ..streaming.exhibits import (
        ALLOWED_ELEMENT_TYPES,
        exhibit_from_present_call,
        mermaid_sanity_error,
    )

    try:
        exhibit = exhibit_from_present_call(
            {"id": id, "title": title, "layout": layout, "elements": elements}
        )
    except ValidationError as e:
        allowed = ", ".join(sorted(ALLOWED_ELEMENT_TYPES))
        return {
            "error": (
                f"Invalid exhibit: {e.error_count()} problem(s). Each element needs "
                f"a 'type' (one of: {allowed}) and 'content'. Details: {e.errors()[:3]}"
            ),
            "success": False,
        }

    for idx, el in enumerate(exhibit.elements):
        if el.type == "mermaid":
            err = mermaid_sanity_error(el.content)
            if err:
                return {
                    "error": f"element[{idx}]: {err}",
                    "success": False,
                }

    return {
        "exhibit_id": exhibit.id,
        "element_count": len(exhibit.elements),
        "note": "Exhibit presented to the user.",
        "success": True,
    }


def _memory_for_ctx():
    """Resolve the AgentMemory for the active tool context, or (None, error)."""
    from .internal_context import current_context

    ctx = current_context()
    if ctx is None:
        return None, {
            "error": "No active agent context — this tool can only be called during a chat turn.",
            "success": False,
        }
    try:
        from ..kit.memory_utils import get_agent_memory
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"Memory system unavailable: {e}", "success": False}

    memory = get_agent_memory(
        user_id=ctx.user_id,
        channel=ctx.channel or "_default",
        agent_id=ctx.agent_id,
    )
    if memory is None:
        return None, {"error": "Memory system unavailable", "success": False}
    return memory, None


@register_tool(
    name="remember_this",
    description=(
        "Mark a known fact as important so it survives memory decay and ranks "
        "higher in future recall. Pass the `fact_id` from a fact you saw in the "
        "injected memory context. Use when the user signals something matters "
        "long-term (\"remember that…\", \"this is important\")."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "fact_id": {
                "type": "string",
                "description": "ID of the fact to boost (from injected memory context).",
            },
        },
        "required": ["fact_id"],
    },
)
def remember_this(fact_id: str) -> dict[str, Any]:
    """Boost a fact's salience via AgentMemory."""
    if not fact_id or not fact_id.strip():
        return {"error": "fact_id is required", "success": False}

    memory, err = _memory_for_ctx()
    if memory is None:
        return err or {"error": "Memory system unavailable", "success": False}

    try:
        updated = memory.boost_salience(fact_id.strip())
    except Exception as e:  # noqa: BLE001
        return {"error": f"Boost failed: {e}", "success": False}

    if updated is None:
        return {"error": "Fact not found", "fact_id": fact_id, "success": False}
    return {"fact_id": fact_id, "salience": updated.get("salience"), "success": True}


@register_tool(
    name="forget",
    description=(
        "Forget a known fact. By default this is a soft retire — the fact is "
        "marked as past and de-prioritized in recall, but kept for provenance. "
        "Set `hard=true` to delete it permanently. Pass the `fact_id` from a "
        "fact you saw in the injected memory context. Use when the user asks you "
        "to forget something or corrects a now-wrong fact."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "fact_id": {
                "type": "string",
                "description": "ID of the fact to forget (from injected memory context).",
            },
            "hard": {
                "type": "boolean",
                "description": "Permanently delete instead of soft-retiring.",
                "default": False,
            },
        },
        "required": ["fact_id"],
    },
)
def forget(fact_id: str, hard: bool = False) -> dict[str, Any]:
    """Soft-retire (default) or hard-delete a fact via AgentMemory."""
    if not fact_id or not fact_id.strip():
        return {"error": "fact_id is required", "success": False}

    memory, err = _memory_for_ctx()
    if memory is None:
        return err or {"error": "Memory system unavailable", "success": False}

    try:
        return memory.forget_fact(fact_id.strip(), hard=bool(hard))
    except Exception as e:  # noqa: BLE001
        return {"error": f"Forget failed: {e}", "success": False}


@register_tool(
    name="detect_language",
    description=(
        "Detect the language of a piece of text. Covers ~20 common languages and "
        "returns an ISO 639-1 code (e.g. 'fr', 'es', 'ja') plus a confidence score. "
        "Useful before translating, or to answer the user in their own language."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text whose language should be detected.",
            },
        },
        "required": ["text"],
    },
)
def detect_language(text: str) -> dict[str, Any]:
    """Detect the language of ``text`` via the TranslationKit."""
    if not text or not text.strip():
        return {"error": "text is required", "success": False}

    try:
        from ..kit.translation import get_translation_kit

        language, confidence = get_translation_kit().detect_language_level_i(text)
    except Exception as e:  # noqa: BLE001 - model load / inference failure
        return {"error": f"Language detection failed: {e}", "success": False}

    return {
        "language": language,
        "confidence": confidence,
        "success": True,
    }


@register_tool(
    name="translate_text",
    description=(
        "Translate text into a target language using the on-device NLLB-200 model "
        "(200+ languages). Pass the target as an ISO 639-1 code ('fr', 'es', 'zh') "
        "for common languages, or a full NLLB code ('fra_Latn', 'zho_Hans') for the "
        "wider set. Source language is auto-handled. On an unsupported code the "
        "result lists the common supported languages."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to translate.",
            },
            "target_language": {
                "type": "string",
                "description": "Target language: ISO 639-1 (e.g. 'fr') or NLLB code (e.g. 'fra_Latn').",
            },
        },
        "required": ["text", "target_language"],
    },
)
def translate_text(text: str, target_language: str) -> dict[str, Any]:
    """Translate ``text`` into ``target_language`` via the TranslationKit."""
    if not text or not text.strip():
        return {"error": "text is required", "success": False}
    if not target_language or not target_language.strip():
        return {"error": "target_language is required", "success": False}

    # NLLB codes look like 'fra_Latn' (script suffix); bare codes/names are level 1.
    level = 2 if "_" in target_language else 1

    try:
        from ..kit.translation import get_translation_kit

        translated = get_translation_kit().translate_text(
            text, target_language, target_language_level=level
        )
    except ValueError as e:
        # Unsupported language — surface the common options for discovery.
        from ..kit.translation import LanguageLexicon

        return {
            "error": str(e),
            "supported": LanguageLexicon().level_i_languages,
            "hint": "Use an ISO 639-1 code above, or a full NLLB code like 'fra_Latn'.",
            "success": False,
        }
    except Exception as e:  # noqa: BLE001 - model load / inference failure
        return {"error": f"Translation failed: {e}", "success": False}

    return {
        "translated_text": translated,
        "target_language": target_language,
        "success": True,
    }


# =============================================================================
# Web search / research (Tavily SDK primary, Brave REST fallback)
# =============================================================================

# Short-TTL in-process cache of identical queries: key -> (expiry_epoch, payload)
_SEARCH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
# Bound the cache so a long research turn can't grow it without limit. Expired
# entries are only dropped lazily on read, so we also prune on insert.
_SEARCH_CACHE_MAX = 256
_SEARCH_TIMEOUT = 15.0  # default cap; overridable via `search.timeout`


def _cache_put(key: str, expiry: float, payload: dict[str, Any]) -> None:
    """Insert into the search cache, pruning to stay under ``_SEARCH_CACHE_MAX``.

    Drops expired entries first, then the soonest-to-expire, so the cache can't
    grow unbounded across a deep research turn.
    """
    _SEARCH_CACHE[key] = (expiry, payload)
    if len(_SEARCH_CACHE) <= _SEARCH_CACHE_MAX:
        return
    now = time.time()
    for k in [k for k, (exp, _) in _SEARCH_CACHE.items() if exp <= now]:
        _SEARCH_CACHE.pop(k, None)
    while len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX:
        oldest = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
        _SEARCH_CACHE.pop(oldest, None)


def _search_timeout() -> float:
    """Per-call wall-clock cap for web_search backends (seconds)."""
    from ..config import get_config_manager

    try:
        return float(get_config_manager().get("search.timeout", _SEARCH_TIMEOUT) or _SEARCH_TIMEOUT)
    except (TypeError, ValueError):
        return _SEARCH_TIMEOUT


def _resolve_search_key(config_key: str, env_var: str) -> str | None:
    """Config value first, then env var fallback."""
    from ..config import get_config_manager

    val = get_config_manager().get(config_key)
    if val:
        return str(val)
    import os

    return os.environ.get(env_var) or None


def _http_get_json(url: str, *, headers: dict, params: dict) -> dict[str, Any]:
    """GET with small retry/backoff on transient errors (timeouts, 429, 5xx)."""
    import httpx

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=_search_timeout()) as client:
                resp = client.get(url, headers=headers, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"transient {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Capability registry — the single source of truth for what each web tool
# advertises (per active backend) and which params its handler forwards. The
# model only ever sees the *active* backend's real capabilities, assembled at
# tool-listing time by the pre-check in `get_internal_tools`.
# -----------------------------------------------------------------------------

# Ordered backend preference when none is explicitly configured / available.
_BACKEND_ORDER: tuple[str, ...] = ("tavily", "brave")

# Tavily-only tools — advertised to the model only when Tavily is the active
# backend, but always registered (so a stale call self-guards instead of 404ing).
_CAPABILITY_GATED_TOOLS: frozenset[str] = frozenset(
    {"web_extract", "web_map", "web_crawl", "web_research"}
)

# Tavily `time_range` → Brave `freshness` codes.
_FRESHNESS_MAP = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


def _enum(values: list[str], desc: str) -> dict[str, Any]:
    return {"type": "string", "enum": values, "description": desc}


def _strarr(desc: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": desc}


# Per-backend, per-tool extra params (beyond each tool's backend-independent base).
SEARCH_CAPABILITIES: dict[str, dict[str, Any]] = {
    "tavily": {
        "label": "Tavily",
        "tools": {
            "web_search": {
                "summary": (
                    "search depth, topic (news/finance), time range, domain "
                    "include/exclude, and an optional LLM-generated answer"
                ),
                "params": {
                    "search_depth": _enum(
                        ["basic", "advanced"],
                        "Search depth; 'advanced' digs deeper (slower, costs more).",
                    ),
                    "topic": _enum(
                        ["general", "news", "finance"],
                        "Search topic; 'news' favors recent reporting.",
                    ),
                    "time_range": _enum(
                        ["day", "week", "month", "year"],
                        "Restrict to results from the last day/week/month/year.",
                    ),
                    "include_domains": _strarr("Only return results from these domains."),
                    "exclude_domains": _strarr("Never return results from these domains."),
                    "include_answer": {
                        "type": "boolean",
                        "description": "Include a short LLM-generated answer to the query.",
                    },
                },
            },
            "web_extract": {"summary": "pull the full cleaned content of specific URLs", "params": {}},
            "web_map": {"summary": "discover a site's URL graph from a base URL", "params": {}},
            "web_crawl": {"summary": "follow links from a base URL and extract pages", "params": {}},
            "web_research": {
                "summary": "agentic deep-research report with citations (slow; minutes)",
                "params": {},
            },
        },
    },
    "brave": {
        "label": "Brave",
        "tools": {
            "web_search": {
                "summary": (
                    "safe-search level, a freshness window, a result-type filter, "
                    "and extra per-result snippets"
                ),
                "params": {
                    "safesearch": _enum(
                        ["off", "moderate", "strict"], "Adult-content filtering level."
                    ),
                    "time_range": _enum(
                        ["day", "week", "month", "year"],
                        "Restrict to results from the last day/week/month/year.",
                    ),
                    "result_filter": {
                        "type": "string",
                        "description": (
                            "Comma-separated result types to include "
                            "(e.g. 'web,news,discussions')."
                        ),
                    },
                    "extra_snippets": {
                        "type": "boolean",
                        "description": (
                            "Return up to 5 extra excerpts per result for richer "
                            "grounding in one call (fewer follow-up extractions)."
                        ),
                    },
                },
            },
        },
    },
}

_TOOL_BASE_DESC: dict[str, str] = {
    "web_search": (
        "Search the public web and get back a ranked list of results "
        "({title, url, snippet}). Use for current events, facts you're unsure "
        "about, documentation lookups, or anything beyond your training data. "
        "Web results are recorded as sources automatically — no need to repeat "
        "them as inline links."
    ),
    "web_extract": (
        "Extract the full cleaned text/markdown of one or more specific web pages "
        "(when a search snippet isn't enough). Pass the `urls` you want to read in "
        "depth; large content is stored and retrievable."
    ),
    "web_map": (
        "Map a website's structure: given a base `url`, return the graph of "
        "discovered page URLs (no page content). Use to scope a site before "
        "extracting specific pages with web_extract."
    ),
    "web_crawl": (
        "Crawl a website from a base `url`, following links and extracting page "
        "content (bounded by depth/breadth/limit). Use when you need many pages of "
        "a site, not just one; large output is stored and retrievable."
    ),
    "web_research": (
        "Run an agentic deep-research task on a `query`: the provider explores "
        "multiple sources and returns a synthesized report with citations. "
        "SLOW (can take minutes) — use only for genuinely involved research, not a "
        "quick lookup (use web_search for those). The report is stored; its sources "
        "are recorded automatically."
    ),
}


def _base_tool_schema(tool: str) -> dict[str, Any]:
    """Backend-independent core schema for a web tool."""
    if tool == "web_search":
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default from config, usually 5).",
                },
            },
            "required": ["query"],
        }
    if tool == "web_extract":
        return {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more page URLs to extract full content from (max 20).",
                },
                "extract_depth": _enum(
                    ["basic", "advanced"], "'advanced' parses more (tables, embeds); slower."
                ),
                "format": _enum(["markdown", "text"], "Output format for extracted content."),
            },
            "required": ["urls"],
        }
    if tool == "web_map":
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Base URL whose site structure to map."},
                "max_depth": {
                    "type": "integer",
                    "description": "How many links deep to traverse (default 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max URLs to return (default 50, capped at 200).",
                },
            },
            "required": ["url"],
        }
    if tool == "web_crawl":
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Base URL to crawl from."},
                "max_depth": {"type": "integer", "description": "Link depth to follow (default 1)."},
                "limit": {
                    "type": "integer",
                    "description": "Max pages to return (default 20, capped at 50).",
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional natural-language guidance for what to crawl.",
                },
            },
            "required": ["url"],
        }
    if tool == "web_research":
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The research question."},
                "depth": _enum(
                    ["auto", "mini", "pro"],
                    "Research effort: 'mini' (fast), 'pro' (deep), 'auto' (provider decides).",
                ),
            },
            "required": ["query"],
        }
    return {"type": "object", "properties": {}}


def build_tool_schema(tool: str, backend: str) -> dict[str, Any]:
    """Base schema + the active backend's extra params for `tool`."""
    schema = _base_tool_schema(tool)
    caps = SEARCH_CAPABILITIES.get(backend, {}).get("tools", {}).get(tool)
    if caps:
        for pname, pschema in caps["params"].items():
            schema["properties"][pname] = pschema
    return schema


def build_tool_description(tool: str, backend: str) -> str:
    """Base description + a one-line capability summary for the active backend."""
    base = _TOOL_BASE_DESC.get(tool, "")
    caps = SEARCH_CAPABILITIES.get(backend, {}).get("tools", {}).get(tool)
    label = SEARCH_CAPABILITIES.get(backend, {}).get("label", backend)
    if caps and caps.get("summary"):
        return f"{base}\nCapabilities ({label}): {caps['summary']}."
    return base


def _backend_has_key(name: str) -> bool:
    if name == "tavily":
        return bool(_resolve_search_key("search.tavily_api_key", "TAVILY_API_KEY"))
    if name == "brave":
        return bool(_resolve_search_key("search.brave_api_key", "BRAVE_API_KEY"))
    return False


def resolve_active_search_backend() -> str | None:
    """The backend `web_search` will hit first: the configured primary if its key
    is present, else the first backend with a key. None when none is configured."""
    from ..config import get_config_manager

    primary = (get_config_manager().get("search.backend", "tavily") or "tavily").lower()
    order = [primary] + [b for b in _BACKEND_ORDER if b != primary]
    for name in order:
        if _backend_has_key(name):
            return name
    return None


# ---- Backends ---------------------------------------------------------------

def _tavily_client():
    """Lazily build a Tavily SDK client from the resolved key. Raises RuntimeError
    when the key is missing or the SDK isn't installed (callers convert to error
    dicts). Sync client matches the synchronous tool-execution path."""
    key = _resolve_search_key("search.tavily_api_key", "TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Tavily API key not configured (search.tavily_api_key / TAVILY_API_KEY)")
    try:
        from tavily import TavilyClient
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise RuntimeError(f"tavily-python is not installed: {e}") from e
    return TavilyClient(api_key=key)


def _tavily_search(query: str, max_results: int, **opts: Any) -> dict[str, Any]:
    """Query Tavily via the SDK; forwards only Tavily-supported params. Deliberately
    omits `include_raw_content` (full content is web_extract's job) so the echoed
    result stays compact + auto-capturable. Raises on failure."""
    client = _tavily_client()
    # Cap the call so a slow/hung search can't block the turn (the Tavily SDK
    # otherwise defaults to ~60s). Brave is capped in `_http_get_json`.
    kwargs: dict[str, Any] = {"max_results": max_results, "timeout": _search_timeout()}
    for p in ("search_depth", "topic", "time_range", "include_domains", "exclude_domains", "include_answer"):
        if opts.get(p) is not None:
            kwargs[p] = opts[p]
    data = client.search(query=query, **kwargs)
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": r.get("score"),
            "published_date": r.get("published_date"),
        }
        for r in (data.get("results") or [])
    ]
    payload: dict[str, Any] = {"results": results}
    if data.get("answer"):
        payload["answer"] = data["answer"]
    return payload


def _brave_search(query: str, max_results: int, **opts: Any) -> dict[str, Any]:
    """Query Brave REST; forwards only Brave-supported params (mapping
    time_range→freshness). Raises on failure."""
    key = _resolve_search_key("search.brave_api_key", "BRAVE_API_KEY")
    if not key:
        raise RuntimeError("Brave API key not configured (search.brave_api_key / BRAVE_API_KEY)")
    params: dict[str, Any] = {"q": query, "count": max_results}
    if opts.get("safesearch"):
        params["safesearch"] = opts["safesearch"]
    if opts.get("time_range") in _FRESHNESS_MAP:
        params["freshness"] = _FRESHNESS_MAP[opts["time_range"]]
    if opts.get("result_filter"):
        params["result_filter"] = opts["result_filter"]
    if opts.get("extra_snippets"):
        params["extra_snippets"] = "true"
    data = _http_get_json(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params=params,
    )
    web = (data.get("web") or {}).get("results") or []
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
            "published_date": r.get("page_age") or r.get("age"),
            **({"extra_snippets": r["extra_snippets"]} if r.get("extra_snippets") else {}),
        }
        for r in web
    ]
    return {"results": results}


_SEARCH_BACKENDS = {"tavily": _tavily_search, "brave": _brave_search}


def _budget_block() -> dict[str, Any]:
    """Budget + cost awareness to stamp onto a search result so the model can pace
    itself by both remaining calls and estimated spend. ``limit`` 0 ⇒ unlimited."""
    from ..agent.search_budget import snapshot

    used, limit, remaining, cost_used = snapshot()
    return {
        "used": used,
        "limit": limit,
        "remaining": "unlimited" if remaining is None else remaining,
        "est_cost_usd": round(cost_used, 4),
    }


def _check_search_budget(weight: int = 1) -> dict[str, Any] | None:
    """Charge ``weight`` calls against the per-turn search budget (Foundation #5).

    Returns a budget-error dict when the turn's window is exhausted (caller must
    return it without hitting a backend), or ``None`` when the call may proceed.
    No active window ⇒ unlimited. ``weight`` lets a costly call (deep web_research)
    charge more than a basic web_search.
    """
    from ..agent.search_budget import consume

    allowed, used, limit = consume(weight)
    if allowed:
        return None
    return {
        "results": [],
        "count": 0,
        "success": False,
        "error": f"web-search budget exhausted for this turn ({used}/{limit} calls; "
        "raise the research/search per-turn limit, or set it to 0 to disable)",
        "budget": _budget_block(),
    }


def _record_search_spend(backend: str, credits: int) -> None:
    """Best-effort log of one search call to the usage ledger (source='search') and
    charge its estimated cost to the active per-turn budget window.

    Tavily bills per credit; Brave bills per request. Both are estimates. Never
    raises — metering must not break a turn (mirrors usage_ledger's own contract).
    """
    try:
        from ..config import get_config_manager
        from ..agent.usage_ledger import record_usage
        from ..agent.search_budget import attribution, charge_cost

        cfg = get_config_manager()
        if backend == "brave":
            # Brave bills per request (~$5/1k); `credits` here counts requests.
            per_unit = float(cfg.get("search.brave_cost_per_request_usd", 0.005) or 0.0)
            units = {"queries": 1, "requests": credits}
            pricing = {"per_request_usd": per_unit, "requests": credits}
        else:  # tavily (and any Tavily-only tool)
            per_unit = float(cfg.get("search.cost_per_credit_usd", 0.008) or 0.0)
            units = {"queries": 1, "credits": credits}
            pricing = {"per_credit_usd": per_unit, "credits": credits}
        cost_total = round(per_unit * credits, 6)
        charge_cost(cost_total)  # surface running spend to the model via _budget_block
        conv_id, agent_id = attribution()
        record_usage(
            source="search",
            model=backend,
            provider=backend,
            conversation_id=conv_id,
            agent_id=agent_id,
            units=units,
            cost={"cost_total": cost_total, "currency": "USD", "pricing_snapshot": pricing},
        )
    except Exception as e:  # noqa: BLE001 — metering is best-effort
        logger.debug(f"search usage_ledger record failed (backend={backend}): {e}")


@register_tool(
    name="web_search",
    description=_TOOL_BASE_DESC["web_search"],
    input_schema=_base_tool_schema("web_search"),
)
def web_search(query: str, max_results: int | None = None, **opts: Any) -> dict[str, Any]:
    """Search the web via the active backend (Tavily SDK) with Brave fallback.

    Extra keyword params are backend-specific knobs advertised by the capability
    pre-check; each backend forwards only what it supports and ignores the rest.
    """
    if not query or not query.strip():
        return {"error": "query is required", "success": False, "results": []}

    from ..config import get_config_manager

    cfg = get_config_manager()
    backend = (cfg.get("search.backend", "tavily") or "tavily").lower()
    fallback_enabled = bool(cfg.get("search.fallback_enabled", True))
    if max_results is None:
        max_results = int(cfg.get("search.max_results", 5))
    ttl = int(cfg.get("search.cache_ttl_seconds", 300))

    # Cache check (keyed by backend + normalized query + count + opts)
    cache_key = f"{backend}:{max_results}:{query.strip().lower()}:{sorted(opts.items())!r}"
    now = time.time()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and cached[0] > now:
        # A cache hit spends nothing, so it bypasses the budget gate.
        return {**cached[1], "cached": True, "success": True, "budget": _budget_block()}

    # Per-turn budget gate (spend only — cached hits above are free).
    budget_error = _check_search_budget()
    if budget_error is not None:
        return budget_error

    # Credits estimate for the ledger: Tavily bills advanced depth at 2 credits.
    credits = 2 if str(opts.get("search_depth", "")).lower() == "advanced" else 1

    # Backend order: configured primary, then the other (if fallback enabled)
    primary = backend if backend in _SEARCH_BACKENDS else "tavily"
    order = [primary]
    if fallback_enabled:
        order += [b for b in _BACKEND_ORDER if b != primary]

    errors: list[str] = []
    for name in order:
        try:
            payload = _SEARCH_BACKENDS[name](query, max_results, **opts)
        except Exception as e:  # noqa: BLE001 - backend/network failure → try fallback
            errors.append(f"{name}: {e}")
            logger.warning(f"web_search backend '{name}' failed: {e}")
            continue
        results = payload.get("results") or []
        if not results:
            errors.append(f"{name}: no results")
            continue
        response: dict[str, Any] = {
            "results": results,
            "count": len(results),
            "backend": name,
            "cached": False,
            "success": True,
        }
        if payload.get("answer"):
            response["answer"] = payload["answer"]
        if ttl > 0:
            cacheable = {k: response[k] for k in ("results", "count", "backend")}
            if "answer" in response:
                cacheable["answer"] = response["answer"]
            _cache_put(cache_key, now + ttl, cacheable)
        _record_search_spend(name, credits)
        # Stamp budget/cost AFTER recording spend so it reflects this call.
        response["budget"] = _budget_block()
        return response

    return {
        "results": [],
        "count": 0,
        "success": False,
        "error": "all search backends failed or returned no results: " + "; ".join(errors),
        "budget": _budget_block(),
    }


@register_tool(
    name="web_extract",
    description=_TOOL_BASE_DESC["web_extract"],
    input_schema=_base_tool_schema("web_extract"),
)
def web_extract(
    urls: list[str] | str,
    extract_depth: str = "basic",
    format: str = "markdown",
    **opts: Any,
) -> dict[str, Any]:
    """Extract full page content for specific URLs via the Tavily SDK.

    Tavily-only (no Brave equivalent): returns a clear error when Tavily isn't
    configured. Large content rides the normal oversize/stored-output handling.
    """
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for u in (urls or []) if isinstance(u, str) and u.strip()][:20]  # abuse cap
    if not urls:
        return {"error": "urls is required", "success": False}

    depth = extract_depth if extract_depth in ("basic", "advanced") else "basic"
    fmt = format if format in ("markdown", "text") else "markdown"

    # Cache identical extractions: research turns re-read the same pages while
    # verifying claims, and each re-extract re-bills (1 credit / 5 URLs). Pages
    # are stable within a turn; cache hits are free (no budget/spend).
    from ..config import get_config_manager

    ttl = int(get_config_manager().get("search.cache_ttl_seconds", 300))
    cache_key = f"extract:{depth}:{fmt}:{sorted(urls)!r}"
    now = time.time()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return {**cached[1], "cached": True, "success": True}

    try:
        client = _tavily_client()
    except RuntimeError as e:
        return {"error": f"web_extract requires Tavily: {e}", "success": False}

    # Normalize to the SDK's accepted values, then pass via an Any-typed kwargs
    # dict so the Literal-typed SDK params accept our (validated) strings.
    kwargs: dict[str, Any] = {
        "urls": urls,
        "extract_depth": depth,
        "format": fmt,
    }
    try:
        data = client.extract(**kwargs)
    except Exception as e:  # noqa: BLE001 - SDK/network failure
        return {"error": f"Extract failed: {e}", "success": False}

    results = [
        {"url": r.get("url", ""), "content": r.get("raw_content") or r.get("content", "")}
        for r in (data.get("results") or [])
    ]
    # Tavily bills extraction at 1 credit per 5 successful URLs.
    if results:
        _record_search_spend("tavily", (len(results) + 4) // 5)
    response = {
        "results": results,
        "failed": data.get("failed_results") or [],
        "count": len(results),
        "success": True,
    }
    # Bound cached-entry size: extracted pages can be large; don't let a few
    # jumbo extractions dominate the in-process cache's memory.
    if ttl > 0 and results and sum(len(r["content"]) for r in results) <= 200_000:
        _cache_put(cache_key, now + ttl, {"results": results, "failed": response["failed"],
                                          "count": len(results)})
    return response


@register_tool(
    name="web_map",
    description=_TOOL_BASE_DESC["web_map"],
    input_schema=_base_tool_schema("web_map"),
)
def web_map(url: str, max_depth: int = 1, limit: int = 50, **opts: Any) -> dict[str, Any]:
    """Map a site's URL graph via the Tavily SDK (Tavily-only). Output is hard-capped."""
    if not url or not url.strip():
        return {"error": "url is required", "success": False}
    limit = max(1, min(int(limit or 50), 200))

    try:
        client = _tavily_client()
    except RuntimeError as e:
        return {"error": f"web_map requires Tavily: {e}", "success": False}

    try:
        data = client.map(url=url, max_depth=max_depth, limit=limit)
    except Exception as e:  # noqa: BLE001 - SDK/network failure
        return {"error": f"Map failed: {e}", "success": False}

    found = data.get("results") or data.get("urls") or []
    found = found[:limit]
    # Tavily bills map at 1 credit per 10 pages.
    if found:
        _record_search_spend("tavily", (len(found) + 9) // 10)
    return {"base_url": url, "urls": found, "count": len(found), "success": True}


@register_tool(
    name="web_crawl",
    description=_TOOL_BASE_DESC["web_crawl"],
    input_schema=_base_tool_schema("web_crawl"),
)
def web_crawl(
    url: str,
    max_depth: int = 1,
    limit: int = 20,
    instructions: str | None = None,
    **opts: Any,
) -> dict[str, Any]:
    """Crawl a site from a base URL via the Tavily SDK (Tavily-only). Page count
    is hard-capped; large content rides the existing oversize/stored-output path."""
    if not url or not url.strip():
        return {"error": "url is required", "success": False}
    limit = max(1, min(int(limit or 20), 50))

    try:
        client = _tavily_client()
    except RuntimeError as e:
        return {"error": f"web_crawl requires Tavily: {e}", "success": False}

    kwargs: dict[str, Any] = {"url": url, "max_depth": max_depth, "limit": limit}
    if instructions and instructions.strip():
        kwargs["instructions"] = instructions
    try:
        data = client.crawl(**kwargs)
    except Exception as e:  # noqa: BLE001 - SDK/network failure
        return {"error": f"Crawl failed: {e}", "success": False}

    pages = [
        {"url": r.get("url", ""), "content": r.get("raw_content") or r.get("content", "")}
        for r in (data.get("results") or [])
    ][:limit]
    # Tavily bills crawl at 1 credit per 10 pages.
    if pages:
        _record_search_spend("tavily", (len(pages) + 9) // 10)
    return {"base_url": url, "pages": pages, "count": len(pages), "success": True}


def _poll_research(client: Any, request_id: str) -> dict[str, Any]:
    """Poll Tavily ``get_research`` until the task completes/fails or the deadline hits.

    Tavily's Research API is async — ``research()`` only initiates a task; the
    report arrives by polling. Returns the completed payload, or an
    ``{"error": ...}`` dict on failure/timeout/cancellation. Checks the ambient
    run-cancel flag between polls (best-effort) so a user's Stop isn't blocked
    for minutes; transient poll errors retry until the deadline.
    """
    from ..config import get_config_manager

    cfg = get_config_manager()
    # None-check, not `or`: an explicit 0 means "don't wait" (immediate timeout),
    # and must not silently become the default.
    _raw_timeout = cfg.get("web_research.poll_timeout_seconds", 240)
    timeout_s = float(240 if _raw_timeout is None else _raw_timeout)
    _raw_interval = cfg.get("web_research.poll_interval_seconds", 5)
    interval_s = max(1.0, float(5 if _raw_interval is None else _raw_interval))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:  # cancel check is best-effort — never let it break the poll
            from ..streaming.tool_loop import _ambient_cancel_check
            if _ambient_cancel_check():
                return {"error": "Research cancelled (run stopped while waiting for the report)"}
        except Exception:  # noqa: BLE001
            pass
        try:
            data = client.get_research(request_id)
        except Exception as e:  # noqa: BLE001 - transient poll failure → retry until deadline
            logger.debug(f"web_research poll error (request_id={request_id}): {e}")
            data = None
        if isinstance(data, dict):
            status = str(data.get("status") or "").lower()
            if status == "completed":
                return data
            if status == "failed":
                return {"error": f"Research task failed: {data.get('error') or 'provider reported failure'}"}
        time.sleep(interval_s)
    return {
        "error": f"Research timed out after {int(timeout_s)}s (request_id={request_id}); "
        "try depth='mini' or a narrower query"
    }


@register_tool(
    name="web_research",
    description=_TOOL_BASE_DESC["web_research"],
    input_schema=_base_tool_schema("web_research"),
)
def web_research(query: str, depth: str = "auto", **opts: Any) -> dict[str, Any]:
    """Agentic deep research via the Tavily SDK (Tavily-only). Long-running, so it
    runs with a bounded timeout and is gated by `web_research.enabled` (default on);
    a truly long research is the background-job path (future). Returns a report plus
    normalized `results` ({title, url}) so its sources are auto-captured as citations.
    """
    from ..config import get_config_manager

    cfg = get_config_manager()
    if not query or not query.strip():
        return {"error": "query is required", "success": False}
    if not cfg.get("web_research.enabled", True):
        return {"error": "web_research is disabled (web_research.enabled)", "success": False}

    model = depth if depth in ("auto", "mini", "pro") else "auto"

    # Cache identical (query, depth) research: a deep report costs 5–20 credits,
    # so re-serving a repeated query is a big saver. Cache hits are free (no gate).
    research_ttl = int(cfg.get("web_research.cache_ttl_seconds", 1800))
    cache_key = f"research:{model}:{query.strip().lower()}"
    now = time.time()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return {**cached[1], "cached": True, "success": True, "budget": _budget_block()}

    # Per-turn budget gate — deep research is one call, but a costly one, so it
    # charges a heavier weight against the budget.
    weight = int(cfg.get("web_research.budget_weight", 3) or 1)
    budget_error = _check_search_budget(weight)
    if budget_error is not None:
        return {"error": budget_error["error"], "success": False, "budget": budget_error["budget"]}

    try:
        client = _tavily_client()
    except RuntimeError as e:
        return {"error": f"web_research requires Tavily: {e}", "success": False}

    # Initiate the async research task ({request_id, status, ...} in ~200ms —
    # the `timeout` here is only the HTTP timeout on the initiation POST).
    try:
        init = client.research(input=query, model=model, timeout=_search_timeout())
    except Exception as e:  # noqa: BLE001 - SDK/network failure
        return {"error": f"Research failed to start: {e}", "success": False}
    if not isinstance(init, dict):
        return {"error": "Research returned no structured result", "success": False}

    # Deep research burns many credits; Tavily runs (and bills) the task
    # server-side once initiated, so record spend now — not on completion.
    _record_search_spend("tavily", {"mini": 5, "auto": 10, "pro": 20}.get(model, 10))

    request_id = init.get("request_id")
    if request_id:
        data = _poll_research(client, str(request_id))
        if data.get("error"):
            return {"error": data["error"], "success": False, "budget": _budget_block()}
    else:
        # Defensive: an SDK/plan that returns the finished report inline.
        data = init

    # The completed payload carries the report in `content` (+ `sources`);
    # older/alternate shapes are checked as fallbacks. Normalize citation-like
    # entries into {title, url} so the tool-loop auto-capture records sources.
    report = data.get("content") or data.get("answer") or data.get("report") or ""
    raw_sources = data.get("sources") or data.get("results") or data.get("citations") or []
    results = []
    for s in raw_sources:
        if isinstance(s, dict):
            results.append({"title": s.get("title") or s.get("url", ""), "url": s.get("url", "")})
        elif isinstance(s, str):
            results.append({"title": s, "url": s})
    if not str(report).strip():
        # Never success:True with nothing — an empty report is a failure the
        # model should react to, not silently accept.
        return {
            "error": "Research completed but returned an empty report — try a "
            "narrower query or depth='mini'",
            "success": False,
            "budget": _budget_block(),
        }
    response = {"report": report, "results": results, "count": len(results), "success": True}
    if research_ttl > 0:
        _cache_put(cache_key, now + research_ttl, {"report": report, "results": results,
                                                   "count": len(results)})
    response["budget"] = _budget_block()
    return response


# =============================================================================
# Public API
# =============================================================================

_WEB_TOOLS: frozenset[str] = frozenset(
    {"web_search", "web_extract", "web_map", "web_crawl", "web_research"}
)


def _advertise_tool(tool: InternalTool, backend: str | None) -> ToolInfo:
    """Build the model-facing ToolInfo, tailoring web tools to the active backend."""
    if backend and tool.name in _WEB_TOOLS:
        return ToolInfo(
            name=tool.name,
            description=build_tool_description(tool.name, backend),
            input_schema=build_tool_schema(tool.name, backend),
            server_name=INTERNAL_SERVER_NAME,
        )
    return ToolInfo(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        server_name=INTERNAL_SERVER_NAME,
    )


def get_internal_tools() -> list[ToolInfo]:
    """
    Get all registered internal tools as ToolInfo objects (model-facing list).

    Capability pre-check: inventories the active search backend and advertises
    web tools tailored to it — `web_extract`/`web_map` only when Tavily is active,
    and `web_search` only when *some* backend is configured. Tools stay registered
    (executable) regardless; this only governs what the model is told about.
    """
    backend = resolve_active_search_backend()
    shell_on = _shell_allowed()
    doc_write_on = _document_write_tools_enabled()
    tools: list[ToolInfo] = []
    for tool in _INTERNAL_TOOLS.values():
        if tool.name in _CAPABILITY_GATED_TOOLS and backend != "tavily":
            continue
        if tool.name == "web_search" and backend is None:
            continue
        if tool.name in _SHELL_TOOL_NAMES and not shell_on:
            continue  # agent shells are opt-in per-workspace (workspace.allow_shell)
        if tool.name in _DOCUMENT_WRITE_TOOL_NAMES and not doc_write_on:
            continue  # default-on, config opt-out (workspace_agent_write_tools)
        if tool.name in _STATE_TOOL_NAMES and not _conversation_state_enabled():
            continue  # default-on, config opt-out (context.conversation_state_enabled)
        tools.append(_advertise_tool(tool, backend))
    return tools


_SHELL_TOOL_NAMES: frozenset[str] = frozenset({"run_command", "write_file", "read_file", "list_files"})
_DOCUMENT_WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {"create_document", "update_document", "append_to_document", "edit_document",
     "delete_document", "rename_document"}
)
_STATE_TOOL_NAMES: frozenset[str] = frozenset({"update_conversation_state"})


def _conversation_state_enabled() -> bool:
    """Structured conversation state is ON by default; settings opt-out."""
    try:
        from ..config import get_config_manager

        return bool(get_config_manager().get("context.conversation_state_enabled", True))
    except Exception:  # pragma: no cover - defensive
        return True


def _document_write_tools_enabled() -> bool:
    """Project document write tools are ON by default; settings opt-out."""
    try:
        from ..kit.agent_memory.config import get_settings

        return bool(get_settings().workspace_agent_write_tools)
    except Exception:  # pragma: no cover - defensive
        return True


def _shell_allowed() -> bool:
    """Shell tools are gated **per-workspace**: the turn's attached workspace must have
    ``allow_shell=true``. No workspace attached → no shell."""
    workspace_id, _conv = _shell_context()
    if not workspace_id:
        return False
    from ..kit.workspaces import repository

    ws = repository.get_workspace(workspace_id)
    return bool(ws and ws.get("allow_shell"))


# Legacy → current tool names. Old conversations, procedural-memory records, and
# per-profile tool lists may still say `workspace_search` (pre-Projects rename);
# they must keep resolving/executing.
_TOOL_ALIASES: dict[str, str] = {"workspace_search": "project_search"}


def resolve_tool_name(name: str) -> str:
    """Map a (possibly legacy) tool name to its current registered name."""
    return _TOOL_ALIASES.get(name, name)


def legacy_names_for(name: str) -> list[str]:
    """Reverse alias lookup: the legacy names that map to ``name`` (for
    name-keyed filters like per-profile allow/block lists)."""
    return [old for old, new in _TOOL_ALIASES.items() if new == name]


def find_internal_tool(name: str) -> ToolInfo | None:
    """
    Find an internal tool by name.

    Args:
        name: Tool name to look up

    Returns:
        ToolInfo if found, None otherwise
    """
    tool = _INTERNAL_TOOLS.get(resolve_tool_name(name))
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
    tool = _INTERNAL_TOOLS.get(resolve_tool_name(name))
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
    """Check if a tool name is an internal tool (legacy aliases included)."""
    return resolve_tool_name(name) in _INTERNAL_TOOLS

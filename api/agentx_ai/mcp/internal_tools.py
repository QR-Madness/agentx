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
    "recall_user_history",
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
    name="checkpoint",
    description=(
        "Record a checkpoint of where you are in the current task. The "
        "checkpoint is re-injected into the system prompt every turn and "
        "survives automatic context compression. Use when you have made a "
        "non-trivial decision, completed a sub-step, or want to lock in "
        "intent before continuing."
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
        },
        "required": ["summary"],
    },
)
def checkpoint(
    summary: str,
    decisions: list[str] | None = None,
    next_step: str = "",
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
    )
    total = len(list_checkpoints(ctx.conversation_id))

    return {
        "stored": entry,
        "checkpoint_count": total,
        "note": "Checkpoint will be re-injected into system context every turn.",
        "success": True,
    }


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
        "a `label` (and optional `url`, `quote`, `source_type`). Mark a source "
        "`kind:'active'` (with a short `quote`) only if it's a key reference you "
        "may point back to; otherwise leave it `passive` (record-keeping). Add "
        "genuinely useful sources only — don't pad.\n"
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
# Web search (Tavily primary, Brave REST fallback)
# =============================================================================

# Short-TTL in-process cache of identical queries: key -> (expiry_epoch, results)
_SEARCH_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_SEARCH_TIMEOUT = 15.0


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
            with httpx.Client(timeout=_SEARCH_TIMEOUT) as client:
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


def _http_post_json(url: str, *, headers: dict, json_body: dict) -> dict[str, Any]:
    """POST with small retry/backoff on transient errors (timeouts, 429, 5xx)."""
    import httpx

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=_SEARCH_TIMEOUT) as client:
                resp = client.post(url, headers=headers, json=json_body)
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


def _tavily_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Query Tavily; returns normalized {title, url, snippet} list. Raises on failure."""
    key = _resolve_search_key("search.tavily_api_key", "TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Tavily API key not configured (search.tavily_api_key / TAVILY_API_KEY)")
    data = _http_post_json(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in (data.get("results") or [])
    ]


def _brave_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Query Brave REST; returns normalized {title, url, snippet} list. Raises on failure."""
    key = _resolve_search_key("search.brave_api_key", "BRAVE_API_KEY")
    if not key:
        raise RuntimeError("Brave API key not configured (search.brave_api_key / BRAVE_API_KEY)")
    data = _http_get_json(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params={"q": query, "count": max_results},
    )
    web = (data.get("web") or {}).get("results") or []
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        }
        for r in web
    ]


_SEARCH_BACKENDS = {"tavily": _tavily_search, "brave": _brave_search}


@register_tool(
    name="web_search",
    description=(
        "Search the public web and get back a ranked list of results "
        "({title, url, snippet}). Use this for current events, facts you're "
        "unsure about, documentation lookups, or anything beyond your training "
        "data. Returns concise snippets; follow up with a page-fetch tool if you "
        "need full content."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results to return (default from config, usually 5).",
            },
        },
        "required": ["query"],
    },
)
def web_search(query: str, max_results: int | None = None) -> dict[str, Any]:
    """Search the web via the configured backend (Tavily) with Brave fallback."""
    if not query or not query.strip():
        return {"error": "query is required", "success": False, "results": []}

    from ..config import get_config_manager

    cfg = get_config_manager()
    backend = (cfg.get("search.backend", "tavily") or "tavily").lower()
    fallback_enabled = bool(cfg.get("search.fallback_enabled", True))
    if max_results is None:
        max_results = int(cfg.get("search.max_results", 5))
    ttl = int(cfg.get("search.cache_ttl_seconds", 300))

    # Cache check (keyed by backend + normalized query + count)
    cache_key = f"{backend}:{max_results}:{query.strip().lower()}"
    now = time.time()
    cached = _SEARCH_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return {
            "results": cached[1],
            "count": len(cached[1]),
            "backend": backend,
            "cached": True,
            "success": True,
        }

    # Backend order: configured primary, then the other (if fallback enabled)
    primary = backend if backend in _SEARCH_BACKENDS else "tavily"
    order = [primary]
    if fallback_enabled:
        order += [b for b in _SEARCH_BACKENDS if b != primary]

    errors: list[str] = []
    for name in order:
        try:
            results = _SEARCH_BACKENDS[name](query, max_results)
        except Exception as e:  # noqa: BLE001 - backend/network failure → try fallback
            errors.append(f"{name}: {e}")
            logger.warning(f"web_search backend '{name}' failed: {e}")
            continue
        if not results:
            errors.append(f"{name}: no results")
            continue
        if ttl > 0:
            _SEARCH_CACHE[cache_key] = (now + ttl, results)
        return {
            "results": results,
            "count": len(results),
            "backend": name,
            "cached": False,
            "success": True,
        }

    return {
        "results": [],
        "count": 0,
        "success": False,
        "error": "all search backends failed or returned no results: " + "; ".join(errors),
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

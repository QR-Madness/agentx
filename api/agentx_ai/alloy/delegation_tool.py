"""
Builds the per-workflow ``delegate_to`` tool descriptor.

The tool is context-aware: its allowed ``agent_id`` enum and surfaced
description depend on the workflow's specialists. It is *not* registered as a
pure-function internal tool because execution requires the live agent
context — see ``alloy/executor.py`` and the special-case branch in
``agent/core.py:_execute_tool_calls``.
"""

from .models import Workflow

DELEGATION_TOOL_NAME = "delegate_to"


def build_delegation_tool(workflow: Workflow) -> dict:
    """
    Build the OpenAI/Anthropic-style tool descriptor for ``delegate_to``,
    scoped to the specialists of the given workflow.
    """
    specialists = workflow.specialists()
    if not specialists:
        # No specialists → the tool would be useless; let the caller decide
        # whether to include it. We still return a well-formed descriptor.
        enum_values: list[str] = []
        bullet_lines = "  (no specialists configured)"
    else:
        enum_values = [m.agent_id for m in specialists]
        bullet_lines = "\n".join(
            f"  - {m.agent_id}: {m.delegation_hint or '(no hint provided)'}"
            for m in specialists
        )

    description = (
        "Delegate a focused task to a specialist agent on your team. "
        "The specialist runs independently and returns its result to you. "
        "The specialist will NOT see your conversation history — include all "
        "context it needs in the task description. Available specialists:\n"
        f"{bullet_lines}"
    )

    schema: dict = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "The specialist agent_id to delegate to.",
            },
            "task": {
                "type": "string",
                "description": (
                    "Self-contained task description. The specialist will not "
                    "see your conversation history; include all relevant "
                    "context here."
                ),
            },
        },
        "required": ["agent_id", "task"],
    }
    if enum_values:
        schema["properties"]["agent_id"]["enum"] = enum_values

    return {
        "name": DELEGATION_TOOL_NAME,
        "description": description,
        "input_schema": schema,
    }

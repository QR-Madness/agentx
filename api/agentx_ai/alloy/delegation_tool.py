"""
Builds the per-workflow ``delegate_to`` tool descriptor.

The tool is context-aware: its allowed ``agent_id`` enum and surfaced
description depend on the workflow's specialists. It is *not* registered as a
pure-function internal tool because execution requires the live agent
context — see ``alloy/executor.py`` and the special-case branch in
``agent/core.py:_execute_tool_calls``.
"""

from .models import Workflow
from .prompts import resolve_specialist_names

DELEGATION_TOOL_NAME = "delegate_to"


def _build_descriptor(entries: list[tuple[str, str, str]]) -> dict:
    """Build the ``delegate_to`` descriptor from ``(agent_id, name, hint)`` rows.

    Shared by the workflow (Agent Alloy) and ad-hoc (Phase 16.4) builders so the
    schema, tool name, and framing stay identical across both.
    """
    if entries:
        enum_values = [aid for aid, _, _ in entries]
        bullet_lines = "\n".join(
            f"  - {name} (id: {aid}): {hint or '(no hint provided)'}"
            for aid, name, hint in entries
        )
    else:
        enum_values = []
        bullet_lines = "  (no agents available)"

    description = (
        "Delegate a focused task to another agent. "
        "That agent runs independently and returns its result to you. "
        "It will NOT see your conversation history — include all "
        "context it needs in the task description. Available agents:\n"
        f"{bullet_lines}"
    )

    schema: dict = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": (
                    "The target agent's agent_id (the slug shown in parentheses "
                    "for each agent above)."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "Self-contained task description. The target agent will not "
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


def build_delegation_tool(workflow: Workflow) -> dict:
    """
    Build the ``delegate_to`` descriptor scoped to the specialists of a workflow.
    """
    names = resolve_specialist_names(workflow)
    entries = [
        (m.agent_id, names.get(m.agent_id, m.agent_id), m.delegation_hint or "")
        for m in workflow.specialists()
    ]
    return _build_descriptor(entries)


def build_adhoc_delegation_tool(self_agent_id: str) -> dict:
    """
    Build the ``delegate_to`` descriptor for ad-hoc (non-workflow) delegation
    (Phase 16.4): every agent profile except the delegating agent is a target.
    """
    from ..agent.profiles import get_profile_manager

    entries = [
        (p.agent_id, p.name, p.description or "")
        for p in get_profile_manager().list_profiles()
        if getattr(p, "agent_id", None) and p.agent_id != self_agent_id
    ]
    return _build_descriptor(entries)

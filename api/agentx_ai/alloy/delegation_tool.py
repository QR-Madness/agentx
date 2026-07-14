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
DELEGATION_START_TOOL_NAME = "delegate_start"


def _build_descriptor(
    entries: list[tuple[str, str, str]],
    *,
    tool_name: str = DELEGATION_TOOL_NAME,
    background: bool = False,
) -> dict:
    """Build a delegation tool descriptor from ``(agent_id, name, hint)`` rows.

    Shared by the workflow (Agent Alloy) and ad-hoc (Phase 16.4) builders so the
    schema and framing stay identical across both. ``background=True`` produces
    the non-blocking ``delegate_start`` variant (same schema, dispatch-receipt
    semantics).
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

    if background:
        description = (
            "Dispatch a focused task to another agent as a background work "
            "order and keep working — this returns immediately with a dispatch "
            "receipt. The agent's report is delivered to you automatically "
            "later in this same turn as a message; do not wait or poll for it. "
            "The agent will NOT see your conversation history — include all "
            "context it needs in the task description. Use delegate_to instead "
            "when you cannot proceed without the result. Available agents:\n"
            f"{bullet_lines}"
        )
    else:
        description = (
            "Delegate a focused task to another agent. "
            "That agent runs independently and returns its result to you. "
            "It will NOT see your conversation history — include all "
            "context it needs in the task description. "
            "To run several agents at once on independent subtasks, emit multiple "
            "delegate_to calls in the same turn — they execute concurrently and "
            "their results return together. Available agents:\n"
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
        "name": tool_name,
        "description": description,
        "input_schema": schema,
    }


def _workflow_entries(workflow: Workflow) -> list[tuple[str, str, str]]:
    """``(agent_id, name, hint)`` rows for a workflow's specialists."""
    from ..agent.profiles import get_profile_manager

    names = resolve_specialist_names(workflow)
    # Per-member hint wins; the profile's own delegation_hint is the fallback so
    # a team member with no hint still surfaces its profile specialty.
    profile_hints = {
        p.agent_id: p.delegation_hint
        for p in get_profile_manager().list_profiles()
        if getattr(p, "agent_id", None)
    }
    return [
        (
            m.agent_id,
            names.get(m.agent_id, m.agent_id),
            m.delegation_hint or profile_hints.get(m.agent_id) or "",
        )
        for m in workflow.specialists()
    ]


def build_delegation_tool(workflow: Workflow) -> dict:
    """
    Build the ``delegate_to`` descriptor scoped to the specialists of a workflow.
    """
    return _build_descriptor(_workflow_entries(workflow))


def build_delegation_start_tool(workflow: Workflow) -> dict:
    """``delegate_start`` (non-blocking work order) scoped to a workflow."""
    return _build_descriptor(
        _workflow_entries(workflow),
        tool_name=DELEGATION_START_TOOL_NAME,
        background=True,
    )


def list_adhoc_delegation_targets(self_agent_id: str) -> list[tuple[str, str, str]]:
    """List ``(agent_id, name, hint)`` rows for ad-hoc delegation targets.

    Single source of truth for who is delegable — shared by the ad-hoc tool
    descriptor and the roster system-prompt block so they can never disagree.
    Filters: has an agent_id, not the delegator itself, opted into the roster
    (``available_for_delegation``), and agent-kind (ambassadors are not chat
    agents → never delegation targets).
    """
    from ..agent.profiles import get_profile_manager

    return [
        (p.agent_id, p.name, getattr(p, "delegation_hint", None) or p.description or "")
        for p in get_profile_manager().list_profiles()
        if getattr(p, "agent_id", None)
        and p.agent_id != self_agent_id
        and getattr(p, "available_for_delegation", True)
        and getattr(p, "kind", "agent") == "agent"
    ]


def build_adhoc_delegation_tool(self_agent_id: str) -> dict:
    """
    Build the ``delegate_to`` descriptor for ad-hoc (non-workflow) delegation
    (Phase 16.4): every opted-in agent profile except the delegator is a target.
    """
    return _build_descriptor(list_adhoc_delegation_targets(self_agent_id))


def build_adhoc_delegation_start_tool(self_agent_id: str) -> dict:
    """``delegate_start`` (non-blocking work order) for ad-hoc delegation."""
    return _build_descriptor(
        list_adhoc_delegation_targets(self_agent_id),
        tool_name=DELEGATION_START_TOOL_NAME,
        background=True,
    )

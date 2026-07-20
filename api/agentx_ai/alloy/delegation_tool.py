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


def modality_suffix(agent_id: str) -> str:
    """Media-capability tag for a roster/tool entry, e.g. `` [sees images · makes images]``.

    Derived from the profile's default model via **cached** provider caps only
    (no catalog warm — this renders per turn, inside sync builders; a cold
    catalog just means no tags, which is advisory-only anyway). Empty string on
    any failure or for a plain text model. This is what lets a supervisor route
    media work to an agent that can actually handle the medium.
    """
    try:
        from ..agent.profiles import get_profile_manager
        from ..providers.capabilities import has_input_modality, has_output_modality
        from ..providers.registry import get_registry

        profile = get_profile_manager().get_profile_by_agent_id(agent_id)
        model = getattr(profile, "default_model", None)
        if not model:
            return ""
        provider, model_id = get_registry().get_provider_for_model(model)
        caps = provider.get_capabilities(model_id)
        tags = []
        if getattr(caps, "supports_vision", False) or has_input_modality(caps, "image"):
            tags.append("sees images")
        if has_input_modality(caps, "audio"):
            tags.append("hears audio")
        if has_output_modality(caps, "image"):
            tags.append("makes images")
        return f" [{' · '.join(tags)}]" if tags else ""
    except Exception:  # noqa: BLE001 — tags are advisory, never break a turn
        return ""


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
            f"  - {name} (id: {aid}){modality_suffix(aid)}: {hint or '(no hint provided)'}"
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
            "media": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional document_ids of images/audio the agent needs (from "
                    "attachment lines, the image catalog, or file listings). "
                    "ALWAYS pass the source image ids when delegating image "
                    "edit/restyle/variation work — the target can't fetch media "
                    "itself, and image models have no tools at all. Media is "
                    "delivered inside the target's first message, capability-"
                    "gated (a non-vision target gets a note; audio degrades to "
                    "a transcript)."
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


def _annotate_chain_hint(target) -> str:
    """Prefix a ChainTarget's hint with its org relation, keeping the
    ``(agent_id, name, hint)`` row shape unchanged for the descriptor/roster."""
    if target.relation == "down_lead":
        prefix = f"[lead of your team '{target.team_name}']"
    elif target.relation == "down_member":
        prefix = f"[your team member — '{target.team_name}']"
    else:  # up_lead
        prefix = "[your lead — escalation only]"
    return f"{prefix} {target.hint}".strip()


def list_adhoc_delegation_targets(self_agent_id: str) -> list[tuple[str, str, str]]:
    """List ``(agent_id, name, hint)`` rows for ad-hoc delegation targets.

    Single source of truth for who is delegable — shared by the ad-hoc tool
    descriptor and the roster system-prompt block so they can never disagree.

    Chain of command (Agentic Orgs): when the knob is on and the delegator is
    an org participant, the row set IS its chain adjacency (manager → leads of
    owned teams, lead → own members, member ↑ own lead), relation-annotated.
    Chain edges deliberately ignore ``available_for_delegation`` — adjacency
    implies delegability; the flag keeps governing only the flat roster below.

    Flat roster (org-free) filters: has an agent_id, not the delegator itself,
    opted into the roster (``available_for_delegation``), and agent-kind
    (ambassadors are not chat agents → never delegation targets).
    """
    from ..agent.profiles import get_profile_manager
    from . import org_chart

    if org_chart.chain_of_command_enabled() and org_chart.in_org(self_agent_id):
        return [
            (t.agent_id, t.name, _annotate_chain_hint(t))
            for t in org_chart.chain_targets(self_agent_id)
        ]

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

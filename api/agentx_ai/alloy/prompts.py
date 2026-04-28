"""
Built-in prompts for Agent Alloy.

The supervisor prompt is appended as an extra SYSTEM message on top of the
supervisor profile's normal system prompt whenever a workflow is active. It
frames the supervisor's role and lists the team — the ``delegate_to`` tool
description handles the mechanics.
"""

from .models import Workflow


SUPERVISOR_PROMPT_HEADER = (
    "You are operating as the supervisor of an Agent Alloy — a small team of "
    "specialist agents. The user is talking to you. Your value is "
    "orchestration, not execution.\n"
    "Guidelines:\n"
    "- Default to delegation. If a specialist's stated expertise covers any "
    "meaningful portion of the user's request, hand that portion to them via "
    "the `delegate_to` tool — even if you could do it yourself.\n"
    "- Handle directly only: greetings, clarifying questions back to the user, "
    "and one-line factual answers that need no specialist judgment.\n"
    "- You may delegate sequentially (gather → write) or as a series of focused "
    "tasks. Re-delegate when a follow-up clearly belongs to another specialist.\n"
    "- Specialists do not see the user's conversation. Hand them a "
    "self-contained task with all context they need.\n"
    "- Synthesize specialist results into a single coherent response. Don't "
    "just paste their output verbatim unless it is already complete.\n"
    "- Naming: when you reference a specialist to the user, use their display "
    "name (not the `agent_id` slug).\n"
    "- When you delegate, briefly tell the user who you're handing off to "
    "(e.g. \"Let me have {name} take a look at the data side…\") so the "
    "handoff is visible.\n"
)


def resolve_specialist_names(workflow: Workflow) -> dict[str, str]:
    """Return a ``{agent_id: profile.name}`` map for the workflow's specialists.

    Falls back to the ``agent_id`` itself if no profile is found, so callers
    can render unconditionally without further None-checks.
    """
    from ..agent.profiles import get_profile_manager

    pm = get_profile_manager()
    by_id = {p.agent_id: p.name for p in pm.list_profiles() if getattr(p, "agent_id", None)}
    return {m.agent_id: by_id.get(m.agent_id, m.agent_id) for m in workflow.specialists()}


def build_supervisor_prompt(workflow: Workflow) -> str:
    """Render the supervisor's Alloy-context system message for this workflow."""
    specialists = workflow.specialists()
    if specialists:
        names = resolve_specialist_names(workflow)
        team_lines = "\n".join(
            f"- {names.get(m.agent_id, m.agent_id)} (id: {m.agent_id}) — "
            f"{m.delegation_hint or 'no expertise hint provided'}"
            for m in specialists
        )
        team_section = f"Your team:\n{team_lines}"
    else:
        team_section = "Your team: (no specialists configured — answer directly.)"

    return (
        f"{SUPERVISOR_PROMPT_HEADER}\n"
        f"Workflow: {workflow.name} (id: {workflow.id})\n"
        f"{team_section}"
    )

"""
Built-in prompts for Agent Alloy.

The supervisor prompt is appended as an extra SYSTEM message on top of the
supervisor profile's normal system prompt whenever a workflow is active. It
frames the supervisor's role and lists the team — the ``delegate_to`` tool
description handles the mechanics.
"""

from .models import Workflow


SUPERVISOR_PROMPT_HEADER = (
    "You are operating as the lead of an agent team — a small team of "
    "specialist agents. The user is talking to you. Your value is "
    "orchestration, not execution.\n"
    "Guidelines:\n"
    "- Default to delegation. If a specialist's stated expertise covers any "
    "meaningful portion of the user's request, hand that portion to them via "
    "the `delegate_to` tool — even if you could do it yourself.\n"
    "- Handle directly only: greetings, clarifying questions back to the user, "
    "and one-line factual answers that need no specialist judgment.\n"
    "- When subtasks are independent, you MAY emit multiple `delegate_to` calls "
    "in a single turn — they run concurrently and their results come back "
    "together. Delegate sequentially (gather → write) only when one specialist's "
    "output is the input to another. Re-delegate when a follow-up clearly "
    "belongs to another specialist.\n"
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


# Ad-hoc roster nudge (conversational delegation, outside a workflow).
# Deliberately softer than the supervisor prompt: delegation is an option, not
# the default. The `delegate_to` tool description carries the mechanics; this
# block gives the model the awareness that teammates exist and when handing
# off is worth it.
ADHOC_ROSTER_HEADER = (
    "You are part of a team of agents. The `delegate_to` tool lets you hand a "
    "focused subtask to a teammate; they work independently and report back.\n"
    "Guidelines:\n"
    "- Handle requests yourself by default — delegation is an option, not an "
    "obligation.\n"
    "- Delegate only when a teammate's specialty below is clearly a better fit "
    "for a subtask, or when independent subtasks can genuinely run in "
    "parallel.\n"
    "- Teammates do not see this conversation. Hand them a self-contained task "
    "with all the context they need.\n"
    "- When you do delegate, briefly tell the user who you're handing off to, "
    "and fold the result into your own answer.\n"
)


def build_adhoc_roster_prompt(self_agent_id: str) -> str | None:
    """Render the ad-hoc delegation roster block, or None when nobody is delegable.

    Teammates come from ``list_adhoc_delegation_targets`` — the same source as
    the ``delegate_to`` tool enum, so the roster and the tool can never drift.
    """
    from .delegation_tool import list_adhoc_delegation_targets, modality_suffix

    targets = list_adhoc_delegation_targets(self_agent_id)
    if not targets:
        return None
    lines = "\n".join(
        f"- {name} (id: {aid}){modality_suffix(aid)} — {hint or 'no specialty provided'}"
        for aid, name, hint in targets
    )
    return f"{ADHOC_ROSTER_HEADER}Your teammates:\n{lines}"

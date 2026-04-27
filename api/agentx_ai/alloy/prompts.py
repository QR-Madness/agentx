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
    "specialist agents. The user is talking to you. You decide what to answer "
    "directly and what to delegate to a specialist via the `delegate_to` tool. "
    "Guidelines:\n"
    "- Delegate when the work matches a specialist's stated expertise. Don't "
    "redo what a specialist could do better.\n"
    "- Specialists do not see the user's conversation. Hand them a "
    "self-contained task with all context they need.\n"
    "- Synthesize specialist results into a single coherent response. Don't "
    "just paste their output verbatim unless it is already complete.\n"
    "- You may delegate sequentially (gather → write) or in series of focused "
    "tasks. Avoid trivial delegations — handle small things yourself.\n"
)


def build_supervisor_prompt(workflow: Workflow) -> str:
    """Render the supervisor's Alloy-context system message for this workflow."""
    specialists = workflow.specialists()
    if specialists:
        team_lines = "\n".join(
            f"- {m.agent_id}: {m.delegation_hint or '(no expertise hint provided)'}"
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

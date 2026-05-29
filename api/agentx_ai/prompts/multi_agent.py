"""Multi-agent conversation prompt awareness (Phase 16.2).

When more than one agent has spoken in a single conversation, each agent's
system prompt gains a roster of the others so it knows it is collaborating
rather than working alone. Modeled on ``alloy/prompts.build_supervisor_prompt``,
but for free-form (non-workflow) conversations — inside an Alloy workflow the
supervisor prompt already frames the team, so this block is suppressed there.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..agent.models import AgentProfile


def build_participants_block(
    current_agent_id: Optional[str],
    participants: dict[str, "AgentProfile"],
) -> Optional[str]:
    """Render a roster of the OTHER agents present in the conversation.

    Args:
        current_agent_id: agent_id of the agent this prompt is for (excluded
            from the roster).
        participants: ``{agent_id: AgentProfile}`` for everyone who has spoken.

    Returns:
        A SYSTEM-message string, or ``None`` when there is no one else present.
    """
    others = [p for aid, p in participants.items() if aid != current_agent_id]
    if not others:
        return None

    lines = "\n".join(f"- {p.name} (id: {p.agent_id})" for p in others)
    return (
        "You are sharing this conversation with other agents. They may have "
        "authored earlier turns; messages are attributed to whoever wrote them.\n"
        f"Other agents present:\n{lines}"
    )

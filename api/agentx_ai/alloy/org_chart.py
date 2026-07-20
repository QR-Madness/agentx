"""
Agentic Organizations — pure chain-of-command derivation.

The org is *derivable* from the two existing stores (INV: no org-membership
database): a profile declares its tier (``org_level``), a workflow declares its
owner (``manager_agent_id``); edges, roster grouping, and the canvas all derive.
``org_level`` plays NO role in edge derivation — tier on the profile, structure
on the workflow; mismatches are log-warns at workflow save time.

Chain relations (strict adjacency — no level-skipping, no peer-to-peer):

- ``down_lead``   — manager → lead (supervisor) of a team it owns
- ``down_member`` — lead → member (specialist) of its own team
- ``up_lead``     — member → its own lead (escalation)

Deliberate asymmetry (documented in the design doc): chain edges IGNORE
``available_for_delegation`` — adjacency implies delegability, exactly as
workflow supervisors already delegate to specialists regardless of the flag.
The flag keeps governing only the flat org-free roster.

``in_org`` is deliberately **manager-anchored**: an agent is in the org iff it
manages a team, or any team it leads / belongs to has a ``manager_agent_id``.
A naive "is in any workflow" definition would silently replace every existing
lead's flat ad-hoc roster on upgrade — org-free installs must stay
byte-identical.

Known consequence: leads have no upward edge (lead → manager escalation is not
a v1 relation), so an in-org lead whose teams have no specialists derives an
empty target list.

This module is the SINGLE derivation source consumed by both enforcement
points (the ``delegate_to`` tool enum and the executor validation — INV: dual
enforcement), and the single patch point for tests. It must never import the
ambassador family (ADR-11 / CapabilitySeamBoundaryTest — ``alloy`` is a core
package).
"""

import logging
from dataclasses import dataclass
from typing import Literal

from ..agent.profiles import get_profile_manager
from ..config import get_config_manager
from .manager import get_workflow_manager
from .models import MemberRole, Workflow

logger = logging.getLogger(__name__)

Relation = Literal["down_lead", "down_member", "up_lead"]


@dataclass(frozen=True)
class ChainTarget:
    """One legal delegation target derived from the org chart."""

    agent_id: str
    name: str
    hint: str
    relation: Relation
    team_id: str
    team_name: str


def chain_of_command_enabled() -> bool:
    """The ``alloy.chain_of_command`` knob (default ON)."""
    return bool(get_config_manager().get("alloy.chain_of_command", True))


def teams_managed_by(agent_id: str) -> list[Workflow]:
    """Teams whose ``manager_agent_id`` is ``agent_id``."""
    if not agent_id:
        return []
    return [
        wf for wf in get_workflow_manager().list() if wf.manager_agent_id == agent_id
    ]


def teams_led_by(agent_id: str) -> list[Workflow]:
    """Teams whose supervisor (lead) is ``agent_id``."""
    if not agent_id:
        return []
    return [
        wf for wf in get_workflow_manager().list() if wf.supervisor_agent_id == agent_id
    ]


def teams_of_member(agent_id: str) -> list[Workflow]:
    """Teams where ``agent_id`` participates as a specialist (member)."""
    if not agent_id:
        return []
    return [
        wf
        for wf in get_workflow_manager().list()
        if any(
            m.agent_id == agent_id and m.role == MemberRole.SPECIALIST
            for m in wf.members
        )
    ]


def in_org(agent_id: str) -> bool:
    """Manager-anchored org membership (see module docstring)."""
    if not agent_id:
        return False
    if teams_managed_by(agent_id):
        return True
    return any(
        wf.manager_agent_id
        for wf in (*teams_led_by(agent_id), *teams_of_member(agent_id))
    )


def chain_targets(agent_id: str) -> list[ChainTarget]:
    """Every legal delegation target for ``agent_id``, adjacency-only.

    Order: ``down_lead`` → ``down_member`` → ``up_lead``; deduped by target
    ``agent_id`` (first relation wins); self excluded; ids that don't resolve
    to a ``kind=='agent'`` profile are skipped (ambassador exclusion rides the
    existing ``get_profile_by_agent_id`` filter).
    """
    pm = get_profile_manager()
    out: list[ChainTarget] = []
    seen: set[str] = {agent_id}

    def _add(target_id: str, relation: Relation, wf: Workflow, member_hint: str | None = None) -> None:
        if not target_id or target_id in seen:
            return
        profile = pm.get_profile_by_agent_id(target_id)
        if profile is None:  # dangling id or non-agent kind — skip
            return
        hint = (
            member_hint
            or getattr(profile, "delegation_hint", None)
            or getattr(profile, "description", None)
            or ""
        )
        out.append(
            ChainTarget(
                agent_id=target_id,
                name=profile.name,
                hint=hint,
                relation=relation,
                team_id=wf.id,
                team_name=wf.name,
            )
        )
        seen.add(target_id)

    # NOTE: down/up edges are derived from EVERY team the agent leads or
    # belongs to (manager-owned or not) — adjacency within one's own team is
    # always legitimate. Whether the chain gate applies at all is decided
    # solely by the manager-anchored ``in_org`` above.
    for wf in teams_managed_by(agent_id):
        _add(wf.supervisor_agent_id, "down_lead", wf)
    for wf in teams_led_by(agent_id):
        for m in wf.members:
            if m.role == MemberRole.SPECIALIST:
                _add(m.agent_id, "down_member", wf, member_hint=m.delegation_hint)
    for wf in teams_of_member(agent_id):
        _add(wf.supervisor_agent_id, "up_lead", wf)

    return out

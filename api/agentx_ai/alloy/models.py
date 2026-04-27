"""
Data models for Agent Alloy workflows.

A workflow declares one supervisor and zero or more specialists, all
referenced by their immutable ``agent_id``. Profile names can change
without breaking workflow references.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


WORKFLOW_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class MemberRole(str, Enum):
    """A member's role within a workflow."""
    SUPERVISOR = "supervisor"
    SPECIALIST = "specialist"


@dataclass
class WorkflowMember:
    """An agent participating in a workflow, referenced by agent_id."""
    agent_id: str
    role: MemberRole
    delegation_hint: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"agent_id": self.agent_id, "role": self.role.value}
        if self.delegation_hint:
            d["delegation_hint"] = self.delegation_hint
        return d


@dataclass
class WorkflowRoute:
    """
    Declarative route between two members. Schema only in v1 — the executor
    does not act on routes; all delegation flows through the ``delegate_to``
    tool. Stored so the Factory editor can persist user intent.
    """
    from_agent_id: str
    to_agent_id: str
    when: str

    def to_dict(self) -> dict:
        return {
            "from_agent_id": self.from_agent_id,
            "to_agent_id": self.to_agent_id,
            "when": self.when,
        }


@dataclass
class Workflow:
    """A multi-agent workflow ("Alloy")."""
    id: str
    name: str
    supervisor_agent_id: str
    members: list[WorkflowMember]
    description: Optional[str] = None
    routes: list[WorkflowRoute] = field(default_factory=list)
    shared_channel: str = ""  # auto-derived in __post_init__ when blank
    canvas: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.shared_channel:
            self.shared_channel = f"_alloy_{self.id}"

    def specialists(self) -> list[WorkflowMember]:
        return [m for m in self.members if m.role == MemberRole.SPECIALIST]

    def get_member(self, agent_id: str) -> Optional[WorkflowMember]:
        for m in self.members:
            if m.agent_id == agent_id:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "supervisor_agent_id": self.supervisor_agent_id,
            "members": [m.to_dict() for m in self.members],
            "routes": [r.to_dict() for r in self.routes],
            "shared_channel": self.shared_channel,
            "canvas": self.canvas,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

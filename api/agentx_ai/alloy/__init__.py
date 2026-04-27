"""
Agent Alloy — multi-agent orchestration.

A Workflow ("Alloy") binds a supervisor profile to one or more specialist
profiles. The supervisor delegates focused tasks to specialists via the
``delegate_to`` tool. Specialists share a workflow-scoped memory channel
(``_alloy_{workflow_id}``) but do not see the user's full conversation.
"""

from .models import MemberRole, Workflow, WorkflowMember, WorkflowRoute
from .manager import WorkflowManager, get_workflow_manager

__all__ = [
    "MemberRole",
    "Workflow",
    "WorkflowMember",
    "WorkflowRoute",
    "WorkflowManager",
    "get_workflow_manager",
]

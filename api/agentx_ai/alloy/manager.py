"""
WorkflowManager: CRUD + YAML persistence for Agent Alloy workflows.

Mirrors the patterns in ``api/agentx_ai/agent/profiles.py``. Workflows are
persisted to ``data/workflows.yaml`` (created on first save).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from ..agent.profiles import get_profile_manager
from .models import (
    MemberRole,
    Workflow,
    WorkflowMember,
    WorkflowRoute,
    WORKFLOW_ID_PATTERN,
)

logger = logging.getLogger(__name__)


def _parse_member(data: dict) -> WorkflowMember:
    return WorkflowMember(
        agent_id=data["agent_id"],
        role=MemberRole(data["role"]),
        delegation_hint=data.get("delegation_hint"),
    )


def _parse_route(data: dict) -> WorkflowRoute:
    return WorkflowRoute(
        from_agent_id=data["from_agent_id"],
        to_agent_id=data["to_agent_id"],
        when=data["when"],
    )


def _parse_workflow(data: dict) -> Workflow:
    created = data.get("created_at")
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    updated = data.get("updated_at")
    if isinstance(updated, str):
        updated = datetime.fromisoformat(updated)

    return Workflow(
        id=data["id"],
        name=data["name"],
        description=data.get("description"),
        supervisor_agent_id=data["supervisor_agent_id"],
        members=[_parse_member(m) for m in data.get("members", [])],
        routes=[_parse_route(r) for r in data.get("routes", [])],
        shared_channel=data.get("shared_channel") or "",
        canvas=data.get("canvas") or {},
        created_at=created,
        updated_at=updated,
    )


class WorkflowManager:
    """Loads, validates, and persists Agent Alloy workflows."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent.parent / "data" / "workflows.yaml"
            )
        self.config_path = config_path
        self._workflows: dict[str, Workflow] = {}
        if config_path.exists():
            self._load_config(config_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> None:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            if not config or "workflows" not in config:
                return
            for entry in config["workflows"]:
                wf = _parse_workflow(entry)
                self._workflows[wf.id] = wf
            logger.info(f"Loaded {len(self._workflows)} workflows from {path}")
        except Exception as e:
            logger.error(f"Failed to load workflows: {e}")

    def save_config(self, path: Optional[Path] = None) -> None:
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config = {"workflows": []}
        for wf in self._workflows.values():
            entry = wf.to_dict()
            entry = {k: v for k, v in entry.items() if v is not None}
            config["workflows"].append(entry)
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved {len(self._workflows)} workflows to {save_path}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, wf: Workflow, *, is_new: bool) -> None:
        if not WORKFLOW_ID_PATTERN.match(wf.id):
            raise ValueError(
                f"workflow id {wf.id!r} must match {WORKFLOW_ID_PATTERN.pattern}"
            )
        if is_new and wf.id in self._workflows:
            raise ValueError(f"workflow id {wf.id!r} already exists")
        if not wf.name.strip():
            raise ValueError("workflow name is required")
        if not wf.members:
            raise ValueError("workflow must have at least one member")

        supervisors = [m for m in wf.members if m.role == MemberRole.SUPERVISOR]
        if len(supervisors) != 1:
            raise ValueError(
                f"workflow must have exactly one supervisor (found {len(supervisors)})"
            )
        if supervisors[0].agent_id != wf.supervisor_agent_id:
            raise ValueError(
                "supervisor_agent_id must reference the member with role 'supervisor'"
            )

        seen_ids: set[str] = set()
        for m in wf.members:
            if m.agent_id in seen_ids:
                raise ValueError(f"duplicate member agent_id: {m.agent_id}")
            seen_ids.add(m.agent_id)

        # All member agent_ids must resolve to a known profile.
        pm = get_profile_manager()
        known_agent_ids = {p.agent_id for p in pm.list_profiles() if p.agent_id}
        for m in wf.members:
            if m.agent_id not in known_agent_ids:
                raise ValueError(
                    f"member agent_id {m.agent_id!r} does not match any profile"
                )

        if wf.routes:
            logger.warning(
                f"workflow {wf.id!r} declares {len(wf.routes)} routes; "
                "declarative routes are accepted but not executed in v1"
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list(self) -> list[Workflow]:
        return list(self._workflows.values())

    def get(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def create(self, workflow: Workflow) -> Workflow:
        self._validate(workflow, is_new=True)
        now = datetime.utcnow()
        workflow.created_at = now
        workflow.updated_at = now
        self._workflows[workflow.id] = workflow
        self.save_config()
        return workflow

    def update(self, workflow_id: str, updates: dict[str, Any]) -> Optional[Workflow]:
        current = self._workflows.get(workflow_id)
        if current is None:
            return None

        # ID is immutable; created_at preserved.
        updates.pop("id", None)
        updates.pop("created_at", None)

        merged = current.to_dict()
        merged.update(updates)
        merged["id"] = current.id
        merged["created_at"] = (
            current.created_at.isoformat() if current.created_at else None
        )
        if "members" in merged and merged["members"] and isinstance(
            merged["members"][0], dict
        ):
            pass  # already plain dicts; _parse_workflow handles them
        if "routes" in merged and merged["routes"] and isinstance(
            merged["routes"][0], dict
        ):
            pass

        new_wf = _parse_workflow(merged)
        self._validate(new_wf, is_new=False)
        new_wf.updated_at = datetime.utcnow()
        self._workflows[workflow_id] = new_wf
        self.save_config()
        return new_wf

    def delete(self, workflow_id: str) -> bool:
        if workflow_id not in self._workflows:
            return False
        del self._workflows[workflow_id]
        self.save_config()
        return True


_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """Return the global WorkflowManager singleton."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
        logger.info("WorkflowManager initialized")
    return _workflow_manager

"""
Agent Skills — named instruction packs with progressive disclosure.

A Skill is know-how, not a tool: a markdown body of instructions the agent can
pull into context *on demand*. Only a compact index (id — name: description)
rides the system prompt (see ``views._skills_block``); the full body loads
through the ``use_skill`` internal tool (``mcp/internal_tools.py``) when the
agent decides the skill is relevant. This keeps per-turn prompt cost flat no
matter how large the skill library grows.

Persistence mirrors ``ProfileManager`` (``agent/profiles.py``): a YAML store at
``data/skills.yaml``, shipped defaults seeded exactly once via YAML-root
``seeded_defaults`` markers (deleting a seeded skill sticks), and a module
singleton. Access control mirrors MCP servers: ``allowed_agent_ids`` is
``None`` = every agent, ``[]`` = no agents, else an agent_id whitelist.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Skill(BaseModel):
    """One named instruction pack."""

    id: str
    name: str
    description: str = ""
    body: str = ""
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    # None = all agents; [] = none; else whitelist by AgentProfile.agent_id.
    allowed_agent_ids: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or f"skill-{uuid.uuid4().hex[:8]}"


# Shipped default skills, seeded ONCE per store (marker → skill). Deliberately
# abstract/thinking-flavored — AgentX is a thinking platform, not a code tool.
_DECISION_BRIEF_SEED = Skill(
    id="decision-brief",
    name="Structured Decision Brief",
    description="Turn a fuzzy choice between options into a compact, opinionated decision brief.",
    tags=["thinking", "writing"],
    body="""When a decision needs to be made between competing options, produce a Decision Brief with this shape:

1. **Frame** — one sentence stating the decision and its stakes/deadline.
2. **Options** — for each option: its strongest argument, its weakest point, and what it costs (time, money, risk).
3. **Criteria** — the 3-5 factors that actually matter to the decider, ranked. If you had to guess the ranking, say so and ask.
4. **Recommendation** — one option, stated plainly, with the two strongest reasons and the main risk being accepted.
5. **Reversal test** — the new fact that would flip the recommendation, and the cheapest way to check it.

Keep the whole brief under 300 words. Never end at "it depends" — a brief without a recommendation has failed. If information is missing, recommend anyway and mark the assumption that could change it.""",
)

_ONE_TIME_SEEDS: dict[str, Skill] = {
    "decision-brief-v1": _DECISION_BRIEF_SEED,
}


class SkillsManager:
    """CRUD + persistence for the skill library (``data/skills.yaml``)."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "skills.yaml"
        self.config_path = config_path
        self._skills: dict[str, Skill] = {}
        # One-time-seed markers persisted at the YAML root as `seeded_defaults`
        # — a recorded marker never seeds again, so deletions stick.
        self._seeded_markers: list[str] = []

        if config_path.exists():
            self._load_config(config_path)
        else:
            self._init_defaults()
        self._ensure_seeded_defaults()

    # --- persistence -------------------------------------------------------

    def _init_defaults(self) -> None:
        """Fresh install: ship the default skills and record every marker."""
        for seed in _ONE_TIME_SEEDS.values():
            self._skills[seed.id] = seed.model_copy(update={"created_at": datetime.now()})
        self._seeded_markers = list(_ONE_TIME_SEEDS.keys())
        self.save_config()
        logger.info("Initialized skill library with defaults")

    def _ensure_seeded_defaults(self) -> None:
        """Apply one-time seeds this store hasn't seen (marker semantics as in
        ``ProfileManager._ensure_seeded_defaults`` — absent → seed unless the
        user already has that id/name; present → never again)."""
        changed = False
        for marker, seed in _ONE_TIME_SEEDS.items():
            if marker in self._seeded_markers:
                continue
            collision = seed.id in self._skills or any(
                s.name == seed.name for s in self._skills.values()
            )
            if not collision:
                self._skills[seed.id] = seed.model_copy(update={"created_at": datetime.now()})
                logger.info(f"Seeded default skill {seed.name!r} ({marker})")
            self._seeded_markers.append(marker)
            changed = True
        if changed:
            self.save_config()

    def _load_config(self, path: Path) -> None:
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            if not config or "skills" not in config:
                self._init_defaults()
                return
            self._seeded_markers = list(config.get("seeded_defaults") or [])
            for data in config["skills"]:
                for key in ("created_at", "updated_at"):
                    if isinstance(data.get(key), str):
                        data[key] = datetime.fromisoformat(data[key])
                skill = Skill(**data)
                self._skills[skill.id] = skill
            logger.info(f"Loaded {len(self._skills)} skills from {path}")
        except Exception as e:
            logger.error(f"Failed to load skills: {e}")
            self._init_defaults()

    def save_config(self, path: Path | None = None) -> None:
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config: dict[str, Any] = {
            "skills": [
                {
                    k: v
                    for k, v in {
                        "id": s.id,
                        "name": s.name,
                        "description": s.description,
                        "body": s.body,
                        "tags": s.tags,
                        "enabled": s.enabled,
                        "allowed_agent_ids": s.allowed_agent_ids,
                        "created_at": s.created_at.isoformat() if s.created_at else None,
                        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    }.items()
                    # allowed_agent_ids None (= all agents) drops like other Nones;
                    # the loader's pydantic default restores it.
                    if v is not None
                }
                for s in self._skills.values()
            ]
        }
        if self._seeded_markers:
            config["seeded_defaults"] = list(self._seeded_markers)
        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # --- CRUD ---------------------------------------------------------------

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get_skill(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def resolve(self, ref: str) -> Skill | None:
        """Look up by id, falling back to a case-insensitive name match —
        models frequently pass the display name from the index."""
        skill = self._skills.get(ref)
        if skill is not None:
            return skill
        needle = ref.strip().lower()
        return next(
            (s for s in self._skills.values() if s.name.lower() == needle or s.id == needle),
            None,
        )

    def create_skill(
        self,
        name: str,
        description: str = "",
        body: str = "",
        tags: list[str] | None = None,
        enabled: bool = True,
        allowed_agent_ids: list[str] | None = None,
    ) -> Skill:
        base = _slugify(name)
        skill_id = base
        while skill_id in self._skills:
            skill_id = f"{base}-{uuid.uuid4().hex[:6]}"
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            body=body,
            tags=tags or [],
            enabled=enabled,
            allowed_agent_ids=allowed_agent_ids,
            created_at=datetime.now(),
        )
        self._skills[skill_id] = skill
        self.save_config()
        logger.info(f"Created skill: {name} ({skill_id})")
        return skill

    def update_skill(self, skill_id: str, updates: dict[str, Any]) -> Skill | None:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        allowed_fields = {"name", "description", "body", "tags", "enabled", "allowed_agent_ids"}
        data = skill.model_dump()
        data.update({k: v for k, v in updates.items() if k in allowed_fields})
        data["updated_at"] = datetime.now()
        updated = Skill(**data)
        self._skills[skill_id] = updated
        self.save_config()
        return updated

    def delete_skill(self, skill_id: str) -> bool:
        if skill_id not in self._skills:
            return False
        del self._skills[skill_id]
        self.save_config()
        logger.info(f"Deleted skill: {skill_id}")
        return True

    # --- agent-facing -------------------------------------------------------

    def skills_for_agent(self, agent_id: str | None) -> list[Skill]:
        """Enabled skills this agent may use. ``allowed_agent_ids`` semantics
        mirror MCP servers: None = everyone; an unknown/absent agent_id only
        matches unrestricted skills."""
        out = []
        for s in self._skills.values():
            if not s.enabled:
                continue
            if s.allowed_agent_ids is None:
                out.append(s)
            elif agent_id is not None and agent_id in s.allowed_agent_ids:
                out.append(s)
        return out


# --- module singleton --------------------------------------------------------

_skills_manager: SkillsManager | None = None


def get_skills_manager() -> SkillsManager:
    global _skills_manager
    if _skills_manager is None:
        _skills_manager = SkillsManager()
    return _skills_manager

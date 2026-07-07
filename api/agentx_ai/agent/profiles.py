"""
Agent Profile Manager for AgentX.

Manages agent profile CRUD operations and YAML storage.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from .models import AgentProfile, AmbassadorConfig, ReasoningStrategy

logger = logging.getLogger(__name__)


def _propagate_agent_rename(agent_id: str, old_name: str, new_name: str) -> None:
    """Rename the agent's first-class memory entity and keep the old name as an alias.

    Best-effort and lazily imported so profile CRUD never hard-depends on the
    memory layer (or a running Neo4j). Updates every ``agent:{agent_id}`` Entity
    node across all users/channels in one statement; agent_id is the stable key,
    so the rename is unambiguous.
    """
    try:
        from ..kit.agent_memory.connections import Neo4jConnection
    except Exception as e:  # noqa: BLE001
        logger.debug(f"agent rename propagation skipped (no memory layer): {e}")
        return
    try:
        with Neo4jConnection.session() as session:
            session.run(
                """
                MATCH (e:Entity)
                WHERE coalesce(e.type, '') = 'Agent'
                  AND e.id ENDS WITH (':' + $agent_id)
                SET e.aliases = CASE
                        WHEN $old_name IS NULL
                             OR toLower($old_name) = toLower($new_name)
                             OR toLower($old_name) IN [a IN coalesce(e.aliases, []) | toLower(a)]
                        THEN coalesce(e.aliases, [])
                        ELSE coalesce(e.aliases, []) + [$old_name]
                    END,
                    e.name = $new_name
                """,
                agent_id=agent_id,
                old_name=old_name,
                new_name=new_name,
            )
    except Exception as e:  # noqa: BLE001 — a rename must never break profile save
        logger.warning(f"Failed to propagate agent rename for {agent_id}: {e}")


# =============================================================================
# Default Profile
# =============================================================================

# pyright doesn't recognize pydantic v2 positional Field() defaults, so it treats
# defaulted fields as required here — they're not. Suppress that false positive.
DEFAULT_PROFILE = AgentProfile(  # pyright: ignore[reportCallIssue]
    id="default",
    name="AgentX",
    avatar="sparkles",
    description="Default assistant profile with balanced settings",
    temperature=0.7,
    reasoning_strategy=ReasoningStrategy.AUTO,
    enable_memory=True,
    memory_channel="_global",
    enable_tools=True,
    is_default=True,
)


def _warn_unqualified_tool_names(profile: AgentProfile) -> None:
    """
    Flag entries in ``allowed_tools`` / ``blocked_tools`` that lack a ``.``
    separator. Phase 18.9.x switched gating to fully-qualified ``server.tool``
    keys (e.g. ``_internal.checkpoint``); bare names match nothing under the
    new scheme. We only warn — the data still loads — so a user editing
    YAML by hand sees what to fix without their profile bouncing.
    """
    unqualified: list[str] = []
    for entry in (profile.allowed_tools or []):
        if "." not in entry:
            unqualified.append(entry)
    for entry in profile.blocked_tools:
        if "." not in entry:
            unqualified.append(entry)
    if unqualified:
        logger.warning(
            "Profile %r has unqualified tool names %r; use `server.tool` "
            "(e.g. `_internal.checkpoint`) — bare names match nothing.",
            profile.id,
            unqualified,
        )


class ProfileManager:
    """
    Manages agent profiles and storage.

    Handles:
    - Loading/saving profiles from YAML configuration
    - Profile management (CRUD operations)
    - Default profile selection
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize the ProfileManager.

        Args:
            config_path: Path to agent_profiles.yaml (defaults to data/agent_profiles.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "data" / "agent_profiles.yaml"

        self.config_path = config_path
        self._profiles: dict[str, AgentProfile] = {}

        # Load from file or initialize defaults
        if config_path.exists():
            self._load_config(config_path)
        else:
            self._init_defaults()

        # Ensure the Ambassador-as-profile-kind invariants (migrate legacy, seed
        # a default ambassador). Idempotent — only writes when something changed.
        self._ensure_ambassador_defaults()

    def _init_defaults(self) -> None:
        """Initialize with default profiles."""
        self._profiles[DEFAULT_PROFILE.id] = DEFAULT_PROFILE

        # Save defaults to disk
        self.save_config()
        logger.info("Initialized agent profiles with defaults")

    def _ensure_ambassador_defaults(self) -> None:
        """Migrate legacy ambassadors to `kind='ambassador'` and guarantee a default
        ambassador exists — without ever converting the default *agent*.

        Before the pivot, an "ambassador" was any profile with `ambassador.enabled`,
        selected via `config.ambassador.profile_id` (which *defaults to the default
        agent*). So we only convert an *explicitly dedicated* ambassador, never the
        default agent, and otherwise seed a fresh default ambassador so briefings
        work out of the box.
        """
        changed = False
        legacy_id: str | None = None
        try:
            from ..config import get_config_manager
            legacy_id = get_config_manager().get("ambassador.profile_id", None)
        except Exception:
            legacy_id = None

        # 1. Migrate dedicated legacy ambassadors (never the default agent).
        for pid, p in list(self._profiles.items()):
            if p.kind == "ambassador":
                continue
            amb = p.ambassador
            dedicated = bool(amb and amb.enabled) and not p.is_default
            is_legacy_target = bool(legacy_id) and pid == legacy_id and not p.is_default
            if dedicated or is_legacy_target:
                data = p.model_dump()
                data["kind"] = "ambassador"
                self._profiles[pid] = AgentProfile(**data)
                changed = True

        ambs = self.list_profiles_by_kind("ambassador")
        if ambs:
            # 2a. Ensure exactly one default ambassador.
            if not any(p.is_default_ambassador for p in ambs):
                target = self._profiles.get(legacy_id) if legacy_id else None
                if target is None or target.kind != "ambassador":
                    target = ambs[0]
                data = target.model_dump()
                data["is_default_ambassador"] = True
                self._profiles[target.id] = AgentProfile(**data)
                changed = True
        else:
            # 2b. Seed a fresh default ambassador (its own agent_id).
            seed_id = "ambassador" if "ambassador" not in self._profiles else f"ambassador-{uuid.uuid4().hex[:8]}"
            seed = AgentProfile(  # pyright: ignore[reportCallIssue]
                id=seed_id,
                name="Ambassador",
                kind="ambassador",
                avatar="radio",
                description="Briefs you on the conversation in parallel — never enters it.",
                is_default_ambassador=True,
                ambassador=AmbassadorConfig(enabled=True),
            )
            self._profiles[seed_id] = seed
            changed = True

        if changed:
            self.save_config()
            logger.info("Ensured ambassador profile defaults (migrate/seed)")

    def _load_config(self, path: Path) -> None:
        """Load profiles from YAML file."""
        try:
            with open(path) as f:
                config = yaml.safe_load(f)

            if not config or "profiles" not in config:
                self._init_defaults()
                return

            for profile_data in config["profiles"]:
                # Handle datetime fields
                if "created_at" in profile_data and isinstance(profile_data["created_at"], str):
                    profile_data["created_at"] = datetime.fromisoformat(profile_data["created_at"])
                if "updated_at" in profile_data and isinstance(profile_data["updated_at"], str):
                    profile_data["updated_at"] = datetime.fromisoformat(profile_data["updated_at"])

                profile = AgentProfile(**profile_data)
                self._profiles[profile.id] = profile
                _warn_unqualified_tool_names(profile)

            logger.info(f"Loaded {len(self._profiles)} agent profiles from {path}")

        except Exception as e:
            logger.error(f"Failed to load agent profiles: {e}")
            self._init_defaults()

    def save_config(self, path: Path | None = None) -> None:
        """Save current profiles to YAML file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "profiles": [
                {
                    "id": p.id,
                    "name": p.name,
                    "kind": p.kind,
                    "agent_id": p.agent_id,
                    "avatar": p.avatar,
                    "description": p.description,
                    "default_model": p.default_model,
                    "temperature": p.temperature,
                    "prompt_profile_id": p.prompt_profile_id,
                    "system_prompt": p.system_prompt,
                    # Coerce enum → its string value; a raw enum attribute (e.g. a
                    # code-constructed profile's default) would dump as an
                    # unloadable !!python/object tag.
                    "reasoning_strategy": getattr(
                        p.reasoning_strategy, "value", p.reasoning_strategy
                    ),
                    "enable_memory": p.enable_memory,
                    "memory_channel": p.memory_channel,
                    "enable_tools": p.enable_tools,
                    "direct_mode": p.direct_mode,
                    # Per-profile tool gating — previously dropped here, so gating
                    # was lost on restart (Phase 18.2+ regression). blocked_tools
                    # defaults to [] (kept); allowed_tools is None=all (filtered out).
                    "allowed_tools": p.allowed_tools,
                    "blocked_tools": p.blocked_tools,
                    "available_for_delegation": p.available_for_delegation,
                    "delegation_hint": p.delegation_hint,
                    # Ambassador section (previously dropped here → lost on restart).
                    "ambassador": p.ambassador.model_dump() if p.ambassador else None,
                    "is_default": p.is_default,
                    "is_default_ambassador": p.is_default_ambassador,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                }
                for p in self._profiles.values()
            ]
        }

        # Filter out None values from each profile dict
        config["profiles"] = [
            {k: v for k, v in p.items() if v is not None}
            for p in config["profiles"]
        ]

        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved agent profiles to {save_path}")

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def list_profiles(self) -> list[AgentProfile]:
        """List all profiles."""
        return list(self._profiles.values())

    def get_profile(self, profile_id: str) -> AgentProfile | None:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)

    def get_profile_by_agent_id(self, agent_id: str) -> AgentProfile | None:
        """Get a profile by its Docker-style agent_id (e.g. 'bold-cosmic-falcon').

        The canonical routing/attribution key for multi-agent conversations
        (Phase 16) — distinct from the profile ``id`` used by ``get_profile``.
        """
        return next(
            (p for p in self._profiles.values()
             if getattr(p, "agent_id", None) == agent_id and p.kind == "agent"),
            None,
        )

    def get_profile_by_name(self, name: str) -> AgentProfile | None:
        """Get a profile by its display name (case-insensitive exact match).

        Used as the @-mention fallback (Phase 16.5) when a token isn't a known
        agent_id — so `@Mobius` resolves as readily as `@bright-grand-fern`.
        """
        lowered = name.lower()
        return next(
            (p for p in self._profiles.values()
             if p.name.lower() == lowered and p.kind == "agent"),
            None,
        )

    def list_profiles_by_kind(self, kind: str) -> list[AgentProfile]:
        """All profiles of a given kind ('agent' or 'ambassador')."""
        return [p for p in self._profiles.values() if p.kind == kind]

    def get_default_profile(self) -> AgentProfile | None:
        """Get the default *agent* profile (never an ambassador)."""
        agents = self.list_profiles_by_kind("agent")
        for profile in agents:
            if profile.is_default:
                return profile
        # Fall back to the first agent profile.
        return next(iter(agents), None)

    def get_default_ambassador(self) -> AgentProfile | None:
        """Get the default *ambassador* profile (briefings resolve this)."""
        ambs = self.list_profiles_by_kind("ambassador")
        for profile in ambs:
            if profile.is_default_ambassador:
                return profile
        return next(iter(ambs), None)

    def set_default_ambassador(self, profile_id: str) -> bool:
        """Mark an ambassador-kind profile as the default ambassador (one per kind)."""
        target = self._profiles.get(profile_id)
        if target is None or target.kind != "ambassador":
            return False
        for pid, profile in self._profiles.items():
            if profile.kind != "ambassador":
                continue
            data = profile.model_dump()
            data["is_default_ambassador"] = (pid == profile_id)
            data["updated_at"] = datetime.utcnow() if pid == profile_id else profile.updated_at
            self._profiles[pid] = AgentProfile(**data)
        self.save_config()
        return True

    def create_profile(self, profile: AgentProfile) -> AgentProfile:
        """Create a new profile."""
        profile.created_at = datetime.utcnow()
        profile.updated_at = datetime.utcnow()
        self._profiles[profile.id] = profile
        self.save_config()
        return profile

    def update_profile(self, profile_id: str, updates: dict) -> AgentProfile | None:
        """Update an existing profile."""
        if profile_id not in self._profiles:
            return None

        current = self._profiles[profile_id]
        updates.pop("agent_id", None)  # agent_id is immutable once generated
        # Use model_dump to carry every field forward, so adding new fields to
        # AgentProfile (e.g. allowed_tools/blocked_tools in Phase 18.2) doesn't
        # silently reset them on any edit.
        old_name = current.name
        updated_data = current.model_dump()
        updated_data.update(updates)
        updated_data["updated_at"] = datetime.utcnow()

        self._profiles[profile_id] = AgentProfile(**updated_data)
        self.save_config()

        # Keep the agent's first-class memory entity rename-safe: on a display-name
        # change, fold the old name into the Agent entity's aliases so historical
        # prose and recall-by-old-name keep resolving. agent_id is the stable key.
        new_name = self._profiles[profile_id].name
        if new_name != old_name:
            _propagate_agent_rename(current.agent_id, old_name, new_name)

        return self._profiles[profile_id]

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by ID."""
        if profile_id not in self._profiles:
            return False

        # Prevent deleting the last profile
        if len(self._profiles) <= 1:
            raise ValueError("Cannot delete the last profile")

        profile = self._profiles[profile_id]
        del self._profiles[profile_id]

        # If the deleted profile was the default agent, promote another agent.
        if profile.is_default:
            next_agent = next(iter(self.list_profiles_by_kind("agent")), None)
            if next_agent is not None:
                data = next_agent.model_dump()
                data["is_default"] = True
                data["updated_at"] = datetime.utcnow()
                self._profiles[next_agent.id] = AgentProfile(**data)
        # If the deleted profile was the default ambassador, promote another.
        if profile.is_default_ambassador:
            next_amb = next(iter(self.list_profiles_by_kind("ambassador")), None)
            if next_amb is not None:
                data = next_amb.model_dump()
                data["is_default_ambassador"] = True
                data["updated_at"] = datetime.utcnow()
                self._profiles[next_amb.id] = AgentProfile(**data)

        self.save_config()
        return True

    def set_default_profile(self, profile_id: str) -> bool:
        """Set an *agent*-kind profile as the default agent. (Ambassadors use
        set_default_ambassador and can never become the default agent.)"""
        target = self._profiles.get(profile_id)
        if target is None or target.kind != "agent":
            return False

        for pid, profile in self._profiles.items():
            if profile.kind != "agent":
                continue
            data = profile.model_dump()
            data["is_default"] = (pid == profile_id)
            data["updated_at"] = datetime.utcnow() if pid == profile_id else profile.updated_at
            self._profiles[pid] = AgentProfile(**data)

        self.save_config()
        return True


# =============================================================================
# Singleton instance
# =============================================================================

_profile_manager: ProfileManager | None = None


def get_profile_manager() -> ProfileManager:
    """Get or create the global ProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
        logger.info("ProfileManager initialized")
    return _profile_manager

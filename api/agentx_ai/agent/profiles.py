"""
Agent Profile Manager for AgentX.

Manages agent profile CRUD operations and YAML storage.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .models import AgentProfile, ReasoningStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# Default Profile
# =============================================================================

DEFAULT_PROFILE = AgentProfile(
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


class ProfileManager:
    """
    Manages agent profiles and storage.

    Handles:
    - Loading/saving profiles from YAML configuration
    - Profile management (CRUD operations)
    - Default profile selection
    """

    def __init__(self, config_path: Optional[Path] = None):
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

    def _init_defaults(self) -> None:
        """Initialize with default profiles."""
        self._profiles[DEFAULT_PROFILE.id] = DEFAULT_PROFILE

        # Save defaults to disk
        self.save_config()
        logger.info("Initialized agent profiles with defaults")

    def _load_config(self, path: Path) -> None:
        """Load profiles from YAML file."""
        try:
            with open(path, "r") as f:
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

            logger.info(f"Loaded {len(self._profiles)} agent profiles from {path}")

        except Exception as e:
            logger.error(f"Failed to load agent profiles: {e}")
            self._init_defaults()

    def save_config(self, path: Optional[Path] = None) -> None:
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
                    "agent_id": p.agent_id,
                    "avatar": p.avatar,
                    "description": p.description,
                    "default_model": p.default_model,
                    "temperature": p.temperature,
                    "prompt_profile_id": p.prompt_profile_id,
                    "system_prompt": p.system_prompt,
                    "reasoning_strategy": p.reasoning_strategy,
                    "enable_memory": p.enable_memory,
                    "memory_channel": p.memory_channel,
                    "enable_tools": p.enable_tools,
                    "is_default": p.is_default,
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

    def get_profile(self, profile_id: str) -> Optional[AgentProfile]:
        """Get a profile by ID."""
        return self._profiles.get(profile_id)

    def get_default_profile(self) -> Optional[AgentProfile]:
        """Get the default profile."""
        for profile in self._profiles.values():
            if profile.is_default:
                return profile
        # Fall back to first profile
        return next(iter(self._profiles.values()), None)

    def create_profile(self, profile: AgentProfile) -> AgentProfile:
        """Create a new profile."""
        profile.created_at = datetime.utcnow()
        profile.updated_at = datetime.utcnow()
        self._profiles[profile.id] = profile
        self.save_config()
        return profile

    def update_profile(self, profile_id: str, updates: dict) -> Optional[AgentProfile]:
        """Update an existing profile."""
        if profile_id not in self._profiles:
            return None

        current = self._profiles[profile_id]
        updates.pop("agent_id", None)  # agent_id is immutable once generated
        # Use model_dump to carry every field forward, so adding new fields to
        # AgentProfile (e.g. allowed_tools/blocked_tools in Phase 18.2) doesn't
        # silently reset them on any edit.
        updated_data = current.model_dump()
        updated_data.update(updates)
        updated_data["updated_at"] = datetime.utcnow()

        self._profiles[profile_id] = AgentProfile(**updated_data)
        self.save_config()
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

        # If deleted profile was default, set another as default
        if profile.is_default and self._profiles:
            first_profile = next(iter(self._profiles.values()))
            data = first_profile.model_dump()
            data["is_default"] = True
            data["updated_at"] = datetime.utcnow()
            self._profiles[first_profile.id] = AgentProfile(**data)

        self.save_config()
        return True

    def set_default_profile(self, profile_id: str) -> bool:
        """Set a profile as the default."""
        if profile_id not in self._profiles:
            return False

        for pid, profile in self._profiles.items():
            data = profile.model_dump()
            data["is_default"] = (pid == profile_id)
            data["updated_at"] = datetime.utcnow() if pid == profile_id else profile.updated_at
            self._profiles[pid] = AgentProfile(**data)

        self.save_config()
        return True


# =============================================================================
# Singleton instance
# =============================================================================

_profile_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """Get or create the global ProfileManager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
        logger.info("ProfileManager initialized")
    return _profile_manager

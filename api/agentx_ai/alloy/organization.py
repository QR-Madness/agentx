"""
Organization chart layout — where the user placed things on the org chart.

**Layout only.** The org's structure always derives from the profiles (tier) and
the workflows (who manages, leads and belongs to each team); this store never
holds membership or edges. It keeps the canvas positions the user dragged nodes
to, keyed by stable ids — ``agent:{agent_id}`` and ``team:{workflow_id}`` — so a
rename never orphans a position (INV-3). Anything without a stored position falls
back to the derived layout, so an absent file means "the derived chart".

Persisted to ``data/organization.yaml`` (created on first save). ``Workflow.canvas``
stays reserved for a future per-team editor.
"""

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ``agent:{agent_id}`` or ``team:{workflow_id}``: ids, never display names.
POSITION_KEY = re.compile(r"^(agent|team):[A-Za-z0-9_.:-]{1,200}$")
MAX_POSITIONS = 2000
MAX_COORDINATE = 1_000_000.0


def validate_positions(positions: Any) -> dict[str, dict[str, float]]:
    """Return a clean ``{key: {"x": float, "y": float}}`` map, or raise ValueError.

    Keys must be ``agent:{id}`` or ``team:{id}``; coordinates must be finite
    numbers within ±1e6; at most 2000 entries.
    """
    if not isinstance(positions, dict):
        raise ValueError("layout.positions must be an object")
    if len(positions) > MAX_POSITIONS:
        raise ValueError(f"layout.positions holds more than {MAX_POSITIONS} entries")
    clean: dict[str, dict[str, float]] = {}
    for key, value in positions.items():
        if not isinstance(key, str) or not POSITION_KEY.match(key):
            raise ValueError(f"invalid position key {key!r}: use agent:{{id}} or team:{{id}}")
        if not isinstance(value, dict):
            raise ValueError(f"position {key!r} must be an object with x and y")
        point: dict[str, float] = {}
        for axis in ("x", "y"):
            coordinate = value.get(axis)
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(f"position {key!r} needs a numeric {axis}")
            if not math.isfinite(coordinate) or abs(coordinate) > MAX_COORDINATE:
                raise ValueError(f"position {key!r} has an out-of-range {axis}")
            point[axis] = float(coordinate)
        clean[key] = point
    return clean


class OrganizationLayoutStore:
    """Loads and saves the org chart layout (``data/organization.yaml``)."""

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent.parent / "data" / "organization.yaml"
            )
        self.config_path = config_path
        self._positions: dict[str, dict[str, float]] = {}
        if config_path.exists():
            self._load_config(config_path)

    def _load_config(self, path: Path) -> None:
        try:
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            positions = (config.get("layout") or {}).get("positions") or {}
            self._positions = validate_positions(positions)
            logger.info(f"Loaded {len(self._positions)} org chart positions from {path}")
        except Exception as e:  # noqa: BLE001 — a bad layout file must not break the org
            logger.error(f"Failed to load the org chart layout (using the derived layout): {e}")
            self._positions = {}

    def save_config(self, path: Path | None = None) -> None:
        """Write atomically (temp file, then replace): drags save often, and an
        interrupted write must never leave a half-written file behind."""
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = save_path.with_name(save_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            yaml.dump({"layout": {"positions": self._positions}}, f,
                      default_flow_style=False, sort_keys=True)
        os.replace(tmp_path, save_path)
        logger.info(f"Saved {len(self._positions)} org chart positions to {save_path}")

    def layout(self) -> dict[str, Any]:
        """The stored layout: ``{"positions": {key: {"x", "y"}}}`` (a copy)."""
        return {"positions": {k: dict(v) for k, v in self._positions.items()}}

    def replace(self, positions: Any) -> dict[str, Any]:
        """Validate and store `positions` in place of the current ones, then save.
        An empty map resets the chart to its derived layout."""
        self._positions = validate_positions(positions)
        self.save_config()
        return self.layout()


_layout_store: OrganizationLayoutStore | None = None


def get_organization_layout_store() -> OrganizationLayoutStore:
    """Return the global OrganizationLayoutStore singleton."""
    global _layout_store
    if _layout_store is None:
        _layout_store = OrganizationLayoutStore()
    return _layout_store

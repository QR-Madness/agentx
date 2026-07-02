"""Per-deployment manager state: the spec + config-file hashes.

`.manager-state.json` lives in the cluster dir (repo mode) or bundle root.
Hashes cover every single-file bind mount plus the .env — that's how the
manager knows a `restart` isn't enough and the affected service needs
`up -d --force-recreate` (bind-mounted single files pin their inode at
container-create time, so in-place edits are invisible to a plain restart).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .spec import ClusterSpec

STATE_FILE = ".manager-state.json"


@dataclass
class ManagerState:
    spec: ClusterSpec | None
    hashes: dict[str, str]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_files(cluster_dir: Path, gateway_dir: Path, env_file: Path) -> dict[str, Path]:
    """Single-file mounts (+ .env) whose edits require a force-recreate.

    Keyed by the compose service the file feeds, prefixed for uniqueness.
    """
    return {
        "env:.env": env_file,
        "nginx:nginx.conf": gateway_dir / "nginx.conf",
        "cloudflared:config.yml": gateway_dir / "cloudflared" / "config.yml",
        "cloudflared:credentials.json": gateway_dir / "cloudflared" / "credentials.json",
    }


def compute_hashes(files: dict[str, Path]) -> dict[str, str]:
    return {key: _hash_file(path) for key, path in files.items() if path.is_file()}


def load_state(state_dir: Path) -> ManagerState | None:
    path = state_dir / STATE_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    spec_data = data.get("spec")
    spec = ClusterSpec.from_dict(spec_data) if spec_data else None
    return ManagerState(spec=spec, hashes=data.get("hashes", {}))


def save_state(state_dir: Path, spec: ClusterSpec, hashes: dict[str, str]) -> None:
    path = state_dir / STATE_FILE
    path.write_text(json.dumps({"spec": spec.to_dict(), "hashes": hashes}, indent=2) + "\n")


def changed_services(old: dict[str, str], new: dict[str, str]) -> set[str]:
    """Services whose tracked files changed (added, removed, or edited).

    An `env:` change touches everything — returns {"*"}.
    """
    changed: set[str] = set()
    for key in set(old) | set(new):
        if old.get(key) != new.get(key):
            service = key.partition(":")[0]
            if service == "env":
                return {"*"}
            changed.add(service)
    return changed

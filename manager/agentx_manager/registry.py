"""Deployment discovery.

Two roots the manager can point at (auto-detected, or forced via
AGENTX_MANAGER_ROOT):

- **repo mode**: a source checkout — deployments are `clusters/<name>/` dirs
  (each with its own .env), compose files live at the repo root, kind=source.
- **bundle mode**: an untarred deploy bundle — the root itself is one
  deployment (`.env` + compose files side by side), kind=image. The cluster
  name comes from AGENTX_CLUSTER_NAME in .env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .envfile import parse_env
from .spec import ClusterSpec, spec_from_files
from .state import load_state

TEMPLATE_DIR = "template"


@dataclass
class Cluster:
    """A discovered deployment: its spec plus where it lives."""

    spec: ClusterSpec
    root: Path        # directory holding the compose files
    cluster_dir: Path  # directory holding .env / config / db / gateway files
    env_file: Path

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def env(self) -> dict[str, str]:
        return parse_env(self.env_file)


def detect_mode(root: Path) -> str:
    """'repo' when a clusters/ tree exists next to the compose files, else 'bundle'."""
    if (root / "clusters").is_dir() and (root / "Taskfile.yml").is_file():
        return "repo"
    return "bundle"


def _cluster_from_dir(root: Path, cluster_dir: Path) -> Cluster | None:
    env_file = cluster_dir / ".env"
    if not env_file.is_file():
        return None
    name = cluster_dir.name
    state = load_state(cluster_dir)
    if state and state.spec:
        spec = state.spec
    else:
        spec = spec_from_files(name, cluster_dir, kind="source")
    return Cluster(spec=spec, root=root, cluster_dir=cluster_dir, env_file=env_file)


def discover(root: Path) -> list[Cluster]:
    """All deployments under a manager root."""
    root = root.resolve()
    if detect_mode(root) == "repo":
        clusters: list[Cluster] = []
        for entry in sorted((root / "clusters").iterdir()):
            if not entry.is_dir() or entry.name == TEMPLATE_DIR:
                continue
            cluster = _cluster_from_dir(root, entry)
            if cluster is not None:
                clusters.append(cluster)
        return clusters

    # Bundle mode: the root is the single deployment.
    env_file = root / ".env"
    env = parse_env(env_file)
    name = env.get("AGENTX_CLUSTER_NAME", "agentx")
    state = load_state(root)
    if state and state.spec:
        spec = state.spec
    else:
        gateway_dir = Path(env.get("AGENTX_GATEWAY_DIR", "./gateway"))
        if not gateway_dir.is_absolute():
            gateway_dir = root / gateway_dir
        spec = ClusterSpec(
            name=name,
            kind="image",
            gateway=(gateway_dir / "nginx.conf").is_file(),
            tunnel="named" if (gateway_dir / "cloudflared" / "config.yml").is_file() else (
                "token" if env.get("TUNNEL_TOKEN") else "none"
            ),
        )
    return [Cluster(spec=spec, root=root, cluster_dir=root, env_file=env_file)]


def get(root: Path, name: str) -> Cluster:
    for cluster in discover(root):
        if cluster.name == name:
            return cluster
    raise KeyError(f"no cluster named {name!r} under {root}")


def gateway_dir(cluster: Cluster) -> Path:
    """Where this deployment's gateway files (nginx.conf, cloudflared/) live."""
    raw = cluster.env.get("AGENTX_GATEWAY_DIR", "")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else cluster.root / path
    if cluster.cluster_dir == cluster.root:
        return cluster.root / "gateway"
    return cluster.cluster_dir

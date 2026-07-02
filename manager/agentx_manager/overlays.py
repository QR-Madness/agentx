"""Pure overlay assembly: ClusterSpec → docker compose argv.

The single source of truth for which `-f` files, `--profile`s, and project
name a deployment runs with. No I/O — the unit-test truth table lives on top
of this module.
"""

from __future__ import annotations

from pathlib import Path

from .spec import ClusterSpec

BASE = "docker-compose.yml"
BUILD = "docker-compose.build.yml"
GATEWAY = "docker-compose.gateway.yml"
GATEWAY_EXPOSE = "docker-compose.gateway.expose.yml"
TUNNEL_TOKEN = "docker-compose.tunnel.yml"  # noqa: S105 — compose filename, not a secret
TUNNEL_NAMED = "docker-compose.tunnel.named.yml"
GPU = "docker-compose.gpu.yml"
SHELL = "docker-compose.shell.yml"


def compose_files(spec: ClusterSpec) -> list[str]:
    """Ordered overlay list. Later files override earlier ones, so the base
    always comes first and exposure overlays come after the gateway they
    modify."""
    files = [BASE]
    if spec.kind == "source":
        files.append(BUILD)
    if spec.gateway:
        files.append(GATEWAY)
        if spec.expose:
            files.append(GATEWAY_EXPOSE)
    if spec.tunnel == "token":
        files.append(TUNNEL_TOKEN)
    elif spec.tunnel == "named":
        files.append(TUNNEL_NAMED)
    if spec.gpu:
        files.append(GPU)
    if spec.shell:
        files.append(SHELL)
    return files


def project_name(spec: ClusterSpec) -> str:
    """Explicit per-cluster compose project.

    Without `-p`, every cluster inherits the project directory's name, so
    `docker compose down` in the repo can cross-target other clusters and all
    clusters share one default network. `agentx-<name>` isolates them.
    """
    return f"agentx-{spec.name}"


def compose_argv(
    spec: ClusterSpec,
    root: Path,
    env_file: Path,
    subcommand: list[str],
) -> list[str]:
    """Full `docker compose …` argv for a subcommand (e.g. ["up", "-d"])."""
    argv = ["docker", "compose", "-p", project_name(spec), "--env-file", str(env_file)]
    for f in compose_files(spec):
        argv += ["-f", str(root / f)]
    argv += ["--profile", "production"]
    argv += subcommand
    return argv

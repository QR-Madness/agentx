"""Health + resource usage for a deployment.

Health has a state the raw Docker healthcheck can't express: a fresh
deployment's first boot downloads models and initializes schemas for
minutes, during which Docker reports "unhealthy" but nothing is wrong.
`api_container_phase` folds container state + recent logs into:

    absent | starting | initializing | healthy | unhealthy | exited

Resources aggregate `docker stats` over the cluster's compose project:
CPU% (sum) and memory (sum used / smallest limit seen), with a per-service
breakdown for drill-in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .compose import ComposeRunner, SubprocessRunner
from .overlays import compose_argv, project_name
from .registry import Cluster

# Log lines that indicate first-boot initialization is progressing.
_INIT_PATTERNS = re.compile(
    r"(downloading|download complete|fetching \d+ files|schema initialization"
    r"|applying|applied \d+ migration|load pretrained|seeding|auto-init"
    r"|alembic|init_memory_schema|migrat)",
    re.IGNORECASE,
)


def _container_name(cluster: Cluster, service: str) -> str:
    prefix = cluster.env.get("AGENTX_CLUSTER_NAME", "agent")
    return f"{prefix}-{service}"


def api_container_phase(cluster: Cluster, runner: ComposeRunner | None = None) -> str:
    runner = runner or SubprocessRunner()
    name = _container_name(cluster, "api")
    inspect = runner.run(
        ["docker", "inspect", "-f",
         "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
         name]
    )
    if not inspect.ok:
        return "absent"
    parts = inspect.stdout.split()
    if len(parts) != 2:
        return "unknown"
    status, health = parts
    if status == "exited":
        return "exited"
    if status != "running":
        return status
    if health in ("healthy", "none"):
        return "healthy"
    if health == "starting":
        return "starting"
    # running + unhealthy: initialization or a real failure — read recent logs.
    logs = runner.run(["docker", "logs", "--tail", "40", name])
    if logs.ok and _INIT_PATTERNS.search(logs.stdout + logs.stderr):
        return "initializing"
    return "unhealthy"


@dataclass
class ServiceStatus:
    service: str
    state: str
    health: str


@dataclass
class ClusterStatus:
    name: str
    phase: str                     # aggregate: down | initializing | degraded | up
    services: list[ServiceStatus] = field(default_factory=list)


def status(cluster: Cluster, runner: ComposeRunner | None = None) -> ClusterStatus:
    runner = runner or SubprocessRunner()
    result = runner.run(
        compose_argv(cluster.spec, cluster.root, cluster.env_file, ["ps", "--format", "json"]),
        cwd=cluster.root,
    )
    services: list[ServiceStatus] = []
    if result.ok and result.stdout.strip():
        # compose ps --format json emits one JSON object per line.
        for line in result.stdout.strip().splitlines():
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            services.append(ServiceStatus(
                service=row.get("Service", row.get("Name", "?")),
                state=row.get("State", "?"),
                health=row.get("Health", "") or "",
            ))

    if not services:
        return ClusterStatus(name=cluster.name, phase="down", services=[])

    api_phase = api_container_phase(cluster, runner)
    if api_phase in ("initializing", "starting"):
        phase = "initializing"
    elif all(s.state == "running" and s.health in ("", "healthy") for s in services) and api_phase == "healthy":
        phase = "up"
    else:
        phase = "degraded"
    return ClusterStatus(name=cluster.name, phase=phase, services=services)


@dataclass
class ServiceUsage:
    service: str
    cpu_percent: float
    mem_used_bytes: int
    mem_limit_bytes: int
    mem_percent: float


@dataclass
class ClusterUsage:
    name: str
    cpu_percent: float
    mem_used_bytes: int
    mem_limit_bytes: int
    mem_percent: float
    services: list[ServiceUsage] = field(default_factory=list)


_SIZE = {"B": 1, "kB": 1000, "KB": 1000, "KiB": 1024, "MB": 1000**2, "MiB": 1024**2,
         "GB": 1000**3, "GiB": 1024**3, "TB": 1000**4, "TiB": 1024**4}


def _parse_size(text: str) -> int:
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)", text.strip())
    if not match:
        return 0
    value, unit = match.groups()
    return int(float(value) * _SIZE.get(unit, 1))


def usage(cluster: Cluster, runner: ComposeRunner | None = None) -> ClusterUsage:
    runner = runner or SubprocessRunner()
    project = project_name(cluster.spec)
    ids = runner.run([
        "docker", "ps", "-q",
        "--filter", f"label=com.docker.compose.project={project}",
    ])
    container_ids = ids.stdout.split() if ids.ok else []
    services: list[ServiceUsage] = []
    if container_ids:
        stats = runner.run([
            "docker", "stats", "--no-stream", "--format", "json", *container_ids,
        ])
        if stats.ok:
            for line in stats.stdout.strip().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mem_used, _, mem_limit = row.get("MemUsage", "0B / 0B").partition("/")
                used = _parse_size(mem_used)
                limit = _parse_size(mem_limit)
                services.append(ServiceUsage(
                    service=row.get("Name", "?"),
                    cpu_percent=float(row.get("CPUPerc", "0%").rstrip("%") or 0),
                    mem_used_bytes=used,
                    mem_limit_bytes=limit,
                    mem_percent=float(row.get("MemPerc", "0%").rstrip("%") or 0),
                ))
    total_used = sum(s.mem_used_bytes for s in services)
    limit = max((s.mem_limit_bytes for s in services), default=0)
    return ClusterUsage(
        name=cluster.name,
        cpu_percent=round(sum(s.cpu_percent for s in services), 1),
        mem_used_bytes=total_used,
        mem_limit_bytes=limit,
        mem_percent=round(total_used / limit * 100, 1) if limit else 0.0,
        services=services,
    )

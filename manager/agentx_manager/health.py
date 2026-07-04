"""Health + resource usage for a deployment.

Health has a state the raw Docker healthcheck can't express: a fresh
deployment's first boot downloads models and initializes schemas for
minutes, during which Docker reports "unhealthy" but nothing is wrong.
`api_container_phase` folds container state + recent logs into:

    absent | starting | initializing | healthy | unhealthy | exited

Resources aggregate `docker stats` over the cluster's compose project:
CPU% (sum) and memory (sum used / smallest limit seen), with a per-service
breakdown for drill-in.

In bundle mode the manager's own container shares the deployment's compose
project *by design* (so a plain `docker compose ps/down` in the bundle dir
sees the containers Up started). Raw project-scoped queries therefore see
the manager as an orphan "service"; `status` and `usage` filter project
containers to the config-derived service set (`stack_services`) so a
manager-only project reads as down and gauges cover only the app stack.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
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


def stack_services(cluster: Cluster, runner: ComposeRunner) -> set[str] | None:
    """Service names this deployment's compose config defines (production
    profile) — ground truth for which project containers belong to the stack.
    None = config unresolvable; callers keep unfiltered behavior."""
    result = runner.run(
        compose_argv(cluster.spec, cluster.root, cluster.env_file, ["config", "--services"]),
        cwd=cluster.root,
    )
    if not result.ok:
        return None
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return names or None


def stack_ps(
    cluster: Cluster, runner: ComposeRunner, expected: set[str] | None = None,
) -> list[ServiceStatus] | None:
    """`compose ps` rows filtered to the stack's own services (see the module
    docstring: the bundle's manager container is a same-project orphan and
    must not count). None = the ps call itself failed."""
    result = runner.run(
        compose_argv(cluster.spec, cluster.root, cluster.env_file, ["ps", "--format", "json"]),
        cwd=cluster.root,
    )
    if not result.ok:
        return None
    services: list[ServiceStatus] = []
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
    if services:
        if expected is None:
            expected = stack_services(cluster, runner)
        if expected is not None:
            services = [s for s in services if s.service in expected]
    return services


def status(
    cluster: Cluster, runner: ComposeRunner | None = None, expected: set[str] | None = None,
) -> ClusterStatus:
    runner = runner or SubprocessRunner()
    services = stack_ps(cluster, runner, expected) or []

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
    net_rx_bytes: int = 0          # cumulative since container start
    net_tx_bytes: int = 0


@dataclass
class ClusterUsage:
    name: str
    cpu_percent: float
    mem_used_bytes: int
    mem_limit_bytes: int
    mem_percent: float
    net_rx_bytes: int = 0          # cumulative, summed over services
    net_tx_bytes: int = 0
    net_rx_rate: float = 0.0       # loose bytes/sec between two usage() polls
    net_tx_rate: float = 0.0
    services: list[ServiceUsage] = field(default_factory=list)


class NetRateTracker:
    """Loose ↓/↑ bytes-per-second from successive cumulative NetIO samples.

    `docker stats` only reports lifetime rx/tx totals, so a rate needs two
    samples — the server keeps one tracker per process and feeds every
    usage() poll into it. Counters reset when a container restarts, so
    negative deltas clamp to zero instead of going backwards.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._last: dict[str, tuple[float, int, int]] = {}  # key → (t, rx, tx)

    def rates(self, key: str, rx_bytes: int, tx_bytes: int) -> tuple[float, float]:
        now = self._clock()
        prev = self._last.get(key)
        self._last[key] = (now, rx_bytes, tx_bytes)
        if prev is None:
            return 0.0, 0.0
        elapsed = now - prev[0]
        if elapsed <= 0:
            return 0.0, 0.0
        return (max(0, rx_bytes - prev[1]) / elapsed,
                max(0, tx_bytes - prev[2]) / elapsed)


_SIZE = {"B": 1, "kB": 1000, "KB": 1000, "KiB": 1024, "MB": 1000**2, "MiB": 1024**2,
         "GB": 1000**3, "GiB": 1024**3, "TB": 1000**4, "TiB": 1024**4}


def _parse_size(text: str) -> int:
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)", text.strip())
    if not match:
        return 0
    value, unit = match.groups()
    return int(float(value) * _SIZE.get(unit, 1))


def usage(
    cluster: Cluster,
    runner: ComposeRunner | None = None,
    expected: set[str] | None = None,
    rates: NetRateTracker | None = None,
) -> ClusterUsage:
    runner = runner or SubprocessRunner()
    project = project_name(cluster.spec)
    listing = runner.run([
        "docker", "ps",
        "--filter", f"label=com.docker.compose.project={project}",
        "--format", '{{.ID}} {{.Label "com.docker.compose.service"}}',
    ])
    container_ids: list[str] = []
    if listing.ok and listing.stdout.strip():
        if expected is None:
            expected = stack_services(cluster, runner)
        for line in listing.stdout.strip().splitlines():
            cid, _, service = line.partition(" ")
            if cid and (expected is None or service.strip() in expected):
                container_ids.append(cid)
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
                net_rx, _, net_tx = row.get("NetIO", "0B / 0B").partition("/")
                services.append(ServiceUsage(
                    service=row.get("Name", "?"),
                    cpu_percent=float(row.get("CPUPerc", "0%").rstrip("%") or 0),
                    mem_used_bytes=used,
                    mem_limit_bytes=limit,
                    mem_percent=float(row.get("MemPerc", "0%").rstrip("%") or 0),
                    net_rx_bytes=_parse_size(net_rx),
                    net_tx_bytes=_parse_size(net_tx),
                ))
    total_used = sum(s.mem_used_bytes for s in services)
    limit = max((s.mem_limit_bytes for s in services), default=0)
    total_rx = sum(s.net_rx_bytes for s in services)
    total_tx = sum(s.net_tx_bytes for s in services)
    rx_rate = tx_rate = 0.0
    if rates is not None:
        rx_rate, tx_rate = rates.rates(cluster.name, total_rx, total_tx)
    return ClusterUsage(
        name=cluster.name,
        cpu_percent=round(sum(s.cpu_percent for s in services), 1),
        mem_used_bytes=total_used,
        mem_limit_bytes=limit,
        mem_percent=round(total_used / limit * 100, 1) if limit else 0.0,
        net_rx_bytes=total_rx,
        net_tx_bytes=total_tx,
        net_rx_rate=round(rx_rate, 1),
        net_tx_rate=round(tx_rate, 1),
        services=services,
    )

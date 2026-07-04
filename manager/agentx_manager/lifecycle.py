"""Cluster lifecycle: up / down / restart / destroy / adopt.

Two behaviors the Taskfile never had, both fixed by construction here:

- **Config-change awareness**: tracked single-file mounts are hashed
  (state.py); when they changed since the last `up`, the affected services
  get `up -d --force-recreate` instead of a plain restart — a plain restart
  keeps the old file inode and silently serves stale config.
- **First-boot patience**: a fresh deployment's first boot can spend minutes
  downloading models before the API healthcheck passes, which makes
  `compose up` fail with "dependency failed to start" for dependents like
  nginx. `up` recognizes that state (API running + still initializing),
  waits, and re-runs instead of failing.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field

from .compose import ComposeRunner, RunResult, SubprocessRunner
from .health import api_container_phase, stack_ps
from .overlays import compose_argv
from .registry import Cluster, gateway_dir
from .state import compute_hashes, load_state, save_state, changed_services, tracked_files

FIRST_BOOT_TIMEOUT = 1200  # seconds to tolerate model download + schema init
POLL_INTERVAL = 10


@dataclass
class LifecycleResult:
    ok: bool
    detail: str
    runs: list[RunResult] = field(default_factory=list)


def _argv(cluster: Cluster, subcommand: list[str]) -> list[str]:
    return compose_argv(cluster.spec, cluster.root, cluster.env_file, subcommand)


def _current_hashes(cluster: Cluster) -> dict[str, str]:
    files = tracked_files(cluster.cluster_dir, gateway_dir(cluster), cluster.env_file)
    return compute_hashes(files)


def _stale_services(cluster: Cluster) -> set[str]:
    state = load_state(cluster.cluster_dir)
    if state is None:
        return set()
    return changed_services(state.hashes, _current_hashes(cluster))


def _persist(cluster: Cluster) -> None:
    save_state(cluster.cluster_dir, cluster.spec, _current_hashes(cluster))


def _ensure_identity(cluster: Cluster) -> None:
    """Backfill AGENTX_CLUSTER_NAME into legacy .envs that predate it.

    Without it every service falls back to the `agent-` container_name prefix,
    which collides across clusters even though compose projects are separate.
    """
    if not cluster.env.get("AGENTX_CLUSTER_NAME"):
        from .envfile import upsert_env

        upsert_env(cluster.env_file, {"AGENTX_CLUSTER_NAME": cluster.name})


def _ensure_secrets(cluster: Cluster) -> list[str]:
    """Generate missing secrets into .env so a fresh deployment is zero-config.

    DJANGO_SECRET_KEY is generated whenever empty (stable once written —
    Django refuses an empty key). Database passwords are generated only while
    the databases have never been initialized (data dirs absent); after that
    an empty value keeps compose's legacy default rather than locking the
    user out of their existing data. Returns the generated key names.
    """
    import secrets

    from .envfile import upsert_env

    updates: dict[str, str] = {}
    if not cluster.env.get("DJANGO_SECRET_KEY"):
        updates["DJANGO_SECRET_KEY"] = secrets.token_urlsafe(48)

    db_dir = cluster.root / cluster.env.get("AGENTX_DB_DIR", "./data")
    if not any((db_dir / d).is_dir() for d in ("postgres", "neo4j")):
        for key in ("NEO4J_PASSWORD", "POSTGRES_PASSWORD"):
            if not cluster.env.get(key):
                updates[key] = secrets.token_urlsafe(24)

    if updates:
        upsert_env(cluster.env_file, updates)
        cluster.env.update(updates)
    return sorted(updates)


def up(cluster: Cluster, runner: ComposeRunner | None = None, *, wait_first_boot: bool = True) -> LifecycleResult:
    runner = runner or SubprocessRunner()
    _ensure_identity(cluster)
    generated = _ensure_secrets(cluster)
    runs: list[RunResult] = []

    stale = _stale_services(cluster)
    if "*" in stale:
        recreate: list[str] = ["--force-recreate"]
    elif stale:
        recreate = ["--force-recreate", *sorted(stale)]
    else:
        recreate = []

    result = runner.run(_argv(cluster, ["up", "-d", *recreate]), cwd=cluster.root)
    runs.append(result)

    if not result.ok and wait_first_boot and "dependency failed to start" in (result.stderr + result.stdout):
        deadline = time.monotonic() + FIRST_BOOT_TIMEOUT
        while time.monotonic() < deadline:
            phase = api_container_phase(cluster, runner)
            if phase == "healthy":
                retry = runner.run(_argv(cluster, ["up", "-d"]), cwd=cluster.root)
                runs.append(retry)
                if retry.ok:
                    _persist(cluster)
                    return LifecycleResult(True, "up (waited out first-boot initialization)", runs)
                return LifecycleResult(False, f"up retry failed: {retry.stderr.strip()}", runs)
            if phase in ("initializing", "starting"):
                time.sleep(POLL_INTERVAL)
                continue
            return LifecycleResult(False, f"api container is {phase}; not waiting", runs)
        return LifecycleResult(False, f"api still initializing after {FIRST_BOOT_TIMEOUT}s", runs)

    if result.ok:
        _persist(cluster)
        detail = "up"
        if recreate:
            which = "all services" if "*" in stale else ", ".join(sorted(stale))
            detail = f"up (config changed → force-recreated {which})"
        if generated:
            detail += f" (generated {', '.join(generated)})"
        return LifecycleResult(True, detail, runs)
    return LifecycleResult(False, result.stderr.strip() or result.stdout.strip(), runs)


def down(cluster: Cluster, runner: ComposeRunner | None = None, *, volumes: bool = False) -> LifecycleResult:
    runner = runner or SubprocessRunner()
    args = ["down", "--volumes"] if volumes else ["down"]
    result = runner.run(_argv(cluster, args), cwd=cluster.root)
    return LifecycleResult(result.ok, "down" if result.ok else result.stderr.strip(), [result])


def restart(cluster: Cluster, runner: ComposeRunner | None = None) -> LifecycleResult:
    """Restart — config-aware (changed services are recreated, not restarted),
    and down-aware (a compose `restart` with no containers is a silent no-op,
    so a stopped cluster falls through to `up`)."""
    runner = runner or SubprocessRunner()
    stale = _stale_services(cluster)
    if stale:
        return up(cluster, runner)
    # stack_ps filters same-project orphans (the bundle's manager container),
    # so a down stack still falls through to `up` in bundle mode.
    rows = stack_ps(cluster, runner)
    if rows is not None and not rows:
        return up(cluster, runner)
    result = runner.run(_argv(cluster, ["restart"]), cwd=cluster.root)
    return LifecycleResult(result.ok, "restart" if result.ok else result.stderr.strip(), [result])


def rebuild(cluster: Cluster, runner: ComposeRunner | None = None) -> LifecycleResult:
    runner = runner or SubprocessRunner()
    if cluster.spec.kind != "source":
        return LifecycleResult(False, "rebuild only applies to source clusters (image deployments pull instead)")
    build = runner.run(_argv(cluster, ["build"]), cwd=cluster.root)
    if not build.ok:
        return LifecycleResult(False, build.stderr.strip(), [build])
    result = up(cluster, runner)
    result.runs.insert(0, build)
    return result


def destroy(cluster: Cluster, runner: ComposeRunner | None = None, *, remove_data: bool = True) -> LifecycleResult:
    """Tear down containers + volumes, then remove the cluster's data dirs.

    Confirmation (typed cluster name) is the caller's job — CLI and GUI both
    require it before invoking this.
    """
    runner = runner or SubprocessRunner()
    result = down(cluster, runner, volumes=True)
    if not result.ok:
        return result
    if remove_data:
        if cluster.cluster_dir == cluster.root:
            # Bundle mode: never delete the deployment root itself (it holds the
            # compose files + .env the user authored); remove data dirs only.
            for sub in ("data",):
                target = cluster.root / sub
                if target.is_dir():
                    shutil.rmtree(target)
            detail = "destroyed (containers + volumes + ./data)"
        else:
            shutil.rmtree(cluster.cluster_dir)
            detail = f"destroyed (containers + volumes + {cluster.cluster_dir})"
        return LifecycleResult(True, detail, result.runs)
    return LifecycleResult(True, "destroyed (containers + volumes; data kept)", result.runs)


def adopt(cluster: Cluster, runner: ComposeRunner | None = None) -> LifecycleResult:
    """One-time migration of a pre-manager cluster onto the `agentx-<name>`
    compose project.

    Legacy clusters ran under the project-directory default name, so their
    containers carry that project label; a manager `up` would collide on
    container names. Adopt = legacy down (default project) → manager up.
    Brief downtime; supervised.
    """
    runner = runner or SubprocessRunner()
    legacy_argv = ["docker", "compose", "--env-file", str(cluster.env_file)]
    from .overlays import compose_files  # local import to keep module deps one-way

    for f in compose_files(cluster.spec):
        legacy_argv += ["-f", str(cluster.root / f)]
    legacy_argv += ["--profile", "production", "down"]
    legacy_down = runner.run(legacy_argv, cwd=cluster.root)
    if not legacy_down.ok:
        return LifecycleResult(False, f"legacy down failed: {legacy_down.stderr.strip()}", [legacy_down])
    result = up(cluster, runner)
    result.runs.insert(0, legacy_down)
    if result.ok:
        result.detail = f"adopted onto project agentx-{cluster.name}"
    return result

"""Persistent per-workspace Docker containers (the 'container' shell backend).

A workspace with ``shell_backend='container'`` gets one long-lived container
(``agentx-ws-{id}``) the agent can install into; it persists across conversations until
the workspace is deleted or idle-GC'd. Managed via the ``docker`` CLI against whatever
``DOCKER_HOST`` is configured (dev: host Docker; prod: the dind sidecar).

The container is **clean** (no host secrets / no ``data/`` mount, only its own
``/workspace`` volume) on an isolated bridge, so network + root inside it are safe.
Files are moved in with ``docker cp`` (bind paths don't cross the dind boundary).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from .sandbox import ExecResult
from .workdir import _shell_base, ensure_workdir

logger = logging.getLogger(__name__)

_LABEL = "agentx.workspace"
_PREFIX = "agentx-ws-"
_META = "containers.json"  # API-side last-used registry under data/shell/


class ContainerError(Exception):
    """Docker/container operation failed."""


def _cfg() -> dict:
    from ...config import get_config_manager

    c = get_config_manager()
    return {
        "enabled": bool(c.get("shell.docker.enabled", False)),
        "image": c.get("shell.docker.image", "python:3.12-slim"),
        "memory": str(c.get("shell.docker.memory", "2g")),
        "cpus": str(c.get("shell.docker.cpus", "2")),
        "pids_limit": int(c.get("shell.docker.pids_limit", 512)),
        "network": c.get("shell.docker.network", "agentx-shell-net"),
        "idle_ttl_days": int(c.get("shell.docker.idle_ttl_days", 7)),
        "timeout": float(c.get("shell.timeout_seconds", 20)),
    }


def _name(workspace_id: str) -> str:
    return _PREFIX + workspace_id


def _docker(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run a ``docker`` CLI command (uses ambient env, incl. DOCKER_HOST)."""
    cmd = ["docker", *args]  # docker CLI on PATH (DOCKER_HOST picks the daemon)
    return subprocess.run(  # noqa: S603 - docker CLI managing the workspace sandbox
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )


def docker_available() -> bool:
    """Whether a Docker daemon is reachable (CLI present + `docker info` works)."""
    if shutil.which("docker") is None:
        return False
    try:
        return _docker(["info", "--format", "{{.ServerVersion}}"], timeout=10).returncode == 0
    except Exception:
        return False


# --- last-used registry (API-side; drives idle-GC + the UI countdown) --------

def _meta_path() -> Path:
    return _shell_base() / _META


def _load_meta() -> dict:
    p = _meta_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return {}


def _save_meta(meta: dict) -> None:
    p = _meta_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta))


def _stamp_used(workspace_id: str) -> None:
    meta = _load_meta()
    meta.setdefault(workspace_id, {})["last_used"] = int(time.time())
    _save_meta(meta)


# --- image / network provisioning -------------------------------------------

def _image_present(image: str) -> bool:
    return _docker(["image", "inspect", image], timeout=15).returncode == 0


def pull_image_async(image: str | None = None) -> None:
    """Pre-pull the base image in the background so first use doesn't block a turn."""
    img = image or _cfg()["image"]

    def _pull() -> None:
        try:
            logger.info("🐳 SHELL pulling base image %s", img)
            _docker(["pull", img], timeout=900)
        except Exception as e:  # pragma: no cover
            logger.warning("base image pull failed for %s: %s", img, e)

    import threading
    threading.Thread(target=_pull, daemon=True, name="img-pull").start()


def _ensure_network(network: str) -> None:
    if _docker(["network", "inspect", network], timeout=10).returncode != 0:
        _docker(["network", "create", "--internal=false", network], timeout=20)


# --- container lifecycle -----------------------------------------------------

def _state(workspace_id: str) -> str:
    """'none' | 'running' | 'stopped'."""
    r = _docker(["inspect", "-f", "{{.State.Status}}", _name(workspace_id)], timeout=10)
    if r.returncode != 0:
        return "none"
    status = r.stdout.strip()
    return "running" if status == "running" else "stopped"


def ensure_container(workspace_id: str) -> tuple[str, bool]:
    """Ensure the workspace container exists + is running. Returns (name, ready).

    ready=False means the base image is still being pulled (provisioning) — the caller
    should tell the agent to retry shortly rather than block.
    """
    cfg = _cfg()
    name = _name(workspace_id)
    if not _image_present(cfg["image"]):
        pull_image_async(cfg["image"])
        return name, False

    _ensure_network(cfg["network"])
    state = _state(workspace_id)
    if state == "none":
        vol = name
        run = _docker([
            "run", "-d", "--name", name,
            "--label", f"{_LABEL}={workspace_id}",
            "--network", cfg["network"],
            "--memory", cfg["memory"], "--cpus", cfg["cpus"],
            "--pids-limit", str(cfg["pids_limit"]),
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "-v", f"{vol}:/workspace", "-w", "/workspace",
            cfg["image"], "sleep", "infinity",
        ], timeout=120)
        if run.returncode != 0:
            raise ContainerError(f"docker run failed: {run.stderr.strip()[:300]}")
        _materialize(workspace_id, name)
    elif state == "stopped":
        if _docker(["start", name], timeout=30).returncode != 0:
            raise ContainerError("docker start failed")
    _stamp_used(workspace_id)
    return name, True


def _materialize(workspace_id: str, name: str) -> None:
    """Stage the workspace's files (via ensure_workdir) and copy them into /workspace."""
    try:
        staged = ensure_workdir(workspace_id, "_container")
    except Exception as e:  # pragma: no cover
        logger.debug("container materialize skipped: %s", e)
        return
    # `docker cp <dir>/. <name>:/workspace` copies the directory contents.
    _docker(["cp", f"{staged}/.", f"{name}:/workspace"], timeout=120)


def exec_command(workspace_id: str, command: str, timeout: float | None = None) -> ExecResult:
    """Run a command in the workspace container (in-container `timeout` + API backstop)."""
    cfg = _cfg()
    name, ready = ensure_container(workspace_id)
    if not ready:
        return ExecResult(
            stdout="", stderr="The workspace container is still provisioning (pulling its "
            "base image). Try again in a few seconds.",
            exit_code=-1, timed_out=False, duration_ms=0, sandbox="container",
        )
    t = float(timeout) if timeout else cfg["timeout"]
    start = time.monotonic()
    try:
        proc = _docker(
            ["exec", name, "timeout", str(int(t)), "sh", "-lc", command],
            timeout=t + 15,
        )
        timed_out = proc.returncode == 124  # `timeout` exit code
        return ExecResult(
            stdout=proc.stdout or "", stderr=proc.stderr or "",
            exit_code=proc.returncode, timed_out=timed_out,
            duration_ms=int((time.monotonic() - start) * 1000), sandbox="container",
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            stdout="", stderr=f"[killed: exceeded {t:.0f}s]", exit_code=-1, timed_out=True,
            duration_ms=int((time.monotonic() - start) * 1000), sandbox="container",
        )
    finally:
        _stamp_used(workspace_id)


# --- file ops (container mode; path-jailed to /workspace) --------------------

import shlex  # noqa: E402


def _safe_rel(path: str) -> str:
    """Validate a relative path stays under /workspace (no abs / `..` escape)."""
    if not path or path.startswith(("/", "~")):
        raise ContainerError("path must be relative to the workspace")
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ContainerError("path escapes the workspace")
    return "/".join(parts)


def write_file(workspace_id: str, path: str, content: str) -> dict:
    rel = _safe_rel(path)
    name, ready = ensure_container(workspace_id)
    if not ready:
        return {"error": "container is still provisioning", "success": False}
    target = f"/workspace/{rel}"
    _docker(["exec", name, "sh", "-lc", f"mkdir -p $(dirname {shlex.quote(target)})"], timeout=30)
    write_cmd = ["docker", "exec", "-i", name, "sh", "-lc", f"cat > {shlex.quote(target)}"]
    proc = subprocess.run(  # noqa: S603 - docker CLI; sandboxed container exec
        write_cmd, input=content or "", text=True, capture_output=True, timeout=60, check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[:300] or "write failed", "success": False}
    _stamp_used(workspace_id)
    return {"path": rel, "bytes": len((content or "").encode("utf-8")), "success": True}


def read_file(workspace_id: str, path: str, offset: int = 0, limit: int = 12000) -> dict:
    rel = _safe_rel(path)
    name, ready = ensure_container(workspace_id)
    if not ready:
        return {"error": "container is still provisioning", "success": False}
    proc = _docker(["exec", name, "sh", "-lc", f"cat {shlex.quote('/workspace/' + rel)}"], timeout=30)
    if proc.returncode != 0:
        return {"error": f"No such file: {path}", "success": False}
    text = proc.stdout
    sliced = text[offset:offset + limit]
    return {
        "path": rel, "content": sliced, "offset": offset, "limit": limit,
        "total_chars": len(text), "has_more": (offset + len(sliced)) < len(text), "success": True,
    }


def list_files(workspace_id: str) -> dict:
    name, ready = ensure_container(workspace_id)
    if not ready:
        return {"error": "container is still provisioning", "success": False}
    proc = _docker(
        ["exec", name, "sh", "-lc", "cd /workspace && find . -type f -not -path './.*' | sed 's|^\\./||' | head -500"],
        timeout=30,
    )
    files = [{"path": line} for line in proc.stdout.splitlines() if line.strip()]
    return {"files": files, "count": len(files), "success": True}


# --- status / actions (UI) ---------------------------------------------------

def status(workspace_id: str) -> dict:
    """Status + stats for the workspace container (drives the UI resource card)."""
    cfg = _cfg()
    name = _name(workspace_id)
    state = _state(workspace_id)
    meta = _load_meta().get(workspace_id, {})
    last_used = meta.get("last_used")
    out: dict = {
        "state": state,
        "image": cfg["image"],
        "last_used_at": last_used,
        "idle_gc_at": (last_used + cfg["idle_ttl_days"] * 86400) if last_used else None,
        "memory_bytes": None, "cpu_percent": None, "install_bytes": None,
    }
    if state == "none":
        return out
    insp = _docker(["inspect", "-f", "{{.State.StartedAt}}", name], timeout=10)
    if insp.returncode == 0:
        out["started_at"] = insp.stdout.strip()
    if state == "running":
        st = _docker(
            ["stats", "--no-stream", "--format", "{{.MemUsage}};{{.CPUPerc}}", name],
            timeout=15,
        )
        if st.returncode == 0 and ";" in st.stdout:
            mem, cpu = st.stdout.strip().split(";", 1)
            out["memory_usage"] = mem.strip()
            out["cpu_percent"] = cpu.strip()
        # Writable-layer (install) size via `docker ps -s`.
        ps = _docker(
            ["ps", "-a", "--no-trunc", "--filter", f"name=^{name}$",
             "--format", "{{.Size}}"], timeout=20,
        )
        if ps.returncode == 0:
            out["install_size"] = ps.stdout.strip()  # e.g. "12.3MB (virtual 150MB)"
    return out


def start(workspace_id: str) -> dict:
    if _state(workspace_id) == "none":
        ensure_container(workspace_id)
    else:
        _docker(["start", _name(workspace_id)], timeout=30)
    return status(workspace_id)


def stop(workspace_id: str) -> dict:
    if _state(workspace_id) != "none":
        _docker(["stop", _name(workspace_id)], timeout=30)
    return status(workspace_id)


def reset(workspace_id: str) -> dict:
    """Drop the writable layer (installs) but keep the workspace files: rm + recreate."""
    _docker(["rm", "-f", _name(workspace_id)], timeout=30)
    ensure_container(workspace_id)  # recreates from base image + re-materializes files
    return status(workspace_id)


def remove(workspace_id: str) -> None:
    """Remove the container + its volume (on workspace delete)."""
    _docker(["rm", "-f", _name(workspace_id)], timeout=30)
    _docker(["volume", "rm", "-f", _name(workspace_id)], timeout=30)
    meta = _load_meta()
    meta.pop(workspace_id, None)
    _save_meta(meta)


def gc(idle_days: int | None = None) -> int:
    """Remove idle workspace containers (last-used older than the TTL). Returns count."""
    ttl = idle_days if idle_days is not None else _cfg()["idle_ttl_days"]
    if ttl <= 0:
        return 0
    cutoff = time.time() - ttl * 86400
    meta = _load_meta()
    removed = 0
    for ws_id, info in list(meta.items()):
        if info.get("last_used", 0) < cutoff:
            remove(ws_id)
            removed += 1
    if removed:
        logger.info("🐳 SHELL GC removed %d idle container(s)", removed)
    return removed

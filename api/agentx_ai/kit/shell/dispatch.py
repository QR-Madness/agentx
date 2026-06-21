"""Route shell ops to the workspace's backend ('bubblewrap' | 'container').

The internal-tool layer calls these and stays backend-agnostic; deny-policy, output
capping, and audit live in the tools (they apply to both backends).
"""

from __future__ import annotations

from typing import Any

from ..workspaces import repository
from . import container
from .sandbox import ExecResult, get_sandbox
from .workdir import WorkdirError, ensure_workdir, resolve_in_workdir


class ShellUnavailable(Exception):
    """No usable shell backend (e.g. bubblewrap missing, or Docker unreachable)."""


def backend_for(workspace_id: str | None) -> str:
    if not workspace_id:
        return "bubblewrap"
    ws = repository.get_workspace(workspace_id)
    return (ws.get("shell_backend") if ws else None) or "bubblewrap"


def _bwrap_cfg() -> dict:
    from ...config import get_config_manager

    c = get_config_manager()
    return {
        "allow_unsandboxed": bool(c.get("shell.allow_unsandboxed", False)),
        "allow_network": bool(c.get("shell.allow_network", False)),
        "timeout": float(c.get("shell.timeout_seconds", 20)),
        "max_materialize_bytes": int(c.get("shell.max_materialize_bytes", 134217728)),
        "cleanup_days": int(c.get("shell.workdir_cleanup_days", 7)),
    }


def _bwrap_workdir(workspace_id: str | None, conversation_id: str):
    cfg = _bwrap_cfg()
    return ensure_workdir(
        workspace_id, conversation_id,
        max_materialize_bytes=cfg["max_materialize_bytes"], cleanup_days=cfg["cleanup_days"],
    )


def run(workspace_id: str | None, conversation_id: str, command: str,
        timeout: float | None = None) -> ExecResult:
    """Execute a command in the workspace's shell backend."""
    if backend_for(workspace_id) == "container" and workspace_id:
        return container.exec_command(workspace_id, command, timeout=timeout)

    cfg = _bwrap_cfg()
    sandbox = get_sandbox(allow_unsandboxed=cfg["allow_unsandboxed"])
    if sandbox is None:
        raise ShellUnavailable(
            "No shell sandbox available (bubblewrap missing). Install bubblewrap or set "
            "shell.allow_unsandboxed."
        )
    workdir = _bwrap_workdir(workspace_id, conversation_id)
    t = min(float(timeout) if timeout else cfg["timeout"], max(cfg["timeout"], 300.0))
    return sandbox.run(command, cwd=workdir, timeout=t, allow_network=cfg["allow_network"])


def write_file(workspace_id: str | None, conversation_id: str, path: str, content: str) -> dict[str, Any]:
    if backend_for(workspace_id) == "container" and workspace_id:
        return container.write_file(workspace_id, path, content)
    workdir = _bwrap_workdir(workspace_id, conversation_id)
    try:
        target = resolve_in_workdir(workdir, path)
    except WorkdirError as e:
        return {"error": str(e), "success": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    data = content or ""
    target.write_text(data)
    return {"path": path, "bytes": len(data.encode("utf-8")), "success": True}


def read_file(workspace_id: str | None, conversation_id: str, path: str,
              offset: int = 0, limit: int = 12000) -> dict[str, Any]:
    if backend_for(workspace_id) == "container" and workspace_id:
        return container.read_file(workspace_id, path, offset=offset, limit=limit)
    workdir = _bwrap_workdir(workspace_id, conversation_id)
    try:
        target = resolve_in_workdir(workdir, path)
    except WorkdirError as e:
        return {"error": str(e), "success": False}
    if not target.is_file():
        return {"error": f"No such file: {path}", "success": False}
    text = target.read_text(errors="replace")
    sliced = text[offset:offset + limit]
    return {
        "path": path, "content": sliced, "offset": offset, "limit": limit,
        "total_chars": len(text), "has_more": (offset + len(sliced)) < len(text), "success": True,
    }


def list_files(workspace_id: str | None, conversation_id: str) -> dict[str, Any]:
    if backend_for(workspace_id) == "container" and workspace_id:
        return container.list_files(workspace_id)
    from .workdir import _MARKER

    workdir = _bwrap_workdir(workspace_id, conversation_id)
    files: list[dict[str, Any]] = []
    for p in sorted(workdir.rglob("*")):
        if p.is_file() and p.name != _MARKER:
            files.append({"path": str(p.relative_to(workdir)), "bytes": p.stat().st_size})
        if len(files) >= 500:
            break
    return {"files": files, "count": len(files), "success": True}

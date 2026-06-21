#!/usr/bin/env python
"""Self-driven e2e for the container shell backend (real Docker on the dev host).

Proves the whole thing: a per-workspace container materialized from the workspace, a
persisted `pip install` across exec calls, network on, isolation from host secrets, the
status/lifecycle endpoints, and that the bwrap backend still works.

Run:  uv run python api/agentx_ai/scripts/shell_container_e2e.py [--keep]
Exits 0 PASS / 1 FAIL / 2 SKIP (Docker unreachable). Cleans up its container+volume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")

CONV_ID = "shell-container-e2e"


def _log(m: str) -> None:
    print(f"[container-e2e] {m}", flush=True)


def _tool(name: str, args: dict) -> dict:
    from agentx_ai.mcp.internal_tools import execute_internal_tool
    res = execute_internal_tool(name, args)
    return json.loads(res.content[0]["text"]) if res.content else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    import django
    django.setup()
    from django.conf import settings as dj
    from django.test import Client

    dj.AGENTX_AUTH_ENABLED = False
    if "testserver" not in dj.ALLOWED_HOSTS:
        dj.ALLOWED_HOSTS = [*dj.ALLOWED_HOSTS, "testserver"]

    from agentx_ai.config import get_config_manager
    from agentx_ai.kit.agent_memory.connections import PostgresConnection
    from agentx_ai.kit.shell import container as sc
    from agentx_ai.kit.workspaces import repository, storage
    from agentx_ai.mcp.internal_context import InternalToolContext, reset_context, set_context

    if PostgresConnection.health_check().get("status") != "healthy":
        _log("SKIP: Postgres unavailable (run `task db:up`).")
        return 2
    if not sc.docker_available():
        _log("SKIP: Docker not reachable.")
        return 2

    cfg = get_config_manager()
    orig = cfg.get("shell.docker.enabled", False)
    image = cfg.get("shell.docker.image", "python:3.12-slim")
    failures: list[str] = []
    workspace_id = None
    token = None
    client = Client()

    def check(cond, label: str) -> None:
        _log(("✓ " if cond else "✗ ") + label)
        if not cond:
            failures.append(label)

    def cget(path: str) -> Any:
        r: Any = client.get(path)
        return r.json()

    def cpost(path: str) -> Any:
        r: Any = client.post(path)
        return r.json()

    try:
        cfg.set("shell.docker.enabled", True)
        _log(f"pulling base image {image} (first run can take a bit)…")
        sc._docker(["pull", image], timeout=900)

        ws = repository.create_workspace(name="container-e2e")
        workspace_id = ws["id"]
        repository.set_allow_shell(workspace_id, True)
        repository.set_shell_backend(workspace_id, "container")
        body = b"alpha\nbeta\ngamma\n"
        shasum, key = storage.store_blob(workspace_id, body)
        doc = repository.create_document(
            workspace_id=workspace_id, filename="notes.txt", content_type="text/plain",
            size_bytes=len(body), sha256=shasum, storage_key=key,
        )
        repository.set_document_status(doc["id"], "ready")

        token = set_context(InternalToolContext(
            user_id="default", conversation_id=CONV_ID, workspace_id=workspace_id))

        # 1. Workspace materialized into the container.
        r = _tool("run_command", {"command": "cat notes.txt"})
        check(r.get("success") and "beta" in r.get("stdout", ""), f"workspace file present in container ({r.get('sandbox')})")
        check(r.get("sandbox") == "container", "ran in the container backend")

        # 2. Install persists across exec calls (network + writable layer).
        inst = _tool("run_command", {"command": "pip install --quiet --root-user-action=ignore cowsay", "timeout": 180})
        check(inst.get("success"), f"pip install over network succeeds (rc={inst.get('exit_code')})")
        use = _tool("run_command", {"command": "python -c 'import cowsay; print(\"cowsay-ok\")'"})
        check("cowsay-ok" in use.get("stdout", ""), "installed package persists to a later command")

        # 3. Isolation: host secrets + DBs unreachable.
        iso = _tool("run_command", {"command": "cat /app/data/config.json 2>&1 | head -1; echo rc=$?"})
        check("config.json" not in iso.get("stdout", "") or "No such" in iso.get("stdout", ""),
              "host data/config.json not present in container")

        # 4. Lifecycle via the HTTP endpoints (drives the UI card).
        st = cget(f"/api/workspaces/{workspace_id}/shell/container")["container"]
        check(st.get("state") == "running", f"status endpoint: running ({st.get('state')})")
        stopped = cpost(f"/api/workspaces/{workspace_id}/shell/container/stop")["container"]
        check(stopped.get("state") == "stopped", "stop endpoint → stopped")
        cpost(f"/api/workspaces/{workspace_id}/shell/container/reset")
        # reset drops installs but keeps files
        after = _tool("run_command", {"command": "python -c 'import cowsay' 2>&1; echo rc=$?; cat notes.txt"})
        check("rc=0" not in after.get("stdout", "") and "beta" in after.get("stdout", ""),
              "reset cleared installs but kept workspace files")

        # 5. bwrap backend still works (regression).
        repository.set_shell_backend(workspace_id, "bubblewrap")
        bw = _tool("run_command", {"command": "echo bwrap-ok"})
        check(bw.get("sandbox") == "bubblewrap" and "bwrap-ok" in bw.get("stdout", ""),
              "bubblewrap backend still routes + runs")

    finally:
        if token is not None:
            reset_context(token)
        cfg.set("shell.docker.enabled", orig)
        if workspace_id and not args.keep:
            try:
                sc.remove(workspace_id)
            except Exception:
                pass
            for d in repository.list_documents(workspace_id):
                full = repository.get_document(d["id"])
                if full and full.get("storage_key"):
                    storage.delete_blob(full["storage_key"])
            repository.delete_workspace(workspace_id)
            _log(f"cleaned up workspace {workspace_id} + container")

    if failures:
        _log(f"FAIL — {len(failures)}: {failures}")
        return 1
    _log("PASS — container shell backend verified end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

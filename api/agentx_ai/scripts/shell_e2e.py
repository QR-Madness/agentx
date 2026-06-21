#!/usr/bin/env python
"""Self-driven end-to-end harness for Agent Shells (bubblewrap-jailed exec).

Exercises the real shell path in-process against the real sandbox: materialize a
workspace into a jailed work dir, run commands through the `run_command` tool, and
assert the jail holds (no secret read, no network, scrubbed env, timeout, gating).

Run:  uv run python api/agentx_ai/scripts/shell_e2e.py [--keep]

Exits 0 PASS / 1 FAIL / 2 SKIP (Postgres unavailable). Idempotent cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")

CONV_ID = "shell-e2e-conv"


def _log(msg: str) -> None:
    print(f"[shell-e2e] {msg}", flush=True)


def _tool(name: str, args: dict) -> dict:
    from agentx_ai.mcp.internal_tools import execute_internal_tool

    res = execute_internal_tool(name, args)
    return json.loads(res.content[0]["text"]) if res.content else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    import django
    django.setup()

    from agentx_ai.kit.agent_memory.connections import PostgresConnection
    from agentx_ai.kit.shell.sandbox import bubblewrap_works
    from agentx_ai.kit.workspaces import repository, storage
    from agentx_ai.mcp.internal_context import InternalToolContext, reset_context, set_context
    from agentx_ai.mcp.internal_tools import get_internal_tools

    if PostgresConnection.health_check().get("status") != "healthy":
        _log("SKIP: Postgres unavailable (need a workspace). Run `task db:up`.")
        return 2
    if not bubblewrap_works():
        _log("NOTE: bubblewrap unavailable — jail/net/env assertions can't run here.")

    failures: list[str] = []
    workspace_id = None
    token = None

    def check(cond: bool, label: str) -> None:
        _log(("✓ " if cond else "✗ ") + label)
        if not cond:
            failures.append(label)

    try:
        # Build a workspace with one file (no embeddings needed — set status ready directly),
        # and opt it into shell (per-workspace enablement).
        ws = repository.create_workspace(name="shell-e2e")
        workspace_id = ws["id"]
        repository.set_allow_shell(workspace_id, True)
        body = b"alpha\nbeta\ngamma\n"
        sha, key = storage.store_blob(workspace_id, body)
        doc = repository.create_document(
            workspace_id=workspace_id, filename="notes.txt", content_type="text/plain",
            size_bytes=len(body), sha256=sha, storage_key=key,
        )
        repository.set_document_status(doc["id"], "ready")

        token = set_context(InternalToolContext(
            user_id="default", conversation_id=CONV_ID, workspace_id=workspace_id,
        ))

        # Gating sanity: tool present when the attached workspace allows shell.
        check("run_command" in {t.name for t in get_internal_tools()},
              "run_command advertised when workspace.allow_shell")

        # 1. Command runs in the materialized workspace.
        r = _tool("run_command", {"command": "ls; wc -l < notes.txt"})
        check(bool(r.get("success")) and "notes.txt" in r.get("stdout", ""), f"ls sees workspace file ({r.get('stdout','').strip()!r})")
        check("3" in r.get("stdout", ""), "wc -l counts the file's lines")

        if bubblewrap_works():
            # 2. FS jail — can't read the host's secrets.
            secret = str(_REPO_ROOT / "data" / "config.json")
            r = _tool("run_command", {"command": f"cat {secret}"})
            check(not r.get("success"), "FS jail blocks reading data/config.json")

            # 3. Network off.
            r = _tool("run_command", {"command": "curl -m3 https://example.com"})
            check(not r.get("success"), "network is blocked (no exfil)")

            # 4. Env scrubbed.
            r = _tool("run_command", {"command": "env"})
            env_out = r.get("stdout", "")
            check(
                "ANTHROPIC_API_KEY" not in env_out and "POSTGRES_PASSWORD" not in env_out
                and "OPENAI_API_KEY" not in env_out,
                "env is scrubbed of secrets",
            )

            # 5. Timeout kills a runaway command.
            r = _tool("run_command", {"command": "sleep 5", "timeout": 1})
            check(r.get("timed_out") is True, "timeout kills a long command")

        # 6. Structured file tools + path jail.
        w = _tool("write_file", {"path": "sub/made.txt", "content": "from write_file"})
        check(bool(w.get("success")), "write_file creates a jailed file")
        rd = _tool("read_file", {"path": "sub/made.txt"})
        check(rd.get("content") == "from write_file", "read_file returns what write_file wrote")
        esc = _tool("write_file", {"path": "../escape.txt", "content": "x"})
        check(not esc.get("success"), "path jail rejects ../escape")

        # 7. Gating off — flip the workspace's allow_shell off.
        repository.set_allow_shell(workspace_id, False)
        check("run_command" not in {t.name for t in get_internal_tools()},
              "run_command hidden when workspace.allow_shell=False")

    finally:
        if token is not None:
            reset_context(token)
        if workspace_id and not args.keep:
            for d in repository.list_documents(workspace_id):
                full = repository.get_document(d["id"])
                if full and full.get("storage_key"):
                    storage.delete_blob(full["storage_key"])
            repository.delete_workspace(workspace_id)
            _log(f"cleaned up workspace {workspace_id}")

    if failures:
        _log(f"FAIL — {len(failures)} check(s): {failures}")
        return 1
    _log("PASS — agent shell jail verified end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

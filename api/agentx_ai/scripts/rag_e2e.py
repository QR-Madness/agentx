#!/usr/bin/env python
"""Self-driven end-to-end harness for File Workspaces & Document RAG.

Drives the WHOLE RAG path against the real stack (your Docker Postgres + pgvector,
local embeddings) so nobody has to upload files by hand:

  Slice 1 (now):  create workspace → upload the seed PDF through the real HTTP
                  endpoint → poll until ready → assert blob + chunks + embeddings.
  Slice 2 (later): + retrieve via query_chunks / the workspace tools and assert
                  the right passage comes back; optional one chat turn.

Run:  uv run python api/agentx_ai/scripts/rag_e2e.py [--keep] [--pdf PATH]

Exits 0 on PASS, 1 on FAIL, 2 on SKIP (Postgres/embeddings unavailable). Idempotent:
cleans up its test workspace unless --keep.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")

DEFAULT_PDF = "/home/redacted/Downloads/Unsorted/report-2024-0fda7c20-66f7-48b2-8a25-6da8e6f6662f.pdf"
POLL_TIMEOUT_S = 240
POLL_INTERVAL_S = 3


def _log(msg: str) -> None:
    print(f"[rag-e2e] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Workspaces & Document RAG e2e harness")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="seed document to ingest")
    parser.add_argument("--keep", action="store_true", help="don't delete the test workspace")
    args = parser.parse_args()

    import django

    django.setup()
    from django.conf import settings as dj_settings
    from django.test import Client

    dj_settings.AGENTX_AUTH_ENABLED = False  # harness drives endpoints without a token
    if "testserver" not in dj_settings.ALLOWED_HOSTS:  # Client uses HTTP_HOST=testserver
        dj_settings.ALLOWED_HOSTS = [*dj_settings.ALLOWED_HOSTS, "testserver"]

    from agentx_ai.kit.agent_memory.connections import PostgresConnection, get_postgres_session
    from agentx_ai.kit.workspaces import storage

    # --- 0. Preconditions -------------------------------------------------
    health = PostgresConnection.health_check()
    if health.get("status") != "healthy":
        _log(f"SKIP: Postgres unavailable ({health.get('error')}). Run `task db:up && task db:migrate:pg`.")
        return 2

    src = Path(args.pdf)
    if not src.exists():
        _log(f"SKIP: seed document not found at {src}")
        return 2

    seed_dir = _REPO_ROOT / "data" / "seed-docs"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_dir / src.name
    if not seed.exists():
        shutil.copy2(src, seed)
    _log(f"seed document: {seed} ({seed.stat().st_size} bytes)")

    client = Client()
    workspace_id = None
    failures: list[str] = []

    def post(path: str, **kw) -> Any:
        return client.post(path, **kw)

    def get(path: str) -> Any:
        return client.get(path)

    def check(cond: bool, label: str) -> None:
        _log(("✓ " if cond else "✗ ") + label)
        if not cond:
            failures.append(label)

    try:
        # --- 1. Create workspace -----------------------------------------
        resp = post(
            "/api/workspaces",
            data='{"name": "rag-e2e harness"}',
            content_type="application/json",
        )
        check(resp.status_code == 201, f"create workspace → 201 (got {resp.status_code})")
        workspace_id = resp.json()["workspace"]["id"]
        _log(f"workspace_id = {workspace_id}")

        # --- 2. Upload the seed PDF through the real multipart endpoint ----
        with seed.open("rb") as fh:
            resp = post(f"/api/workspaces/{workspace_id}/documents", data={"file": fh})
        check(resp.status_code == 201, f"upload document → 201 (got {resp.status_code})")
        doc = resp.json()["document"]
        document_id = doc["id"]
        check(doc["status"] == "pending", f"initial status pending (got {doc['status']})")
        check(bool(doc["sha256"]), "document has sha256")

        # --- 3. Poll until ingestion settles -----------------------------
        status, deadline = "pending", time.monotonic() + POLL_TIMEOUT_S
        current: dict[str, Any] | None = None
        while status == "pending" and time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL_S)
            listing = get(f"/api/workspaces/{workspace_id}/documents").json()["documents"]
            current = next((d for d in listing if d["id"] == document_id), None)
            status = current["status"] if current else "missing"
            _log(f"  …status={status}")
        check(status == "ready", f"ingestion reached ready (final={status})")
        if current:
            _log(f"  tags={current.get('tags')} summary={(current.get('summary') or '')[:80]!r}")

        # --- 4. Assert the three stores agree ----------------------------
        full = get(f"/api/workspaces/{workspace_id}/documents/{document_id}").json()["document"]
        blob = storage.read_blob(_storage_key(document_id))
        check(blob is not None and len(blob) == full["size_bytes"], "blob present on disk, size matches")

        with get_postgres_session() as s:
            from sqlalchemy import text
            total = s.execute(
                text("SELECT COUNT(*) FROM document_chunks WHERE document_id = :id"),
                {"id": document_id},
            ).scalar()
            embedded = s.execute(
                text("SELECT COUNT(*) FROM document_chunks WHERE document_id = :id AND embedding IS NOT NULL"),
                {"id": document_id},
            ).scalar()
        check((total or 0) > 0, f"document_chunks populated (n={total})")
        check(total == embedded, f"every chunk embedded ({embedded}/{total})")

        # --- 5. Retrieve (deterministic, no chat LLM needed) -------------
        from agentx_ai.kit.workspaces import retrieval

        query = "programming languages and total coding time"
        hits = retrieval.query_chunks(workspace_id, query, top_k=3)
        check(len(hits) > 0, f"document_query returns chunks for {query!r} (n={len(hits)})")
        if hits:
            top = hits[0]
            check(top["filename"] == seed.name, f"top hit is the seed doc ({top['filename']})")
            check(top["score"] > 0.1, f"top hit has a real similarity score ({top['score']})")
            _log(f"  top chunk #{top['chunk_index']} score={top['score']} :: {top['text'][:70]!r}")

        manifest = retrieval.search_manifest(workspace_id, seed.stem[:8])
        check(len(manifest) > 0, f"workspace_search finds the file by name (n={len(manifest)})")

        # The agent-facing tools, scoped via the per-turn internal context.
        from agentx_ai.mcp.internal_context import (
            InternalToolContext, reset_context, set_context,
        )
        from agentx_ai.mcp.internal_tools import execute_internal_tool
        tok = set_context(InternalToolContext(user_id="default", workspace_id=workspace_id))
        try:
            tool_res = execute_internal_tool("document_query", {"query": query, "top_k": 3})
            import json as _json
            payload = _json.loads(tool_res.content[0]["text"]) if tool_res.content else {}
            check(tool_res.success and payload.get("count", 0) > 0,
                  f"document_query tool returns results (count={payload.get('count')})")
            read_res = execute_internal_tool("read_document", {"document_id": document_id, "limit": 500})
            read_payload = _json.loads(read_res.content[0]["text"]) if read_res.content else {}
            check(read_res.success and bool(read_payload.get("content")),
                  "read_document tool returns content")
        finally:
            reset_context(tok)

    finally:
        if workspace_id and not args.keep:
            client.delete(f"/api/workspaces/{workspace_id}")
            _log(f"cleaned up workspace {workspace_id}")
        elif workspace_id:
            _log(f"kept workspace {workspace_id} (--keep)")

    if failures:
        _log(f"FAIL — {len(failures)} check(s) failed: {failures}")
        return 1
    _log("PASS — full ingest + retrieval path verified end to end")
    return 0


def _storage_key(document_id: str) -> str:
    from agentx_ai.kit.workspaces import repository

    doc = repository.get_document(document_id)
    return doc["storage_key"] if doc else ""


if __name__ == "__main__":
    raise SystemExit(main())

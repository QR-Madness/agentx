"""Per-conversation shell work dirs materialized from a workspace.

Layout: ``${AGENTX_DB_DIR:-<repo>/data}/shell/{workspace_id|_scratch}/{conversation_id}/``.
A workspace's ready documents are written as ``filename → bytes`` once per conversation
(idempotent via a manifest marker). Agent-created files persist across commands in the
conversation; stale dirs are GC'd by mtime. All paths are jailed under the work dir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

from ..workspaces import repository, storage

logger = logging.getLogger(__name__)

_MARKER = ".agentx-manifest"
_SCRATCH = "_scratch"
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


class WorkdirError(Exception):
    """Materialization refused (e.g. workspace exceeds the size cap)."""


def _shell_base() -> Path:
    env = os.environ.get("AGENTX_DB_DIR")
    base = Path(env) if env else Path(__file__).resolve().parents[4] / "data"
    return base / "shell"


def _safe_segment(value: str) -> str:
    """A filesystem-safe single path segment (no separators / traversal)."""
    return _SAFE_ID.sub("_", value or "")[:128] or "_"


def ensure_workdir(
    workspace_id: str | None,
    conversation_id: str,
    *,
    max_materialize_bytes: int = 128 * 1024 * 1024,
    cleanup_days: int = 7,
) -> Path:
    """Return the conversation's work dir, materializing the workspace into it once.

    Raises :class:`WorkdirError` if the workspace's documents exceed
    ``max_materialize_bytes``. Opportunistically GCs stale work dirs.
    """
    gc_workdirs(cleanup_days)

    ws_seg = _safe_segment(workspace_id) if workspace_id else _SCRATCH
    workdir = _shell_base() / ws_seg / _safe_segment(conversation_id)
    workdir.mkdir(parents=True, exist_ok=True)

    if not workspace_id:
        return workdir

    docs = [d for d in repository.list_documents(workspace_id) if d.get("status") == "ready"]
    manifest = sorted((d["filename"], d.get("sha256") or "") for d in docs)
    marker = workdir / _MARKER
    digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    # Materialize once per conversation; skip if already done (don't clobber agent edits).
    if marker.exists() and marker.read_text().strip() == digest:
        return workdir

    total = sum(int(d.get("size_bytes") or 0) for d in docs)
    if total > max_materialize_bytes:
        raise WorkdirError(
            f"workspace is {total} bytes; shell.max_materialize_bytes is {max_materialize_bytes}"
        )

    for d in docs:
        name = Path(d["filename"]).name  # basename only — no path traversal
        if not name or name in {".", ".."}:
            continue
        full = repository.get_document(d["id"])
        key = (full or {}).get("storage_key")
        raw = storage.read_blob(key) if key else None
        if raw is None:
            continue
        (workdir / name).write_bytes(raw)

    marker.write_text(digest)
    logger.info("🐚 SHELL materialized %d file(s) into %s", len(docs), workdir)
    return workdir


def gc_workdirs(cleanup_days: int) -> int:
    """Remove conversation work dirs whose mtime is older than ``cleanup_days``.

    Cheap + opportunistic (called from ``ensure_workdir``); returns the count removed.
    """
    base = _shell_base()
    if cleanup_days <= 0 or not base.exists():
        return 0
    cutoff = time.time() - cleanup_days * 86400
    removed = 0
    for ws_dir in base.iterdir():
        if not ws_dir.is_dir():
            continue
        for conv_dir in ws_dir.iterdir():
            try:
                if conv_dir.is_dir() and conv_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(conv_dir, ignore_errors=True)
                    removed += 1
            except OSError:  # pragma: no cover - racey fs
                continue
    if removed:
        logger.info("🐚 SHELL GC removed %d stale work dir(s)", removed)
    return removed


def resolve_in_workdir(workdir: Path, relpath: str) -> Path:
    """Resolve ``relpath`` strictly inside ``workdir`` (rejects abs paths / ``..`` escapes)."""
    if not relpath or relpath.startswith(("/", "~")):
        raise WorkdirError("path must be relative to the workspace")
    target = (workdir / relpath).resolve()
    workroot = workdir.resolve()
    if target != workroot and workroot not in target.parents:
        raise WorkdirError("path escapes the workspace")
    return target

"""Content-addressed blob store for workspace documents.

Bytes are keyed by sha256 (Git-like): free dedup + integrity. Layout mirrors the
existing ``./data`` bind-mount pattern:

    ${AGENTX_DB_DIR:-<repo>/data}/workspaces/{workspace_id}/{sha256}

Swappable for MinIO/S3 later (Phase 17 prod/cluster path) — callers only ever see
the opaque ``storage_key`` returned by :func:`store_blob`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _base_dir() -> Path:
    """Root of the on-disk store. Honors ``AGENTX_DB_DIR``; defaults to ``<repo>/data``."""
    env = os.environ.get("AGENTX_DB_DIR")
    base = Path(env) if env else Path(__file__).resolve().parents[4] / "data"
    return base / "workspaces"


def sha256_bytes(raw: bytes) -> str:
    """Hex sha256 digest of ``raw``."""
    return hashlib.sha256(raw).hexdigest()


def storage_key_for(workspace_id: str, sha256: str) -> str:
    """The relative storage key (stable, store-backend-agnostic)."""
    return f"{workspace_id}/{sha256}"


def store_blob(workspace_id: str, raw: bytes) -> tuple[str, str]:
    """Persist ``raw`` content-addressed. Returns ``(sha256, storage_key)``.

    Idempotent: re-storing identical bytes is a no-op (the path already exists).
    """
    digest = sha256_bytes(raw)
    key = storage_key_for(workspace_id, digest)
    path = _base_dir() / key
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp sibling then atomically rename, so a crash mid-write can't
        # leave a corrupt blob at the content-addressed key.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return digest, key


def read_blob(storage_key: str) -> bytes | None:
    """Read bytes by storage key, or ``None`` if missing."""
    path = _base_dir() / storage_key
    if not path.exists():
        return None
    return path.read_bytes()


def delete_blob(storage_key: str) -> bool:
    """Delete a blob by storage key. Returns True if a file was removed."""
    path = _base_dir() / storage_key
    if path.exists():
        path.unlink()
        return True
    return False

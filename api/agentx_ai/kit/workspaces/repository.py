"""Postgres persistence for workspaces, documents, and chunks.

Thin SQL over :func:`get_postgres_session` (no ORM models — matches the rest of the
memory PG layer). Embeddings are written as pgvector text literals (no pgvector
Python dep needed); the cosine operator ``<=>`` backs retrieval (Slice 2).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text

from ..agent_memory.connections import get_postgres_session


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def vector_literal(values: list[float]) -> str:
    """Format an embedding as a pgvector literal: ``[0.1,0.2,...]``."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


# --- Workspaces -------------------------------------------------------------

def create_workspace(name: str, user_id: str = "default") -> dict[str, Any]:
    ws_id = _new_id("ws")
    with get_postgres_session() as s:
        s.execute(
            text(
                "INSERT INTO workspaces (id, user_id, name) VALUES (:id, :uid, :name)"
            ),
            {"id": ws_id, "uid": user_id, "name": name},
        )
        s.commit()
    return get_workspace(ws_id)  # type: ignore[return-value]


# Reserved, well-known workspace id for the user's personal "Home" space (avatars,
# custom icons, personal files). Visible + renamable like any other workspace.
HOME_WORKSPACE_ID = "ws_home"


def ensure_workspace(workspace_id: str, name: str, user_id: str = "default") -> dict[str, Any]:
    """Get-or-create a workspace with an *explicit* id (idempotent). Used for reserved
    workspaces like Home; a rename later won't be clobbered (we only insert if absent)."""
    with get_postgres_session() as s:
        s.execute(
            text(
                """
                INSERT INTO workspaces (id, user_id, name) VALUES (:id, :uid, :name)
                ON CONFLICT (id) DO NOTHING
                """
            ),
            {"id": workspace_id, "uid": user_id, "name": name},
        )
        s.commit()
    return get_workspace(workspace_id)  # type: ignore[return-value]


def ensure_home_workspace(user_id: str = "default") -> dict[str, Any]:
    """The user's reserved personal workspace (avatars + personal files)."""
    return ensure_workspace(HOME_WORKSPACE_ID, "Home", user_id)


def list_workspaces(user_id: str = "default") -> list[dict[str, Any]]:
    with get_postgres_session() as s:
        rows = s.execute(
            text(
                """
                SELECT w.id, w.user_id, w.name, w.allow_shell, w.shell_backend,
                       w.created_at, w.updated_at,
                       COUNT(d.id)                       AS document_count,
                       COALESCE(SUM(d.size_bytes), 0)    AS used_bytes
                FROM workspaces w
                LEFT JOIN documents d ON d.workspace_id = w.id
                WHERE w.user_id = :uid
                GROUP BY w.id
                ORDER BY w.updated_at DESC
                """
            ),
            {"uid": user_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    with get_postgres_session() as s:
        row = s.execute(
            text(
                """
                SELECT w.id, w.user_id, w.name, w.allow_shell, w.shell_backend,
                       w.created_at, w.updated_at,
                       COUNT(d.id)                       AS document_count,
                       COALESCE(SUM(d.size_bytes), 0)    AS used_bytes
                FROM workspaces w
                LEFT JOIN documents d ON d.workspace_id = w.id
                WHERE w.id = :id
                GROUP BY w.id
                """
            ),
            {"id": workspace_id},
        ).mappings().first()
    return dict(row) if row else None


def rename_workspace(workspace_id: str, name: str) -> dict[str, Any] | None:
    with get_postgres_session() as s:
        res = s.execute(
            text(
                "UPDATE workspaces SET name = :name, updated_at = NOW() WHERE id = :id"
            ),
            {"id": workspace_id, "name": name},
        )
        s.commit()
        if getattr(res, "rowcount", 0) == 0:
            return None
    return get_workspace(workspace_id)


def set_allow_shell(workspace_id: str, allow_shell: bool) -> dict[str, Any] | None:
    """Toggle per-workspace agent-shell access (opt-in; default false)."""
    with get_postgres_session() as s:
        res = s.execute(
            text("UPDATE workspaces SET allow_shell = :v, updated_at = NOW() WHERE id = :id"),
            {"id": workspace_id, "v": bool(allow_shell)},
        )
        s.commit()
        if getattr(res, "rowcount", 0) == 0:
            return None
    return get_workspace(workspace_id)


SHELL_BACKENDS = ("bubblewrap", "container")


def set_shell_backend(workspace_id: str, backend: str) -> dict[str, Any] | None:
    """Choose the per-workspace shell sandbox ('bubblewrap' | 'container')."""
    if backend not in SHELL_BACKENDS:
        raise ValueError(f"invalid shell_backend: {backend!r}")
    with get_postgres_session() as s:
        res = s.execute(
            text("UPDATE workspaces SET shell_backend = :b, updated_at = NOW() WHERE id = :id"),
            {"id": workspace_id, "b": backend},
        )
        s.commit()
        if getattr(res, "rowcount", 0) == 0:
            return None
    return get_workspace(workspace_id)


def delete_workspace(workspace_id: str) -> bool:
    """Delete a workspace and (via FK cascade) its documents + chunks.

    Caller is responsible for removing blobs (see views) — bytes live off-DB.
    """
    with get_postgres_session() as s:
        res = s.execute(
            text("DELETE FROM workspaces WHERE id = :id"), {"id": workspace_id}
        )
        s.commit()
        return getattr(res, "rowcount", 0) > 0


def workspace_usage_bytes(workspace_id: str) -> int:
    with get_postgres_session() as s:
        val = s.execute(
            text("SELECT COALESCE(SUM(size_bytes), 0) FROM documents WHERE workspace_id = :id"),
            {"id": workspace_id},
        ).scalar()
    return int(val or 0)


# --- Documents --------------------------------------------------------------

def create_document(
    *,
    workspace_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    sha256: str,
    storage_key: str,
    status: str = "pending",
) -> dict[str, Any]:
    """Create a document row. ``status`` defaults to ``pending`` (text docs await
    ingestion); binary media stored via ``service.store_media`` pass ``ready`` since
    there's no parse/chunk/embed step."""
    doc_id = _new_id("doc")
    with get_postgres_session() as s:
        s.execute(
            text(
                """
                INSERT INTO documents
                    (id, workspace_id, filename, content_type, size_bytes, sha256, storage_key, status)
                VALUES (:id, :wid, :fn, :ct, :sz, :sha, :key, :status)
                """
            ),
            {
                "id": doc_id, "wid": workspace_id, "fn": filename, "ct": content_type,
                "sz": size_bytes, "sha": sha256, "key": storage_key, "status": status,
            },
        )
        s.execute(text("UPDATE workspaces SET updated_at = NOW() WHERE id = :id"), {"id": workspace_id})
        s.commit()
    return get_document(doc_id)  # type: ignore[return-value]


def list_documents(workspace_id: str) -> list[dict[str, Any]]:
    with get_postgres_session() as s:
        rows = s.execute(
            text(
                """
                SELECT id, workspace_id, filename, content_type, size_bytes, sha256,
                       tags, summary, status, error, created_at, updated_at
                FROM documents WHERE workspace_id = :wid ORDER BY created_at DESC
                """
            ),
            {"wid": workspace_id},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_document(document_id: str) -> dict[str, Any] | None:
    with get_postgres_session() as s:
        row = s.execute(
            text(
                """
                SELECT id, workspace_id, filename, content_type, size_bytes, sha256,
                       storage_key, tags, summary, status, error, created_at, updated_at
                FROM documents WHERE id = :id
                """
            ),
            {"id": document_id},
        ).mappings().first()
    return dict(row) if row else None


def set_document_enrichment(
    document_id: str, *, tags: list[str], summary: str, status: str, error: str | None = None
) -> None:
    with get_postgres_session() as s:
        s.execute(
            text(
                """
                UPDATE documents
                SET tags = :tags, summary = :summary, status = :status,
                    error = :error, updated_at = NOW()
                WHERE id = :id
                """
            ),
            {"id": document_id, "tags": tags, "summary": summary, "status": status, "error": error},
        )
        s.commit()


def set_document_status(document_id: str, status: str, error: str | None = None) -> None:
    with get_postgres_session() as s:
        s.execute(
            text("UPDATE documents SET status = :st, error = :err, updated_at = NOW() WHERE id = :id"),
            {"id": document_id, "st": status, "err": error},
        )
        s.commit()


def delete_document(document_id: str) -> str | None:
    """Delete a document (chunks cascade). Returns its ``storage_key`` for blob cleanup."""
    doc = get_document(document_id)
    if not doc:
        return None
    with get_postgres_session() as s:
        s.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
        s.commit()
    return doc.get("storage_key")


def pending_document_ids(limit: int = 25) -> list[str]:
    with get_postgres_session() as s:
        rows = s.execute(
            text("SELECT id FROM documents WHERE status = 'pending' ORDER BY created_at LIMIT :lim"),
            {"lim": limit},
        ).all()
    return [r[0] for r in rows]


# --- Chunks -----------------------------------------------------------------

def replace_chunks(
    document_id: str, workspace_id: str, chunks: list[tuple[int, str, list[float]]]
) -> int:
    """Replace a document's chunks with ``(chunk_index, text, embedding)`` rows.

    Idempotent for re-ingestion: clears existing chunks first.
    """
    with get_postgres_session() as s:
        s.execute(text("DELETE FROM document_chunks WHERE document_id = :id"), {"id": document_id})
        for idx, chunk_text, embedding in chunks:
            s.execute(
                text(
                    """
                    INSERT INTO document_chunks
                        (document_id, workspace_id, chunk_index, text, embedding)
                    VALUES (:did, :wid, :idx, :txt, CAST(:emb AS vector))
                    """
                ),
                {
                    "did": document_id, "wid": workspace_id, "idx": idx,
                    "txt": chunk_text, "emb": vector_literal(embedding),
                },
            )
        s.commit()
    return len(chunks)

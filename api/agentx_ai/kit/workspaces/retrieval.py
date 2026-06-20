"""Two-tier retrieval over a workspace.

  - :func:`search_manifest` — the catalog tier: find the right *file* by
    filename / tag / summary (keyword).
  - :func:`query_chunks` — the semantic tier: find the right *passage* by pgvector
    cosine similarity over ``document_chunks``.
  - :func:`read_document` — paginated full text of one document.

Mirrors the shipped stored-output ``tool_output_section`` (list → fetch) +
``tool_output_query`` (semantic) pattern, just persisted per workspace.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from ..agent_memory.connections import get_postgres_session
from ..agent_memory.embeddings import get_embedder
from . import parsing, repository, storage
from .repository import vector_literal


def search_manifest(workspace_id: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Catalog search: rank documents by filename / tag / summary match.

    Keyword (ILIKE) over the manifest — cheap and exact for "which file is about X".
    Empty/whitespace query returns the most-recent documents.
    """
    q = (query or "").strip()
    with get_postgres_session() as s:
        if not q:
            rows = s.execute(
                text(
                    """
                    SELECT id, filename, content_type, tags, summary, status, size_bytes
                    FROM documents WHERE workspace_id = :wid AND status = 'ready'
                    ORDER BY created_at DESC LIMIT :lim
                    """
                ),
                {"wid": workspace_id, "lim": limit},
            ).mappings().all()
        else:
            rows = s.execute(
                text(
                    """
                    SELECT id, filename, content_type, tags, summary, status, size_bytes,
                           (CASE WHEN filename ILIKE :like THEN 3 ELSE 0 END
                            + CASE WHEN summary  ILIKE :like THEN 2 ELSE 0 END
                            + CASE WHEN array_to_string(tags, ' ') ILIKE :like THEN 2 ELSE 0 END
                           ) AS score
                    FROM documents
                    WHERE workspace_id = :wid AND status = 'ready'
                      AND (filename ILIKE :like OR summary ILIKE :like
                           OR array_to_string(tags, ' ') ILIKE :like)
                    ORDER BY score DESC, created_at DESC LIMIT :lim
                    """
                ),
                {"wid": workspace_id, "like": f"%{q}%", "lim": limit},
            ).mappings().all()
    return [_manifest_row(r) for r in rows]


def query_chunks(workspace_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic search: embed ``query`` and return the closest chunks (cosine).

    ``<=>`` is pgvector cosine distance (0 = identical); score = ``1 - distance``.
    """
    q = (query or "").strip()
    if not q:
        return []
    embedding = get_embedder().embed([q])[0]
    with get_postgres_session() as s:
        rows = s.execute(
            text(
                """
                SELECT c.document_id, c.chunk_index, c.text, d.filename,
                       1 - (c.embedding <=> CAST(:emb AS vector)) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.workspace_id = :wid AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> CAST(:emb AS vector)
                LIMIT :k
                """
            ),
            {"wid": workspace_id, "emb": vector_literal(embedding), "k": top_k},
        ).mappings().all()
    return [
        {
            "document_id": r["document_id"],
            "filename": r["filename"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "score": round(float(r["score"]), 4),
        }
        for r in rows
    ]


def read_document(
    document_id: str, offset: int = 0, limit: int = 12000, workspace_id: str | None = None
) -> dict[str, Any] | None:
    """Paginated plain text of a document (reconstructed from its stored bytes).

    ``workspace_id``, when given, scopes the lookup (so a tool can't read across
    workspaces). Returns ``None`` if not found / not in the workspace.
    """
    doc = repository.get_document(document_id)
    if not doc or (workspace_id is not None and doc["workspace_id"] != workspace_id):
        return None
    raw = storage.read_blob(doc["storage_key"])
    if raw is None:
        return None
    try:
        full = parsing.parse_to_text(raw, doc["filename"])
    except parsing.UnsupportedDocumentError:
        full = ""
    sliced = full[offset : offset + limit]
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "offset": offset,
        "limit": limit,
        "total_chars": len(full),
        "has_more": (offset + len(sliced)) < len(full),
        "content": sliced,
    }


def render_manifest_block(workspace_id: str, max_files: int = 50) -> str:
    """Render the workspace's file list for the turn preamble (stable awareness).

    Names + tags + short summary only (bounded) so the agent knows *what corpus it
    has* before retrieving. Returns "" when the workspace has no ready documents.
    """
    docs = [d for d in repository.list_documents(workspace_id) if d.get("status") == "ready"]
    if not docs:
        return ""
    lines = [
        "Attached workspace — files you can search with `workspace_search` "
        "(by name/tag) and `document_query` (by meaning), then `read_document`:"
    ]
    for d in docs[:max_files]:
        tags = ", ".join(d.get("tags") or [])
        summary = (d.get("summary") or "").strip()
        meta = f" [{tags}]" if tags else ""
        tail = f" — {summary}" if summary else ""
        lines.append(f"- {d['filename']}{meta}{tail}")
    if len(docs) > max_files:
        lines.append(f"…and {len(docs) - max_files} more files.")
    return "\n".join(lines)


def _manifest_row(r: Any) -> dict[str, Any]:
    return {
        "document_id": r["id"],
        "filename": r["filename"],
        "content_type": r["content_type"],
        "tags": list(r["tags"] or []),
        "summary": r["summary"] or "",
        "size_bytes": int(r["size_bytes"] or 0),
    }

"""Upload orchestration: validate → store blob → create manifest row → ingest.

Keeps the Django view thin and the policy (quota, size, type) in one testable place.
Raises :class:`WorkspaceError` with a ``code`` the view maps to an HTTP status.

Also home to the text-document write path (``create_text_document`` /
``update_text_document``) shared by the REST endpoints and the agent's
``create_document``/``update_document`` internal tools.
"""

from __future__ import annotations

from typing import Any

from ..agent_memory.config import get_settings
from . import ingestion, parsing, repository, storage


class WorkspaceError(Exception):
    """A policy/validation failure during upload. ``code`` drives the HTTP status."""

    def __init__(self, code: str, message: str, *, document_id: str | None = None):
        super().__init__(message)
        # "not_found" | "unsupported" | "too_large" | "quota_exceeded" | "conflict"
        self.code = code
        self.message = message
        self.document_id = document_id  # conflict: the existing/current document


def upload_document(
    *, workspace_id: str, filename: str, content_type: str, raw: bytes
) -> dict[str, Any]:
    """Validate + persist an uploaded file, then kick off background ingestion."""
    settings = get_settings()

    if repository.get_workspace(workspace_id) is None:
        raise WorkspaceError("not_found", f"workspace {workspace_id} not found")

    if not parsing.is_supported(filename, settings.workspace_allowed_extensions):
        raise WorkspaceError(
            "unsupported",
            f"file type not supported (allowed: {', '.join(settings.workspace_allowed_extensions)})",
        )

    size = len(raw)
    if size == 0:
        raise WorkspaceError("unsupported", "file is empty")
    if size > settings.workspace_max_file_bytes:
        raise WorkspaceError(
            "too_large",
            f"file is {size} bytes; per-file limit is {settings.workspace_max_file_bytes}",
        )

    used = repository.workspace_usage_bytes(workspace_id)
    if used + size > settings.workspace_quota_bytes:
        raise WorkspaceError(
            "quota_exceeded",
            f"workspace quota exceeded ({used + size} > {settings.workspace_quota_bytes} bytes)",
        )

    sha256, storage_key = storage.store_blob(workspace_id, raw)
    doc = repository.create_document(
        workspace_id=workspace_id,
        filename=filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=size,
        sha256=sha256,
        storage_key=storage_key,
    )
    ingestion.ingest_document_async(doc["id"])
    return doc


# --- Text documents (agent tools + hub editor) ------------------------------

_TEXT_CONTENT_TYPES = {"md": "text/markdown", "markdown": "text/markdown", "txt": "text/plain"}


def normalize_text_filename(filename: str) -> str:
    """Validate + normalize a text-document filename.

    Allows an optional single-level ``folder/`` prefix (the hub's grouping
    convention, cf. ``avatars/``). Raises ``WorkspaceError("unsupported")`` on
    traversal attempts, empty names, or unwritable extensions.
    """
    name = (filename or "").strip().strip("/")
    if not name or "\x00" in name or "\\" in name:
        raise WorkspaceError("unsupported", "invalid filename")
    parts = name.split("/")
    if len(parts) > 2 or any(p in ("", ".", "..") for p in parts):
        raise WorkspaceError(
            "unsupported", "filename may have at most one folder level and no '.'/'..' segments"
        )
    ext = parts[-1].rsplit(".", 1)[-1].lower() if "." in parts[-1] else ""
    allowed = get_settings().workspace_agent_writable_extensions
    if ext not in allowed:
        raise WorkspaceError(
            "unsupported", f"writable file type must be one of: {', '.join(allowed)}"
        )
    return name


def _check_text_size_and_quota(
    workspace_id: str, size: int, *, replacing_bytes: int = 0
) -> None:
    settings = get_settings()
    if size == 0:
        raise WorkspaceError("unsupported", "document content is empty")
    if size > settings.workspace_max_file_bytes:
        raise WorkspaceError(
            "too_large",
            f"content is {size} bytes; per-file limit is {settings.workspace_max_file_bytes}",
        )
    used = repository.workspace_usage_bytes(workspace_id)
    if used - replacing_bytes + size > settings.workspace_quota_bytes:
        raise WorkspaceError(
            "quota_exceeded",
            f"workspace quota exceeded ({used - replacing_bytes + size} > "
            f"{settings.workspace_quota_bytes} bytes)",
        )


def release_blob_if_unreferenced(
    workspace_id: str, storage_key: str, *, exclude_id: str | None = None
) -> bool:
    """Delete a blob only when no (other) document row still references it —
    sha-dedup means two docs can share one file on disk."""
    if repository.count_documents_with_storage_key(
        workspace_id, storage_key, exclude_id=exclude_id
    ) > 0:
        return False
    return storage.delete_blob(storage_key)


def create_text_document(
    *, workspace_id: str, filename: str, content: str
) -> dict[str, Any]:
    """Create a new text/markdown document and kick off ingestion.

    Filename collisions are a typed ``conflict`` (carrying the existing document's
    id) rather than a silent overwrite — update is an explicit, separate act.
    """
    if repository.get_workspace(workspace_id) is None:
        raise WorkspaceError("not_found", f"workspace {workspace_id} not found")
    name = normalize_text_filename(filename)

    existing = repository.get_document_by_filename(workspace_id, name)
    if existing:
        raise WorkspaceError(
            "conflict",
            f"'{name}' already exists in this project — use update_document to change it",
            document_id=existing["id"],
        )

    raw = content.encode("utf-8")
    _check_text_size_and_quota(workspace_id, len(raw))

    sha256, storage_key = storage.store_blob(workspace_id, raw)
    ext = name.rsplit(".", 1)[-1].lower()
    doc = repository.create_document(
        workspace_id=workspace_id,
        filename=name,
        content_type=_TEXT_CONTENT_TYPES.get(ext, "text/plain"),
        size_bytes=len(raw),
        sha256=sha256,
        storage_key=storage_key,
    )
    ingestion.ingest_document_async(doc["id"])
    return doc


def update_text_document(
    *,
    workspace_id: str,
    document_id: str,
    content: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Replace a text document's content (full-content replace) and re-ingest.

    ``expected_sha256`` is an optional optimistic-concurrency check (the hub
    editor sends it; the agent tool doesn't — last-write-wins is fine there).
    """
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        raise WorkspaceError("not_found", f"document {document_id} not found in this project")
    # Only agent-writable text types are editable — images/PDFs are not.
    normalize_text_filename(doc["filename"])

    if expected_sha256 and expected_sha256 != doc["sha256"]:
        raise WorkspaceError(
            "conflict",
            "document changed since it was read (sha256 mismatch)",
            document_id=document_id,
        )

    raw = content.encode("utf-8")
    _check_text_size_and_quota(
        workspace_id, len(raw), replacing_bytes=int(doc["size_bytes"] or 0)
    )

    sha256, storage_key = storage.store_blob(workspace_id, raw)
    if sha256 == doc["sha256"]:
        return doc  # no-op: identical content — skip repoint + re-ingest

    old_key = doc["storage_key"]
    updated = repository.update_document_content(
        document_id, size_bytes=len(raw), sha256=sha256, storage_key=storage_key
    )
    if updated is None:  # row vanished between read and update
        raise WorkspaceError("not_found", f"document {document_id} not found in this project")
    if old_key and old_key != storage_key:
        release_blob_if_unreferenced(workspace_id, old_key)
    ingestion.ingest_document_async(document_id)
    return updated


def rename_document(*, workspace_id: str, document_id: str, new_base: str) -> dict[str, Any]:
    """Rename a document's *base name*, keeping its folder prefix and extension.

    Base-name-only: the file can't change type or move sections, and — since only the
    filename column changes — the doc_id (and any conversation image reference) is
    untouched, with no re-ingest. Collides as a typed ``conflict``.
    """
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        raise WorkspaceError("not_found", f"document {document_id} not found in this project")

    # Base name only — take the last path segment so a typed-in path can't traverse.
    base = (new_base or "").strip().rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    if not base:
        raise WorkspaceError("unsupported", "new name is empty")

    old = doc["filename"]
    slash = old.rfind("/")
    folder = old[: slash + 1] if slash != -1 else ""  # keep trailing slash, or ""
    dot = old.rfind(".")
    ext = old[dot:] if dot > slash else ""  # extension incl. the dot (ignore folder dots)
    # If the user already typed the extension, don't double it.
    if ext and base.lower().endswith(ext.lower()):
        base = base[: -len(ext)]
    new_filename = f"{folder}{base}{ext}"

    if new_filename == old:
        return doc  # no-op
    existing = repository.get_document_by_filename(workspace_id, new_filename)
    if existing and existing["id"] != document_id:
        raise WorkspaceError(
            "conflict",
            f"'{new_filename}' already exists in this project",
            document_id=existing["id"],
        )
    updated = repository.rename_document(document_id, new_filename)
    if updated is None:
        raise WorkspaceError("not_found", f"document {document_id} not found in this project")
    return updated


# Image media the blob store can hold + serve (no text ingestion).
MEDIA_CONTENT_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def store_media(
    *, workspace_id: str, filename: str, content_type: str, raw: bytes
) -> dict[str, Any]:
    """Store a binary image blob + a manifest row, **skipping text ingestion** (images
    aren't parsed/chunked/embedded). The row is ``ready`` immediately. Reuses the same
    size/quota policy as uploads. Raises :class:`WorkspaceError` on a policy failure."""
    settings = get_settings()

    if repository.get_workspace(workspace_id) is None:
        raise WorkspaceError("not_found", f"workspace {workspace_id} not found")
    if content_type not in MEDIA_CONTENT_TYPES:
        raise WorkspaceError(
            "unsupported",
            f"media type not supported (allowed: {', '.join(sorted(MEDIA_CONTENT_TYPES))})",
        )

    size = len(raw)
    if size == 0:
        raise WorkspaceError("unsupported", "image is empty")
    if size > settings.workspace_max_file_bytes:
        raise WorkspaceError(
            "too_large",
            f"image is {size} bytes; per-file limit is {settings.workspace_max_file_bytes}",
        )
    used = repository.workspace_usage_bytes(workspace_id)
    if used + size > settings.workspace_quota_bytes:
        raise WorkspaceError(
            "quota_exceeded",
            f"workspace quota exceeded ({used + size} > {settings.workspace_quota_bytes} bytes)",
        )

    sha256, storage_key = storage.store_blob(workspace_id, raw)
    return repository.create_document(
        workspace_id=workspace_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size,
        sha256=sha256,
        storage_key=storage_key,
        status="ready",  # no ingestion for binary media
    )

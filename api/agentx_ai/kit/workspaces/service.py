"""Upload orchestration: validate → store blob → create manifest row → ingest.

Keeps the Django view thin and the policy (quota, size, type) in one testable place.
Raises :class:`WorkspaceError` with a ``code`` the view maps to an HTTP status.
"""

from __future__ import annotations

from typing import Any

from ..agent_memory.config import get_settings
from . import ingestion, parsing, repository, storage


class WorkspaceError(Exception):
    """A policy/validation failure during upload. ``code`` drives the HTTP status."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # "not_found" | "unsupported" | "too_large" | "quota_exceeded"
        self.message = message


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

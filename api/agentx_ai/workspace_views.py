"""HTTP API for File Workspaces & Document RAG (todo/backlog/workspaces.md).

Endpoints (base ``/api``):
  GET/POST   /workspaces                         list / create
  GET/PATCH/DELETE /workspaces/{id}              detail / rename+fields / delete
  GET/POST   /workspaces/{id}/documents          manifest list / multipart upload
  POST       /workspaces/{id}/documents/text     create text/markdown doc (JSON)
  PUT        /workspaces/{id}/documents/{doc_id}/text  replace text content (JSON)
  GET/DELETE /workspaces/{id}/documents/{doc_id} detail / delete
  GET        /workspaces/{id}/conversations      the project's conversations
  PUT/DELETE /workspaces/{id}/conversations/{conv_id}  link (move) / unlink

Workspaces surface to the user as **Projects** (Projects v1): PATCH also accepts
``description`` and ``instructions`` (instructions ride every turn's preamble).

Bytes live in a content-addressed blob store; manifest in Postgres; vectors in
pgvector. Upload validates type/size/quota then ingests in the background.
"""

from __future__ import annotations

import logging
from typing import Any

from django.views.decorators.csrf import csrf_exempt

from .kit.workspaces import repository, storage
from .kit.workspaces.service import (
    WorkspaceError,
    create_text_document,
    release_blob_if_unreferenced,
    update_text_document,
    upload_document,
)
from .utils.responses import json_error, json_success, parse_json_body, require_methods

logger = logging.getLogger(__name__)

_ERROR_STATUS = {
    "not_found": 404,
    "unsupported": 415,
    "too_large": 413,
    "quota_exceeded": 413,
    "conflict": 409,
}


def _workspace_error_response(e: WorkspaceError):
    extra: dict[str, Any] = {}
    if e.code == "conflict":
        extra["code"] = "conflict"
        if e.document_id:
            extra["document_id"] = e.document_id
            doc = repository.get_document(e.document_id)
            if doc:
                extra["current_sha256"] = doc.get("sha256")
    return json_error(e.message, status=_ERROR_STATUS.get(e.code, 400), **extra)


def _iso(value: Any) -> Any:
    """Render datetimes as ISO strings; pass other values through."""
    return value.isoformat() if hasattr(value, "isoformat") else value


def _serialize_workspace(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ws["id"],
        "name": ws["name"],
        "user_id": ws.get("user_id", "default"),
        "description": ws.get("description") or "",
        "instructions": ws.get("instructions") or "",
        "allow_shell": bool(ws.get("allow_shell", False)),
        "shell_backend": ws.get("shell_backend", "bubblewrap"),
        "document_count": int(ws.get("document_count", 0) or 0),
        "used_bytes": int(ws.get("used_bytes", 0) or 0),
        "created_at": _iso(ws.get("created_at")),
        "updated_at": _iso(ws.get("updated_at")),
    }


def _serialize_document(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "workspace_id": doc["workspace_id"],
        "filename": doc["filename"],
        "content_type": doc.get("content_type"),
        "size_bytes": int(doc.get("size_bytes", 0) or 0),
        "sha256": doc.get("sha256"),
        "tags": list(doc.get("tags") or []),
        "summary": doc.get("summary") or "",
        "status": doc.get("status"),
        "error": doc.get("error"),
        "created_at": _iso(doc.get("created_at")),
        "updated_at": _iso(doc.get("updated_at")),
    }


@csrf_exempt
@require_methods("GET", "POST")
def workspaces(request):
    """List workspaces, or create one."""
    if request.method == "GET":
        items = repository.list_workspaces()
        return json_success({"workspaces": [_serialize_workspace(w) for w in items]})

    data, err = parse_json_body(request)
    if err:
        return err
    name = (data.get("name") or "").strip()
    if not name:
        return json_error("Missing required field: name", status=400)
    ws = repository.create_workspace(name=name)
    return json_success({"workspace": _serialize_workspace(ws)}, status=201)


@csrf_exempt
@require_methods("GET", "PATCH", "DELETE")
def workspace_detail(request, workspace_id: str):
    """Get, rename, or delete a workspace."""
    if request.method == "GET":
        ws = repository.get_workspace(workspace_id)
        if not ws:
            return json_error("Workspace not found", status=404)
        return json_success({"workspace": _serialize_workspace(ws)})

    if request.method == "PATCH":
        data, err = parse_json_body(request)
        if err:
            return err
        ws = repository.get_workspace(workspace_id)
        if not ws:
            return json_error("Workspace not found", status=404)
        if "name" in data:
            name = (data.get("name") or "").strip()
            if not name:
                return json_error("name cannot be empty", status=400)
            ws = repository.rename_workspace(workspace_id, name)
        if "description" in data:
            description = str(data.get("description") or "")
            if len(description) > repository.DESCRIPTION_MAX_CHARS:
                return json_error(
                    f"description exceeds {repository.DESCRIPTION_MAX_CHARS} characters",
                    status=400,
                )
            ws = repository.set_description(workspace_id, description)
        if "instructions" in data:
            instructions = str(data.get("instructions") or "")
            if len(instructions) > repository.INSTRUCTIONS_MAX_CHARS:
                return json_error(
                    f"instructions exceed {repository.INSTRUCTIONS_MAX_CHARS} characters",
                    status=400,
                )
            ws = repository.set_instructions(workspace_id, instructions)
        if "allow_shell" in data:
            ws = repository.set_allow_shell(workspace_id, bool(data.get("allow_shell")))
        if "shell_backend" in data:
            try:
                ws = repository.set_shell_backend(workspace_id, str(data.get("shell_backend")))
            except ValueError as e:
                return json_error(str(e), status=400)
            if data.get("shell_backend") == "container":
                _prepull_shell_image()  # warm the base image so first use doesn't block
        if not ws:
            return json_error("Workspace not found", status=404)
        return json_success({"workspace": _serialize_workspace(ws)})

    # DELETE — collect blob keys first (bytes live off-DB), then cascade the rows.
    keys = [
        full["storage_key"]
        for d in repository.list_documents(workspace_id)
        if (full := repository.get_document(d["id"])) and full.get("storage_key")
    ]
    if not repository.delete_workspace(workspace_id):
        return json_error("Workspace not found", status=404)
    for key in keys:
        storage.delete_blob(key)
    _remove_shell_container(workspace_id)  # best-effort: drop any per-workspace container + volume
    return json_success({"status": "deleted", "workspace_id": workspace_id})


def _remove_shell_container(workspace_id: str) -> None:
    """Best-effort removal of a workspace's shell container (no-op if Docker is absent)."""
    try:
        from .kit.shell import container as sc
        if sc.docker_available():
            sc.remove(workspace_id)
    except Exception as e:  # pragma: no cover - best-effort cleanup
        logger.debug("shell container cleanup skipped for %s: %s", workspace_id, e)


@csrf_exempt
@require_methods("GET", "POST")
def workspace_documents(request, workspace_id: str):
    """List the manifest, or upload a document (multipart)."""
    if request.method == "GET":
        if repository.get_workspace(workspace_id) is None:
            return json_error("Workspace not found", status=404)
        docs = repository.list_documents(workspace_id)
        return json_success({"documents": [_serialize_document(d) for d in docs]})

    # POST — multipart upload. Accept the first uploaded file under "file".
    upload = request.FILES.get("file") or next(iter(request.FILES.values()), None)
    if upload is None:
        return json_error("No file provided (multipart field 'file')", status=400)
    try:
        raw = upload.read()
        doc = upload_document(
            workspace_id=workspace_id,
            filename=upload.name,
            content_type=getattr(upload, "content_type", "") or "",
            raw=raw,
        )
    except WorkspaceError as e:
        return _workspace_error_response(e)
    return json_success({"document": _serialize_document(doc)}, status=201)


@csrf_exempt
@require_methods("POST")
def workspace_documents_text(request, workspace_id: str):
    """Create a text/markdown document from JSON (the hub's "New document" and the
    agent's ``create_document`` REST twin). Filename collision → 409 with the
    existing ``document_id``."""
    data, err = parse_json_body(request)
    if err:
        return err
    filename = str(data.get("filename") or "")
    content = data.get("content")
    if not isinstance(content, str):
        return json_error("Missing required field: content (string)", status=400)
    try:
        doc = create_text_document(
            workspace_id=workspace_id, filename=filename, content=content
        )
    except WorkspaceError as e:
        return _workspace_error_response(e)
    return json_success({"document": _serialize_document(doc)}, status=201)


@csrf_exempt
@require_methods("PUT")
def workspace_document_text(request, workspace_id: str, document_id: str):
    """Replace a text document's content. Optional ``expected_sha256`` is an
    optimistic-concurrency check → 409 with ``current_sha256`` on mismatch."""
    data, err = parse_json_body(request)
    if err:
        return err
    content = data.get("content")
    if not isinstance(content, str):
        return json_error("Missing required field: content (string)", status=400)
    try:
        doc = update_text_document(
            workspace_id=workspace_id,
            document_id=document_id,
            content=content,
            expected_sha256=str(data.get("expected_sha256") or "") or None,
        )
    except WorkspaceError as e:
        return _workspace_error_response(e)
    return json_success({"document": _serialize_document(doc)})


@csrf_exempt
@require_methods("GET", "DELETE")
def workspace_document_detail(request, workspace_id: str, document_id: str):
    """Get document metadata, or delete the document + its blob."""
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return json_error("Document not found", status=404)

    if request.method == "GET":
        return json_success({"document": _serialize_document(doc)})

    storage_key = repository.delete_document(document_id)
    if storage_key:
        # sha-dedup: another doc row may still reference the same blob.
        release_blob_if_unreferenced(workspace_id, storage_key)
    return json_success({"status": "deleted", "document_id": document_id})


@csrf_exempt
@require_methods("POST")
def workspace_document_reingest(request, workspace_id: str, document_id: str):
    """Re-run ingestion for a document (typically after a `failed` — e.g. an
    embedding timeout under load). Resets status to pending and re-fires the
    background pipeline; the blob is already stored so no re-upload is needed."""
    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return json_error("Document not found", status=404)
    if doc.get("status") == "pending":
        return json_success({"document": _serialize_document(doc)})  # already queued
    from .kit.workspaces.ingestion import ingest_document_async
    repository.set_document_status(document_id, "pending", None)
    ingest_document_async(document_id)
    refreshed = repository.get_document(document_id) or doc
    return json_success({"document": _serialize_document(refreshed)}, status=202)


@csrf_exempt
@require_methods("GET")
def workspace_document_raw(request, workspace_id: str, document_id: str):
    """Serve a document's raw bytes (the blob) with its content-type. The stable URL for
    stored media — e.g. generated avatars in the Home workspace. The client fetches this
    through the authed API client (then object-URLs it), so it works under auth."""
    from django.http import HttpResponse

    doc = repository.get_document(document_id)
    if not doc or doc["workspace_id"] != workspace_id:
        return json_error("Document not found", status=404)
    raw = storage.read_blob(doc["storage_key"])
    if raw is None:
        return json_error("Blob not found", status=404)
    resp = HttpResponse(raw, content_type=doc.get("content_type") or "application/octet-stream")
    resp["Cache-Control"] = "private, max-age=86400"
    return resp


@csrf_exempt
@require_methods("GET")
def workspace_conversations(request, workspace_id: str):
    """GET /api/workspaces/{id}/conversations — the project's conversation summaries
    (same row shape as /api/conversations, scoped by durable membership)."""
    if repository.get_workspace(workspace_id) is None:
        return json_error("Workspace not found", status=404)
    from typing import cast

    from .kit.agent_memory.connections import PostgresConnection
    from .views import _conversation_summaries

    try:
        pg_conn = cast(Any, PostgresConnection.get_engine().raw_connection())
        try:
            with pg_conn.cursor() as cursor:
                conversations, total = _conversation_summaries(
                    cursor, workspace_id=workspace_id, limit=100
                )
        finally:
            pg_conn.close()
    except Exception as e:
        logger.error("Error listing project conversations: %s", e)
        return json_error(str(e), status=500)
    return json_success({"conversations": conversations, "total": total})


@csrf_exempt
@require_methods("PUT", "DELETE")
def workspace_conversation_detail(request, workspace_id: str, conversation_id: str):
    """PUT = durably link a conversation to this project (upsert; moves an existing
    link), DELETE = unlink. ws_home is not a project and never holds membership."""
    if repository.get_workspace(workspace_id) is None:
        return json_error("Workspace not found", status=404)
    if workspace_id == repository.HOME_WORKSPACE_ID:
        return json_error("The Home workspace is not a project", status=400)

    if request.method == "PUT":
        if not repository.link_conversation(workspace_id, conversation_id):
            return json_error("Invalid conversation id", status=400)
        return json_success({
            "status": "linked",
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
        })

    if not repository.unlink_conversation(workspace_id, conversation_id):
        return json_error("Conversation is not linked to this workspace", status=404)
    return json_success({
        "status": "unlinked",
        "workspace_id": workspace_id,
        "conversation_id": conversation_id,
    })


def _prepull_shell_image() -> None:
    try:
        from .kit.shell import container as sc
        if sc.docker_available():
            sc.pull_image_async()
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("shell image pre-pull skipped: %s", e)


_SHELL_ACTIONS = {"start", "stop", "reset", "remove"}


@csrf_exempt
@require_methods("GET")
def workspace_shell_container(request, workspace_id: str):
    """Status + live stats for a workspace's shell container (UI resource card)."""
    if repository.get_workspace(workspace_id) is None:
        return json_error("Workspace not found", status=404)
    from .kit.shell import container as sc
    if not sc.docker_available():
        return json_success({"container": {"state": "unavailable"}})
    return json_success({"container": sc.status(workspace_id)})


@csrf_exempt
@require_methods("POST")
def workspace_shell_container_action(request, workspace_id: str, action: str):
    """Lifecycle action on a workspace's shell container: start|stop|reset|remove."""
    if action not in _SHELL_ACTIONS:
        return json_error(f"Unknown action: {action}", status=400)
    if repository.get_workspace(workspace_id) is None:
        return json_error("Workspace not found", status=404)
    from .kit.shell import container as sc
    if not sc.docker_available():
        return json_error("Docker is not available", status=503)
    if action == "remove":
        sc.remove(workspace_id)
        return json_success({"container": {"state": "none"}})
    result = {"start": sc.start, "stop": sc.stop, "reset": sc.reset}[action](workspace_id)
    return json_success({"container": result})

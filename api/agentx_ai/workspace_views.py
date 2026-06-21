"""HTTP API for File Workspaces & Document RAG (todo/backlog/workspaces.md).

Endpoints (base ``/api``):
  GET/POST   /workspaces                         list / create
  GET/PATCH/DELETE /workspaces/{id}              detail / rename / delete
  GET/POST   /workspaces/{id}/documents          manifest list / multipart upload
  GET/DELETE /workspaces/{id}/documents/{doc_id} detail / delete

Bytes live in a content-addressed blob store; manifest in Postgres; vectors in
pgvector. Upload validates type/size/quota then ingests in the background.
"""

from __future__ import annotations

import logging
from typing import Any

from django.views.decorators.csrf import csrf_exempt

from .kit.workspaces import repository, storage
from .kit.workspaces.service import WorkspaceError, upload_document
from .utils.responses import json_error, json_success, parse_json_body, require_methods

logger = logging.getLogger(__name__)

_ERROR_STATUS = {
    "not_found": 404,
    "unsupported": 415,
    "too_large": 413,
    "quota_exceeded": 413,
}


def _iso(value: Any) -> Any:
    """Render datetimes as ISO strings; pass other values through."""
    return value.isoformat() if hasattr(value, "isoformat") else value


def _serialize_workspace(ws: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ws["id"],
        "name": ws["name"],
        "user_id": ws.get("user_id", "default"),
        "allow_shell": bool(ws.get("allow_shell", False)),
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
        if "allow_shell" in data:
            ws = repository.set_allow_shell(workspace_id, bool(data.get("allow_shell")))
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
    return json_success({"status": "deleted", "workspace_id": workspace_id})


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
        return json_error(e.message, status=_ERROR_STATUS.get(e.code, 400))
    return json_success({"document": _serialize_document(doc)}, status=201)


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
        storage.delete_blob(storage_key)
    return json_success({"status": "deleted", "document_id": document_id})

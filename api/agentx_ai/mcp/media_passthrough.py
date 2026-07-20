"""MCP media passthrough — surface non-text tool content instead of dropping it.

External MCP tools can return image/audio/resource content blocks. Historically
``ToolResult.text`` flattened text blocks only, so a screenshot tool's image (or
a synthesizer's audio) was invisible to both the model and the user. This module
flattens a tool result's content-block list into the model-facing string while
**storing** image/audio payloads as workspace media — the streaming tool loop
then renders them as exhibits (and persists synthetic ``present_exhibit`` turns
so they survive reload).

Model-facing shape: when a result is text-only the output is byte-identical to
the legacy flatten. When media/links are present the output becomes a JSON dict
(``{"text", "stored_media", "resource_links"}``) — compact served-blob refs, so
base64 never burns model context (the model can ``view_image`` a stored image).

Abuse caps: an external server can't spam the disk — at most
:data:`MAX_STORED_BLOCKS` blobs per result, :data:`MAX_STORED_BYTES` each;
beyond that a block degrades to a short note. Never raises.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Per-tool-result store caps (an external MCP server is untrusted input).
MAX_STORED_BLOCKS = 3
MAX_STORED_BYTES = 10 * 1024 * 1024

# Embedded text resources are inlined up to this many chars (then truncated).
MAX_RESOURCE_TEXT_CHARS = 4000

# content-type → stored-file extension for passthrough blobs.
_MEDIA_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif",
    "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
    "audio/ogg": "ogg", "audio/webm": "webm", "audio/mp4": "m4a", "audio/x-m4a": "m4a",
    "audio/aac": "aac", "audio/flac": "flac",
}


def _slug(s: str, *, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return slug[:max_len].rstrip("-") or "mcp"


def store_media_block(
    data_b64: str, mime_type: str, *, tool_name: str, server_name: str
) -> dict[str, Any] | None:
    """Decode + store one media payload as workspace media. Returns the
    ``{url, media_type, filename}`` stored entry, or ``None`` (logged) on any
    policy/decode failure."""
    try:
        from ..kit.workspaces import repository
        from ..kit.workspaces.service import store_media
        from .internal_context import current_context

        ext = _MEDIA_EXT.get(mime_type)
        if ext is None:
            return None
        # Cheap size gate before decoding (b64 inflates ~4/3).
        if len(data_b64) * 3 // 4 > MAX_STORED_BYTES:
            logger.info(f"MCP media block from {tool_name} exceeds store cap; skipped")
            return None
        raw = base64.b64decode(data_b64, validate=True)
        if not raw or len(raw) > MAX_STORED_BYTES:
            return None

        ctx = current_context()
        user_id = ctx.user_id if ctx else "default"
        ws_id = (ctx.workspace_id if ctx and ctx.workspace_id else None) or (
            repository.ensure_home_workspace(user_id)["id"]
        )
        suffix = datetime.now(UTC).strftime("%H%M%S%f")[-6:]
        filename = f"mcp/{_slug(server_name)}-{_slug(tool_name)}-{suffix}.{ext}"
        doc = store_media(
            workspace_id=ws_id, filename=filename, content_type=mime_type, raw=raw,
        )
        return {
            "url": f"/api/workspaces/{ws_id}/documents/{doc['id']}/raw",
            "media_type": mime_type,
            "filename": filename,
        }
    except Exception as e:  # noqa: BLE001 — passthrough is best-effort
        logger.warning(f"MCP media passthrough store failed ({tool_name}): {e}")
        return None


def flatten_result_content(
    content: list[dict[str, Any]], *, tool_name: str, server_name: str
) -> str:
    """Flatten a tool result's content-block wire list into the model-facing string.

    Text-only results return the legacy ``"\\n"``-joined text unchanged. Results
    carrying media/resource blocks return a JSON dict string with ``stored_media``
    (served-blob refs for stored image/audio) and ``resource_links``.
    """
    texts: list[str] = []
    stored: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    for item in content or []:
        if not isinstance(item, dict):
            continue
        btype = item.get("type")
        if btype == "text":
            texts.append(item.get("text", ""))
        elif btype in ("image", "audio"):
            entry = None
            if len(stored) < MAX_STORED_BLOCKS:
                entry = store_media_block(
                    item.get("data") or "", item.get("mimeType") or "",
                    tool_name=tool_name, server_name=server_name,
                )
            if entry is not None:
                stored.append(entry)
            else:
                texts.append(f"[{btype} content omitted — not stored]")
        elif btype == "resource_link":
            links.append({
                k: item[k]
                for k in ("uri", "name", "description", "mimeType")
                if item.get(k) is not None
            })
        elif btype == "resource":
            res = item.get("resource")
            if isinstance(res, dict):
                text = res.get("text")
                if isinstance(text, str) and text:
                    clipped = text[:MAX_RESOURCE_TEXT_CHARS]
                    suffix = "… [truncated]" if len(text) > MAX_RESOURCE_TEXT_CHARS else ""
                    texts.append(f"[resource {res.get('uri', '')}]\n{clipped}{suffix}")
                elif res.get("blob"):
                    mime = res.get("mimeType") or ""
                    if mime in _MEDIA_EXT and len(stored) < MAX_STORED_BLOCKS:
                        entry = store_media_block(
                            res.get("blob") or "", mime,
                            tool_name=tool_name, server_name=server_name,
                        )
                        if entry is not None:
                            stored.append(entry)
                            continue
                    texts.append(f"[binary resource {res.get('uri', '')} ({mime}) — not inlined]")
            else:
                # Legacy stringified resource shape — keep it visible.
                texts.append(f"[resource] {res}")

    if not stored and not links:
        return "\n".join(texts)
    payload: dict[str, Any] = {"text": "\n".join(texts)}
    if stored:
        payload["stored_media"] = stored
    if links:
        payload["resource_links"] = links
    return json.dumps(payload)

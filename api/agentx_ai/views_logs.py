"""Log API — read-only views over the in-memory ring buffer + on-disk archive.

These power the Tauri client's Log panel (the client is a thin HTTP shell; the
API is a separate process, so logs must travel over HTTP, not stdout capture).

Auth is enforced globally by ``AgentXAuthMiddleware`` when ``AGENTX_AUTH_ENABLED``
— these views only add the ``AGENTX_LOG_API_ENABLED`` kill-switch. Records are
already redacted at capture time (see ``logging_kit.redaction``).

This module's logger is excluded from the ring buffer (see ``_SELF_PREFIXES`` in
``logging_kit.ring_buffer``) so serving the log stream can't feed itself.
"""

from __future__ import annotations

import json
import logging
import queue

from django.http import FileResponse, JsonResponse, StreamingHttpResponse

from .logging_kit.archive import list_segments, resolve_segment
from .logging_kit.categories import serializable_catalog
from .logging_kit.flags import read_flags
from .logging_kit.ring_buffer import get_ring_handler

logger = logging.getLogger(__name__)

_HEARTBEAT_SECONDS = 10.0


def _api_disabled() -> JsonResponse | None:
    if not read_flags().api_enabled:
        return JsonResponse({"error": "log API disabled"}, status=404)
    return None


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def logs_recent(request):
    """GET /api/logs — recent records (filters: level, category, run_id, search, since, limit)."""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    if (disabled := _api_disabled()) is not None:
        return disabled
    ring = get_ring_handler()
    if ring is None:
        return JsonResponse({"logs": [], "available": False})

    try:
        limit = min(max(int(request.GET.get("limit", 500)), 1), 2000)
    except (TypeError, ValueError):
        limit = 500
    since_id = None
    raw_since = request.GET.get("since")
    if raw_since:
        try:
            since_id = int(raw_since)
        except (TypeError, ValueError):
            since_id = None

    logs = ring.snapshot(
        level=request.GET.get("level"),
        category=request.GET.get("category"),
        run_id=request.GET.get("run_id"),
        search=request.GET.get("search"),
        since_id=since_id,
        limit=limit,
    )
    return JsonResponse({"logs": logs, "available": True})


def logs_categories(request):
    """GET /api/logs/categories — the category registry (client color map)."""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    return JsonResponse({"categories": serializable_catalog()})


def logs_stream(request):
    """GET /api/logs/stream — replay the buffer then follow live via SSE."""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    if (disabled := _api_disabled()) is not None:
        return disabled
    ring = get_ring_handler()
    if ring is None:
        return JsonResponse({"error": "log buffer unavailable"}, status=404)

    sub = ring.subscribe()
    if sub is None:
        return JsonResponse({"error": "too many log subscribers"}, status=503)

    def _stream():
        try:
            # Replay current buffer first, remembering the high-water id so the
            # live feed doesn't double-emit what we just replayed.
            replay = ring.snapshot(limit=2000)
            last_id = replay[-1]["id"] if replay else 0
            for entry in replay:
                yield _sse("log", entry)
            while True:
                try:
                    entry = sub.get(timeout=_HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": ping\n\n"  # heartbeat → surfaces client disconnects
                    continue
                if entry["id"] <= last_id:
                    continue
                last_id = entry["id"]
                yield _sse("log", entry)
        except (GeneratorExit, BrokenPipeError):
            pass
        finally:
            ring.unsubscribe(sub)

    response = StreamingHttpResponse(_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def logs_archive_list(request):
    """GET /api/logs/archive — list compressed archive segments."""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    if (disabled := _api_disabled()) is not None:
        return disabled
    return JsonResponse({"segments": list_segments()})


def logs_archive_download(request, name: str):
    """GET /api/logs/archive/<name> — download a single archive segment."""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    if (disabled := _api_disabled()) is not None:
        return disabled
    path = resolve_segment(name)
    if path is None:
        return JsonResponse({"error": "segment not found"}, status=404)
    content_type = "application/gzip" if path.suffix == ".gz" else "text/plain"
    return FileResponse(open(path, "rb"), as_attachment=True, filename=path.name, content_type=content_type)

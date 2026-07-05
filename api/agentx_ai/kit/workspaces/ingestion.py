"""Document ingestion: parse → chunk → embed → store, then auto-tag + summarize.

Runs off the request path (a daemon thread fired from the upload view). A document
moves ``pending`` → ``ready`` (or ``failed`` with a reason). Enrichment (LLM tags +
summary) is best-effort: if it fails the document still goes ``ready`` — the chunks
and embeddings (what retrieval needs) are already stored.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from ...agent.tool_output_chunker import chunk_text
from ...utils.async_bridge import run_coro_sync
from ..agent_memory.config import get_settings
from ..agent_memory.embeddings import get_embedder
from . import parsing, repository, storage

logger = logging.getLogger(__name__)

_DEFAULT_SUMMARY_MODEL = "anthropic:claude-haiku-4-5-20251001"
_ENRICH_INPUT_CHARS = 6000  # cap text sent to the summarizer

# Embedding a large document goes through the shared embedding queue in bounded
# slices: one giant submit would monopolize the worker (starving chat recall) and
# blow the default 30s request timeout (sized for a recall query, not an
# 1800-chunk PDF). Each slice gets a generous background-work timeout instead —
# it must also absorb *queue wait* behind other ingests and chat traffic.
_EMBED_SLICE_SIZE = 32
_EMBED_SLICE_TIMEOUT_S = 300.0


def ingest_document(document_id: str) -> dict[str, Any]:
    """Run the full ingestion pipeline for one document (synchronous)."""
    doc = repository.get_document(document_id)
    if not doc:
        return {"status": "failed", "error": "document not found"}

    try:
        raw = storage.read_blob(doc["storage_key"])
        if raw is None:
            raise FileNotFoundError(f"blob missing for {doc['storage_key']}")

        text = parsing.parse_to_text(raw, doc["filename"])
        settings = get_settings()
        chunks = chunk_text(
            text,
            chunk_size=settings.workspace_chunk_size,
            overlap=settings.workspace_chunk_overlap,
        )
        if not chunks:
            repository.set_document_status(document_id, "failed", "no extractable text")
            return {"status": "failed", "error": "no extractable text"}

        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_SLICE_SIZE):
            vectors.extend(embedder.embed(
                texts[start:start + _EMBED_SLICE_SIZE], timeout=_EMBED_SLICE_TIMEOUT_S,
            ))
        rows = [(c["index"], c["text"], vec) for c, vec in zip(chunks, vectors, strict=True)]
        repository.replace_chunks(document_id, doc["workspace_id"], rows)

        tags, summary = _enrich(doc["filename"], text)
        repository.set_document_enrichment(
            document_id, tags=tags, summary=summary, status="ready"
        )
        logger.info(
            "📄 WORKSPACE ingested %s (%d chunks) ready", doc["filename"], len(rows)
        )
        return {"status": "ready", "chunks": len(rows), "tags": tags}

    except parsing.UnsupportedDocumentError as e:
        repository.set_document_status(document_id, "failed", str(e))
        return {"status": "failed", "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("workspace ingestion failed for %s", document_id)
        # A bare TimeoutError stringifies to "" — always leave a readable reason
        # on the manifest row (it surfaces in the Projects hub).
        reason = str(e) or type(e).__name__
        if isinstance(e, TimeoutError):
            reason = "embedding timed out — the server may be busy; retry ingestion"
        repository.set_document_status(document_id, "failed", reason)
        return {"status": "failed", "error": reason}


def ingest_document_async(document_id: str) -> None:
    """Fire-and-forget ingestion on a daemon thread (called from the upload view)."""
    threading.Thread(
        target=ingest_document, args=(document_id,), daemon=True,
        name=f"ingest-{document_id}",
    ).start()


def ingest_pending_documents(limit: int = 25) -> dict[str, Any]:
    """Sweep documents stuck in ``pending`` (e.g. after a restart) and ingest them."""
    ids = repository.pending_document_ids(limit=limit)
    results = [ingest_document(doc_id) for doc_id in ids]
    return {
        "processed": len(results),
        "ready": sum(1 for r in results if r.get("status") == "ready"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
    }


# --- Enrichment (best-effort) ----------------------------------------------

def _enrich(filename: str, text: str) -> tuple[list[str], str]:
    """LLM auto-tag + summary. Degrades to ([], heuristic snippet) on any failure."""
    snippet = text[:_ENRICH_INPUT_CHARS]
    try:
        from ...providers.base import Message, MessageRole
        from ...providers.registry import get_registry

        settings = get_settings()
        model = settings.workspace_summary_model or _DEFAULT_SUMMARY_MODEL
        provider, model_id, _ = get_registry().resolve_with_fallback(model)

        prompt = (
            "You are cataloging a document for a searchable knowledge base. "
            "Return STRICT JSON: {\"tags\": [up to 6 short lowercase topic tags], "
            "\"summary\": \"1-2 sentence description of what this document is and covers\"}.\n\n"
            f"Filename: {filename}\n\nContent (truncated):\n{snippet}"
        )
        result = run_coro_sync(
            provider.complete(
                [Message(role=MessageRole.USER, content=prompt)],
                model_id, temperature=0.2, max_tokens=300,
            ),
            timeout=60.0,
        )
        tags, summary = _parse_enrichment(result.content)
        if summary:
            return tags, summary
    except Exception as e:
        logger.debug("workspace enrichment failed for %s, using fallback: %s", filename, e)
    # Fallback: no tags, a plain snippet so the manifest still says something.
    return [], _fallback_summary(snippet)


def _parse_enrichment(content: str) -> tuple[list[str], str]:
    raw = (content or "").strip()
    if raw.startswith("```"):  # strip markdown code fences if the model added them
        raw = raw.split("```", 2)[1].lstrip("json").strip() if "```" in raw[3:] else raw.strip("`")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return [], ""
    tags = [str(t).strip().lower() for t in (data.get("tags") or []) if str(t).strip()][:6]
    summary = str(data.get("summary") or "").strip()
    return tags, summary


def _fallback_summary(snippet: str) -> str:
    flat = " ".join(snippet.split())
    return (flat[:200] + "…") if len(flat) > 200 else flat

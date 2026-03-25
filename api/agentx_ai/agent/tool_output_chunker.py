"""
Tool Output Chunker — utilities for intent-aware retrieval of stored tool outputs.

Provides chunking, section detection, JSON path resolution, and semantic search
over stored tool outputs. Used by the Phase 14.3 internal MCP tools.
"""

import json
import logging
import math
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Markdown heading pattern
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# Separator patterns
_SEPARATOR_RE = re.compile(r'^(-{3,}|={3,})$', re.MULTILINE)


def chunk_text(
    content: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    """
    Split text into chunks with positional metadata.

    Tries structural splitting (markdown headings) first, falls back
    to fixed-size chunking with overlap.

    Returns:
        List of {"text": str, "start": int, "end": int, "index": int}
    """
    if not content:
        return []

    # Try structural split by markdown headings
    headings = list(_HEADING_RE.finditer(content))
    if len(headings) >= 2:
        chunks = []
        for i, match in enumerate(headings):
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            text = content[start:end].strip()
            if text:
                chunks.append({
                    "text": text,
                    "start": start,
                    "end": end,
                    "index": i,
                })
        # If there's content before the first heading, include it
        if headings[0].start() > 0:
            pre = content[:headings[0].start()].strip()
            if pre:
                chunks.insert(0, {
                    "text": pre,
                    "start": 0,
                    "end": headings[0].start(),
                    "index": 0,
                })
                # Re-index
                for j, c in enumerate(chunks):
                    c["index"] = j
        return chunks

    # Fall back to fixed-size chunking with overlap
    chunks = []
    pos = 0
    idx = 0
    while pos < len(content):
        end = min(pos + chunk_size, len(content))
        text = content[pos:end]
        if text.strip():
            chunks.append({
                "text": text,
                "start": pos,
                "end": end,
                "index": idx,
            })
            idx += 1
        pos += chunk_size - overlap
        if pos >= len(content):
            break

    return chunks


def detect_sections(content: str) -> list[dict[str, Any]]:
    """
    Detect named sections in content.

    Priority: (1) JSON top-level keys, (2) markdown headings,
    (3) separator-delimited blocks, (4) blank-line paragraphs.

    Returns:
        List of {"name": str, "start": int, "end": int, "level": int}
    """
    if not content.strip():
        return []

    # (1) Try JSON: if entire content is valid JSON, use top-level keys
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed:
            sections = []
            # Re-serialize each key to find approximate positions
            for key in parsed:
                value_str = json.dumps(parsed[key], indent=2, default=str)
                sections.append({
                    "name": key,
                    "start": 0,
                    "end": len(value_str),
                    "level": 1,
                    "size": len(value_str),
                })
            return sections
    except (json.JSONDecodeError, TypeError):
        pass

    # (2) Markdown headings
    headings = list(_HEADING_RE.finditer(content))
    if headings:
        sections = []
        for i, match in enumerate(headings):
            level = len(match.group(1))
            name = match.group(2).strip()
            start = match.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            sections.append({
                "name": name,
                "start": start,
                "end": end,
                "level": level,
                "size": end - start,
            })
        return sections

    # (3) Separator-delimited blocks
    separators = list(_SEPARATOR_RE.finditer(content))
    if len(separators) >= 2:
        sections = []
        boundaries = [0] + [m.end() for m in separators] + [len(content)]
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            block = content[start:end].strip()
            if block:
                # Use first line as name
                first_line = block.split('\n')[0].strip()[:80]
                sections.append({
                    "name": first_line or f"Block {i + 1}",
                    "start": start,
                    "end": end,
                    "level": 1,
                    "size": end - start,
                })
        return sections

    # (4) Blank-line paragraphs (max 30 to avoid noise)
    paragraphs = re.split(r'\n\s*\n', content)
    if 2 <= len(paragraphs) <= 30:
        sections = []
        pos = 0
        for i, para in enumerate(paragraphs):
            para_stripped = para.strip()
            if not para_stripped:
                pos += len(para) + 1
                continue
            start = content.find(para_stripped, pos)
            if start == -1:
                start = pos
            end = start + len(para_stripped)
            first_line = para_stripped.split('\n')[0].strip()[:80]
            sections.append({
                "name": first_line or f"Paragraph {i + 1}",
                "start": start,
                "end": end,
                "level": 1,
                "size": end - start,
            })
            pos = end
        return sections

    return []


def get_section_content(
    content: str,
    section_name: str,
    limit: int = 5000,
) -> Optional[dict[str, Any]]:
    """
    Extract content for a named section (case-insensitive fuzzy match).

    Returns:
        {"name": str, "content": str, "start": int, "end": int} or None
    """
    sections = detect_sections(content)
    if not sections:
        return None

    query = section_name.lower()

    # Try exact match first, then substring match
    for section in sections:
        if section["name"].lower() == query:
            text = content[section["start"]:section["end"]][:limit]
            return {
                "name": section["name"],
                "content": text,
                "start": section["start"],
                "end": section["end"],
            }

    for section in sections:
        if query in section["name"].lower() or section["name"].lower() in query:
            text = content[section["start"]:section["end"]][:limit]
            return {
                "name": section["name"],
                "content": text,
                "start": section["start"],
                "end": section["end"],
            }

    return None


def resolve_json_path(content: str, path: str) -> dict[str, Any]:
    """
    Resolve a dot-notation JSON path against content.

    Supports: "key", "key.nested", "items[0]", "items[*].name"

    Returns:
        {"value": Any, "type": str, "path": str} or {"error": str}
    """
    # Parse JSON from content
    data = None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    if data is None:
        # Try extracting JSON from markdown code blocks
        from .output_parser import validate_json_output
        is_valid, parsed, _ = validate_json_output(content)
        if is_valid and parsed is not None:
            data = parsed

    if data is None:
        return {"error": "Content is not valid JSON", "success": False}

    # Parse and resolve path
    try:
        result = _walk_path(data, path)
    except (KeyError, IndexError, TypeError) as e:
        return {"error": f"Path resolution failed: {e}", "path": path, "success": False}

    # Serialize result, truncate if needed
    try:
        serialized = json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        serialized = str(result)

    truncated = False
    if len(serialized) > 10000:
        serialized = serialized[:10000] + "\n...[truncated]"
        truncated = True

    return {
        "value": serialized,
        "type": type(result).__name__,
        "path": path,
        "truncated": truncated,
        "success": True,
    }


def _walk_path(data: Any, path: str) -> Any:
    """Walk a dot-notation path with array indexing through data."""
    if not path:
        return data

    # Split path into segments, respecting brackets
    segments = _parse_path_segments(path)
    current = data

    for segment in segments:
        if segment == "*":
            # Wildcard — must be inside an array context, handled by caller
            raise TypeError("Bare wildcard '*' not supported; use 'key[*]'")

        # Check for array index: "key[N]" or "key[*]"
        bracket_match = re.match(r'^([^[]*)\[(\d+|\*)\]$', segment)
        if bracket_match:
            key = bracket_match.group(1)
            index_str = bracket_match.group(2)

            if key:
                if isinstance(current, dict):
                    current = current[key]
                else:
                    raise TypeError(f"Cannot access key '{key}' on {type(current).__name__}")

            if index_str == "*":
                # Wildcard: map remaining path over array elements
                if not isinstance(current, list):
                    raise TypeError(f"Wildcard [*] requires a list, got {type(current).__name__}")
                return current  # Return the full list for wildcard at end
            else:
                idx = int(index_str)
                if isinstance(current, list):
                    current = current[idx]
                else:
                    raise TypeError(f"Cannot index {type(current).__name__} with [{idx}]")
        else:
            # Simple key access
            if isinstance(current, dict):
                current = current[segment]
            elif isinstance(current, list):
                # Try integer index
                try:
                    current = current[int(segment)]
                except ValueError:
                    raise KeyError(f"Cannot access '{segment}' on a list")
            else:
                raise TypeError(f"Cannot access '{segment}' on {type(current).__name__}")

    return current


def _parse_path_segments(path: str) -> list[str]:
    """Parse a dot-notation path into segments, preserving bracket notation."""
    segments = []
    current = ""

    i = 0
    while i < len(path):
        ch = path[i]
        if ch == '.':
            if current:
                segments.append(current)
                current = ""
        elif ch == '[':
            # Find closing bracket
            j = path.find(']', i)
            if j == -1:
                current += path[i:]
                break
            current += path[i:j + 1]
            i = j
        else:
            current += ch
        i += 1

    if current:
        segments.append(current)

    return segments


def semantic_search_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Semantic search over chunks using embeddings.

    Falls back to keyword matching if embedding provider is unavailable.

    Returns:
        Top-k chunks augmented with "score" field, sorted descending.
    """
    if not chunks or not query:
        return []

    try:
        from ..kit.agent_memory.embeddings import get_embedder
        embedder = get_embedder()

        texts = [c["text"] for c in chunks]
        chunk_embeddings = embedder.embed(texts)
        query_embedding = embedder.embed_single(query)

        # Cosine similarity
        scored = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            score = _cosine_similarity(query_embedding, chunk_emb)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            result = dict(chunks[idx])
            result["score"] = round(score, 4)
            results.append(result)
        return results

    except Exception as e:
        logger.warning(f"Embedding-based search failed, falling back to keyword matching: {e}")
        return _keyword_search(chunks, query, top_k)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_search(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fallback: rank chunks by keyword overlap with query."""
    query_words = set(query.lower().split())
    if not query_words:
        return chunks[:top_k]

    scored = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk["text"].lower()
        hits = sum(1 for w in query_words if w in chunk_lower)
        score = hits / len(query_words) if query_words else 0.0
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        result = dict(chunks[idx])
        result["score"] = round(score, 4)
        results.append(result)
    return results

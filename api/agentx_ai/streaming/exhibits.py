"""
Exhibits — typed, declarative agent-authored content presented to the user.

An **Exhibit** is one presented unit the agent builds and arranges declaratively
by calling the ``present_exhibit`` internal tool (see
:mod:`agentx_ai.mcp.internal_tools`). It is a constrained tree of typed
**Elements** (see :data:`ALLOWED_ELEMENT_TYPES`) arranged by a ``layout``
(``stack`` or ``grid``). A conversation accumulates exhibits into a **Gallery**.

The agent declares the *desired state* (with a stable ``id``); re-declaring the
same ``id`` across turns is an amend (replace-in-place). This module owns the
schema + validation so both the producer tool and the streaming layer agree on
one definition. The element-type allow-list (:data:`ALLOWED_ELEMENT_TYPES`) is
the security boundary — the agent can only emit registered, typed elements,
never raw HTML.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

# Bump only on a breaking change to the Exhibit wire shape (honors the v0.20
# "migratable across platforms" rule — every exhibit carries this). New element
# types and layouts are ADDITIVE (old clients fall back to source-as-code /
# stack), so audio/video/text + grid did not bump it.
EXHIBIT_SCHEMA_VERSION = 1

# The allow-list. Adding an element type = add it here + a client renderer.
ALLOWED_ELEMENT_TYPES: frozenset[str] = frozenset(
    {"mermaid", "choice", "table", "citation", "image", "audio", "video", "text"}
)

# Media element `url`s must be served-blob paths — the client fetches them through
# the authed API client. An agent-authored exhibit can never point at an external
# or arbitrary URL (exfiltration/markup-injection guard).
_SERVED_BLOB_RE = re.compile(r"^/api/workspaces/[A-Za-z0-9_-]+/documents/[A-Za-z0-9_-]+/raw$")


def _validate_served_blob_url(url: str) -> str:
    s = (url or "").strip()
    if not _SERVED_BLOB_RE.match(s):
        raise ValueError(
            "media element url must be a served-blob path "
            "(/api/workspaces/{ws}/documents/{doc}/raw)"
        )
    return s

# Upper bound on choice options — keep the rendered button set usable.
MAX_CHOICE_OPTIONS = 10
# Wide tables render terribly in a chat column — cap columns hard.
MAX_TABLE_COLUMNS = 12
# Citations can be many (record-keeping), but keep a sane ceiling.
MAX_CITATION_SOURCES = 50

# Recognized mermaid diagram-type keywords (the first non-blank token of a
# diagram). Used for a cheap server-side sanity check — full syntax validation
# happens client-side at render time (no JS mermaid engine in Python).
MERMAID_DIAGRAM_KEYWORDS: frozenset[str] = frozenset({
    "graph", "flowchart", "sequenceDiagram", "classDiagram",
    "stateDiagram", "stateDiagram-v2", "erDiagram", "gantt", "pie",
    "mindmap", "journey", "gitGraph", "timeline", "quadrantChart",
    "requirementDiagram", "C4Context", "sankey", "xychart-beta", "block-beta",
})


class MermaidElement(BaseModel):
    """A mermaid diagram element (rendered client-side to SVG)."""

    type: Literal["mermaid"]
    content: str
    title: str | None = None


class ChoiceElement(BaseModel):
    """An interactive choice — the user picks an option, which is fed back to
    the agent as their next message (next-turn round-trip)."""

    type: Literal["choice"]
    prompt: str | None = None
    options: list[str] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)
    title: str | None = None

    @field_validator("options")
    @classmethod
    def _clean_options(cls, v: list[str]) -> list[str]:
        """Strip, drop blanks, de-dupe (order-preserving); require ≥1 remains."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for opt in v:
            s = (opt or "").strip()
            if s and s not in seen:
                seen.add(s)
                cleaned.append(s)
        if not cleaned:
            raise ValueError("choice element needs at least one non-blank option")
        return cleaned


class TableElement(BaseModel):
    """A structured table (sortable / scrollable / responsive client-side)."""

    type: Literal["table"]
    columns: list[str] = Field(min_length=1, max_length=MAX_TABLE_COLUMNS)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None
    title: str | None = None

    @field_validator("rows", mode="before")
    @classmethod
    def _normalize_rows(cls, rows: Any, info: Any) -> list[list[str]]:
        """Stringify cells (None → ""), pad/truncate each row to the column count.

        Runs **before** pydantic's `str` coercion so numeric/None cells (which a
        model commonly emits, e.g. `[["opus", 0.4, 1200]]`) don't fail validation.
        """
        ncols = len(info.data.get("columns") or [])
        if ncols == 0 or not isinstance(rows, (list, tuple)):
            return []
        out: list[list[str]] = []
        for row in rows:
            cells = list(row) if isinstance(row, (list, tuple)) else [row]
            norm = ["" if c is None else str(c) for c in cells[:ncols]]
            norm += [""] * (ncols - len(norm))
            out.append(norm)
        return out


class CitationSource(BaseModel):
    """One cited source. `active` = a working reference (carries a `quote`, folds
    out); `passive` = record-keeping (archived). Default passive."""

    label: str
    url: str | None = None
    quote: str | None = None
    kind: Literal["active", "passive"] = "passive"
    source_type: Literal["web", "memory", "doc"] | None = None

    @field_validator("label")
    @classmethod
    def _label_non_blank(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("citation source needs a non-blank label")
        return s


class CitationElement(BaseModel):
    """A set of cited sources presented to the user."""

    type: Literal["citation"]
    sources: list[CitationSource] = Field(min_length=1, max_length=MAX_CITATION_SOURCES)
    title: str | None = None


class ImageElement(BaseModel):
    """A generated/stored image, displayed inline. ``url`` is a served-blob path
    (``/api/workspaces/{ws}/documents/{doc}/raw``); the client fetches it through the
    authed API client and object-URLs it (a raw <img src> can't carry auth)."""

    type: Literal["image"]
    url: str
    alt: str | None = None
    title: str | None = None

    @field_validator("url")
    @classmethod
    def _url_is_served_blob(cls, v: str) -> str:
        return _validate_served_blob_url(v)


class AudioElement(BaseModel):
    """A generated/stored audio clip, rendered as an inline player. Same
    served-blob ``url`` contract (and validator) as :class:`ImageElement`."""

    type: Literal["audio"]
    url: str
    caption: str | None = None
    title: str | None = None

    @field_validator("url")
    @classmethod
    def _url_is_served_blob(cls, v: str) -> str:
        return _validate_served_blob_url(v)


class VideoElement(BaseModel):
    """A stored video, rendered as an inline ``<video>`` player (render-only —
    video never enters model context). Same served-blob ``url`` contract."""

    type: Literal["video"]
    url: str
    caption: str | None = None
    title: str | None = None

    @field_validator("url")
    @classmethod
    def _url_is_served_blob(cls, v: str) -> str:
        return _validate_served_blob_url(v)


class TextElement(BaseModel):
    """A markdown text passage rendered through the client's chat markdown
    pipeline (same sanitization — never raw HTML)."""

    type: Literal["text"]
    content: str
    title: str | None = None

    @field_validator("content")
    @classmethod
    def _content_non_blank(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("text element content is empty")
        return v


# Element union — discriminated on `type`; the discriminator enforces the
# allow-list (an unknown type raises). Widen as new element types ship.
Element = Annotated[
    MermaidElement
    | ChoiceElement
    | TableElement
    | CitationElement
    | ImageElement
    | AudioElement
    | VideoElement
    | TextElement,
    Field(discriminator="type"),
]


class Exhibit(BaseModel):
    """A declarative, agent-authored presentation unit."""

    schema_version: int = EXHIBIT_SCHEMA_VERSION
    id: str
    title: str | None = None
    # `grid` flows elements into responsive columns (client-side; degrades to
    # stack on narrow viewports and on clients that predate it).
    layout: Literal["stack", "grid"] = "stack"
    elements: list[Element] = Field(min_length=1)


def mermaid_sanity_error(content: str) -> str | None:
    """Return a human-readable error if mermaid source looks invalid, else None.

    Cheap structural gate only (non-empty + a recognized diagram-type keyword);
    real validation is the client render + its error fallback.
    """
    if not content or not content.strip():
        return "mermaid element content is empty"
    first = content.strip().split(None, 1)[0].rstrip(";")
    if first not in MERMAID_DIAGRAM_KEYWORDS:
        sample = ", ".join(sorted(MERMAID_DIAGRAM_KEYWORDS)[:6])
        return (
            f"unrecognized mermaid diagram type {first!r}; the diagram must "
            f"start with a mermaid keyword (e.g. {sample}, ...)"
        )
    return None


def citation_exhibit_from_web_search(
    results: Any, *, exhibit_id: str
) -> Exhibit | None:
    """Build a passive ``citation`` exhibit from ``web_search`` results.

    Each result becomes a ``passive`` web :class:`CitationSource` (record-keeping,
    never re-enters model context); sources are de-duped by URL and capped at
    :data:`MAX_CITATION_SOURCES`. Returns ``None`` when there's nothing to show.

    Shared by the live auto-capture (streaming tool loop) and — mirrored in the
    client — the history-restore path, so a searched-then-cited turn looks the
    same live and on reload. The dedupe is the seed of the future Bibliography.
    """
    if not isinstance(results, (list, tuple)):
        return None
    sources: list[CitationSource] = []
    seen: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        label = title or url
        if not label:
            continue
        key = url or label
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            CitationSource(label=label, url=url or None, source_type="web", kind="passive")
        )
        if len(sources) >= MAX_CITATION_SOURCES:
            break
    if not sources:
        return None
    return Exhibit(
        schema_version=EXHIBIT_SCHEMA_VERSION,
        id=exhibit_id,
        layout="stack",
        elements=[CitationElement(type="citation", sources=sources)],
    )


def citation_exhibit_from_document_query(
    results: Any, *, exhibit_id: str
) -> Exhibit | None:
    """Build a passive ``doc`` citation exhibit from ``document_query`` results.

    Each hit (``{document_id, filename, text, score}``) becomes a passive
    :class:`CitationSource` with ``source_type="doc"`` — deduped by document, a
    short quote from the matched chunk. Returns ``None`` when there's nothing to show.
    Mirrors :func:`citation_exhibit_from_web_search` for the workspace corpus.
    """
    if not isinstance(results, (list, tuple)):
        return None
    sources: list[CitationSource] = []
    seen: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        doc_id = (r.get("document_id") or "").strip()
        label = (r.get("filename") or doc_id or "").strip()
        if not label or doc_id in seen:
            continue
        seen.add(doc_id)
        quote = " ".join((r.get("text") or "").split())[:240] or None
        sources.append(
            CitationSource(
                label=label,
                url=f"/workspaces/documents/{doc_id}" if doc_id else None,
                quote=quote,
                source_type="doc",
                kind="passive",
            )
        )
        if len(sources) >= MAX_CITATION_SOURCES:
            break
    if not sources:
        return None
    return Exhibit(
        schema_version=EXHIBIT_SCHEMA_VERSION,
        id=exhibit_id,
        layout="stack",
        elements=[CitationElement(type="citation", sources=sources)],
    )


def image_exhibit_from_generate(url: str, *, exhibit_id: str, alt: str | None = None) -> Exhibit | None:
    """Build an ``image`` exhibit from a ``generate_image`` tool result. ``url`` is the
    served-blob path; ``alt`` is the generating prompt. Returns ``None`` when there's no url."""
    if not url:
        return None
    return Exhibit(
        schema_version=EXHIBIT_SCHEMA_VERSION,
        id=exhibit_id,
        layout="stack",
        elements=[ImageElement(type="image", url=url, alt=alt)],
    )


def audio_exhibit_from_generate(
    url: str, *, exhibit_id: str, caption: str | None = None
) -> Exhibit | None:
    """Build an ``audio`` exhibit from a stored clip (``generate_speech`` result or
    MCP audio passthrough). Mirrors :func:`image_exhibit_from_generate`."""
    if not url:
        return None
    return Exhibit(
        schema_version=EXHIBIT_SCHEMA_VERSION,
        id=exhibit_id,
        layout="stack",
        elements=[AudioElement(type="audio", url=url, caption=caption)],
    )


def media_exhibit_from_stored(
    stored: dict[str, Any], *, exhibit_id: str
) -> Exhibit | None:
    """Build an image/audio exhibit from one stored-media passthrough entry
    (``{url, media_type, filename?}`` — see ``mcp.media_passthrough``). Returns
    ``None`` for missing urls or non-media types (never raises: passthrough is
    best-effort)."""
    url = (stored.get("url") or "").strip()
    media_type = (stored.get("media_type") or "").lower()
    label = stored.get("filename") or None
    try:
        if media_type.startswith("image/"):
            return image_exhibit_from_generate(url, exhibit_id=exhibit_id, alt=label)
        if media_type.startswith("audio/"):
            return audio_exhibit_from_generate(url, exhibit_id=exhibit_id, caption=label)
    except Exception:  # noqa: BLE001 — a malformed entry must not break the turn
        return None
    return None


def exhibit_from_present_call(arguments: dict[str, Any]) -> Exhibit:
    """Build + validate an :class:`Exhibit` from ``present_exhibit`` tool args.

    Assigns a generated ``id`` when omitted (so a one-shot exhibit still has a
    stable handle). Raises ``pydantic.ValidationError`` on a malformed tree
    (missing/invalid fields, a non-allow-listed element type).
    """
    args = dict(arguments or {})
    if not args.get("id"):
        args["id"] = f"exh_{uuid.uuid4().hex[:12]}"
    args.setdefault("layout", "stack")
    args.setdefault("schema_version", EXHIBIT_SCHEMA_VERSION)
    return Exhibit.model_validate(args)

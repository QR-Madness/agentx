"""
Exhibits — typed, declarative agent-authored content presented to the user.

An **Exhibit** is one presented unit the agent builds and arranges declaratively
by calling the ``present_exhibit`` internal tool (see
:mod:`agentx_ai.mcp.internal_tools`). It is a constrained tree of typed
**Elements** (Slice 1: ``mermaid`` only) arranged by a ``layout``. A conversation
accumulates exhibits into a **Gallery**.

The agent declares the *desired state* (with a stable ``id``); re-declaring the
same ``id`` across turns is an amend (replace-in-place). This module owns the
schema + validation so both the producer tool and the streaming layer agree on
one definition. The element-type allow-list (:data:`ALLOWED_ELEMENT_TYPES`) is
the security boundary — the agent can only emit registered, typed elements,
never raw HTML.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# Bump only on a breaking change to the Exhibit wire shape (honors the v0.20
# "migratable across platforms" rule — every exhibit carries this).
EXHIBIT_SCHEMA_VERSION = 1

# The allow-list. Adding an element type = add it here + a client renderer.
ALLOWED_ELEMENT_TYPES: frozenset[str] = frozenset({"mermaid"})

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


# Element union — single member today; widen as new element types ship.
Element = MermaidElement


class Exhibit(BaseModel):
    """A declarative, agent-authored presentation unit."""

    schema_version: int = EXHIBIT_SCHEMA_VERSION
    id: str
    title: str | None = None
    layout: Literal["stack"] = "stack"
    elements: list[MermaidElement] = Field(min_length=1)


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

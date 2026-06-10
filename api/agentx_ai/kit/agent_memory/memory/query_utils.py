"""Shared Cypher query helpers for the memory subsystem."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # LiteralString lives in typing only on 3.11+; the project targets >=3.10,
    # so it is referenced for type-checking only (annotations are strings at
    # runtime via `from __future__ import annotations`).
    from typing import LiteralString


def get_channel_filter_cypher(channel: str | None, var: LiteralString = "g") -> LiteralString:
    """Return the Cypher channel-access clause for a node bound to ``var``.

    When a non-global channel is active, results are scoped to that channel plus
    the shared ``_global`` channel; otherwise only ``_global`` is visible. The
    clause is designed to follow a ``WHERE true`` so it can be appended with a
    leading ``AND``, and expects a ``$channel`` parameter to be supplied to the
    query when a non-global channel is active.

    Returns a ``LiteralString`` so callers can keep interpolating it into
    ``session.run(...)`` queries without tripping neo4j's injection guard.
    """
    if channel and channel != "_global":
        return f"AND ({var}.channel = $channel OR {var}.channel = '_global')"
    return f"AND {var}.channel = '_global'"

"""PostgreSQL twins of the memory classes (Decisions.md ADR-21, Slice 3).

One module per memory class, each a subclass with the same public methods, built
on the tables of Alembic ``0009`` and the row-level-security scopes of
``memory/scope.py``. Nothing in this package may reach Neo4j
(``tests_memory.MemoryTwinBoundaryTest``).

Shared here: the few things every twin needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

_ITERATIVE_SCAN = text("SET LOCAL hnsw.iterative_scan = relaxed_order")


def require_user(user_id: str | None) -> str:
    """The owner of a row or a query. The graph let a caller leave it out and then
    spanned every user; these tables have no rows without an owner."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("the PostgreSQL memory store needs a user id for every operation")
    return user_id


def vector_literal(embedding: Sequence[float] | None) -> str | None:
    """A vector as pgvector reads it (``[0.1,0.2,…]``), or None for no vector."""
    if embedding is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def embedder_name() -> str:
    """The model that produces this install's vectors, as the export envelope
    names it (``local:BAAI/bge-m3``). Stored beside every vector, so a change of
    model can find the rows it has to redo (Memory-Roadmap §1.3)."""
    from ...portability.schema import current_embedder_info

    return current_embedder_info().provider_model


def prepare_vector_search(session: Session) -> None:
    """Make the nearest-neighbour queries of this transaction keep scanning the
    index until they have enough rows that pass the row filters.

    The index knows distances, not owners: row-level security and every other
    condition are applied to what it returns. Measured on these tables (20,000
    turns, a user holding 60 of them, pgvector 0.8.2): without this, a search for
    that user's ten nearest turns returned none; in strict order it returned ten,
    nine of them right; in relaxed order, ten of ten. Relaxed order may hand rows
    back slightly out of order, so `nearest_rows` sorts them again.

    Needs pgvector 0.8 or later; says so when the database has less.
    """
    try:
        session.execute(_ITERATIVE_SCAN)
    except DBAPIError as err:
        raise RuntimeError(
            "The PostgreSQL memory store needs pgvector 0.8 or later (iterative index "
            "scans). Update the extension: ALTER EXTENSION vector UPDATE;"
        ) from err


# A filtered set of at most this many rows is ranked exactly, without the index.
# Measured on these tables (pgvector 0.8.2, 1024 dimensions, warm): an exact pass
# costs about 6 ms per 1,000 admitted rows (6 ms, 18 ms and 67 ms for 1,000, 5,000
# and 20,000), with every result right. The index took 3 ms for a user who holds
# nearly all of a table, but 39 ms at 0.94 recall for one who holds 60 of 20,000
# rows, where the exact pass took 0.3 ms. Slice 4 tunes this against real recall.
EXACT_SCAN_ROWS = 2000


def nearest_rows(
    session: Session,
    *,
    table: str,
    columns: str,
    where: str,
    params: dict[str, Any],
    embedding: Sequence[float],
    limit: int,
) -> list[dict[str, Any]]:
    """The `limit` rows of `table` nearest `embedding`, nearest first, each with
    its cosine ``distance``, among the rows that `where` and the session's scope
    admit. `table`, `columns` and `where` are the caller's own SQL, never input;
    `where` is empty or starts with ``AND``.

    Few admitted rows (``EXACT_SCAN_ROWS`` or fewer) are ranked exactly: faster
    than the index there, and never wrong. Otherwise the index is asked, and if it
    comes back with fewer rows than asked for, the admitted rows are ranked exactly
    after all. A short answer therefore always means there were no more rows to
    find, never that the index stopped looking (``hnsw.max_scan_tuples``).
    """
    limit = max(1, int(limit))
    bound = {**params, "embedding": vector_literal(embedding), "limit": limit}
    exactly = text(f"""
        WITH admitted AS MATERIALIZED (
            SELECT {columns}, embedding FROM {table} WHERE embedding IS NOT NULL {where}
        )
        SELECT {columns}, embedding <=> CAST(:embedding AS vector) AS distance
          FROM admitted
         ORDER BY distance
         LIMIT :limit
    """)  # noqa: S608 — table, columns and filter are the caller's constants; values are bound

    def exact() -> list[dict[str, Any]]:
        return [dict(row) for row in session.execute(exactly, bound).mappings()]

    counted = text(f"""
        SELECT count(*) FROM (
            SELECT 1 FROM {table} WHERE embedding IS NOT NULL {where} LIMIT :cap
        ) AS admitted
    """)  # noqa: S608 — as above
    admitted = session.execute(counted, {**params, "cap": EXACT_SCAN_ROWS + 1}).scalar_one()
    if admitted <= EXACT_SCAN_ROWS:
        return exact()

    prepare_vector_search(session)
    by_index = text(f"""
        WITH nearest AS MATERIALIZED (
            SELECT {columns}, embedding <=> CAST(:embedding AS vector) AS distance
              FROM {table}
             WHERE embedding IS NOT NULL {where}
             ORDER BY embedding <=> CAST(:embedding AS vector)
             LIMIT :limit
        )
        SELECT * FROM nearest ORDER BY distance
    """)  # noqa: S608 — as above
    rows = [dict(row) for row in session.execute(by_index, bound).mappings()]
    return rows if len(rows) >= limit else exact()


def similarity(distance: Any) -> float:
    """The score recall expects for a cosine distance: the graph's scale, where 1
    is identical and 0.5 is unrelated, so thresholds mean the same on both stores."""
    return 1.0 - float(distance) / 2.0

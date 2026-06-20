# pyright: reportMissingImports=false
# (agentx_ai resolves at runtime via alembic.ini `prepend_sys_path = api`.)
"""Shared helpers for Alembic revisions (memory Postgres).

Keeps the vector-dimension substitution in one place so the baseline and any
future revision that touches a pgvector column stay correct on non-default
installs (local bge-m3 = 1024, OpenAI = 1536, etc.).
"""

from __future__ import annotations

import re

_VECTOR_RE = re.compile(r"vector\(\d+\)")


def configured_vector_dims() -> int:
    """The embedding dimension this deployment is configured for."""
    from agentx_ai.kit.agent_memory.config import get_settings

    return get_settings().embedding_dimensions


def with_vector_dims(sql: str) -> str:
    """Rewrite every ``vector(N)`` literal to the configured embedding dimension.

    Mirrors ``init_memory_schema._substitute_postgres_dims`` so DDL authored
    against the placeholder ``vector(1024)`` lands at the right width.
    """
    return _VECTOR_RE.sub(f"vector({configured_vector_dims()})", sql)


def exec_sql_file(path: str) -> None:
    """Execute a raw multi-statement .sql file (dim-substituted) on the bound
    connection.

    Mirrors the legacy ``init_memory_schema``'s ``session.execute(text(file))`` —
    the proven path for this exact DDL (its ``::`` casts are handled; there are no
    single-colon bind-like literals). Runs on Alembic's managed connection so it
    shares the migration transaction.
    """
    from pathlib import Path

    import sqlalchemy as sa
    from alembic import op

    sql = with_vector_dims(Path(path).read_text())
    op.get_bind().execute(sa.text(sql))

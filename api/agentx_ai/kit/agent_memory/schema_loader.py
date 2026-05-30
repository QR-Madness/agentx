"""Helpers for loading and parsing the memory-system schema/migration files.

The Neo4j baseline (``queries/neo4j_schemas.cypher``) and the incremental
migrations (``queries/neo4j_migrations/*.cypher``) are split into individual
statements before execution. Splitting must be comment-aware: a ``//`` comment
may legitimately contain a ``;`` (e.g. an inline example like
``id is "<conv_id>:<agent_id>"``), and a naive ``text.split(";")`` would break
the comment in two and glue the orphaned tail onto the following statement,
producing a Cypher syntax error.

``split_cypher_statements`` strips ``//`` comments (full-line and trailing)
*before* splitting on ``;``, which makes statement boundaries immune to
semicolons that live inside comments.
"""

from __future__ import annotations


def _strip_line_comment(line: str) -> str:
    """Remove a trailing ``//`` comment from a single line.

    Safe for the memory schema/migration files because none of their string
    literals contain ``//`` (the only strings are things like ``'cosine'`` and
    ``'_global'``). If that ever changes, this would need string-literal
    awareness; a guard test over the real schema files covers the assumption.
    """
    idx = line.find("//")
    return line[:idx] if idx != -1 else line


def split_cypher_statements(text: str) -> list[str]:
    """Split a Cypher schema/migration blob into executable statements.

    Strips ``//`` comments before splitting on ``;`` so a semicolon inside a
    comment cannot break statement boundaries. Blank chunks and bare
    ``RETURN 1`` markers (used to satisfy parsers / mark applied versions) are
    dropped.
    """
    # Remove comments first so their semicolons can't split statements.
    decommented = "\n".join(
        stripped
        for line in text.splitlines()
        if (stripped := _strip_line_comment(line).rstrip())
    )

    statements: list[str] = []
    for chunk in decommented.split(";"):
        cleaned = "\n".join(
            line for line in chunk.splitlines() if line.strip()
        ).strip()
        if cleaned and cleaned != "RETURN 1":
            statements.append(cleaned)
    return statements

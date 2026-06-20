"""baseline — full memory Postgres schema as of Alembic adoption

Revision ID: 0001_baseline
Revises:
Create Date: 2026-06-19

The starting point: applies the frozen `alembic/baseline.sql` (the former
`queries/postgres_builder.sql`, verbatim) — every table, index, function and
partition, including `usage_events` and `agentx_auth`. All DDL is
`CREATE ... IF NOT EXISTS`, so running this against an already-populated database
(the cutover case) is a safe no-op that simply records `alembic_version`.

Do NOT edit baseline.sql after this — author a new revision for any change.
"""

from collections.abc import Sequence
from pathlib import Path

# revision identifiers, used by Alembic.
revision: str = "0001_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BASELINE_SQL = Path(__file__).resolve().parent.parent / "baseline.sql"


def upgrade() -> None:
    """Apply the frozen baseline schema (vector dims substituted)."""
    from helpers import exec_sql_file

    exec_sql_file(str(_BASELINE_SQL))


def downgrade() -> None:
    """No automatic teardown — dropping the entire memory schema must be a
    deliberate operator action, never an `alembic downgrade`."""
    raise NotImplementedError(
        "The baseline is not auto-reversible; drop the schema manually if needed."
    )

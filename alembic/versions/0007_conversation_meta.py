"""conversation_meta — user-set conversation metadata (title, archived)

Conversation titles have always been derived (first user message, truncated)
and never stored; archiving didn't exist. The Ambassador v3 conversation-
management belt (rename/archive/delete as user-confirmed writes) needs a
durable home for both — a TTL'd Redis key would silently revert a custom
title after 30 days, so this is Postgres-only with no hot cache (the listing
paths that consume it already hit Postgres).

TEXT key for the same reason as conversation_state (0006): session ids may be
arbitrary strings, and meta may be written before a first turn persists (or
survive after turns are pruned) — no FK to conversation_logs.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_conversation_meta"
down_revision: str | Sequence[str] | None = "0006_conversation_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_meta (
            conversation_id TEXT PRIMARY KEY,
            title           TEXT,
            archived        BOOLEAN NOT NULL DEFAULT FALSE,
            archived_at     TIMESTAMPTZ,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS conversation_meta")

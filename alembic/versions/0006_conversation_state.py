"""conversation_state — durable copy of the per-conversation working state

The structured conversation state (goals/decisions/open_threads/artifacts/
narrative slots + the rolling compaction digest — the INV-CTX-1 coverage
surface) lived only in Redis with a 30-day TTL: a long-parked conversation
came back with its verbatim tail but its compaction coverage evaporated.
This table is the durable copy; Redis stays the hot cache (write-through on
save, read-through + re-warm on a Redis miss).

TEXT key (not UUID like conversation_logs): state is keyed by the session
id, which background/harness callers may supply as an arbitrary string — the
durable copy must never fail on a non-UUID id, and an unbounded key means the
FULL id is stored (no truncation, so it can never collide two long ids or
disagree with the full-id Redis key). No FK to conversation_logs: state can
precede its first persisted turn.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_conversation_state"
down_revision: str | Sequence[str] | None = "0005_workspace_projects"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_state (
            conversation_id TEXT PRIMARY KEY,
            state           JSONB NOT NULL,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS conversation_state")

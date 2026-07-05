"""workspace projects — description/instructions + durable conversation membership

Projects v1 evolves the workspace into a Claude-Projects-style container:
  - description/instructions: user-authored; instructions ride every turn's
    preamble as a stable ledger block (see kit/workspaces/retrieval.py).
  - workspace_conversations: server-side conversation→project membership,
    replacing the client-only localStorage attach. PK on conversation_id
    encodes the v1 invariant: one project per conversation (upsert moves it).
    No FK to conversation_logs — membership can precede/outlive log rows.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005_workspace_projects"
down_revision: str | Sequence[str] | None = "0004_workspace_shell_backend"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS instructions TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS workspace_conversations (
            conversation_id UUID PRIMARY KEY,
            workspace_id    VARCHAR(100) NOT NULL
                            REFERENCES workspaces(id) ON DELETE CASCADE,
            user_id         VARCHAR(100) NOT NULL DEFAULT 'default',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ws_conv_workspace "
        "ON workspace_conversations (workspace_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS workspace_conversations")
    op.execute("ALTER TABLE workspaces DROP COLUMN IF EXISTS instructions")
    op.execute("ALTER TABLE workspaces DROP COLUMN IF EXISTS description")

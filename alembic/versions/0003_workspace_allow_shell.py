"""workspace allow_shell — per-workspace opt-in for agent shells

Shell access is a property of the workspace you point the agent at (not a global flag):
agents only get the sandboxed shell tools when their conversation is attached to a
workspace with ``allow_shell = true``. Defaults false (opt-in).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_workspace_allow_shell"
down_revision: str | Sequence[str] | None = "0002_workspaces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS allow_shell BOOLEAN NOT NULL DEFAULT FALSE"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE workspaces DROP COLUMN IF EXISTS allow_shell")

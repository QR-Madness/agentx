"""workspace shell_backend — choose the per-workspace shell sandbox

A workspace with shell enabled (allow_shell) can run its shell in one of two backends:
  - 'bubblewrap' (default): lightweight, no install, no network — the locked-down jail.
  - 'container':  a persistent per-workspace Docker container (installs + network).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_workspace_shell_backend"
down_revision: str | Sequence[str] | None = "0003_workspace_allow_shell"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS "
        "shell_backend VARCHAR(20) NOT NULL DEFAULT 'bubblewrap'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE workspaces DROP COLUMN IF EXISTS shell_backend")

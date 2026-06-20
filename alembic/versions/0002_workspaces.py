"""workspaces — file workspaces & document RAG schema

Adds the three-store-separation tables for the Workspaces feature
(todo/backlog/workspaces.md):
  - workspaces       : named container, user-scoped (manifest/metadata in PG)
  - documents        : per-file manifest row (filename, type, size, sha256,
                       storage_key, auto tags+summary, ingest status)
  - document_chunks  : text chunks + pgvector embedding (the rebuildable vectors)

Bytes live in a content-addressed blob store on disk (kit/workspaces/storage.py),
NOT in Postgres. Embedding width is substituted by ``with_vector_dims`` so this
lands at the configured dim (bge-m3 = 1024, OpenAI = 1536).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_workspaces"
down_revision: str | Sequence[str] | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id          VARCHAR(100) PRIMARY KEY,
    user_id     VARCHAR(100) NOT NULL DEFAULT 'default',
    name        VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_workspaces_user ON workspaces (user_id);

CREATE TABLE IF NOT EXISTS documents (
    id            VARCHAR(100) PRIMARY KEY,
    workspace_id  VARCHAR(100) NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename      VARCHAR(512) NOT NULL,
    content_type  VARCHAR(150) NOT NULL DEFAULT 'text/plain',
    size_bytes    BIGINT       NOT NULL DEFAULT 0,
    sha256        VARCHAR(64)  NOT NULL,
    storage_key   TEXT         NOT NULL,
    tags          TEXT[]       NOT NULL DEFAULT '{}',
    summary       TEXT         NOT NULL DEFAULT '',
    status        VARCHAR(20)  NOT NULL DEFAULT 'pending',  -- pending | ready | failed
    error         TEXT,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents (workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents (status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   VARCHAR(100) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id  VARCHAR(100) NOT NULL,  -- denormalized for workspace-scoped vector search
    chunk_index   INTEGER      NOT NULL,
    text          TEXT         NOT NULL,
    embedding     vector(1024),           -- bge-m3 = 1024 (OpenAI = 1536); dim-substituted
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_document  ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_workspace ON document_chunks (workspace_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON document_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def upgrade() -> None:
    from helpers import with_vector_dims

    op.get_bind().execute(sa.text(with_vector_dims(_SCHEMA)))


def downgrade() -> None:
    op.execute(
        "DROP TABLE IF EXISTS document_chunks; "
        "DROP TABLE IF EXISTS documents; "
        "DROP TABLE IF EXISTS workspaces;"
    )

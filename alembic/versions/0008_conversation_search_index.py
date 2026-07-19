"""conversation_logs full-text search index

Lexical search over conversation transcripts (the Ambassador's
``search_conversations`` belt tool — "which conversation discussed X?").
An *expression* GIN index (no stored column → no table rewrite) over
``to_tsvector('simple', content)``; queries pair it with
``websearch_to_tsquery('simple', …)`` for multi-word AND semantics, quoted
phrases and -negation.

The ``simple`` config is deliberate: no stemming, language-neutral
lowercase/whitespace tokenization — an ``english`` config would corrupt
non-English recall on a translation platform. Known limit: unsegmented CJK
collapses to long tokens (substring search weak there); the installed-but-
unused pg_trgm extension is the additive later fix, and semantic search over
the existing embedding column is the "later" tier.

Plain CREATE INDEX blocks writes for the build — fine at self-hosted scale
(use autocommit + CONCURRENTLY manually on a huge deployment).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008_conversation_search_index"
down_revision: str | Sequence[str] | None = "0007_conversation_meta"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_logs_content_fts
        ON conversation_logs USING GIN (to_tsvector('simple', content))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_conversation_logs_content_fts")

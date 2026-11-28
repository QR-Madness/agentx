"""memory store tables — the PostgreSQL home of the memory store (ADR-21, Slice 2)

Ten empty tables that the memory classes' PostgreSQL twins will read and write
(Slice 3). Nothing reads them yet. Design and reasoning:
todo/backlog/memory-store-consolidation.md ("Slice 2 — the schema").

Shape, in short:

- **Every row has an owner.** `user_id` is NOT NULL and never empty, and it leads
  every primary key: a row's identity is `(user_id, id)`, so two users can hold
  the same id (an imported pack, say) without touching each other.
- **References stay inside a user.** Foreign keys are composite, `(user_id, …)`,
  so a turn cannot point at another user's conversation. Row-level security
  (revision 0010) does not cover this by itself: foreign-key checks bypass it.
- **A turn position is stored once.** `memory_turns` is unique on
  `(user_id, conversation_id, turn_index)`, the transcript's own key
  (`conversation_logs`), so re-storing a turn replaces it and a projection row
  always maps to one transcript row.
- **Many-to-many relationships are rows in `memory_edges`**; ownership and
  parent links are columns. A node's edges go when the node goes (the
  `memory_detach_edges` trigger), which a polymorphic table can't get from
  foreign keys.
- **Vectors** are `vector(1024)` here and land at the configured embedding width
  (`with_vector_dims`), with an HNSW cosine index where recall searches them.
  HNSW indexes take at most 2,000 dimensions, the same ceiling the ivfflat
  indexes of 0001 and 0002 already impose.
- `embedder` names the model that produced a row's vector (Memory-Roadmap §1.3),
  so a change of model can find the rows it has to re-embed.

Requires PostgreSQL 15+ (`ON DELETE SET NULL (column)`) and pgvector 0.5+ (HNSW);
the bundled image is PostgreSQL 16 with pgvector 0.8.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009_memory_store_tables"
down_revision: str | Sequence[str] | None = "0008_conversation_search_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Dependency order: a table follows the tables it references.
TABLES = (
    "memory_conversations",
    "memory_turns",
    "memory_entities",
    "memory_facts",
    "memory_goals",
    "memory_strategies",
    "memory_procedures",
    "memory_tools",
    "memory_tool_invocations",
    "memory_edges",
)

# Node types an edge may join, and the table each lives in. Deleting a row of
# one of these tables removes the edges that touch it.
EDGE_ENDPOINTS = {
    "conversation": "memory_conversations",
    "entity": "memory_entities",
    "fact": "memory_facts",
    "strategy": "memory_strategies",
    "procedure": "memory_procedures",
}

_SCHEMA = """
-- A conversation as memory sees it: its scope, its consolidation state and, when
-- one was recorded, how it turned out. The transcript itself is conversation_logs.
CREATE TABLE memory_conversations (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    channel             TEXT        NOT NULL DEFAULT '_global',
    agent_id            TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consolidated        TIMESTAMPTZ,            -- last sweep of its user turns
    self_consolidated   TIMESTAMPTZ,            -- last sweep of its assistant turns
    archived            BOOLEAN     NOT NULL DEFAULT FALSE,
    patterns_extracted  BOOLEAN     NOT NULL DEFAULT FALSE,
    outcome_success     BOOLEAN,                -- NULL: no outcome recorded
    outcome_task_type   TEXT,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX idx_memory_conversations_scope ON memory_conversations (user_id, channel);

-- Memory's projection of a turn: what recall searches and consolidation sweeps.
CREATE TABLE memory_turns (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    conversation_id     TEXT        NOT NULL,
    turn_index          INTEGER     NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    role                TEXT        NOT NULL,
    content             TEXT        NOT NULL,
    token_count         INTEGER,
    channel             TEXT        NOT NULL DEFAULT '_global',
    agent_id            TEXT,
    agent_name          TEXT,
    consolidated        TIMESTAMPTZ,            -- user turns: when facts were extracted
    self_consolidated   TIMESTAMPTZ,            -- assistant turns: when self-facts were
    archived            BOOLEAN     NOT NULL DEFAULT FALSE,
    embedding           vector(1024),
    embedder            TEXT,
    PRIMARY KEY (user_id, id),
    UNIQUE (user_id, conversation_id, turn_index),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES memory_conversations (user_id, id) ON DELETE CASCADE
);
CREATE INDEX idx_memory_turns_recent ON memory_turns (user_id, timestamp DESC);
-- Consolidation discovery spans users and wants only the turns still to do, so
-- these stay as small as the backlog.
CREATE INDEX idx_memory_turns_pending_user ON memory_turns (conversation_id, turn_index)
    WHERE role = 'user' AND consolidated IS NULL;
CREATE INDEX idx_memory_turns_pending_assistant ON memory_turns (conversation_id, turn_index)
    WHERE role = 'assistant' AND self_consolidated IS NULL;
CREATE INDEX idx_memory_turns_embedding ON memory_turns
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE memory_entities (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    name                TEXT        NOT NULL,
    type                TEXT        NOT NULL,
    aliases             TEXT[]      NOT NULL DEFAULT '{}',
    description         TEXT,
    salience            DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    properties          JSONB       NOT NULL DEFAULT '{}',
    channel             TEXT        NOT NULL DEFAULT '_global',
    first_seen          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count        INTEGER     NOT NULL DEFAULT 0,
    promoted_from       TEXT[]      NOT NULL DEFAULT '{}',  -- channels it was promoted from
    promoted_at         TIMESTAMPTZ,
    last_promoted       TIMESTAMPTZ,
    embedding           vector(1024),
    embedder            TEXT,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX idx_memory_entities_scope ON memory_entities (user_id, channel);
CREATE INDEX idx_memory_entities_name  ON memory_entities (user_id, lower(name));
CREATE INDEX idx_memory_entities_embedding ON memory_entities
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE memory_facts (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    claim               TEXT        NOT NULL,
    claim_hash          TEXT,
    confidence          DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    source              TEXT        NOT NULL,   -- extraction | user_stated | inferred | …
    source_turn_id      TEXT,                   -- the turn it came from; may be gone
    channel             TEXT        NOT NULL DEFAULT '_global',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count        INTEGER     NOT NULL DEFAULT 0,
    salience            DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    temporal_context    TEXT,                   -- current | past | future
    superseded_at       TIMESTAMPTZ,
    superseded_by_id    TEXT,
    supersedes_id       TEXT,
    flagged_for_review  BOOLEAN     NOT NULL DEFAULT FALSE,
    promoted_from       TEXT[]      NOT NULL DEFAULT '{}',
    promoted_at         TIMESTAMPTZ,
    last_promoted       TIMESTAMPTZ,
    embedding           vector(1024),
    embedder            TEXT,
    -- 'simple': no stemming, language-neutral, as conversation search (0008).
    claim_tsv           tsvector GENERATED ALWAYS AS (to_tsvector('simple', claim)) STORED,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX idx_memory_facts_scope      ON memory_facts (user_id, channel);
CREATE INDEX idx_memory_facts_claim_hash ON memory_facts (user_id, claim_hash);
CREATE INDEX idx_memory_facts_claim_tsv  ON memory_facts USING gin (claim_tsv);
CREATE INDEX idx_memory_facts_embedding  ON memory_facts
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE memory_goals (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    description         TEXT        NOT NULL,
    status              TEXT        NOT NULL DEFAULT 'active',
    priority            INTEGER     NOT NULL DEFAULT 3,
    parent_goal_id      TEXT,
    channel             TEXT        NOT NULL DEFAULT '_global',
    conversation_id     TEXT,                   -- the conversation that opened it
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deadline            TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    result              TEXT,
    embedding           vector(1024),           -- stored, not searched: no index
    embedder            TEXT,
    PRIMARY KEY (user_id, id),
    FOREIGN KEY (user_id, parent_goal_id)
        REFERENCES memory_goals (user_id, id) ON DELETE SET NULL (parent_goal_id)
);
CREATE INDEX idx_memory_goals_status       ON memory_goals (user_id, status);
CREATE INDEX idx_memory_goals_conversation ON memory_goals (user_id, conversation_id);
CREATE INDEX idx_memory_goals_parent       ON memory_goals (user_id, parent_goal_id);

CREATE TABLE memory_strategies (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    description         TEXT        NOT NULL,
    context_pattern     TEXT        NOT NULL,
    tool_sequence       TEXT[]      NOT NULL DEFAULT '{}',
    success_count       INTEGER     NOT NULL DEFAULT 0,
    failure_count       INTEGER     NOT NULL DEFAULT 0,
    channel             TEXT        NOT NULL DEFAULT '_global',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used           TIMESTAMPTZ,
    embedding           vector(1024),
    embedder            TEXT,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX idx_memory_strategies_scope ON memory_strategies (user_id, channel);
CREATE INDEX idx_memory_strategies_embedding ON memory_strategies
    USING hnsw (embedding vector_cosine_ops);

-- `channel` is the procedure's scope (the model's `scope` field).
CREATE TABLE memory_procedures (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    id                  TEXT        NOT NULL,
    trigger             TEXT        NOT NULL,
    trigger_features    JSONB       NOT NULL DEFAULT '{}',
    body                TEXT        NOT NULL,
    rationale           TEXT        NOT NULL DEFAULT '',
    channel             TEXT        NOT NULL DEFAULT '_global',
    agent_id            TEXT,
    strength            INTEGER     NOT NULL DEFAULT 1,
    evidence_refs       TEXT[]      NOT NULL DEFAULT '{}',
    signal_kinds        TEXT[]      NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_reinforced     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding           vector(1024),
    embedder            TEXT,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX idx_memory_procedures_scope ON memory_procedures (user_id, channel);
CREATE INDEX idx_memory_procedures_embedding ON memory_procedures
    USING hnsw (embedding vector_cosine_ops);

-- A user's running statistics for a tool, kept apart from the invocations so
-- they outlive the conversations those belong to.
CREATE TABLE memory_tools (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    name                TEXT        NOT NULL,
    usage_count         INTEGER     NOT NULL DEFAULT 0,
    success_count       INTEGER     NOT NULL DEFAULT 0,
    avg_latency_ms      DOUBLE PRECISION,
    PRIMARY KEY (user_id, name)
);

CREATE TABLE memory_tool_invocations (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    conversation_id     TEXT        NOT NULL,
    turn_index          INTEGER,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name           TEXT        NOT NULL,
    tool_input          JSONB,
    tool_output         JSONB,
    success             BOOLEAN,
    latency_ms          INTEGER,
    error_message       TEXT,
    channel             TEXT        NOT NULL DEFAULT '_global',
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES memory_conversations (user_id, id) ON DELETE CASCADE
);
CREATE INDEX idx_memory_tool_invocations_conversation
    ON memory_tool_invocations (user_id, conversation_id, turn_index);
CREATE INDEX idx_memory_tool_invocations_tool
    ON memory_tool_invocations (user_id, tool_name);

-- Relationships between nodes. `rel` is the relationship (ABOUT, RELATES_TO,
-- MENTIONS, …); `label` is its meaning where one relationship carries several
-- ("works for", "part of"), so two entities can be related in more than one way.
CREATE TABLE memory_edges (
    user_id             TEXT        NOT NULL CHECK (user_id <> ''),
    src_type            TEXT        NOT NULL
        CHECK (src_type IN ('conversation', 'entity', 'fact', 'strategy', 'procedure')),
    src_id              TEXT        NOT NULL,
    rel                 TEXT        NOT NULL,
    label               TEXT        NOT NULL DEFAULT '',
    dst_type            TEXT        NOT NULL
        CHECK (dst_type IN ('conversation', 'entity', 'fact', 'strategy', 'procedure')),
    dst_id              TEXT        NOT NULL,
    channel             TEXT,                   -- where it was asserted, when recorded
    confidence          DOUBLE PRECISION,
    props               JSONB       NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, src_type, src_id, rel, label, dst_type, dst_id)
);
-- The primary key serves a node's outgoing edges; this serves its incoming ones.
CREATE INDEX idx_memory_edges_dst ON memory_edges (user_id, dst_type, dst_id, rel);

-- A deleted node takes its edges with it. Filtered by the node's own user, so it
-- is correct whichever role fires it (a cascade runs as the table owner).
CREATE FUNCTION memory_detach_edges() RETURNS trigger
LANGUAGE plpgsql AS $fn$
BEGIN
    DELETE FROM memory_edges
     WHERE user_id = OLD.user_id AND src_type = TG_ARGV[0] AND src_id = OLD.id;
    DELETE FROM memory_edges
     WHERE user_id = OLD.user_id AND dst_type = TG_ARGV[0] AND dst_id = OLD.id;
    RETURN OLD;
END
$fn$;
"""


def upgrade() -> None:
    from helpers import with_vector_dims

    bind = op.get_bind()
    server = int(bind.execute(sa.text("SHOW server_version_num")).scalar() or 0)
    if server < 150000:
        raise RuntimeError(
            "The memory tables need PostgreSQL 15 or newer "
            f"(this server reports version number {server})."
        )
    bind.execute(sa.text(with_vector_dims(_SCHEMA)))
    for node_type, table in EDGE_ENDPOINTS.items():
        # Identifiers come from the constants above, never from input.
        bind.execute(sa.text(
            f"CREATE TRIGGER {table}_detach_edges AFTER DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION memory_detach_edges('{node_type}')"
        ))


def downgrade() -> None:
    for table in reversed(TABLES):
        op.execute(f"DROP TABLE IF EXISTS {table}")
    op.execute("DROP FUNCTION IF EXISTS memory_detach_edges()")

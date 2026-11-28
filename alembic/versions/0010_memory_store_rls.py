"""memory store row-level security — a user's memory is that user's alone (ADR-21, Slice 2)

Turns on row-level security for the ten memory tables of 0009 and creates the two
roles that queries run under. Design and reasoning:
todo/backlog/memory-store-consolidation.md ("Slice 2 — row-level security").

**The rule.** A transaction sees and writes the rows of one user, the one it named
with `SET LOCAL agentx.user_id`. One that named nobody sees nothing and can write
nothing. Only the system role sees every user's rows, for the jobs that
legitimately cross users (consolidation discovery, maintenance, admin commands).

**Two roles, neither of which can log in.**

- `agentx_memory`: the role memory queries run as. It exists because the role the
  app logs in with may be a superuser (the bundled stack's is), and superusers
  bypass row-level security even when it is forced. Dropping to this role inside
  the transaction is what makes the policies apply.
- `agentx_memory_system`: the cross-user role. Only `CrossUserStore`'s PostgreSQL
  twin may use it (`tests_memory.MemoryScopeBoundaryTest`).

Both are entered with `SET LOCAL ROLE`, so they end with the transaction and work
behind a transaction-mode connection pooler. The app only ever enters them through
`agentx_ai.kit.agent_memory.memory.scope`.

**Two policies per table.**

- `memory_tenant`: `user_id` equals the named user. The setting is read through
  `NULLIF(..., '')` because a custom setting reads as an empty string, not NULL,
  once any earlier transaction on the connection has set it; and inside a scalar
  sub-select so it is evaluated once per statement, not once per row.
- `memory_system`: every row, for `agentx_memory_system`. It is granted `TO` that
  role, so it is no part of an ordinary query's filter, which stays the plain
  `user_id = …` an index can serve (two policies for the same role are OR'ed, and
  an OR is not an index condition). It also tests `current_user`, because a policy
  granted to a role reaches every role that inherits from it: without the test, a
  login role given membership with inheritance would see every row unasked.

`FORCE` makes the policies bind the table's owner as well, so a login role that is
not a superuser gets nothing by owning the tables.

**Running this as a role that is not a superuser.** It needs CREATEROLE, or the
two roles created beforehand (NOLOGIN, not superuser, no BYPASSRLS) and granted to
it. A role that later connects without being the one that migrated needs the same
membership: `GRANT agentx_memory, agentx_memory_system TO <login role>`.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010_memory_store_rls"
down_revision: str | Sequence[str] | None = "0009_memory_store_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# The tables of 0009, spelled out: a revision is a snapshot and imports no list
# that later code could change under it.
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

_ROLES = """
DO $roles$
DECLARE
    r text;
    can_set boolean;
    migrator_is_superuser boolean :=
        (SELECT rolsuper FROM pg_roles WHERE rolname = current_user);
BEGIN
    FOREACH r IN ARRAY ARRAY['agentx_memory', 'agentx_memory_system'] LOOP
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
                EXECUTE format(
                    'CREATE ROLE %I NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE', r);
            END IF;
            -- A superuser can SET ROLE to anything; any other role needs a membership
            -- that allows it. PostgreSQL 16 made that a separate option: the role that
            -- creates another is a member with ADMIN only, which SET ROLE refuses.
            IF NOT migrator_is_superuser THEN
                can_set := CASE
                    WHEN current_setting('server_version_num')::int >= 160000
                        THEN pg_has_role(current_user, r, 'SET')
                    ELSE pg_has_role(current_user, r, 'MEMBER')
                END;
                IF NOT can_set THEN
                    EXECUTE format('GRANT %I TO %I', r, current_user);
                END IF;
            END IF;
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE EXCEPTION
                'memory store: % cannot create or be granted role %. Give it CREATEROLE, or as an administrator run: CREATE ROLE % NOLOGIN; GRANT % TO %;',
                current_user, r, r, r, current_user;
        END;
        -- A pre-existing role with either attribute would void every policy below.
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r AND (rolsuper OR rolbypassrls)) THEN
            RAISE EXCEPTION
                'memory store: role % is a superuser or has BYPASSRLS, so row-level security would not apply to it', r;
        END IF;
    END LOOP;
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO agentx_memory, agentx_memory_system',
                   current_schema());
END
$roles$;
"""

_PER_TABLE = """
ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {table} FORCE ROW LEVEL SECURITY;

CREATE POLICY memory_tenant ON {table}
    USING      (user_id = (SELECT NULLIF(current_setting('agentx.user_id', true), '')))
    WITH CHECK (user_id = (SELECT NULLIF(current_setting('agentx.user_id', true), '')));

CREATE POLICY memory_system ON {table} TO agentx_memory_system
    USING      ((SELECT current_user) = 'agentx_memory_system')
    WITH CHECK ((SELECT current_user) = 'agentx_memory_system');

GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO agentx_memory, agentx_memory_system;
"""

_DROP_PER_TABLE = """
DROP POLICY IF EXISTS memory_tenant ON {table};
DROP POLICY IF EXISTS memory_system ON {table};
ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY;
ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;
REVOKE ALL ON {table} FROM agentx_memory, agentx_memory_system;
"""

# Roles belong to the whole server, not to this database: another database may
# still use them, in which case they stay.
_DROP_ROLES = """
DO $roles$
DECLARE
    r text;
BEGIN
    EXECUTE format('REVOKE USAGE ON SCHEMA %I FROM agentx_memory, agentx_memory_system',
                   current_schema());
    FOREACH r IN ARRAY ARRAY['agentx_memory', 'agentx_memory_system'] LOOP
        BEGIN
            EXECUTE format('DROP ROLE IF EXISTS %I', r);
        EXCEPTION WHEN dependent_objects_still_exist OR insufficient_privilege THEN
            RAISE NOTICE 'memory store: role % is still in use elsewhere and was kept', r;
        END;
    END LOOP;
END
$roles$;
"""


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text(_ROLES))
    for table in TABLES:
        bind.execute(sa.text(_PER_TABLE.format(table=table)))


def downgrade() -> None:
    bind = op.get_bind()
    for table in TABLES:
        bind.execute(sa.text(_DROP_PER_TABLE.format(table=table)))
    bind.execute(sa.text(_DROP_ROLES))

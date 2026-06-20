# Legacy PostgreSQL migrations (frozen — historical only)

These `NNNN_*.sql` files were the **pre-Alembic** PostgreSQL migration runner
(`manage.py migrate_schema`, now Neo4j-only). They are **frozen for reference**
and are no longer applied by anything.

PostgreSQL schema is now managed by **Alembic**:

- Baseline (everything these files + the old `postgres_builder.sql` produced) is
  `alembic/baseline.sql`, applied by the `0001_baseline` revision.
- New PostgreSQL changes are **Alembic revisions** — `task db:revision -- "msg"`,
  then `task db:migrate:pg` (`alembic upgrade head`).

Do **not** add new files here. See `alembic/README` and `Decisions.md`.

> Neo4j migrations still live next door in `queries/neo4j_migrations/` and remain
> on the home-grown runner — a *new graph change is a numbered `.cypher` migration*,
> never an edit to `neo4j_schemas.cypher`.

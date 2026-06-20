# Neo4j migrations (home-grown runner)

Incremental **Neo4j** schema changes — applied by `manage.py migrate_schema`
(Neo4j-only since the PostgreSQL cutover to Alembic; see Decisions.md ADR-9).

- Files are `NNNN_<name>.cypher`, applied in order when their version exceeds the
  current `_SchemaMeta.version` (the runner bumps it).
- Discipline: a **new graph change is a numbered migration here**, never an edit
  to the baseline `queries/neo4j_schemas.cypher`. (Same "don't edit the baseline"
  rule Alembic enforces for Postgres — kept here by convention.)

Apply: `task db:migrate` (PG via Alembic + Neo4j) or `task db:migrate:status`.

> PostgreSQL migrations are **not** here — they live in `alembic/versions/`.
> The frozen `queries/migrations/` dir is the retired pre-Alembic PG runner.

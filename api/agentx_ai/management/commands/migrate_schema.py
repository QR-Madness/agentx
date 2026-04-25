"""
Django management command to apply incremental database schema migrations.

Scans queries/migrations/NNNN_*.sql and queries/neo4j_migrations/NNNN_*.cypher,
checks the current schema version, and applies all pending files in order.

Usage:
    python manage.py migrate_schema
    python manage.py migrate_schema --status
    python manage.py migrate_schema --dry-run
    python manage.py migrate_schema --postgres-only
    python manage.py migrate_schema --neo4j-only
    python manage.py migrate_schema --verbose
"""

import re
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


# Neo4j vector index names that require dimension substitution
_VECTOR_INDEX_RE = re.compile(r"`vector\.dimensions`:\s*\d+")


def _extract_version(filename: str) -> int | None:
    """Extract the leading version number from a migration filename like '0002_foo.sql'."""
    m = re.match(r"^(\d+)_", filename)
    return int(m.group(1)) if m else None


def _description_from_filename(filename: str) -> str:
    """Turn '0002_user_storage.sql' into 'user storage'."""
    stem = Path(filename).stem  # '0002_user_storage'
    parts = stem.split("_", 1)
    return parts[1].replace("_", " ") if len(parts) > 1 else stem


class Command(BaseCommand):
    help = "Apply pending schema migrations (PostgreSQL + Neo4j)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--postgres-only",
            action="store_true",
            help="Only run PostgreSQL migrations",
        )
        parser.add_argument(
            "--neo4j-only",
            action="store_true",
            help="Only run Neo4j migrations",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be applied without executing",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            help="Print current schema versions and list pending migrations",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show detailed output",
        )

    def handle(self, *args, **options):
        postgres_only = options["postgres_only"]
        neo4j_only = options["neo4j_only"]
        dry_run = options["dry_run"]
        status_only = options["status"]
        verbose = options["verbose"]

        do_all = not (postgres_only or neo4j_only)

        project_root = Path(__file__).parent.parent.parent.parent.parent
        pg_migrations_dir = project_root / "queries" / "migrations"
        neo4j_migrations_dir = project_root / "queries" / "neo4j_migrations"

        pg_ok = True
        neo4j_ok = True

        if do_all or postgres_only:
            pg_ok = self._run_postgres(
                pg_migrations_dir, dry_run=dry_run, status_only=status_only, verbose=verbose
            )

        if do_all or neo4j_only:
            neo4j_ok = self._run_neo4j(
                neo4j_migrations_dir, dry_run=dry_run, status_only=status_only, verbose=verbose
            )

        if not (pg_ok and neo4j_ok):
            raise CommandError("One or more migrations failed.")

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    def _get_pg_version(self, session, text) -> int:
        """Return the current max version from schema_version, or 0 if missing."""
        try:
            result = session.execute(text("SELECT MAX(version) FROM schema_version"))
            row = result.fetchone()
            return row[0] if row and row[0] is not None else 0
        except Exception:
            return 0

    def _get_configured_dims(self) -> int:
        from agentx_ai.kit.agent_memory.config import get_settings
        return get_settings().embedding_dimensions

    def _substitute_pg_dims(self, sql: str, dims: int) -> str:
        return re.sub(r"vector\(\d+\)", f"vector({dims})", sql)

    def _run_postgres(
        self, migrations_dir: Path, *, dry_run: bool, status_only: bool, verbose: bool
    ) -> bool:
        self.stdout.write("\n🐘 PostgreSQL Migrations")
        self.stdout.write("-" * 40)

        try:
            from agentx_ai.kit.agent_memory.connections import get_postgres_session
            from sqlalchemy import text
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f"  Import error: {e}"))  # type: ignore[attr-defined]
            return False

        if not migrations_dir.exists():
            self.stdout.write(self.style.ERROR(f"  Migrations dir not found: {migrations_dir}"))  # type: ignore[attr-defined]
            return False

        # Collect and sort migration files
        files = sorted(
            (f for f in migrations_dir.glob("*.sql") if _extract_version(f.name) is not None),
            key=lambda f: _extract_version(f.name),  # type: ignore[arg-type, return-value]
        )

        try:
            with get_postgres_session() as session:
                current_version = self._get_pg_version(session, text)
                self.stdout.write(f"  Current version: {current_version}")

                pending = [f for f in files if _extract_version(f.name) > current_version]  # type: ignore[operator]

                if not pending:
                    self.stdout.write("  No pending migrations.")
                    return True

                if status_only or dry_run:
                    label = "Pending" if not dry_run else "Would apply"
                    self.stdout.write(f"  {label} ({len(pending)}):")
                    for f in pending:
                        self.stdout.write(f"    {f.name}")
                    return True

                dims = self._get_configured_dims()
                applied = 0

                for mfile in pending:
                    version = _extract_version(mfile.name)
                    description = _description_from_filename(mfile.name)
                    sql = self._substitute_pg_dims(mfile.read_text(), dims)

                    # Skip pure marker files (only comments/whitespace)
                    executable = "\n".join(
                        line for line in sql.splitlines()
                        if line.strip() and not line.strip().startswith("--")
                    ).strip()

                    if verbose:
                        self.stdout.write(f"  Applying {mfile.name}...")

                    try:
                        if executable:
                            session.execute(text(sql))
                        # Record as applied (idempotent: skip if already present)
                        session.execute(text(
                            "INSERT INTO schema_version (version, description, filename) "
                            "SELECT :v, :d, :f WHERE NOT EXISTS "
                            "(SELECT 1 FROM schema_version WHERE version = :v)"
                        ), {"v": version, "d": description, "f": mfile.name})
                        session.commit()
                        applied += 1
                        self.stdout.write(f"  ✓ {mfile.name}")
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"  ✗ {mfile.name}: {e}"))  # type: ignore[attr-defined]
                        return False

                self.stdout.write(f"  Applied {applied} migration(s).")
                return True

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Error: {e}"))  # type: ignore[attr-defined]
            return False

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------

    def _get_neo4j_version(self, session) -> int:
        """Return the current version from _SchemaMeta, or 0 if not found."""
        try:
            result = session.run(
                "MATCH (m:_SchemaMeta {id: 'schema'}) RETURN m.version AS v"
            )
            record = result.single()
            return int(record["v"]) if record and record["v"] is not None else 0
        except Exception:
            return 0

    def _substitute_neo4j_dims(self, cypher: str, dims: int) -> str:
        return _VECTOR_INDEX_RE.sub(f"`vector.dimensions`: {dims}", cypher)

    def _run_neo4j(
        self, migrations_dir: Path, *, dry_run: bool, status_only: bool, verbose: bool
    ) -> bool:
        self.stdout.write("\n📊 Neo4j Migrations")
        self.stdout.write("-" * 40)

        try:
            from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f"  Import error: {e}"))  # type: ignore[attr-defined]
            return False

        if not migrations_dir.exists():
            self.stdout.write(self.style.ERROR(f"  Migrations dir not found: {migrations_dir}"))  # type: ignore[attr-defined]
            return False

        files = sorted(
            (f for f in migrations_dir.glob("*.cypher") if _extract_version(f.name) is not None),
            key=lambda f: _extract_version(f.name),  # type: ignore[arg-type, return-value]
        )

        try:
            driver = Neo4jConnection.get_driver()

            with driver.session() as session:
                current_version = self._get_neo4j_version(session)
                self.stdout.write(f"  Current version: {current_version}")

                pending = [f for f in files if _extract_version(f.name) > current_version]  # type: ignore[operator]

                if not pending:
                    self.stdout.write("  No pending migrations.")
                    return True

                if status_only or dry_run:
                    label = "Pending" if not dry_run else "Would apply"
                    self.stdout.write(f"  {label} ({len(pending)}):")
                    for f in pending:
                        self.stdout.write(f"    {f.name}")
                    return True

                dims = self._get_configured_dims()
                applied = 0

                for mfile in pending:
                    cypher = self._substitute_neo4j_dims(mfile.read_text(), dims)

                    # Split on semicolons, filter out blank/comment-only blocks
                    statements = []
                    for stmt in cypher.split(";"):
                        lines = [
                            ln for ln in stmt.strip().splitlines()
                            if ln.strip() and not ln.strip().startswith("//")
                        ]
                        cleaned = "\n".join(lines).strip()
                        if cleaned and cleaned != "RETURN 1":
                            statements.append(cleaned)

                    if verbose:
                        self.stdout.write(f"  Applying {mfile.name} ({len(statements)} statements)...")

                    try:
                        for stmt in statements:
                            try:
                                session.run(stmt)
                            except Exception as e:
                                err = str(e).lower()
                                if "already exists" in err:
                                    if verbose:
                                        self.stdout.write("    ○ Already exists (skipped)")
                                else:
                                    raise
                        applied += 1
                        self.stdout.write(f"  ✓ {mfile.name}")
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"  ✗ {mfile.name}: {e}"))  # type: ignore[attr-defined]
                        return False

                self.stdout.write(f"  Applied {applied} migration(s).")
                return True

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  Error: {e}"))  # type: ignore[attr-defined]
            return False

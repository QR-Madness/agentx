"""
Django management command to apply incremental **Neo4j** schema migrations.

Scans queries/neo4j_migrations/NNNN_*.cypher, checks `_SchemaMeta.version`, and
applies all pending files in order. **PostgreSQL migrations are managed by
Alembic** now (`alembic upgrade head` / `task db:migrate:pg`).

Usage:
    python manage.py migrate_schema
    python manage.py migrate_schema --status
    python manage.py migrate_schema --dry-run
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
    help = "Apply pending Neo4j schema migrations (PostgreSQL is managed by Alembic)"

    def add_arguments(self, parser):
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
        dry_run = options["dry_run"]
        status_only = options["status"]
        verbose = options["verbose"]

        project_root = Path(__file__).parent.parent.parent.parent.parent
        neo4j_migrations_dir = project_root / "queries" / "neo4j_migrations"

        if not self._run_neo4j(
            neo4j_migrations_dir, dry_run=dry_run, status_only=status_only, verbose=verbose
        ):
            raise CommandError("One or more migrations failed.")

    def _get_configured_dims(self) -> int:
        from agentx_ai.kit.agent_memory.config import get_settings
        return get_settings().embedding_dimensions

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

                from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements

                for mfile in pending:
                    cypher = self._substitute_neo4j_dims(mfile.read_text(), dims)

                    # Comment-aware split: strips // comments before splitting on
                    # ';' so a semicolon inside a comment can't break a statement.
                    statements = split_cypher_statements(cypher)

                    if verbose:
                        self.stdout.write(f"  Applying {mfile.name} ({len(statements)} statements)...")

                    try:
                        for stmt in statements:
                            try:
                                session.run(stmt)  # type: ignore[arg-type]
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

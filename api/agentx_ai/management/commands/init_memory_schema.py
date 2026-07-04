"""
Django management command to initialize memory system database schemas.

This command sets up the **Neo4j** schema and verifies **Redis**. **PostgreSQL is
managed by Alembic** (`alembic upgrade head` / `task db:migrate:pg`) — this command
no longer touches it (`--postgres-only` is a no-op pointer).

Usage:
    python manage.py init_memory_schema
    python manage.py init_memory_schema --neo4j-only
    python manage.py init_memory_schema --redis-only
    python manage.py init_memory_schema --verify
    python manage.py init_memory_schema --force-recreate-indexes   # Neo4j vector indexes
"""

import re
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


# Neo4j vector index names used by the memory system
VECTOR_INDEXES = [
    "turn_embeddings",
    "entity_embeddings",
    "fact_embeddings",
    "strategy_embeddings",
]


class Command(BaseCommand):
    help = "Initialize memory schemas: Neo4j + Redis (PostgreSQL is managed by Alembic)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--neo4j-only",
            action="store_true",
            help="Only initialize Neo4j schemas",
        )
        parser.add_argument(
            "--postgres-only",
            action="store_true",
            help="Deprecated no-op — PostgreSQL is managed by Alembic (task db:migrate:pg)",
        )
        parser.add_argument(
            "--redis-only",
            action="store_true",
            help="Only verify Redis connectivity and key patterns",
        )
        parser.add_argument(
            "--verify",
            action="store_true",
            help="Verify schemas exist without creating them",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Show detailed output",
        )
        parser.add_argument(
            "--force-recreate-indexes",
            action="store_true",
            help="Drop and recreate all vector indexes (use after changing embedding model/dimensions)",
        )
        parser.add_argument(
            "--validate-embedder",
            action="store_true",
            help="Also load the embedding model to check its output dimensions match "
                 "EMBEDDING_DIMENSIONS (slow: pulls the model into RAM)",
        )

    def handle(self, *args, **options):
        neo4j_only = options["neo4j_only"]
        postgres_only = options["postgres_only"]
        redis_only = options["redis_only"]
        verify_only = options["verify"]
        verbose = options["verbose"]
        force_recreate = options["force_recreate_indexes"]

        # If no specific flag, do all
        do_all = not (neo4j_only or postgres_only or redis_only)

        results: dict[str, dict | None] = {"neo4j": None, "postgres": None, "redis": None}

        validate_embedder = options["validate_embedder"]

        if do_all or neo4j_only:
            if force_recreate:
                results["neo4j"] = self._force_recreate_neo4j_indexes(verbose, validate_embedder)
            else:
                results["neo4j"] = self._handle_neo4j(verify_only, verbose, validate_embedder)

        # PostgreSQL is owned by Alembic now (`alembic upgrade head` /
        # `task db:migrate:pg`). This command only handles Neo4j + Redis; surface a
        # pointer rather than touching PG (incl. for --postgres-only / force-recreate).
        if postgres_only or (force_recreate and not (neo4j_only or redis_only)):
            self.stdout.write(self.style.WARNING(  # type: ignore[attr-defined]
                "PostgreSQL schema is managed by Alembic — run `task db:migrate:pg` "
                "(alembic upgrade head); skipping PostgreSQL here."
            ))
        elif do_all:
            self.stdout.write("\n🐘 PostgreSQL: managed by Alembic (`task db:migrate:pg`)")

        if do_all or redis_only:
            results["redis"] = self._handle_redis(verify_only, verbose)

        # Summary
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write(self.style.SUCCESS("Schema Initialization Summary"))  # type: ignore[attr-defined]
        self.stdout.write("=" * 50)

        baseline_success = True
        for db, result in results.items():
            if result is None:
                continue
            if result["success"]:
                self.stdout.write(f"  ✓ {db.upper()}: {result['message']}")
            else:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ {db.upper()}: {result['message']}")  # type: ignore[attr-defined]
                )
                baseline_success = False

        # Apply any pending incremental migrations on top of the baseline.
        # This runs even when the baseline reported an error, so a single
        # non-fatal baseline hiccup can't silently skip incremental migrations
        # (the failure is still surfaced via the CommandError below).
        migrations_success = True
        if not verify_only and not force_recreate:
            self.stdout.write("\n📦 Applying pending migrations...")
            from django.core.management import call_command
            try:
                call_command(
                    "migrate_schema",
                    verbose=verbose,
                    stdout=self.stdout,
                    stderr=self.stderr,
                )
            except CommandError as e:
                migrations_success = False
                self.stderr.write(self.style.ERROR(f"  ✗ migrate_schema: {e}"))  # type: ignore[attr-defined]

        if baseline_success and migrations_success:
            self.stdout.write(self.style.SUCCESS("\n✅ All schemas initialized!"))  # type: ignore[attr-defined]
        else:
            failed = []
            if not baseline_success:
                failed.append("baseline schema")
            if not migrations_success:
                failed.append("incremental migrations")
            raise CommandError(f"Some schema initializations failed: {', '.join(failed)}")

    def _get_configured_dims(self) -> int:
        """Get configured embedding dimensions from settings."""
        from agentx_ai.kit.agent_memory.config import get_settings
        return get_settings().embedding_dimensions

    def _substitute_neo4j_dims(self, schema_content: str, dims: int) -> str:
        """Replace any vector.dimensions value in Cypher schema with configured dims."""
        return re.sub(
            r"(`vector\.dimensions`:\s*)\d+",
            rf"\g<1>{dims}",
            schema_content,
        )

    def _validate_embedder_dimensions(self, verbose: bool) -> bool:
        """Validate that the embedding model produces the configured dimensions."""
        try:
            from agentx_ai.kit.agent_memory.embeddings import get_embedder
            embedder = get_embedder()
            actual, configured, match = embedder.validate_dimensions()
            if match:
                if verbose:
                    self.stdout.write(f"  ✓ Embedding model output dimensions: {actual} (matches config)")
            else:
                self.stdout.write(
                    self.style.WARNING(  # type: ignore[attr-defined]
                        f"  ⚠ Dimension mismatch: model produces {actual}-dim vectors "
                        f"but EMBEDDING_DIMENSIONS={configured}. "
                        f"Update config or use --force-recreate-indexes after fixing."
                    )
                )
            return match
        except Exception as e:
            if verbose:
                self.stdout.write(f"  ⚠ Could not validate embedder dimensions: {e}")
            return True  # Don't block schema init if model can't be loaded

    def _handle_neo4j(self, verify_only: bool, verbose: bool, validate_embedder: bool = False) -> dict:
        """Initialize or verify Neo4j schemas."""
        self.stdout.write("\n📊 Neo4j Schema Initialization")
        self.stdout.write("-" * 40)

        try:
            from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        except ImportError as e:
            return {"success": False, "message": f"Import error: {e}"}

        # Load schema file (api/agentx_ai/management/commands/ -> project_root/queries/)
        # __file__ -> commands -> management -> agentx_ai -> api -> project_root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        queries_dir = project_root / "queries"
        schema_file = queries_dir / "neo4j_schemas.cypher"

        if not schema_file.exists():
            return {"success": False, "message": f"Schema file not found: {schema_file}"}

        schema_content = schema_file.read_text()

        # Substitute vector dimensions from config
        dims = self._get_configured_dims()
        schema_content = self._substitute_neo4j_dims(schema_content, dims)
        self.stdout.write(f"  Vector dimensions: {dims}")

        # Opt-in only: loading the model just to cross-check dimension ints is
        # what made every container boot pull BGE-M3 into RAM (and leave the
        # non-exiting download threads the entrypoint watchdog existed for).
        # Index creation uses the configured dims either way.
        if validate_embedder:
            self._validate_embedder_dimensions(verbose)

        # Parse individual statements (comment-aware: strips // comments before
        # splitting on ';' so a semicolon inside a comment can't break a statement).
        from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements
        statements = split_cypher_statements(schema_content)

        if verbose:
            self.stdout.write(f"  Found {len(statements)} schema statements")

        try:
            driver = Neo4jConnection.get_driver()

            # Verify APOC is available
            with driver.session() as session:
                try:
                    result = session.run(
                        "RETURN apoc.version() AS version"
                    ).single()
                    apoc_version = result["version"] if result else "unknown"
                    self.stdout.write(f"  ✓ APOC plugin available (v{apoc_version})")
                except Exception:
                    self.stdout.write(
                        self.style.WARNING("  ⚠ APOC plugin not available (some features may not work)")  # type: ignore[attr-defined]
                    )

            if verify_only:
                # Check what exists, and assert the required vector indexes are
                # present AND online — a missing/failed fact_embeddings is what
                # silently breaks the semantic-duplicate check at consolidation.
                with driver.session() as session:
                    index_states = {
                        r["name"]: r["state"]
                        for r in session.run("SHOW INDEXES YIELD name, state RETURN name, state")
                    }
                    constraints = [
                        r["name"] for r in session.run("SHOW CONSTRAINTS YIELD name RETURN name")
                    ]

                self.stdout.write(
                    f"  Found {len(index_states)} indexes, {len(constraints)} constraints"
                )

                missing = [n for n in VECTOR_INDEXES if n not in index_states]
                not_online = [
                    f"{n} ({index_states[n]})"
                    for n in VECTOR_INDEXES
                    if n in index_states and index_states[n] != "ONLINE"
                ]
                if missing or not_online:
                    problems = []
                    if missing:
                        problems.append(f"missing: {', '.join(missing)}")
                    if not_online:
                        problems.append(f"not online: {', '.join(not_online)}")
                    detail = "; ".join(problems)
                    self.stdout.write(
                        self.style.ERROR(  # type: ignore[attr-defined]
                            f"  ✗ Required vector indexes — {detail}. "
                            f"Run: python manage.py init_memory_schema"
                        )
                    )
                    return {"success": False, "message": f"vector indexes {detail}"}

                self.stdout.write("  ✓ All required vector indexes present and online")
                return {
                    "success": True,
                    "message": f"{len(index_states)} indexes, {len(constraints)} constraints",
                }

            # Execute schema statements
            success_count = 0
            error_count = 0

            with driver.session() as session:
                for stmt in statements:
                    try:
                        session.run(stmt)  # type: ignore[arg-type]
                        success_count += 1
                        if verbose:
                            # Extract a description from the statement
                            first_line = stmt.split("\n")[0][:60]
                            self.stdout.write(f"    ✓ {first_line}...")
                    except Exception as e:
                        error_msg = str(e)
                        # Some errors are expected (e.g., constraint already exists)
                        if "already exists" in error_msg.lower():
                            success_count += 1
                            if verbose:
                                self.stdout.write("    ○ Already exists (skipped)")
                        else:
                            error_count += 1
                            if verbose:
                                self.stdout.write(
                                    self.style.ERROR(f"    ✗ Error: {error_msg[:80]}")  # type: ignore[attr-defined]
                                )

            self.stdout.write(f"  Applied {success_count} statements, {error_count} errors")

            if error_count > 0:
                return {"success": False, "message": f"{success_count} applied, {error_count} errors"}

            return {"success": True, "message": f"{success_count} statements applied"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def _force_recreate_neo4j_indexes(self, verbose: bool, validate_embedder: bool = False) -> dict:
        """Drop and recreate all Neo4j vector indexes with correct dimensions."""
        self.stdout.write("\n📊 Neo4j Vector Index Recreation")
        self.stdout.write("-" * 40)

        try:
            from agentx_ai.kit.agent_memory.connections import Neo4jConnection
        except ImportError as e:
            return {"success": False, "message": f"Import error: {e}"}

        dims = self._get_configured_dims()
        self.stdout.write(f"  Target dimensions: {dims}")
        if validate_embedder:
            self._validate_embedder_dimensions(verbose)

        # Load schema to extract CREATE VECTOR INDEX statements
        project_root = Path(__file__).parent.parent.parent.parent.parent
        schema_file = project_root / "queries" / "neo4j_schemas.cypher"
        if not schema_file.exists():
            return {"success": False, "message": f"Schema file not found: {schema_file}"}

        schema_content = self._substitute_neo4j_dims(schema_file.read_text(), dims)

        # Extract CREATE VECTOR INDEX statements from schema
        from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements
        create_stmts = {}
        for cleaned in split_cypher_statements(schema_content):
            if "CREATE VECTOR INDEX" in cleaned:
                for idx_name in VECTOR_INDEXES:
                    if idx_name in cleaned:
                        create_stmts[idx_name] = cleaned
                        break

        try:
            driver = Neo4jConnection.get_driver()

            with driver.session() as session:
                # Drop existing vector indexes
                for idx_name in VECTOR_INDEXES:
                    try:
                        session.run(f"DROP INDEX {idx_name} IF EXISTS")  # type: ignore[arg-type]
                        self.stdout.write(f"  ✓ Dropped index: {idx_name}")
                    except Exception as e:
                        if verbose:
                            self.stdout.write(f"  ⚠ Drop {idx_name}: {e}")

                # NULL out stale embeddings so consolidation re-embeds them
                for label in ["Turn", "Entity", "Fact", "Strategy"]:
                    query = (
                        f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
                        f"SET n.embedding = NULL RETURN count(n) AS cleared"
                    )
                    result = session.run(query)  # type: ignore[arg-type]
                    record = result.single()
                    count = record["cleared"] if record else 0
                    if count > 0:
                        self.stdout.write(f"  ✓ Cleared {count} stale {label} embeddings")

                # Recreate vector indexes with correct dimensions
                for idx_name in VECTOR_INDEXES:
                    if idx_name in create_stmts:
                        # Remove IF NOT EXISTS since we just dropped them
                        stmt = create_stmts[idx_name].replace(" IF NOT EXISTS", "")
                        session.run(stmt)
                        self.stdout.write(f"  ✓ Created index: {idx_name} ({dims} dims)")
                    else:
                        self.stdout.write(
                            self.style.WARNING(f"  ⚠ No CREATE statement found for {idx_name}")  # type: ignore[attr-defined]
                        )

            return {"success": True, "message": f"Recreated {len(create_stmts)} vector indexes at {dims} dims"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def _handle_redis(self, verify_only: bool, verbose: bool) -> dict:
        """Verify Redis connectivity and document key patterns."""
        self.stdout.write("\n🔴 Redis Connectivity Check")
        self.stdout.write("-" * 40)

        try:
            from agentx_ai.kit.agent_memory.connections import RedisConnection
        except ImportError as e:
            return {"success": False, "message": f"Import error: {e}"}

        try:
            client = RedisConnection.get_client()

            # Test connectivity
            pong = client.ping()
            if not pong:
                return {"success": False, "message": "PING failed"}

            self.stdout.write("  ✓ Connection successful (PING/PONG)")

            # Get server info
            info = client.info("server")
            redis_version = info.get("redis_version", "unknown") if isinstance(info, dict) else "unknown"
            self.stdout.write(f"  Redis version: {redis_version}")

            # Document key patterns
            self.stdout.write("\n  📋 Memory Key Patterns:")
            key_patterns = [
                ("working:{user_id}:{channel}:{conversation_id}:*", "Working memory items"),
                ("working:{user_id}:{channel}:{conversation_id}:context", "Current context window"),
                ("consolidation:job:{job_id}", "Consolidation job tracking"),
                ("consolidation:lock:{conversation_id}", "Consolidation locks"),
                ("session:{session_id}:*", "Session-scoped ephemeral data"),
            ]
            for pattern, desc in key_patterns:
                self.stdout.write(f"    • {pattern}")
                if verbose:
                    self.stdout.write(f"      {desc}")

            # Check memory usage
            info_memory = client.info("memory")
            if isinstance(info_memory, dict):
                used_memory = info_memory.get("used_memory_human", "unknown")
                max_memory = info_memory.get("maxmemory_human", "not set")
            else:
                used_memory = "unknown"
                max_memory = "not set"
            self.stdout.write(f"\n  Memory: {used_memory} used / {max_memory} max")

            # List any existing keys (just count)
            key_count = client.dbsize()
            self.stdout.write(f"  Keys in database: {key_count}")

            return {"success": True, "message": f"Connected, v{redis_version}, {key_count} keys"}

        except Exception as e:
            return {"success": False, "message": str(e)}

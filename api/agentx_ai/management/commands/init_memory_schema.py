"""
Django management command to initialize memory system database schemas.

This command sets up the required schemas in Neo4j, PostgreSQL, and Redis
for the agent memory system to function properly.

Usage:
    python manage.py init_memory_schema
    python manage.py init_memory_schema --neo4j-only
    python manage.py init_memory_schema --postgres-only
    python manage.py init_memory_schema --redis-only
    python manage.py init_memory_schema --verify
"""

from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Initialize memory system database schemas (Neo4j, PostgreSQL, Redis)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--neo4j-only",
            action="store_true",
            help="Only initialize Neo4j schemas",
        )
        parser.add_argument(
            "--postgres-only",
            action="store_true",
            help="Only initialize PostgreSQL schemas",
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

    def handle(self, *args, **options):
        neo4j_only = options["neo4j_only"]
        postgres_only = options["postgres_only"]
        redis_only = options["redis_only"]
        verify_only = options["verify"]
        verbose = options["verbose"]

        # If no specific flag, do all
        do_all = not (neo4j_only or postgres_only or redis_only)

        results = {"neo4j": None, "postgres": None, "redis": None}

        if do_all or neo4j_only:
            results["neo4j"] = self._handle_neo4j(verify_only, verbose)

        if do_all or postgres_only:
            results["postgres"] = self._handle_postgres(verify_only, verbose)

        if do_all or redis_only:
            results["redis"] = self._handle_redis(verify_only, verbose)

        # Summary
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write(self.style.SUCCESS("Schema Initialization Summary"))
        self.stdout.write("=" * 50)

        all_success = True
        for db, result in results.items():
            if result is None:
                continue
            if result["success"]:
                self.stdout.write(f"  ✓ {db.upper()}: {result['message']}")
            else:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ {db.upper()}: {result['message']}")
                )
                all_success = False

        if all_success:
            self.stdout.write(self.style.SUCCESS("\n✅ All schemas initialized!"))
        else:
            raise CommandError("Some schema initializations failed")

    def _handle_neo4j(self, verify_only: bool, verbose: bool) -> dict:
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

        # Parse individual statements (split on semicolons, filter empty/comments)
        statements = []
        for stmt in schema_content.split(";"):
            stmt = stmt.strip()
            # Skip empty statements and pure comment blocks
            if not stmt or stmt.startswith("//"):
                continue
            # Skip the final RETURN 1 statement
            if stmt.strip() == "RETURN 1":
                continue
            statements.append(stmt)

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
                        self.style.WARNING("  ⚠ APOC plugin not available (some features may not work)")
                    )

            if verify_only:
                # Just check if indexes exist
                with driver.session() as session:
                    result = session.run("SHOW INDEXES")
                    indexes = [r["name"] for r in result]
                    result = session.run("SHOW CONSTRAINTS")
                    constraints = [r["name"] for r in result]

                self.stdout.write(f"  Found {len(indexes)} indexes, {len(constraints)} constraints")
                return {"success": True, "message": f"{len(indexes)} indexes, {len(constraints)} constraints"}

            # Execute schema statements
            success_count = 0
            error_count = 0

            with driver.session() as session:
                for stmt in statements:
                    try:
                        session.run(stmt)
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
                                    self.style.ERROR(f"    ✗ Error: {error_msg[:80]}")
                                )

            self.stdout.write(f"  Applied {success_count} statements, {error_count} errors")

            if error_count > 0:
                return {"success": False, "message": f"{success_count} applied, {error_count} errors"}

            return {"success": True, "message": f"{success_count} statements applied"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    def _handle_postgres(self, verify_only: bool, verbose: bool) -> dict:
        """Initialize or verify PostgreSQL schemas."""
        self.stdout.write("\n🐘 PostgreSQL Schema Initialization")
        self.stdout.write("-" * 40)

        try:
            from agentx_ai.kit.agent_memory.connections import get_postgres_session
            from sqlalchemy import text
        except ImportError as e:
            return {"success": False, "message": f"Import error: {e}"}

        # Load schema file (api/agentx_ai/management/commands/ -> project_root/queries/)
        project_root = Path(__file__).parent.parent.parent.parent.parent
        queries_dir = project_root / "queries"
        schema_file = queries_dir / "postgres_builder.sql"

        if not schema_file.exists():
            return {"success": False, "message": f"Schema file not found: {schema_file}"}

        schema_content = schema_file.read_text()

        if verify_only:
            try:
                with get_postgres_session() as session:
                    # Check for key tables
                    result = session.execute(text("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name IN (
                            'conversation_logs', 'memory_timeline', 'tool_invocations',
                            'user_profiles', 'memory_audit_log', 'schema_version'
                        )
                    """))
                    tables = [r[0] for r in result.fetchall()]

                    # Check for channel columns
                    result = session.execute(text("""
                        SELECT table_name, column_name FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND column_name = 'channel'
                    """))
                    channel_cols = [(r[0], r[1]) for r in result.fetchall()]

                self.stdout.write(f"  Found {len(tables)} memory tables")
                self.stdout.write(f"  Found {len(channel_cols)} tables with 'channel' column")
                return {
                    "success": True,
                    "message": f"{len(tables)} tables, {len(channel_cols)} with channel",
                }
            except Exception as e:
                return {"success": False, "message": str(e)}

        # Execute schema (PostgreSQL handles IF NOT EXISTS gracefully)
        try:
            with get_postgres_session() as session:
                # Execute the entire schema as a single transaction
                # Split by statements for better error reporting
                session.execute(text(schema_content))
                session.commit()

                # Verify key components
                result = session.execute(text("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN (
                        'conversation_logs', 'memory_timeline', 'tool_invocations',
                        'user_profiles', 'memory_audit_log', 'schema_version'
                    )
                """))
                tables = [r[0] for r in result.fetchall()]

                # Get schema version
                result = session.execute(text(
                    "SELECT version, description FROM schema_version ORDER BY version DESC LIMIT 1"
                ))
                version_row = result.fetchone()
                if version_row:
                    self.stdout.write(f"  Schema version: {version_row[0]} ({version_row[1]})")

                self.stdout.write(f"  ✓ Created/verified {len(tables)} tables")

                # Check audit log partitions
                result = session.execute(text("""
                    SELECT COUNT(*) FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname LIKE 'memory_audit_log_%'
                    AND n.nspname = 'public'
                """))
                partition_count = result.scalar()
                self.stdout.write(f"  ✓ Created/verified {partition_count} audit log partitions")

            return {"success": True, "message": f"{len(tables)} tables, {partition_count} partitions"}

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
            redis_version = info.get("redis_version", "unknown")
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
            used_memory = info_memory.get("used_memory_human", "unknown")
            max_memory = info_memory.get("maxmemory_human", "not set")
            self.stdout.write(f"\n  Memory: {used_memory} used / {max_memory} max")

            # List any existing keys (just count)
            key_count = client.dbsize()
            self.stdout.write(f"  Keys in database: {key_count}")

            return {"success": True, "message": f"Connected, v{redis_version}, {key_count} keys"}

        except Exception as e:
            return {"success": False, "message": str(e)}

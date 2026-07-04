"""
Single-process container bootstrap: everything the api container must do
before uvicorn, in ONE interpreter.

Replaces the entrypoint's four separate processes (Django migrate, alembic
upgrade, init_memory_schema, setup_auth --check), each of which paid full
import cost. Phases:

    1. Django ORM migrations           (call_command("migrate"))
    2. Memory PostgreSQL via Alembic   (alembic.command.upgrade, in-process)
    3. Memory schema fast path         (version stamps + index states; falls
                                        back to full init_memory_schema)
    4. Embedding-warmup signal         (HF cache probe — never loads torch)
    5. Auth setup hint                 (auth_service.is_setup_required())

The fast path never loads the embedding model, so warm boots take seconds
and no watchdog is needed around this command (the entrypoint wraps only
the explicit warmup_embeddings step, and only when warmup=needed).

Stdout contract (greppable by docker/entrypoint.sh; lines are emitted only
by this command, so nested command output can't forge them):

    BOOTSTRAP django_migrate=ok
    BOOTSTRAP alembic=ok head=<rev>
    BOOTSTRAP memory_schema=verified|initialized
    BOOTSTRAP warmup=needed|cached|remote
    BOOTSTRAP auth=setup_required|configured
    BOOTSTRAP_RESULT ok|failed

Exit codes: 0 ok; 1 transient failure (entrypoint retries); 2 configuration
error such as an unresolvable alembic.ini (entrypoint fails fast).

Usage:
    python manage.py bootstrap            # fast path when stamps match
    python manage.py bootstrap --full     # force full init_memory_schema
"""

import os
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from .init_memory_schema import VECTOR_INDEXES
from .migrate_schema import _extract_version, get_neo4j_version


def _find_alembic_ini() -> Path | None:
    """AGENTX_ALEMBIC_INI override, else walk up from this file (repo root in
    checkouts, /app in the container — same idiom as init_memory_schema)."""
    override = os.environ.get("AGENTX_ALEMBIC_INI")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "alembic.ini"
        if candidate.is_file():
            return candidate
    return None


# Weight files that prove the local embedding model is fully cached. A future
# model shipping only sharded safetensors would miss both names — that costs
# one redundant warmup run, never a failure.
_MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


class Command(BaseCommand):
    help = "One-process container bootstrap: Django + Alembic migrations, memory schema, warmup signal, auth hint"

    def add_arguments(self, parser):
        parser.add_argument(
            "--full",
            action="store_true",
            help="Skip the stamp fast path and run full init_memory_schema",
        )

    def handle(self, *args, **options):
        self.stdout.write("Bootstrap: applying database migrations + schema init...")

        # 1) Django ORM (SQLite) migrations.
        call_command("migrate", interactive=False, verbosity=1)
        self.stdout.write("BOOTSTRAP django_migrate=ok")

        # 2) Memory PostgreSQL via Alembic, in-process (env.py imports no Django).
        head = self._alembic_upgrade()
        self.stdout.write(f"BOOTSTRAP alembic=ok head={head}")

        # 3) Neo4j + Redis: stamp fast path, full init on any miss.
        if not options["full"] and self._memory_schema_current():
            self.stdout.write("BOOTSTRAP memory_schema=verified")
        else:
            call_command("init_memory_schema")
            self.stdout.write("BOOTSTRAP memory_schema=initialized")

        # 4) Embedding warmup signal for the entrypoint.
        self.stdout.write(f"BOOTSTRAP warmup={self._warmup_signal()}")

        # 5) Auth hint.
        from agentx_ai.auth.service import get_auth_service

        auth = "setup_required" if get_auth_service().is_setup_required() else "configured"
        self.stdout.write(f"BOOTSTRAP auth={auth}")

        self.stdout.write("BOOTSTRAP_RESULT ok")

    def _alembic_upgrade(self) -> str:
        ini = _find_alembic_ini()
        if ini is None:
            self.stdout.write("BOOTSTRAP alembic=failed reason=\"alembic.ini not found\"")
            self.stdout.write("BOOTSTRAP_RESULT failed")
            raise CommandError(
                "alembic.ini not found (set AGENTX_ALEMBIC_INI or run from a checkout/image layout)",
                returncode=2,
            )
        from alembic import command as alembic_command
        from alembic.config import Config

        alembic_command.upgrade(Config(str(ini)), "head")

        # Report the reached head for the contract line.
        from alembic.script import ScriptDirectory

        return ScriptDirectory.from_config(Config(str(ini))).get_current_head() or "?"

    def _memory_schema_current(self) -> bool:
        """Cheap warm-boot stamps: Neo4j _SchemaMeta.version == latest on-disk
        migration, all vector indexes ONLINE, Redis answers PING. No model
        load, no writes. Any doubt → False (full init is idempotent)."""
        try:
            from agentx_ai.kit.agent_memory.connections import Neo4jConnection, RedisConnection

            migrations_dir = Path(__file__).resolve().parents[4] / "queries" / "neo4j_migrations"
            versions = [
                v for f in migrations_dir.glob("*.cypher")
                if (v := _extract_version(f.name)) is not None
            ]
            latest = max(versions, default=0)

            driver = Neo4jConnection.get_driver()
            with driver.session() as session:
                current = get_neo4j_version(session)
                if current < latest:
                    self.stdout.write(
                        f"  memory stamps: Neo4j schema v{current} < v{latest} — full init"
                    )
                    return False
                index_states = {
                    r["name"]: r["state"]
                    for r in session.run("SHOW INDEXES YIELD name, state RETURN name, state")
                }
            stale = [
                n for n in VECTOR_INDEXES
                if index_states.get(n) != "ONLINE"
            ]
            if stale:
                self.stdout.write(f"  memory stamps: vector indexes not online: {stale} — full init")
                return False

            if not RedisConnection.get_client().ping():
                return False

            self.stdout.write(f"  memory stamps: Neo4j v{current}, {len(VECTOR_INDEXES)} indexes ONLINE, Redis ok")
            return True
        except Exception as exc:  # noqa: BLE001 — any stamp doubt falls back to full init
            self.stdout.write(f"  memory stamps unavailable ({exc}) — full init")
            return False

    def _warmup_signal(self) -> str:
        """'remote' for API providers; for local, 'cached' only when the model
        weights are provably in the HF cache (probe via huggingface_hub — no
        torch / sentence_transformers import). Anything uncertain → 'needed':
        a spurious warmup costs ~30s once; a missed one only means a slow
        first request, never a hang (the entrypoint watchdog wraps warmup)."""
        from agentx_ai.kit.agent_memory.config import get_settings

        settings = get_settings()
        if settings.embedding_provider != "local":
            return "remote"
        try:
            from huggingface_hub import try_to_load_from_cache

            for filename in _MODEL_WEIGHT_FILES:
                if isinstance(
                    try_to_load_from_cache(settings.local_embedding_model, filename), str
                ):
                    return "cached"
        except Exception:  # noqa: BLE001 — cache probe is best-effort
            pass
        return "needed"

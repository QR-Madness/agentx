"""Cluster-wide snapshot / wipe / restore utilities shared by the eval harnesses.

Extracted from ``eval_consolidation`` (which keeps thin delegators) so that
``eval_recall`` and future debug-harness commands reuse one implementation.
Snapshots are text-only bundles of every user's full memory (embeddings are
regenerated on restore) — see ``portability.MemoryExporter``/``MemoryImporter``.

⚠️  ``wipe_cluster``/``restore_snapshot`` are DESTRUCTIVE (restore wipes first
to remove eval residue). Callers own the safety gating.
"""

import json
from collections.abc import Callable
from datetime import datetime, UTC
from pathlib import Path

Log = Callable[[str], object]


def _noop_log(_msg: str) -> None:
    return None


def count_conversations() -> int:
    """Number of conversations in the connected cluster."""
    from agentx_ai.kit.agent_memory.memory.cross_user import CrossUserStore
    return CrossUserStore().count_conversations()


def wipe_cluster(log: Log = _noop_log, err: Log = _noop_log) -> None:
    """DESTROY all memory data: every Neo4j node + the PG mirror tables."""
    from agentx_ai.kit.agent_memory.memory.cross_user import CrossUserStore
    result = CrossUserStore().wipe_all()
    if result.pg_error:
        err(f"  postgres wipe note: {result.pg_error}")
    log(f"  wiped Neo4j nodes={result.graph_nodes}, postgres rows={result.pg_rows}")


def list_user_ids() -> list[str]:
    """Every user id in the store. Snapshots are cluster-wide because the wipe is."""
    from agentx_ai.kit.agent_memory.memory.cross_user import CrossUserStore
    return CrossUserStore().list_user_ids()


def testing_core_path(name: str) -> Path:
    """Vault path for a named **Testing Core** — a promoted, reusable snapshot.

    Naming a snapshot turns a throwaway eval backup into a versioned corpus
    fixture (``data/vault/testing/{name}.json``) the harnesses can restore
    repeatedly — see ``todo/backlog/cores.md`` §Testing cores.
    """
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name).strip("-.")
    if not safe:
        raise ValueError(f"Unusable testing-core name: {name!r}")
    return Path("data") / "vault" / "testing" / f"{safe}.json"


def make_snapshot(path: Path | str, log: Log = _noop_log) -> Path:
    """Export every user's full memory into one bundle file (text-only;
    embeddings are regenerated on restore)."""
    from agentx_ai.kit.agent_memory.portability import MemoryExporter

    users = []
    for uid in list_user_ids():
        export = MemoryExporter(uid, channel="_all").export()
        users.append(export.model_dump(mode="json"))
    bundle = {
        "snapshot_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "users": users,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    log(f"  snapshot: {len(users)} user(s) → {path}")
    return path


def restore_snapshot(path: Path | str, log: Log = _noop_log, err: Log = _noop_log) -> int:
    """Wipe the cluster (removes eval residue) and re-import a snapshot bundle.

    Returns the number of restored nodes.
    """
    from agentx_ai.kit.agent_memory.portability import MemoryImporter

    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    users = bundle.get("users", [])
    wipe_cluster(log, err)
    restored = 0
    for env in users:
        summary = MemoryImporter(env["user_id"]).import_export(env, mode="merge")
        restored += sum(c["total"] for c in summary["imported"].values())
    log(f"  restored {len(users)} user(s), {restored} node(s) from {path}")
    return restored

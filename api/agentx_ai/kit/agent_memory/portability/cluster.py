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
    """Number of Conversation nodes in the connected cluster."""
    from agentx_ai.kit.agent_memory.connections import Neo4jConnection
    with Neo4jConnection.session() as s:
        rec = s.run("MATCH (c:Conversation) RETURN count(c) AS n").single()
    return rec["n"] if rec else 0


def wipe_cluster(log: Log = _noop_log, err: Log = _noop_log) -> None:
    """DESTROY all memory data: every Neo4j node + the PG mirror tables."""
    from agentx_ai.kit.agent_memory.connections import Neo4jConnection, get_postgres_session
    from sqlalchemy import text
    with Neo4jConnection.session() as s:
        rec = s.run("MATCH (n) DETACH DELETE n RETURN count(n) AS n").single()
        neo = rec["n"] if rec else 0
    pg = 0
    try:
        with get_postgres_session() as ses:
            for tbl in ("conversation_logs", "memory_audit_log"):
                pg += ses.execute(text(f"DELETE FROM {tbl}")).rowcount  # pyright: ignore[reportAttributeAccessIssue]
    except Exception as e:
        err(f"  postgres wipe note: {e}")
    log(f"  wiped Neo4j nodes={neo}, postgres rows={pg}")


def list_user_ids() -> list[str]:
    """Every User id in the graph. Snapshots are cluster-wide because the wipe is."""
    from agentx_ai.kit.agent_memory.connections import Neo4jConnection
    with Neo4jConnection.session() as s:
        return [r["uid"] for r in s.run("MATCH (u:User) RETURN u.id AS uid") if r["uid"]]


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

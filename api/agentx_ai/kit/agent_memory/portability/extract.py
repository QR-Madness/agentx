"""Extract — export with verified deletion (Slice 1 of the Cores arc).

The hard invariant, in order and non-negotiable:

    export → write → verify → wipe → receipt

The wipe never runs until the written artifact re-parses cleanly from disk and
its per-family counts reconcile against the live stores. The export runs in
**exact-channel** mode (``include_global=False``) so the artifact ≡ the wiped
set — the equality verify depends on. Artifacts land in the vault
(``data/vault/``, bind-mounted → survives containers) beside a ``.receipt.json``
sidecar; nothing hard-deletes them automatically (the grace window).

While an extract runs, its channels are **consolidation-frozen** via Redis keys
(``extract_freeze:{user_id}:{channel}``) so a sweep can't land facts mid-extract
(silent loss). Deeper write-freezes (``store_turn``) are deliberately deferred —
see ``todo/backlog/cores.md``.

The verify step is envelope-agnostic on purpose: a per-family
``(name, count_query)`` table, so future core envelopes (Agent/Project Cores)
plug into the same verify-before-wipe machinery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, UTC
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field
from sqlalchemy import text
from typing import LiteralString

from ..audit import MemoryAuditLogger, MemoryType, OperationType
from ..connections import Neo4jConnection, RedisConnection, get_postgres_session
from ..query_utils import CypherFilterBuilder, SQLFilterBuilder
from .exporter import MemoryExporter
from .importer import wipe_channels, wipe_pg_mirror
from .schema import MemoryExport

logger = logging.getLogger(__name__)

# Freeze TTL is a safety net — the finally-block thaw is the real cleanup.
FREEZE_TTL_S = 600
_FREEZE_KEY = "extract_freeze:{user_id}:{channel}"

DEFAULT_VAULT_DIR = Path("data") / "vault"

# Envelope-agnostic verify table: (family, live-count Cypher). Every query
# takes $user_id + $channels and must count with the same exact-channel scope
# the export and wipe use (include_global=False).
def _verify_families() -> list[tuple[str, str]]:
    return [
        ("conversations", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
            WHERE true {_chan_clause('c')}
            RETURN count(c) AS c"""),
        ("turns", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                  -[:HAS_TURN]->(t:Turn)
            WHERE true {_chan_clause('t')}
            RETURN count(t) AS c"""),
        ("entities", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_ENTITY]->(e:Entity)
            WHERE true {_chan_clause('e')}
            RETURN count(e) AS c"""),
        ("facts", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_FACT]->(f:Fact)
            WHERE true {_chan_clause('f')}
            RETURN count(f) AS c"""),
        ("goals", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_GOAL]->(g:Goal)
            WHERE true {_chan_clause('g')}
            RETURN count(g) AS c"""),
        ("strategies", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_STRATEGY]->(s:Strategy)
            WHERE true {_chan_clause('s')}
            RETURN count(s) AS c"""),
        ("procedures", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_PROCEDURE]->(p:Procedure)
            WHERE true {_chan_clause('p')}
            RETURN count(p) AS c"""),
        ("tool_invocations", f"""
            MATCH (u:User {{id: $user_id}})-[:HAS_CONVERSATION]->(c:Conversation)
                  -[:USED_TOOL]->(inv:ToolInvocation)
            WHERE true {_chan_clause('inv')}
            RETURN count(inv) AS c"""),
    ]


def _chan_clause(alias: str) -> str:
    """Exact-channel inline clause bound to $channels (never includes _global).

    The builder shapes the clause from the list form only — actual channel
    values bind at query time through the ``$channels`` parameter.
    """
    return CypherFilterBuilder(alias).add_channel_filter(
        ["_shape_only_"], include_global=False
    ).build_inline()


class ExtractError(ValueError):
    """Bad extract request (channels missing / '_all' not allowed in v1)."""


class ExtractReceipt(BaseModel):
    """What an extract did — written beside the artifact and audit-logged."""

    user_id: str
    channels: list[str]
    file: str | None = None
    sha256: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    wiped_counts: dict[str, int] = Field(default_factory=dict)
    exported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verified: bool = False
    wiped: bool = False
    dry_run: bool = False
    error: str | None = None


# -- freeze (consolidation quiesce, v1 scope) ---------------------------


def freeze_key(user_id: str, channel: str) -> str:
    return _FREEZE_KEY.format(user_id=user_id, channel=channel)


def freeze_channels(user_id: str, channels: list[str]) -> None:
    client = RedisConnection.get_client()
    for chan in channels:
        client.set(freeze_key(user_id, chan), "1", ex=FREEZE_TTL_S)


def thaw_channels(user_id: str, channels: list[str]) -> None:
    client = RedisConnection.get_client()
    for chan in channels:
        client.delete(freeze_key(user_id, chan))


def is_channel_frozen(user_id: str, channel: str) -> bool:
    """True while an extract holds `channel`. Fail-open: Redis trouble must
    never stall consolidation (the extract's verify step still protects it)."""
    try:
        return bool(RedisConnection.get_client().exists(freeze_key(user_id, channel)))
    except Exception:  # noqa: BLE001
        return False


# -- pipeline -----------------------------------------------------------


def _normalize_channels(channels: list[str] | str | None) -> list[str]:
    if isinstance(channels, str):
        channels = [channels]
    chans = [c for c in (channels or []) if c and c.strip()]
    if not chans:
        raise ExtractError("Extract requires an explicit channel list.")
    if "_all" in chans:
        raise ExtractError(
            "Extract of '_all' is not supported — name the channels explicitly "
            "(cluster-wide snapshots live in portability.cluster)."
        )
    return list(dict.fromkeys(chans))


def _artifact_path(vault_dir: Path, channels: list[str]) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9_+-]+", "-", "+".join(channels)).strip("-_") or "core"
    return vault_dir / f"memcore_{slug[:80]}_{ts}.json"


def _write_artifact(path: Path, export: MemoryExport) -> str:
    """Write the envelope and return the file's sha256 (of the bytes on disk)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(export.model_dump_json(indent=2), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_counts(user_id: str, channels: list[str]) -> dict[str, int]:
    """Count every family in the live stores under the exact extract scope."""
    counts: dict[str, int] = {}
    params = {"user_id": user_id, "channels": channels}
    with Neo4jConnection.session() as session:
        for family, query in _verify_families():
            rec = session.run(cast(LiteralString, query), **params).single()
            counts[family] = (rec["c"] if rec else 0) or 0

    for table, family in (
        ("conversation_logs", "pg_conversation_logs"),
        ("tool_invocations", "pg_tool_invocations"),
    ):
        builder = SQLFilterBuilder().add_channel_filter(channels, include_global=False)
        with get_postgres_session() as session:
            row = session.execute(
                text(f"SELECT count(*) AS c FROM {table} {builder.build()}"),  # noqa: S608
                builder.params,
            ).first()
            counts[family] = (row.c if row else 0) or 0
    return counts


def verify_artifact(
    path: Path, live_counts: dict[str, int]
) -> tuple[bool, dict[str, int], list[str]]:
    """Re-read the artifact from disk and reconcile it against the live stores.

    Returns ``(ok, artifact_counts, mismatches)``. Envelope-agnostic: compares
    exactly the families present in both count dicts.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    envelope = MemoryExport.model_validate(payload)
    artifact_counts = envelope.counts()
    mismatches = [
        f"{family}: artifact={artifact_counts[family]} live={live}"
        for family, live in live_counts.items()
        if family in artifact_counts and artifact_counts[family] != live
    ]
    return (not mismatches, artifact_counts, mismatches)


def extract_memory(
    user_id: str,
    channels: list[str] | str,
    vault_dir: Path | str | None = None,
    dry_run: bool = False,
) -> ExtractReceipt:
    """Extract `channels` for `user_id`: export → write → verify → wipe → receipt.

    ``dry_run`` returns the counts + prospective vault path without writing or
    wiping anything. On any verify mismatch the artifact is kept and NOTHING is
    deleted (``verified=False`` on the receipt).
    """
    chans = _normalize_channels(channels)
    vault = Path(vault_dir) if vault_dir else DEFAULT_VAULT_DIR
    receipt = ExtractReceipt(user_id=user_id, channels=chans, dry_run=dry_run)

    if dry_run:
        receipt.counts = _live_counts(user_id, chans)
        receipt.file = str(_artifact_path(vault, chans))
        return receipt

    freeze_channels(user_id, chans)
    try:
        # 1. Export — exact-channel scope: the artifact must equal the wiped set.
        export = MemoryExporter(
            user_id=user_id, channels=chans, include_global=False
        ).export()

        # 2. Write + hash.
        path = _artifact_path(vault, chans)
        receipt.file = str(path)
        receipt.sha256 = _write_artifact(path, export)

        # 3. Verify from disk against the live stores. A file that won't even
        #    re-parse is a verify failure, not a crash — the artifact is kept
        #    and nothing gets deleted either way.
        try:
            ok, artifact_counts, mismatches = verify_artifact(
                path, _live_counts(user_id, chans)
            )
        except Exception as e:  # noqa: BLE001
            ok, artifact_counts, mismatches = False, {}, [f"unreadable artifact: {e}"]
        receipt.counts = artifact_counts
        receipt.verified = ok
        if not ok:
            receipt.error = "verify failed: " + "; ".join(mismatches)
            logger.error(
                "Extract verify FAILED (user=%s channels=%s): %s — artifact kept, "
                "nothing deleted", user_id, chans, receipt.error,
            )
            return receipt

        # 4. Wipe — only now, and only the verified scope.
        with Neo4jConnection.session() as session:
            tx = session.begin_transaction()
            try:
                receipt.wiped_counts = wipe_channels(tx, user_id, chans)
                tx.commit()
            except Exception:
                tx.rollback()
                raise
        receipt.wiped_counts.update(wipe_pg_mirror(chans))
        receipt.wiped = True
        return receipt
    finally:
        thaw_channels(user_id, chans)
        _finish_receipt(receipt)


def _finish_receipt(receipt: ExtractReceipt) -> None:
    """Persist the receipt beside the artifact + audit-log the operation."""
    if receipt.file and not receipt.dry_run:
        try:
            receipt_path = Path(receipt.file).with_suffix(".receipt.json")
            receipt_path.write_text(
                receipt.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as e:  # keep the extract result even if the sidecar fails
            logger.warning("Could not write extract receipt sidecar: %s", e)
    try:
        MemoryAuditLogger().log_write(
            operation=OperationType.EXTRACT.value,
            memory_type=MemoryType.COMPOSITE.value,
            user_id=receipt.user_id,
            channel=",".join(receipt.channels),
            success=receipt.dry_run or receipt.wiped,
            error_message=receipt.error,
            metadata={
                "file": receipt.file,
                "sha256": receipt.sha256,
                "counts": receipt.counts,
                "wiped_counts": receipt.wiped_counts,
                "verified": receipt.verified,
                "dry_run": receipt.dry_run,
            },
        )
    except Exception as e:  # noqa: BLE001 — audit must not mask the receipt
        logger.warning("Could not audit-log extract: %s", e)

    logger.info(
        "Extract (user=%s channels=%s dry_run=%s verified=%s wiped=%s): %s",
        receipt.user_id, receipt.channels, receipt.dry_run, receipt.verified,
        receipt.wiped, receipt.file,
    )

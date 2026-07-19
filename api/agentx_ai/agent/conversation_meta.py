"""
User-set conversation metadata — custom title + archived flag.

Conversation titles were always *derived* (first user message, truncated) and
archiving didn't exist. This module owns the durable overrides behind the
Ambassador v3 conversation-management belt (rename/archive/delete as
user-confirmed writes) and the manual rename/restore actions in the client.

Postgres-only (table ``conversation_meta``, Alembic 0007) with no Redis tier:
the listing paths that consume it already hit Postgres (they LEFT JOIN the
table), and a TTL'd cache could silently revert a custom title. Reads here
never raise; writes return ``False`` on failure so the confirmed-write
endpoints can report an honest error instead of a phantom success.

INV-2 note: nothing in this module is reachable from the Ambassador's tool
belt — belt tools only *propose*; these writes execute solely inside the
user-confirmed HTTP endpoints (and the client's manual actions).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

MAX_TITLE_CHARS = 200

_EMPTY_META: dict[str, Any] = {"title": None, "archived": False}


@contextmanager
def _pg_cursor(commit: bool = False):
    """One round-trip: engine-pooled raw connection, cursor, optional commit;
    close() always returns the connection to the pool."""
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


def get_meta(conversation_id: str) -> dict[str, Any]:
    """The meta record for one conversation: ``{"title", "archived"}``.

    Missing row (or any failure) degrades to the empty meta — a custom title
    is always an overlay, never load-bearing.
    """
    if not conversation_id:
        return dict(_EMPTY_META)
    try:
        with _pg_cursor() as cur:
            cur.execute(
                "SELECT title, archived FROM conversation_meta WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = cur.fetchone()
        if row is None:
            return dict(_EMPTY_META)
        return {"title": row[0], "archived": bool(row[1])}
    except Exception as e:
        logger.debug(f"conversation meta read failed for {conversation_id}: {e}")
        return dict(_EMPTY_META)


def get_meta_bulk(conversation_ids: list[str]) -> dict[str, dict[str, Any]]:
    """One query for many ids → ``{conversation_id: {"title", "archived"}}``.

    Ids without a row are simply absent; any failure degrades to ``{}``.
    """
    ids = [c for c in conversation_ids if c]
    if not ids:
        return {}
    try:
        with _pg_cursor() as cur:
            cur.execute(
                "SELECT conversation_id, title, archived FROM conversation_meta"
                " WHERE conversation_id = ANY(%s)",
                (ids,),
            )
            rows = cur.fetchall()
        return {r[0]: {"title": r[1], "archived": bool(r[2])} for r in rows}
    except Exception as e:
        logger.debug(f"conversation meta bulk read failed: {e}")
        return {}


def set_title(conversation_id: str, title: str) -> bool:
    """Upsert the custom title (validated non-empty, clipped to the cap)."""
    cleaned = (title or "").strip()[:MAX_TITLE_CHARS]
    if not conversation_id or not cleaned:
        return False
    try:
        with _pg_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO conversation_meta (conversation_id, title, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET title = EXCLUDED.title, updated_at = NOW()
                """,
                (conversation_id, cleaned),
            )
        return True
    except Exception as e:
        logger.warning(f"conversation meta title write failed for {conversation_id}: {e}")
        return False


def set_archived(conversation_id: str, archived: bool) -> bool:
    """Upsert the archived flag; ``archived_at`` stamps on archive, clears on restore."""
    if not conversation_id:
        return False
    try:
        with _pg_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO conversation_meta (conversation_id, archived, archived_at, updated_at)
                VALUES (%s, %s, CASE WHEN %s THEN NOW() ELSE NULL END, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET archived = EXCLUDED.archived,
                              archived_at = EXCLUDED.archived_at,
                              updated_at = NOW()
                """,
                (conversation_id, archived, archived),
            )
        return True
    except Exception as e:
        logger.warning(f"conversation meta archive write failed for {conversation_id}: {e}")
        return False


def delete_meta(conversation_id: str) -> None:
    """Drop the meta row (rides the conversation-delete cleanup). Best-effort."""
    if not conversation_id:
        return
    try:
        with _pg_cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM conversation_meta WHERE conversation_id = %s",
                (conversation_id,),
            )
    except Exception as e:
        logger.debug(f"conversation meta delete failed for {conversation_id}: {e}")

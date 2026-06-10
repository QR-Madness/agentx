"""
Rehydrate a conversation's verbatim transcript from durable storage.

The in-memory :class:`SessionManager` is process-local, so a conversation resumed
in a new process, after the session-timeout eviction, or restored from history
starts *cold* — the model would see only memory facts + the new message, not the
actual conversation. This loads the recent user/assistant turns back from the
durable ``conversation_logs`` record into the Session, bounded by a token budget
so a very long thread doesn't load whole.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ..providers.base import Message, MessageRole

logger = logging.getLogger(__name__)

# Rough char→token estimate (mirrors ContextManager.estimate_tokens / helpers).
_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 10

# Hard ceiling on rows pulled from the DB regardless of budget (a guard against
# pathologically long threads); the token budget normally bites first.
_MAX_ROWS = 400

TurnReader = Callable[[str, int], list[tuple[str, str]]]
# A conversation lister returns recent conversations newest-first as dicts.
ConversationLister = Callable[[int], list[dict]]


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN + _PER_MESSAGE_OVERHEAD


def _default_reader(conversation_id: str, limit: int) -> list[tuple[str, str]]:
    """Read up to ``limit`` most-recent user/assistant turns (newest first).

    Tool_call/tool_result rows are intentionally excluded — they're ephemeral and
    the in-turn trajectory compressor already handles tool growth.
    """
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content FROM conversation_logs
                WHERE conversation_id = %s AND role IN ('user', 'assistant')
                ORDER BY turn_index DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            return [(r[0], r[1] or "") for r in cur.fetchall()]
    finally:
        conn.close()


def _default_labeled_reader(
    conversation_id: str, limit: int
) -> list[tuple[str, str, Optional[str]]]:
    """Like :func:`_default_reader` but also returns each turn's producing agent
    name (``metadata->>'agent_name'``, Phase 16 attribution) so a reader can label
    each conversation by its *own* agent — used by the ambassador's tools so a
    cross-conversation survey names the right agent per session, not one global name.
    ``None`` for user turns / unstamped assistant turns.
    """
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, metadata->>'agent_name'
                FROM conversation_logs
                WHERE conversation_id = %s AND role IN ('user', 'assistant')
                ORDER BY turn_index DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            return [(r[0], r[1] or "", r[2]) for r in cur.fetchall()]
    finally:
        conn.close()


def load_recent_labeled_turns(
    conversation_id: str,
    *,
    token_budget: int,
    max_rows: int = _MAX_ROWS,
    reader: Optional[Callable[[str, int], list[tuple[str, str, Optional[str]]]]] = None,
) -> list[tuple[str, str, Optional[str]]]:
    """Recent user/assistant turns as ``(role, content, agent_name)`` in chronological
    order, fitting ``token_budget``. ``agent_name`` is the producing agent's display
    name per turn (``None`` for user/unstamped). Empty on error."""
    reader = reader or _default_labeled_reader
    try:
        rows = reader(conversation_id, max_rows)  # newest-first
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"labeled transcript load failed for {conversation_id}: {e}")
        return []
    picked: list[tuple[str, str, Optional[str]]] = []
    used = 0
    for role, content, agent_name in rows:
        tokens = _estimate_tokens(content)
        if picked and used + tokens > token_budget:
            break
        picked.append((role, content, agent_name))
        used += tokens
    picked.reverse()
    return picked


def _default_conversation_lister(limit: int) -> list[dict]:
    """Read the most-recent conversations (newest first) from ``conversation_logs``.

    Read-only and SELECT-only — backs the ambassador's cross-conversation survey
    ("what have my agents discovered?") without touching the main agent's world.
    """
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    cl.conversation_id::text,
                    MAX(cl.timestamp) AS last_at,
                    COUNT(*) AS message_count,
                    (SELECT content FROM conversation_logs s
                     WHERE s.conversation_id = cl.conversation_id AND s.role = 'user'
                     ORDER BY s.turn_index ASC LIMIT 1) AS first_user,
                    (SELECT content FROM conversation_logs s
                     WHERE s.conversation_id = cl.conversation_id
                     ORDER BY s.turn_index DESC LIMIT 1) AS last_message,
                    (SELECT string_agg(DISTINCT s.metadata->>'agent_name', ', ')
                     FROM conversation_logs s
                     WHERE s.conversation_id = cl.conversation_id
                       AND s.role = 'assistant'
                       AND s.metadata->>'agent_name' IS NOT NULL) AS agents
                FROM conversation_logs cl
                GROUP BY cl.conversation_id
                ORDER BY MAX(cl.timestamp) DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [
                {
                    "conversation_id": r[0],
                    "last_at": str(r[1]) if r[1] is not None else "",
                    "message_count": int(r[2] or 0),
                    "first_user": r[3] or "",
                    "last_message": r[4] or "",
                    "agents": r[5] or "",
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def list_recent_conversations(
    limit: int = 20, *, lister: Optional[ConversationLister] = None
) -> list[dict]:
    """Recent conversations, newest-first. Empty on error (read-only, never raises)."""
    lister = lister or _default_conversation_lister
    try:
        return lister(min(max(1, limit), 50))
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"conversation list failed: {e}")
        return []


def load_recent_turns(
    conversation_id: str,
    *,
    token_budget: int,
    max_rows: int = _MAX_ROWS,
    reader: Optional[TurnReader] = None,
) -> list[Message]:
    """Load the most recent user/assistant turns that fit ``token_budget``.

    Returns ``Message`` objects in **chronological** order (oldest→newest). Walks
    newest→oldest accumulating an estimated token count; always keeps at least the
    single most-recent turn even if it alone exceeds the budget. Empty on error or
    when there's no durable history.
    """
    reader = reader or _default_reader
    try:
        rows = reader(conversation_id, max_rows)  # newest-first
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"transcript load failed for {conversation_id}: {e}")
        return []

    picked: list[Message] = []
    used = 0
    for role, content in rows:
        tokens = _estimate_tokens(content)
        if picked and used + tokens > token_budget:
            break
        picked.append(
            Message(
                role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                content=content,
            )
        )
        used += tokens
    picked.reverse()  # chronological
    return picked


def hydrate_session_from_history(
    session,
    conversation_id: str,
    *,
    token_budget: int,
    reader: Optional[TurnReader] = None,
) -> int:
    """Populate an *empty, not-yet-hydrated* session from durable history (once).

    Idempotent: no-ops when the session already has messages (an active in-process
    conversation) or was already hydrated this lifetime — so it never clobbers live
    state or double-loads. Returns the number of turns loaded.
    """
    if session is None or session.messages or session.metadata.get("hydrated"):
        return 0
    # Mark first so a history-less conversation isn't re-queried every turn.
    session.metadata["hydrated"] = True

    # Restore the persisted rolling summary (covers turns older than the budget),
    # unless the live session already carries one.
    if not session.summary:
        try:
            from .conversation_summary_storage import get_summary
            persisted = get_summary(conversation_id)
            if persisted:
                session.summary = persisted
        except Exception as e:  # pragma: no cover - Redis offline
            logger.debug(f"summary restore skipped: {e}")

    msgs = load_recent_turns(conversation_id, token_budget=token_budget, reader=reader)
    if msgs:
        session.messages.extend(msgs)
        logger.info(
            f"Rehydrated {len(msgs)} prior turns into session '{conversation_id}'"
        )
    return len(msgs)

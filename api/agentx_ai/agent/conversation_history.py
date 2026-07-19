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
from typing import Any
from collections.abc import Callable, Sequence

from ..providers.base import Message, MessageRole
from ..tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Hard ceiling on rows pulled from the DB regardless of budget (a guard against
# pathologically long threads); the token budget normally bites first.
_MAX_ROWS = 400

# A turn reader returns rows newest-first as either (role, content) or — for the
# default reader — (role, content, metadata); `load_recent_turns` handles both arities.
TurnReader = Callable[[str, int], Sequence[tuple]]
# A conversation lister returns recent conversations newest-first as dicts.
ConversationLister = Callable[[int], list[dict]]


def _default_reader(conversation_id: str, limit: int) -> list[tuple[str, str, dict | None]]:
    """Read up to ``limit`` most-recent user/assistant turns (newest first).

    Returns ``(role, content, metadata)`` — the metadata carries vision-input image
    refs (``metadata['images']``) on user turns so they can be re-fed to a vision
    model after a cold rehydration. Tool_call/tool_result rows are excluded — they're
    ephemeral and the in-turn trajectory compressor already handles tool growth.
    """
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, metadata FROM conversation_logs
                WHERE conversation_id = %s AND role IN ('user', 'assistant')
                ORDER BY turn_index DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            return [(r[0], r[1] or "", r[2]) for r in cur.fetchall()]
    finally:
        conn.close()


def load_turn_window(
    conversation_id: str,
    *,
    center_turn: int | None = None,
    radius: int = 6,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Verbatim user/assistant turns from the durable transcript
    (``conversation_logs``) — the digest-expansion read behind
    ``read_thread(conversation_id="current")``.

    This reads the same substrate rehydration uses, NOT episodic memory, so
    aged-out turns stay readable even when the memory system is off or Neo4j
    holds nothing for the conversation. With ``center_turn``: the window of
    turns around that 0-based index. Without it: the EARLIEST turns — digest
    expansion wants the aged-out start of the thread; the recent turns are
    already in the live window. Content is capped per turn like the episodic
    reader. Best-effort: returns ``[]`` on any error (including non-UUID
    session ids, which have no durable transcript).
    """
    from ..kit.agent_memory.connections import PostgresConnection

    try:
        conn: Any = PostgresConnection.get_engine().raw_connection()
    except Exception as e:
        logger.debug(f"turn window unavailable: {e}")
        return []
    rows: list[tuple] = []
    try:
        with conn.cursor() as cur:
            if center_turn is not None:
                lo = max(0, int(center_turn) - radius)
                cur.execute(
                    """
                    SELECT turn_index, role, timestamp, content FROM conversation_logs
                    WHERE conversation_id = %s AND role IN ('user', 'assistant')
                      AND turn_index BETWEEN %s AND %s
                    ORDER BY turn_index ASC
                    """,
                    (conversation_id, lo, int(center_turn) + radius),
                )
            else:
                cur.execute(
                    """
                    SELECT turn_index, role, timestamp, content FROM conversation_logs
                    WHERE conversation_id = %s AND role IN ('user', 'assistant')
                    ORDER BY turn_index ASC
                    LIMIT %s
                    """,
                    (conversation_id, limit),
                )
            rows = cur.fetchall()
    except Exception as e:  # non-UUID id, table missing, PG down — never break the tool
        logger.debug(f"turn window read failed: {e}")
    finally:
        conn.close()
    return [
        {
            "index": r[0],
            "role": r[1],
            "timestamp": str(r[2] or ""),
            "content": (r[3] or "")[:1500],
        }
        for r in rows
    ]


def _default_labeled_reader(
    conversation_id: str, limit: int
) -> list[tuple[str, str, str | None]]:
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
    reader: Callable[[str, int], list[tuple[str, str, str | None]]] | None = None,
) -> list[tuple[str, str, str | None]]:
    """Recent user/assistant turns as ``(role, content, agent_name)`` in chronological
    order, fitting ``token_budget``. ``agent_name`` is the producing agent's display
    name per turn (``None`` for user/unstamped). Empty on error."""
    reader = reader or _default_labeled_reader
    try:
        rows = reader(conversation_id, max_rows)  # newest-first
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"labeled transcript load failed for {conversation_id}: {e}")
        return []
    picked: list[tuple[str, str, str | None]] = []
    used = 0
    for role, content, agent_name in rows:
        tokens = estimate_tokens(content)
        if picked and used + tokens > token_budget:
            break
        picked.append((role, content, agent_name))
        used += tokens
    picked.reverse()
    return picked


def latest_agent_name(conversation_id: str) -> str:
    """The display name of the agent that produced the conversation's most recent
    *stamped* assistant turn (``metadata->>'agent_name'``, Phase 16 attribution), or ``""``.

    Used to recover which agent a conversation belongs to when there's no open client tab
    — e.g. the ambassador relaying a message into a conversation headlessly. Sync;
    never raises (→ ``""``)."""
    from ..kit.agent_memory.connections import PostgresConnection

    try:
        conn: Any = PostgresConnection.get_engine().raw_connection()
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"latest_agent_name connect failed for {conversation_id}: {e}")
        return ""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT metadata->>'agent_name'
                FROM conversation_logs
                WHERE conversation_id = %s AND role = 'assistant'
                  AND metadata->>'agent_name' IS NOT NULL
                ORDER BY turn_index DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
            return (row[0] or "").strip() if row else ""
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"latest_agent_name query failed for {conversation_id}: {e}")
        return ""
    finally:
        conn.close()


def list_conversation_images(conversation_id: str, *, limit: int = 12) -> list[dict]:
    """Images created earlier in this conversation, for history *awareness*.

    Scans the durable ``generate_image`` tool-result turns (each carries the stored
    image's ``doc_id``/``workspace_id``/``prompt``) so the agent can be told an image
    exists and choose to ``view_image`` it — we never re-inject the pixels automatically.
    Newest-first, deduped by doc_id. Read-only; empty on any error."""
    from ..kit.agent_memory.connections import PostgresConnection

    try:
        conn: Any = PostgresConnection.get_engine().raw_connection()
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"conversation image scan connect failed for {conversation_id}: {e}")
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content FROM conversation_logs
                WHERE conversation_id = %s AND role = 'tool_result'
                  AND metadata->>'tool' = 'generate_image'
                ORDER BY turn_index DESC
                LIMIT 100
                """,
                (conversation_id,),
            )
            rows = cur.fetchall()
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"conversation image scan failed for {conversation_id}: {e}")
        return []
    finally:
        conn.close()

    import json as _json

    seen: set[str] = set()
    images: list[dict] = []
    for (content,) in rows:
        try:
            data = _json.loads(content or "{}")
        except (ValueError, TypeError):
            continue
        doc_id = data.get("doc_id") or data.get("document_id")
        if not data.get("success") or not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        images.append({
            "doc_id": doc_id,
            "workspace_id": data.get("workspace_id"),
            "prompt": (data.get("prompt") or "").strip(),
        })
        if len(images) >= limit:
            break
    return images


def render_conversation_images_block(conversation_id: str) -> str:
    """A stable awareness note listing images made earlier in this conversation, so the
    agent knows it can ``view_image`` them. Empty when there are none."""
    images = list_conversation_images(conversation_id)
    if not images:
        return ""
    lines = [
        "Images created earlier in this conversation — to actually see one, call "
        "`view_image(document_id=…)` (you won't see it otherwise):"
    ]
    for img in images:
        desc = f" — {img['prompt']}" if img.get("prompt") else ""
        lines.append(f"- document_id={img['doc_id']}{desc}")
    return "\n".join(lines)


def _default_conversation_lister(limit: int, *, include_archived: bool = False) -> list[dict]:
    """Read the most-recent conversations (newest first) from ``conversation_logs``.

    Read-only and SELECT-only — backs the ambassador's cross-conversation survey
    ("what have my agents discovered?") without touching the main agent's world.
    A ``conversation_meta`` row overlays a custom ``title`` and carries the
    archived flag; archived conversations stay out of the survey unless asked for.
    """
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            # (%s OR NOT archived): include_archived=True keeps every row,
            # False reduces to the archived exclusion — static SQL, bound flag.
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
                       AND s.metadata->>'agent_name' IS NOT NULL) AS agents,
                    MAX(cm.title) AS title,
                    BOOL_OR(COALESCE(cm.archived, FALSE)) AS archived
                FROM conversation_logs cl
                LEFT JOIN conversation_meta cm ON cm.conversation_id = cl.conversation_id::text
                WHERE (%s OR COALESCE(cm.archived, FALSE) = FALSE)
                GROUP BY cl.conversation_id
                ORDER BY MAX(cl.timestamp) DESC
                LIMIT %s
                """,
                (include_archived, limit),
            )
            return [
                {
                    "conversation_id": r[0],
                    "last_at": str(r[1]) if r[1] is not None else "",
                    "message_count": int(r[2] or 0),
                    "first_user": r[3] or "",
                    "last_message": r[4] or "",
                    "agents": r[5] or "",
                    "title": r[6] or "",
                    "archived": bool(r[7]),
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def list_recent_conversations(
    limit: int = 20, *, lister: ConversationLister | None = None,
    include_archived: bool = False,
) -> list[dict]:
    """Recent conversations, newest-first. Empty on error (read-only, never raises).

    ``include_archived`` only reaches the default lister — the injected ``lister``
    seam keeps its one-arg signature.
    """
    if lister is None:
        def _lister(n: int) -> list[dict]:
            return _default_conversation_lister(n, include_archived=include_archived)
        lister = _lister
    try:
        return lister(min(max(1, limit), 50))
    except Exception as e:  # pragma: no cover - DB offline
        logger.debug(f"conversation list failed: {e}")
        return []


def search_conversation_logs(
    query: str,
    *,
    limit: int = 8,
    per_conversation: int = 2,
    include_archived: bool = False,
) -> list[dict]:
    """Lexical full-text search over ``conversation_logs`` (the 0008 GIN index).

    ``websearch_to_tsquery('simple', …)`` semantics: bare words AND together,
    ``"quoted phrases"`` match adjacently, ``-word`` negates. Groups hits by
    conversation (best-rank order), keeping ``per_conversation`` snippet(s) per
    conversation with ``**bold**`` markdown match markers. Returns
    ``[{conversation_id, title, last_at, agents, snippets: [{role, snippet}]}]``.
    Read-only, never raises — empty on error/no match.
    """
    q = (query or "").strip()
    if not q:
        return []
    limit = min(max(1, limit), 20)
    per_conversation = min(max(1, per_conversation), 5)
    try:
        from ..kit.agent_memory.connections import PostgresConnection

        conn: Any = PostgresConnection.get_engine().raw_connection()
        try:
            with conn.cursor() as cur:
                # (%s OR NOT archived) — same bound-flag idiom as the lister.
                # ts_headline recomputes only on the <= limit*per_conversation
                # rows that survive; the GIN expression index answers the @@.
                cur.execute(
                    """
                    WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS tsq),
                    hits AS (
                        SELECT cl.conversation_id, cl.role, cl.content,
                               ts_rank(to_tsvector('simple', cl.content), q.tsq) AS rank,
                               ROW_NUMBER() OVER (
                                   PARTITION BY cl.conversation_id
                                   ORDER BY ts_rank(to_tsvector('simple', cl.content), q.tsq) DESC,
                                            cl.turn_index DESC
                               ) AS rn
                        FROM conversation_logs cl CROSS JOIN q
                        WHERE to_tsvector('simple', cl.content) @@ q.tsq
                    ),
                    ranked_convs AS (
                        SELECT h.conversation_id, MAX(h.rank) AS best_rank
                        FROM hits h
                        LEFT JOIN conversation_meta cm
                               ON cm.conversation_id = h.conversation_id::text
                        WHERE (%s OR COALESCE(cm.archived, FALSE) = FALSE)
                        GROUP BY h.conversation_id
                        ORDER BY best_rank DESC
                        LIMIT %s
                    )
                    SELECT
                        h.conversation_id::text,
                        h.role,
                        ts_headline('simple', h.content, q.tsq,
                                    'MaxWords=18,MinWords=6,MaxFragments=2,'
                                    'FragmentDelimiter= … ,StartSel=**,StopSel=**') AS snippet,
                        (SELECT MAX(s.timestamp) FROM conversation_logs s
                         WHERE s.conversation_id = h.conversation_id) AS last_at,
                        (SELECT cm2.title FROM conversation_meta cm2
                         WHERE cm2.conversation_id = h.conversation_id::text) AS title,
                        (SELECT string_agg(DISTINCT s.metadata->>'agent_name', ', ')
                         FROM conversation_logs s
                         WHERE s.conversation_id = h.conversation_id
                           AND s.role = 'assistant'
                           AND s.metadata->>'agent_name' IS NOT NULL) AS agents
                    FROM hits h
                    JOIN ranked_convs rc ON rc.conversation_id = h.conversation_id
                    CROSS JOIN q
                    WHERE h.rn <= %s
                    ORDER BY rc.best_rank DESC, h.conversation_id, h.rn
                    """,
                    (q, include_archived, limit, per_conversation),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        grouped: dict[str, dict] = {}
        for cid, role, snippet, last_at, title, agents in rows:
            entry = grouped.setdefault(cid, {
                "conversation_id": cid,
                "title": title or "",
                "last_at": str(last_at) if last_at is not None else "",
                "agents": agents or "",
                "snippets": [],
            })
            entry["snippets"].append({"role": role or "", "snippet": snippet or ""})
        return list(grouped.values())
    except Exception as e:  # pragma: no cover - DB offline / index missing
        logger.debug(f"conversation search failed: {e}")
        return []


def load_recent_turns(
    conversation_id: str,
    *,
    token_budget: int,
    max_rows: int = _MAX_ROWS,
    reader: TurnReader | None = None,
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

    # Re-feeding base64 images on every turn is expensive, so only the most-recent K
    # user turns that carry images get them back; older turns rehydrate text-only (the
    # durable transcript still keeps the refs for the client to render).
    refeed_recent = _vision_refeed_recent_turns()
    image_turns_used = 0

    picked: list[Message] = []
    used = 0
    for row in rows:  # newest-first
        role, content = row[0], row[1] or ""
        metadata = row[2] if len(row) > 2 else None
        tokens = estimate_tokens(content)
        if picked and used + tokens > token_budget:
            break
        images = None
        if role == "user" and refeed_recent > 0 and image_turns_used < refeed_recent:
            images = _images_from_metadata(metadata)
            if images:
                image_turns_used += 1
        picked.append(
            Message(
                role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                content=content,
                images=images,
            )
        )
        used += tokens
    picked.reverse()  # chronological
    return picked


def _vision_refeed_recent_turns() -> int:
    """How many recent image-bearing user turns to re-feed on rehydration (config)."""
    try:
        from ..config import get_config_manager

        return int(get_config_manager().get("vision.refeed_recent_turns", 2))
    except Exception:  # pragma: no cover - config unavailable
        return 2


def _images_from_metadata(metadata: Any) -> list | None:
    """Reconstruct ImageRef objects from a turn's persisted ``metadata['images']``.

    Tolerates metadata stored as a JSON string or a dict; returns ``None`` when there
    are no valid image refs. Never raises — a malformed entry is skipped."""
    if not metadata:
        return None
    try:
        from ..providers.base import ImageRef

        if isinstance(metadata, str):
            import json

            metadata = json.loads(metadata)
        raw = (metadata or {}).get("images") or []
        refs = [
            ImageRef(workspace_id=r["workspace_id"], doc_id=r["doc_id"], media_type=r["media_type"])
            for r in raw
            if isinstance(r, dict) and r.get("workspace_id") and r.get("doc_id") and r.get("media_type")
        ]
        return refs or None
    except Exception:  # pragma: no cover - defensive
        return None


def hydrate_session_from_history(
    session,
    conversation_id: str,
    *,
    token_budget: int,
    max_rows: int | None = None,
    reader: TurnReader | None = None,
) -> int:
    """Populate an *empty, not-yet-hydrated* session from durable history (once).

    Idempotent: no-ops when the session already has messages (an active in-process
    conversation) or was already hydrated this lifetime — so it never clobbers live
    state or double-loads. Returns the number of turns loaded.

    ``max_rows`` defaults to ``context.rehydrate_max_turns`` (the knob was
    previously defined but never wired — hydrate was silently bound by the
    module ``_MAX_ROWS``). When the row cap is hit AND no persisted summary was
    restored, ``session.metadata["history_overflow"]`` is set. Turns beyond the
    cap were never loaded, so no compaction pass can cover them — the chat path
    surfaces the flag as an honest ``history_overflow_notice`` ledger block
    (the earliest turns stay reachable via memory recall / ``read_thread``)
    instead of letting them silently vanish (INV-CTX-1).
    """
    if session is None or session.messages or session.metadata.get("hydrated"):
        return 0
    # Mark first so a history-less conversation isn't re-queried every turn.
    session.metadata["hydrated"] = True

    if max_rows is None:
        try:
            from ..config import get_config_manager
            max_rows = int(get_config_manager().get("context.rehydrate_max_turns", _MAX_ROWS))
        except Exception:  # pragma: no cover - defensive
            max_rows = _MAX_ROWS

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

    msgs = load_recent_turns(
        conversation_id, token_budget=token_budget, max_rows=max_rows, reader=reader
    )
    if msgs:
        session.messages.extend(msgs)
        logger.info(
            f"Rehydrated {len(msgs)} prior turns into session '{conversation_id}'"
        )
    # Cap hit with no summary restored → turns beyond the cap have NO coverage.
    # Flag it rather than truncating silently; the JIT summarizer picks it up.
    if len(msgs) >= max_rows and not session.summary:
        session.metadata["history_overflow"] = True
        logger.warning(
            f"Rehydration hit the {max_rows}-row cap for '{conversation_id}' with no "
            "persisted summary — flagged history_overflow (the chat path surfaces an "
            "in-prompt notice; earlier turns stay reachable via recall/read_thread)"
        )
    return len(msgs)

"""Episodic memory on PostgreSQL: ``memory_conversations`` and ``memory_turns``.

The twin of ``memory/episodic.py``. Same public methods and the same shapes back;
what it does differently, it does on purpose (Decisions.md ADR-22):

- **Every call names a user.** The graph took ``user_id=None`` to mean every
  user's turns. Here each call runs in that user's row-level-security scope.
- **A turn position is stored once.** Storing a turn at an index that is taken
  replaces it and keeps its id, so the facts drawn from it still point at it; the
  graph added a second turn at the same position. If the text changed, the turn's
  consolidation marks are cleared, so the new text is read again.
- **The roster is read from the turns.** The graph kept a node per agent and
  conversation; who spoke, and under which name last, is already in the turns.
- **Search is scoped, and complete.** The graph ranked every user's turns and
  filtered afterwards, over-fetching to make up for it. Here the scope is part of
  the query, and a search that the index answers short is answered exactly
  (``nearest_rows``), so fewer results than asked for means there were no more.
- **Rows come back without their vector.** Nothing downstream reads it, and it
  is most of a turn's bytes.

``store_turn_log`` and ``get_conversation_agent_ids`` are inherited: they are the
transcript (``conversation_logs``), which was always in PostgreSQL.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from ...models import Turn
from ...query_utils import SQLFilterBuilder, convert_all_datetimes, validate_time_window
from ..episodic import EpisodicMemory
from ..scope import user_scope
from . import embedder_name, nearest_rows, require_user, similarity, vector_literal

# A turn as the graph returned it: `index`, not the column's name, and no vector.
_TURN_COLUMNS = (
    "id, conversation_id, turn_index AS index, timestamp, role, content, token_count, "
    "channel, user_id, agent_id, agent_name, consolidated, self_consolidated, archived"
)

_ENSURE_CONVERSATION = text("""
    INSERT INTO memory_conversations (user_id, id, channel, agent_id)
    VALUES (:user_id, :conversation_id, :channel, :agent_id)
    ON CONFLICT (user_id, id) DO UPDATE
       SET agent_id = COALESCE(memory_conversations.agent_id, EXCLUDED.agent_id)
""")

_STORE_TURN = text("""
    INSERT INTO memory_turns
        (user_id, id, conversation_id, turn_index, timestamp, role, content, token_count,
         channel, agent_id, agent_name, embedding, embedder)
    VALUES
        (:user_id, :id, :conversation_id, :turn_index, :timestamp, :role, :content, :token_count,
         :channel, :agent_id, :agent_name, CAST(:embedding AS vector), :embedder)
    ON CONFLICT (user_id, conversation_id, turn_index) DO UPDATE
       SET timestamp = EXCLUDED.timestamp,
           role = EXCLUDED.role,
           token_count = EXCLUDED.token_count,
           channel = EXCLUDED.channel,
           agent_id = EXCLUDED.agent_id,
           agent_name = EXCLUDED.agent_name,
           embedding = EXCLUDED.embedding,
           embedder = EXCLUDED.embedder,
           consolidated = CASE WHEN memory_turns.content IS DISTINCT FROM EXCLUDED.content
                               THEN NULL ELSE memory_turns.consolidated END,
           self_consolidated = CASE WHEN memory_turns.content IS DISTINCT FROM EXCLUDED.content
                                    THEN NULL ELSE memory_turns.self_consolidated END,
           content = EXCLUDED.content
    RETURNING id
""")

_ROSTER = text("""
    SELECT agent_id,
           COALESCE(
               (array_agg(agent_name ORDER BY turn_index DESC)
                    FILTER (WHERE agent_name IS NOT NULL))[1],
               agent_id
           ) AS name
      FROM memory_turns
     WHERE conversation_id = :conversation_id AND agent_id IS NOT NULL
     GROUP BY agent_id
     ORDER BY min(turn_index)
""")

_COUNT_TURNS = text("SELECT count(*) FROM memory_turns WHERE conversation_id = :conversation_id")
_DELETE_CONVERSATION = text(
    "DELETE FROM memory_conversations WHERE id = :conversation_id RETURNING id"
)


def _row(mapping: Any) -> dict[str, Any]:
    return dict(mapping)


class PgEpisodicMemory(EpisodicMemory):
    """Conversation history on PostgreSQL. See the module docstring for what
    differs from the graph on purpose."""

    def store_turn(
        self, turn: Turn, user_id: str | None = None, channel: str = "_global",
        agent_id: str | None = None,
    ) -> None:
        """Store `turn` for `user_id` in `channel`, creating its conversation on
        first sight (whose agent is the first one given). A turn already stored at
        that position is replaced, and `turn.id` is set to the id that position
        keeps."""
        user = require_user(user_id)
        with user_scope(user) as session:
            session.execute(_ENSURE_CONVERSATION, {
                "user_id": user, "conversation_id": turn.conversation_id,
                "channel": channel, "agent_id": agent_id,
            })
            turn.id = session.execute(_STORE_TURN, {
                "user_id": user,
                "id": turn.id,
                "conversation_id": turn.conversation_id,
                "turn_index": turn.index,
                "timestamp": turn.timestamp,
                "role": turn.role,
                "content": turn.content,
                "token_count": turn.token_count,
                "channel": channel,
                "agent_id": turn.agent_id,
                "agent_name": (turn.metadata or {}).get("agent_name"),
                "embedding": vector_literal(turn.embedding),
                "embedder": embedder_name() if turn.embedding is not None else None,
            }).scalar_one()

    def get_conversation_roster(
        self, conversation_id: str, user_id: str | None = None,
    ) -> list[dict[str, str]]:
        """The agents that produced turns in a conversation, in order of first
        appearance: ``{"agent_id", "name"}``, the name being the latest one a turn
        carried, or the id when none did."""
        with user_scope(require_user(user_id)) as session:
            rows = session.execute(_ROSTER, {"conversation_id": conversation_id})
            return [{"agent_id": r.agent_id, "name": r.name} for r in rows]

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        user_id: str | None = None,
        time_window_hours: int | None = None,
        channel: str | None = None,
    ) -> list[dict[str, Any]]:
        """The user's `top_k` turns nearest `query_embedding`, best first, within
        the time window and channel (a named channel includes ``_global``)."""
        filters = SQLFilterBuilder().add_channel_filter(channel)
        if time_window_hours is not None:
            filters.add_custom(
                "timestamp > NOW() - make_interval(hours => :window_hours)",
                window_hours=validate_time_window(time_window_hours),
            )
        with user_scope(require_user(user_id)) as session:
            rows = nearest_rows(
                session,
                table="memory_turns",
                columns="id, content, role, timestamp, channel, conversation_id",
                where=filters.build_inline(),
                params=filters.params,
                embedding=query_embedding,
                limit=top_k,
            )
        for row in rows:
            row["score"] = similarity(row.pop("distance"))
        return rows

    def get_conversation(
        self, conversation_id: str, user_id: str | None = None, channel: str | None = None,
    ) -> list[Turn]:
        """Every turn of a conversation, in order, as ``Turn`` models."""
        filters = SQLFilterBuilder().add_channel_filter(channel)
        query = text(f"""
            SELECT {_TURN_COLUMNS} FROM memory_turns
             WHERE conversation_id = :conversation_id {filters.build_inline()}
             ORDER BY turn_index
        """)  # noqa: S608 — fixed fragments; values are bound
        with user_scope(require_user(user_id)) as session:
            rows = session.execute(
                query, {**filters.params, "conversation_id": conversation_id},
            ).mappings().all()
        return [
            Turn(
                id=r["id"], conversation_id=r["conversation_id"], index=r["index"],
                timestamp=r["timestamp"], role=r["role"], content=r["content"],
                token_count=r["token_count"], channel=r["channel"], agent_id=r["agent_id"],
                metadata={"agent_name": r["agent_name"]} if r["agent_name"] else {},
            )
            for r in rows
        ]

    def delete_conversation(self, conversation_id: str, user_id: str | None = None) -> dict[str, int]:
        """Delete the user's conversation and its turns. Returns the counts
        (``turns``, ``conversation``); another user's conversation is not found and
        not touched. The transcript, Redis and sidecars are the caller's to clear."""
        with user_scope(require_user(user_id)) as session:
            turns = session.execute(_COUNT_TURNS, {"conversation_id": conversation_id}).scalar_one()
            gone = session.execute(_DELETE_CONVERSATION, {"conversation_id": conversation_id}).all()
        return {"turns": int(turns or 0), "conversation": len(gone)}

    def get_conversation_turns(
        self, conversation_id: str, user_id: str, limit: int = 3, order: str = "desc",
    ) -> list[dict[str, Any]]:
        """The first or last `limit` turns of a conversation (capped at 20)."""
        direction = "DESC" if order == "desc" else "ASC"
        query = text(f"""
            SELECT {_TURN_COLUMNS} FROM memory_turns
             WHERE conversation_id = :conversation_id
             ORDER BY turn_index {direction}
             LIMIT :limit
        """)  # noqa: S608 — the direction is one of two constants
        with user_scope(require_user(user_id)) as session:
            rows = session.execute(query, {
                "conversation_id": conversation_id, "limit": max(1, min(int(limit), 20)),
            }).mappings().all()
        return [_row(r) for r in rows]

    def get_turn_by_id(self, turn_id: str, user_id: str) -> dict[str, Any] | None:
        """One turn of the user's, with its conversation id and ISO timestamps, or
        None when there is none (another user's included)."""
        query = text(f"SELECT {_TURN_COLUMNS} FROM memory_turns WHERE id = :turn_id")  # noqa: S608 — constant
        with user_scope(require_user(user_id)) as session:
            row = session.execute(query, {"turn_id": turn_id}).mappings().first()
        return convert_all_datetimes(_row(row)) if row else None

    def get_turns_around(
        self, conversation_id: str, user_id: str, center_index: int, radius: int = 6,
    ) -> list[dict[str, Any]]:
        """The turns within `radius` (at most 20) of `center_index`, oldest first."""
        radius = max(0, min(int(radius), 20))
        query = text(f"""
            SELECT {_TURN_COLUMNS} FROM memory_turns
             WHERE conversation_id = :conversation_id AND turn_index BETWEEN :lo AND :hi
             ORDER BY turn_index
        """)  # noqa: S608 — constant
        with user_scope(require_user(user_id)) as session:
            rows = session.execute(query, {
                "conversation_id": conversation_id,
                "lo": max(0, int(center_index) - radius),
                "hi": int(center_index) + radius,
            }).mappings().all()
        return [convert_all_datetimes(_row(r)) for r in rows]

    def get_recent_turns(
        self, user_id: str, hours: int = 24, limit: int = 50, channel: str | None = None,
    ) -> list[dict[str, Any]]:
        """The user's newest turns across conversations within `hours` (at most a
        year), newest first, at most `limit` (capped at 500)."""
        filters = SQLFilterBuilder().add_channel_filter(channel)
        query = text(f"""
            SELECT content, role, timestamp, channel, conversation_id
              FROM memory_turns
             WHERE timestamp > NOW() - make_interval(hours => :hours) {filters.build_inline()}
             ORDER BY timestamp DESC
             LIMIT :limit
        """)  # noqa: S608 — fixed fragments; values are bound
        with user_scope(require_user(user_id)) as session:
            rows = session.execute(query, {
                **filters.params,
                "hours": validate_time_window(hours),
                "limit": max(1, min(int(limit), 500)),
            }).mappings().all()
        return [_row(r) for r in rows]

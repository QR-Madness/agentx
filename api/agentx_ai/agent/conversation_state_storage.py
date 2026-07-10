"""
Structured per-conversation state — the Claude-like working-memory block.

Unlike the free-prose rolling summary (:mod:`conversation_summary_storage`), this
is a STRUCTURED, slot-based object: ``goals`` / ``decisions`` / ``open_threads`` /
``artifacts``, plus a freeform ``narrative`` catch-all so content that fits no
named slot is still captured (a general "thinking platform" conversation, not
just coding). Each entry carries provenance — ``source_turn``, ``author``
(``user`` | ``agent``), ``updated_at`` — so we always know who asserted what.

Slice 1a is **additive**: the object rides alongside the prose summary and is
injected as its own ledger block; it does NOT yet become the compaction target
(that is Slice 1c). Stored in Redis keyed by conversation with a 30-day TTL, like
checkpoints/scratchpad; the durable copy of the underlying turns remains the raw
``conversation_logs``.

The pure ``apply_update``/``render_state`` transforms hold the slot bounds and
formatting so they can be unit-tested without Redis; ``update_slot`` wraps them
with get→apply→save.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STATE_PREFIX = "conv_state:"
STATE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
STATE_NEGATIVE_TTL_SECONDS = 60 * 10  # both-tiers miss cached briefly (bounds PG re-checks)
PG_BACKOFF_SECONDS = 60.0  # durable-tier breaker: skip PG this long after a failure

# The one true spelling of the digest-expansion call — shared by the digest
# render and the rehydration-overflow notice so the taught signature can't
# drift between surfaces. (The memory-tools coaching layer repeats it as
# *versioned* prose: layer content only changes via a default_version bump.)
READ_THREAD_CURRENT_CALL = 'read_thread(conversation_id="current", center_turn=N)'

# Named slots + the freeform catch-all. Tuple order is render order.
NAMED_SLOTS: tuple[str, ...] = ("goals", "decisions", "open_threads", "artifacts")
SLOTS: tuple[str, ...] = (*NAMED_SLOTS, "narrative")

SLOT_LABELS: dict[str, str] = {
    "goals": "Goals",
    "decisions": "Decisions",
    "open_threads": "Open threads",
    "artifacts": "Artifacts",
    "narrative": "Narrative",
}

# Bounds — a slot can't grow unbounded (drift + poisoning defense). The narrative
# tolerates longer entries since it's the catch-all for arbitrary content.
MAX_ENTRIES_PER_SLOT = 20
MAX_ENTRY_CHARS = 600
MAX_NARRATIVE_CHARS = 1200

Author = Literal["user", "agent"]


class StateEntry(BaseModel):
    """A single slot entry with provenance."""

    text: str
    author: Author = "agent"
    source_turn: int | None = None
    updated_at: str = ""


class ConversationState(BaseModel):
    """Structured, slot-based working memory for one conversation."""

    goals: list[StateEntry] = Field(default_factory=list)
    decisions: list[StateEntry] = Field(default_factory=list)
    open_threads: list[StateEntry] = Field(default_factory=list)
    artifacts: list[StateEntry] = Field(default_factory=list)
    narrative: list[StateEntry] = Field(default_factory=list)
    # Rolling summary of turns aged out of the verbatim window (Slice 1c). Unlike
    # the append-capped `narrative` slot, this is a SINGLE field re-summarized in
    # place each compaction, so it stays bounded WITHOUT dropping old coverage —
    # it is the compaction target that lets the state object cover INV-CTX-1.
    digest: str = ""

    def entries(self, slot: str) -> list[StateEntry]:
        return getattr(self, slot)

    def is_empty(self) -> bool:
        return not self.digest and not any(self.entries(s) for s in SLOTS)


def _entry_cap(slot: str) -> int:
    return MAX_NARRATIVE_CHARS if slot == "narrative" else MAX_ENTRY_CHARS


def _coerce_author(author: str, default: Author = "agent") -> Author:
    """Only ``user`` or ``agent`` may author state — never ``system``/``tool``/``web``
    (Slice 4 hardening). A forged or injected author coerces to a safe default so a
    poisoned entry can never masquerade as an authoritative system instruction; every
    entry stays attributable to a human or the agent."""
    if author == "user":
        return "user"
    if author == "agent":
        return "agent"
    return default


def apply_update(
    state: ConversationState,
    slot: str,
    entries: list[str],
    *,
    author: Author = "agent",
    source_turn: int | None = None,
    replace: bool = False,
) -> ConversationState:
    """Return a new state with ``entries`` written to ``slot`` (pure — no I/O).

    ``replace=True`` supersedes the whole slot (the author is asserting the
    current set) instead of appending. Empty/blank strings are dropped, each
    entry is truncated to the slot's char cap, and the slot is bounded to the
    newest ``MAX_ENTRIES_PER_SLOT`` so it can't grow without limit.
    """
    if slot not in SLOTS:
        raise ValueError(f"Unknown conversation-state slot: {slot!r} (valid: {', '.join(SLOTS)})")

    now = datetime.now(UTC).isoformat()
    cap_chars = _entry_cap(slot)
    author = _coerce_author(author)
    fresh = [
        StateEntry(
            text=t.strip()[:cap_chars],
            author=author,
            source_turn=source_turn,
            updated_at=now,
        )
        for t in entries
        if t and t.strip()
    ]

    updated = state.model_copy(deep=True)
    current = [] if replace else list(updated.entries(slot))
    combined = (current + fresh)[-MAX_ENTRIES_PER_SLOT:]
    setattr(updated, slot, combined)
    return updated


def set_slot(
    state: ConversationState,
    slot: str,
    entries: list[Any],
) -> ConversationState:
    """Return a new state with ``slot`` set to exactly ``entries`` (pure — no I/O).

    The **user-edit primitive** (Slice 1b), distinct from :func:`apply_update` (the
    agent's append/replace-with-strings tool). Accepts full entry objects
    (``StateEntry`` or ``{text, author?, source_turn?}`` dicts) so a client can
    round-trip existing provenance while adding/editing/removing entries in one
    PATCH. ``updated_at`` is re-stamped now; blanks are dropped; each entry is
    truncated to the slot's char cap; the slot is bounded to the newest
    ``MAX_ENTRIES_PER_SLOT``. Unknown authors coerce to ``user`` (a user-driven
    edit doesn't get to mint an agent-authored entry).
    """
    if slot not in SLOTS:
        raise ValueError(f"Unknown conversation-state slot: {slot!r} (valid: {', '.join(SLOTS)})")

    now = datetime.now(UTC).isoformat()
    cap_chars = _entry_cap(slot)
    out: list[StateEntry] = []
    for raw in entries:
        if isinstance(raw, StateEntry):
            text, author, source_turn = raw.text, raw.author, raw.source_turn
        elif isinstance(raw, dict):
            text = str(raw.get("text", ""))
            author = raw.get("author", "user")
            source_turn = raw.get("source_turn")
        else:
            text, author, source_turn = str(raw), "user", None
        text = text.strip()[:cap_chars]
        if not text:
            continue
        # A user-driven edit defaults unknown/forged authors to `user` — it can't
        # mint an agent-authored (or system-authored) entry.
        out.append(StateEntry(
            text=text, author=_coerce_author(str(author), "user"),
            source_turn=source_turn, updated_at=now,
        ))

    updated = state.model_copy(deep=True)
    setattr(updated, slot, out[-MAX_ENTRIES_PER_SLOT:])
    return updated


def render_state(state: ConversationState) -> str:
    """Format the state object as a system-prompt block, or "" if empty."""
    if state.is_empty():
        return ""

    lines: list[str] = ["## Conversation State (structured, survives compression)"]
    for slot in SLOTS:
        items = state.entries(slot)
        if not items:
            continue
        lines.append(f"### {SLOT_LABELS[slot]}")
        for e in items:
            # Flag user-asserted entries so the model doesn't overwrite what the
            # user themselves put on the record.
            marker = " [user]" if e.author == "user" else ""
            lines.append(f"- {e.text}{marker}")
    # The rolling digest of aged-out turns comes last — it's background continuity
    # for turns no longer in the verbatim window (the INV-CTX-1 coverage surface).
    # The digest is a summary, not the record: the anchor line tells the model the
    # verbatim turns behind it stay readable on demand (pointers, not payloads).
    if state.digest:
        lines.append("### Summary of earlier turns")
        lines.append(state.digest)
        lines.append(
            f"(The verbatim turns behind this summary remain readable: "
            f"{READ_THREAD_CURRENT_CALL} — earliest turns are N=0.)"
        )
    return "\n".join(lines)


# --- Storage: Redis hot cache + durable Postgres copy ------------------------
#
# Redis (30-day TTL) serves every per-turn read; Postgres (`conversation_state`
# table, Alembic 0006) is the durable copy so compaction coverage — the digest
# is the only surviving view of aged-out turns (INV-CTX-1) — outlives the TTL
# and a Redis wipe. Write-through on save; read-through + re-warm on a Redis
# miss (a both-tiers miss is negative-cached briefly so stateless conversations
# don't re-query PG per turn). Both sides are best-effort — storage must never
# fail a turn — and PG failures trip a short breaker so a down/unconfigured
# durable tier degrades to Redis-only instead of stalling on connect timeouts.
# Keys are the FULL conversation id in both tiers (`conversation_state` is
# TEXT-keyed): Redis and Postgres must never disagree on identity.

def _redis():
    from ..kit.agent_memory.connections import RedisConnection

    return RedisConnection.get_client()


def _key(conversation_id: str) -> str:
    return f"{STATE_PREFIX}{conversation_id}"


_pg_retry_at = 0.0  # monotonic deadline; durable-tier ops skipped until then


def _pg_ready() -> bool:
    return time.monotonic() >= _pg_retry_at


def _pg_trip_breaker(op: str, exc: Exception) -> None:
    """One warning per trip, then silence for the backoff window — a down or
    unconfigured Postgres must degrade to Redis-only, not stall every turn on
    a connect timeout."""
    global _pg_retry_at
    _pg_retry_at = time.monotonic() + PG_BACKOFF_SECONDS
    logger.warning(
        f"conversation state durable {op} failed "
        f"(backing off {int(PG_BACKOFF_SECONDS)}s): {exc}"
    )


@contextmanager
def _pg_cursor(commit: bool = False):
    """One durable-tier round-trip: engine-pooled raw connection, cursor,
    optional commit; close() always returns the connection to the pool
    (rolling back any open transaction)."""
    from ..kit.agent_memory.connections import PostgresConnection

    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


def _pg_save(conversation_id: str, state_json: str) -> None:
    """Upsert the durable copy. Raises on failure (caller trips the breaker)."""
    with _pg_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO conversation_state (conversation_id, state, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET state = EXCLUDED.state, updated_at = NOW()
            """,
            (conversation_id, state_json),
        )


def _pg_load(conversation_id: str) -> str | None:
    """Read the durable copy's JSON, or None. Raises on failure (caller trips the breaker)."""
    with _pg_cursor() as cur:
        cur.execute(
            "SELECT state FROM conversation_state WHERE conversation_id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        value = row[0]
        return value if isinstance(value, str) else json.dumps(value)


def _pg_delete(conversation_id: str) -> None:
    with _pg_cursor(commit=True) as cur:
        cur.execute(
            "DELETE FROM conversation_state WHERE conversation_id = %s",
            (conversation_id,),
        )


def _parse_state(raw: str | bytes) -> ConversationState | None:
    try:
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw))
        return ConversationState.model_validate(data)
    except Exception as e:  # pragma: no cover — corrupt payload
        logger.debug(f"conversation state parse failed: {e}")
        return None


def _redis_set(key: str, payload: str, ttl: int = STATE_TTL_SECONDS) -> None:
    """Single atomic SET-with-expiry — no set→expire gap to strand a TTL-less key."""
    _redis().set(key, payload, ex=ttl)


def get_state(conversation_id: str) -> ConversationState:
    """Return the persisted state, or an empty state on miss/error.

    Redis first (the hot path every turn). On a miss — or a corrupt hot-cache
    payload — fall back to the durable Postgres copy and re-warm Redis, so a
    conversation resumed after the Redis TTL keeps its digest coverage instead
    of silently starting blank. A true both-tiers miss is negative-cached
    briefly so stateless conversations don't re-query Postgres on every
    per-turn read.
    """
    raw: Any = None
    try:
        raw = _redis().get(_key(conversation_id))
    except Exception as e:  # pragma: no cover — Redis offline
        logger.debug(f"conversation state read failed: {e}")
    if raw:
        cached = _parse_state(raw)
        if cached is not None:
            return cached
        # Corrupt hot-cache payload: fall through to the durable copy.

    durable: str | None = None
    if _pg_ready():
        try:
            durable = _pg_load(conversation_id)
        except Exception as e:
            _pg_trip_breaker("read", e)
    state = _parse_state(durable) if durable else None
    try:  # re-warm so the next read stays on Redis (miss ⇒ short negative TTL)
        if state is not None and durable is not None:
            _redis_set(_key(conversation_id), durable)
        else:
            _redis_set(
                _key(conversation_id),
                ConversationState().model_dump_json(),
                ttl=STATE_NEGATIVE_TTL_SECONDS,
            )
    except Exception as e:  # pragma: no cover — Redis offline
        logger.debug(f"conversation state re-warm skipped: {e}")
    return state or ConversationState()


def save_state(conversation_id: str, state: ConversationState) -> None:
    """Persist (replace) the state object: Redis hot cache + durable Postgres
    (write-through). Both sides best-effort — storage must never fail a turn."""
    payload = state.model_dump_json()
    try:
        _redis_set(_key(conversation_id), payload)
    except Exception as e:  # pragma: no cover — Redis offline
        logger.warning(f"conversation state write failed: {e}")
    if _pg_ready():
        try:
            _pg_save(conversation_id, payload)
        except Exception as e:
            _pg_trip_breaker("write", e)


def update_slot(
    conversation_id: str,
    slot: str,
    entries: list[str],
    *,
    author: Author = "agent",
    source_turn: int | None = None,
    replace: bool = False,
) -> ConversationState:
    """Get→apply→save one slot mutation. Returns the persisted state."""
    state = apply_update(
        get_state(conversation_id),
        slot,
        entries,
        author=author,
        source_turn=source_turn,
        replace=replace,
    )
    save_state(conversation_id, state)
    return state


def replace_slot(conversation_id: str, slot: str, entries: list[Any]) -> ConversationState:
    """Get→set→save a full-slot user edit (Slice 1b). Returns the persisted state."""
    state = set_slot(get_state(conversation_id), slot, entries)
    save_state(conversation_id, state)
    return state


def update_digest(conversation_id: str, digest: str) -> ConversationState:
    """Replace the rolling compaction digest (Slice 1c). Returns the persisted state."""
    state = get_state(conversation_id)
    state.digest = (digest or "").strip()
    save_state(conversation_id, state)
    return state


def clear_state(conversation_id: str) -> None:
    """Delete the persisted state for a conversation (both stores)."""
    try:
        _redis().delete(_key(conversation_id))
    except Exception as e:  # pragma: no cover — Redis offline
        logger.debug(f"conversation state clear failed: {e}")
    if _pg_ready():
        try:
            _pg_delete(conversation_id)
        except Exception as e:
            _pg_trip_breaker("clear", e)


def render_state_block(conversation_id: str) -> str:
    """Render the persisted state as a system-prompt block, or "" if none."""
    return render_state(get_state(conversation_id))

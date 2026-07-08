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
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STATE_PREFIX = "conv_state:"
STATE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

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
    if state.digest:
        lines.append("### Summary of earlier turns")
        lines.append(state.digest)
    return "\n".join(lines)


# --- Redis storage -----------------------------------------------------------

def _redis():
    from ..kit.agent_memory.connections import RedisConnection

    return RedisConnection.get_client()


def _key(conversation_id: str) -> str:
    return f"{STATE_PREFIX}{conversation_id}"


def get_state(conversation_id: str) -> ConversationState:
    """Return the persisted state, or an empty state on miss/error."""
    try:
        raw = _redis().get(_key(conversation_id))
    except Exception as e:  # pragma: no cover — Redis offline
        logger.debug(f"conversation state read failed: {e}")
        return ConversationState()
    if not raw:
        return ConversationState()
    try:
        data = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw))
        return ConversationState.model_validate(data)
    except Exception as e:  # pragma: no cover — corrupt payload
        logger.debug(f"conversation state parse failed: {e}")
        return ConversationState()


def save_state(conversation_id: str, state: ConversationState) -> None:
    """Persist (replace) the state object for a conversation."""
    try:
        client = _redis()
        key = _key(conversation_id)
        client.set(key, state.model_dump_json())
        client.expire(key, STATE_TTL_SECONDS)
    except Exception as e:  # pragma: no cover — Redis offline
        logger.warning(f"conversation state write failed: {e}")


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
    """Delete the persisted state for a conversation."""
    try:
        _redis().delete(_key(conversation_id))
    except Exception as e:  # pragma: no cover — Redis offline
        logger.debug(f"conversation state clear failed: {e}")


def render_state_block(conversation_id: str) -> str:
    """Render the persisted state as a system-prompt block, or "" if none."""
    return render_state(get_state(conversation_id))

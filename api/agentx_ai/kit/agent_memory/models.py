"""Data models for the agent memory system."""

import re
from datetime import datetime, UTC
from hashlib import sha256
from typing import Any
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4


# Distilled procedure triggers are stored condition-phrased and usually already
# lead with a condition word ("when presenting a recommendation"). Capitalize
# when they do; only prepend "When " when they don't — avoids the doubled
# "When when …" wart in rendered procedures. (Render-only; storage is untouched.)
_TRIGGER_CONDITION_LEAD = re.compile(
    r"^(when|whenever|before|after|while|if|once|upon|during|as)\b", re.IGNORECASE
)


def _prefix_trigger(trigger: str) -> str:
    """Headline form of a procedure trigger, de-doubling a leading condition word."""
    t = (trigger or "").strip()
    if not t:
        return t
    if _TRIGGER_CONDITION_LEAD.match(t):
        return t[0].upper() + t[1:]
    return f"When {t}"


def _coerce_datetime(value: Any) -> Any:
    """Convert neo4j.time.DateTime (or anything with .to_native()) → datetime.

    Neo4j driver returns its own DateTime type for datetime properties on
    nodes; Pydantic v2 rejects it as not a python datetime. This shim runs
    in field validators so models can be constructed directly from
    ``dict(record["g"])`` without per-call conversion.
    """
    if value is None or isinstance(value, datetime):
        return value
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        return to_native()
    return value


def compute_claim_hash(claim: str) -> str:
    """
    Compute a hash of the normalized claim for duplicate detection.

    Normalizes by lowercasing and removing extra whitespace.
    Returns first 16 chars of SHA256 hash.
    """
    normalized = " ".join(claim.lower().split())
    return sha256(normalized.encode()).hexdigest()[:16]


def _utc_now():
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class Turn(BaseModel):
    """Represents a single conversation turn."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    index: int
    timestamp: datetime = Field(default_factory=_utc_now)
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    embedding: list[float] | None = None
    token_count: int | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    channel: str = "_global"
    # Docker-style agent_id of the producing agent (assistant turns only).
    # Enables multi-agent transcript reconstruction (Phase 16 — message
    # attribution).
    agent_id: str | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class Entity(BaseModel):
    """Represents a named entity (person, organization, concept, etc.)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str  # 'Person', 'Organization', 'Concept', etc.
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    embedding: list[float] | None = None
    salience: float = 0.5
    properties: dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=_utc_now)
    last_accessed: datetime = Field(default_factory=_utc_now)
    access_count: int = 0
    channel: str = "_global"

    @field_validator("first_seen", "last_accessed", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)

    @staticmethod
    def compute_embedding_text(name: str, description: str | None, type_: str) -> str:
        """Canonical text used to embed an entity: ``"{name}: {description or type}"``.

        Shared by upsert and update paths so the embedding input stays
        consistent regardless of which code path (re-)embeds the entity.
        """
        return f"{name}: {description or type_}"

    def embedding_text(self) -> str:
        """Canonical embedding text for this entity instance."""
        return self.compute_embedding_text(self.name, self.description, self.type)


class Fact(BaseModel):
    """Represents a factual claim or piece of knowledge."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    claim: str
    claim_hash: str | None = None  # SHA256 hash for indexed duplicate detection
    confidence: float = 0.8
    source: str  # 'extraction', 'user_stated', 'inferred'
    source_turn_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    channel: str = "_global"

    # Access tracking (parity with Entity for reinforcement signal)
    last_accessed: datetime = Field(default_factory=_utc_now)
    access_count: int = 0
    salience: float = 0.5

    # Temporal context (simple: current/past/future)
    temporal_context: str | None = None  # "current", "past", "future", or None

    # Supersession tracking (for corrections and contradictions)
    superseded_at: datetime | None = None
    superseded_by_id: str | None = None
    supersedes_id: str | None = None  # ID of fact this supersedes

    # Review flags
    flagged_for_review: bool = False

    @field_validator(
        "created_at", "last_accessed", "superseded_at", mode="before"
    )
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)

    def model_post_init(self, __context) -> None:
        """Compute claim_hash after initialization if not set."""
        if self.claim_hash is None and self.claim:
            self.claim_hash = compute_claim_hash(self.claim)


class Goal(BaseModel):
    """Represents a user goal or objective."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: str = "active"  # 'active', 'completed', 'abandoned', 'blocked'
    priority: int = 3  # 1-5
    parent_goal_id: str | None = None
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    deadline: datetime | None = None
    channel: str = "_global"

    @field_validator("created_at", "deadline", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class Strategy(BaseModel):
    """Represents a successful procedural pattern or strategy."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    context_pattern: str  # Regex or keywords
    tool_sequence: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None
    channel: str = "_global"

    @field_validator("last_used", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class Procedure(BaseModel):
    """A distilled, scoped procedural rule — the project/user/domain "how we do it
    here" delta a general model wouldn't already do by default (Slice 1).

    Richer than ``Strategy`` (which keys a tool sequence off a context pattern):
    a ``Procedure`` carries a natural-language trigger + a replayable body and is
    strengthened (not duplicated) as the same pattern recurs.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    trigger: str  # NL condition that activates this — "when presenting a recommendation"
    trigger_features: dict[str, Any] = Field(default_factory=dict)  # situation features (later activation)
    body: str  # the replayable approach / instruction
    rationale: str = ""  # why this is the right behavior
    scope: str = "_global"  # channel scope (_global | user | project | _self_{agent})
    agent_id: str | None = None
    strength: int = 1  # replay/reinforce count
    evidence_refs: list[str] = Field(default_factory=list)  # candidate ids + conversation ids
    signal_kinds: list[str] = Field(default_factory=list)  # e.g. ["correction", "explicit_rule"]
    embedding: list[float] | None = None
    created_at: datetime | None = None
    last_reinforced: datetime | None = None

    @field_validator("created_at", "last_reinforced", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class MemoryBundle(BaseModel):
    """Aggregated retrieval result for context injection."""

    relevant_turns: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    strategies: list[dict[str, Any]] = Field(default_factory=list)
    procedures: list[dict[str, Any]] = Field(default_factory=list)
    active_goals: list[dict[str, Any]] = Field(default_factory=list)
    user_context: dict[str, Any] = Field(default_factory=dict)

    def to_context_string(
        self,
        *,
        turn_char_limit: int = 2000,
        max_turns: int = 10,
        roles: set | None = None,
        current_conversation_id: str | None = None,
    ) -> str:
        """Format memory bundle as context for LLM prompt.

        Per-turn truncation and turn count are tunable so callers can give
        the model enough verbatim context to actually continue prior work
        (e.g. write a second paragraph that references the first).

        ``roles`` optionally restricts which turn roles are rendered.
        ``current_conversation_id``, when set, drops turns from other
        conversations (cross-conversation history is opt-in via the
        ``recall_user_history`` tool, not auto-injected).
        """
        sections = []

        if self.relevant_turns:
            def _format_turn(t: dict[str, Any]) -> str:
                content = t["content"] or ""
                if len(content) > turn_char_limit:
                    content = content[:turn_char_limit] + "…[truncated]"
                return f"[{t['timestamp']}] {t['role']}: {content}"

            visible = list(self.relevant_turns)
            if roles is not None:
                visible = [t for t in visible if t.get("role") in roles]
            if current_conversation_id is not None:
                visible = [
                    t for t in visible
                    if t.get("conversation_id") == current_conversation_id
                ]
            if visible:
                turns_text = "\n".join(_format_turn(t) for t in visible[:max_turns])
                sections.append(f"## Relevant Past Conversations\n{turns_text}")

        if self.facts:
            facts_text = "\n".join(
                f"- {f['claim']} (confidence: {f['confidence']:.0%})"
                for f in self.facts[:10]
            )
            sections.append(f"## Known Facts\n{facts_text}")

        if self.entities:
            entities_text = "\n".join(
                f"- {e['name']} ({e['type']}): {e.get('description', 'N/A')}"
                for e in self.entities[:10]
            )
            sections.append(f"## Relevant Entities\n{entities_text}")

        if self.active_goals:
            goals_text = "\n".join(
                f"- [{g['priority']}] {g['description']} ({g['status']})"
                for g in self.active_goals
            )
            sections.append(f"## Active Goals\n{goals_text}")

        # Reflex core (Slice 1): the always-on "how we work here" procedures — the
        # delta a general model wouldn't already do by default. Maintained, not
        # searched, so it's injected every turn (see ProceduralMemory.get_reflex_procedures).
        if self.procedures:
            def _format_proc(p: dict[str, Any]) -> str:
                trigger = (p.get("trigger") or "").strip()
                body = (p.get("body") or "").strip()
                return f"- {_prefix_trigger(trigger)}: {body}" if trigger else f"- {body}"

            procs_text = "\n".join(_format_proc(p) for p in self.procedures)
            sections.append(
                f"## Learned Procedures (how we work here)\n{procs_text}"
            )

        return "\n\n".join(sections)

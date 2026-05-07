"""Data models for the agent memory system."""

from datetime import datetime, timezone
from hashlib import sha256
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4


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
    return datetime.now(timezone.utc)


class Turn(BaseModel):
    """Represents a single conversation turn."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    index: int
    timestamp: datetime = Field(default_factory=_utc_now)
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    embedding: Optional[List[float]] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    channel: str = "_global"
    # Docker-style agent_id of the producing agent (assistant turns only).
    # Enables multi-agent transcript reconstruction (Phase 16 — message
    # attribution).
    agent_id: Optional[str] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class Entity(BaseModel):
    """Represents a named entity (person, organization, concept, etc.)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str  # 'Person', 'Organization', 'Concept', etc.
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    embedding: Optional[List[float]] = None
    salience: float = 0.5
    properties: Dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=_utc_now)
    last_accessed: datetime = Field(default_factory=_utc_now)
    access_count: int = 0
    channel: str = "_global"

    @field_validator("first_seen", "last_accessed", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class Fact(BaseModel):
    """Represents a factual claim or piece of knowledge."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    claim: str
    claim_hash: Optional[str] = None  # SHA256 hash for indexed duplicate detection
    confidence: float = 0.8
    source: str  # 'extraction', 'user_stated', 'inferred'
    source_turn_id: Optional[str] = None
    entity_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=_utc_now)
    channel: str = "_global"

    # Access tracking (parity with Entity for reinforcement signal)
    last_accessed: datetime = Field(default_factory=_utc_now)
    access_count: int = 0
    salience: float = 0.5

    # Temporal context (simple: current/past/future)
    temporal_context: Optional[str] = None  # "current", "past", "future", or None

    # Supersession tracking (for corrections and contradictions)
    superseded_at: Optional[datetime] = None
    superseded_by_id: Optional[str] = None
    supersedes_id: Optional[str] = None  # ID of fact this supersedes

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
    parent_goal_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=_utc_now)
    deadline: Optional[datetime] = None
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
    tool_sequence: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None
    channel: str = "_global"

    @field_validator("last_used", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        return _coerce_datetime(v)


class MemoryBundle(BaseModel):
    """Aggregated retrieval result for context injection."""

    relevant_turns: List[Dict[str, Any]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    strategies: List[Dict[str, Any]] = Field(default_factory=list)
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    user_context: Dict[str, Any] = Field(default_factory=dict)

    def to_context_string(
        self,
        *,
        turn_char_limit: int = 2000,
        max_turns: int = 10,
        roles: Optional[set] = None,
        current_conversation_id: Optional[str] = None,
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
            def _format_turn(t: Dict[str, Any]) -> str:
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

        return "\n\n".join(sections)

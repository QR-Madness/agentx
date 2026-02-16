"""Data models for the agent memory system."""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import uuid4


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


class Fact(BaseModel):
    """Represents a factual claim or piece of knowledge."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    claim: str
    confidence: float = 0.8
    source: str  # 'extraction', 'user_stated', 'inferred'
    source_turn_id: Optional[str] = None
    entity_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=_utc_now)
    channel: str = "_global"


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


class MemoryBundle(BaseModel):
    """Aggregated retrieval result for context injection."""

    relevant_turns: List[Dict[str, Any]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    strategies: List[Dict[str, Any]] = Field(default_factory=list)
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    user_context: Dict[str, Any] = Field(default_factory=dict)

    def to_context_string(self) -> str:
        """Format memory bundle as context for LLM prompt."""
        sections = []

        if self.relevant_turns:
            turns_text = "\n".join(
                f"[{t['timestamp']}] {t['role']}: {t['content'][:200]}..."
                for t in self.relevant_turns[:5]
            )
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

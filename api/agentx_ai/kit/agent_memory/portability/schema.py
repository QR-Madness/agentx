"""Envelope schema for memory import/export.

The export is a single versioned JSON object. Nodes are carried as opaque
property dicts (so the export survives additive schema changes) keyed by their
stable ``id``; edges that are *not* reconstructable from foreign-key-like props
already on the nodes are carried explicitly.

Honors the v0.20 "migratable across platforms" rule: ``schema_version`` gates
imports and ``embedder`` records the vectorizer so an importer can decide
restore-vs-recompute.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import get_settings

# Bump only on a breaking change to the envelope shape. Importers reject an
# unknown (newer) version rather than silently mis-reading it.
SCHEMA_VERSION = 1


class EmbedderInfo(BaseModel):
    """Identifies the vectorizer used for the exported embeddings."""

    provider_model: str  # e.g. "local:BAAI/bge-m3" or "openai:text-embedding-3-small"
    dimensions: int


def current_embedder_info() -> EmbedderInfo:
    """Describe the currently-configured embedder without loading the model.

    Uses the configured (not auto-detected) dimensions so callers don't pay a
    model load just to stamp the envelope; the importer compares this against
    the export's recorded dimensions to decide whether to recompute vectors.
    """
    settings = get_settings()
    provider = settings.embedding_provider
    model = (
        settings.embedding_model
        if provider == "openai"
        else settings.local_embedding_model
    )
    return EmbedderInfo(
        provider_model=f"{provider}:{model}",
        dimensions=settings.embedding_dimensions,
    )


class MemoryExport(BaseModel):
    """Round-trippable snapshot of a user's memory graph (optionally one channel)."""

    schema_version: int = SCHEMA_VERSION
    exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str
    channel: Optional[str] = None  # None / "_all" = every channel for the user
    include_embeddings: bool = True
    embedder: EmbedderInfo

    # Node collections — each item is the node's full property dict (datetimes
    # as ISO strings), plus a couple of relationship-derived helper keys noted
    # inline below.
    conversations: List[Dict[str, Any]] = Field(default_factory=list)
    turns: List[Dict[str, Any]] = Field(default_factory=list)  # + "conversation_id"
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)  # + "entity_ids"
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    strategies: List[Dict[str, Any]] = Field(default_factory=list)  # + "tool_sequence", "succeeded_in", "failed_in"
    tool_invocations: List[Dict[str, Any]] = Field(default_factory=list)  # + "conversation_id"

    # PostgreSQL audit mirror (carries fields not on the graph nodes: model,
    # metadata/cost on logs; tool_input/tool_output on invocations).
    pg_conversation_logs: List[Dict[str, Any]] = Field(default_factory=list)
    pg_tool_invocations: List[Dict[str, Any]] = Field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        """Per-collection item counts (for CLI/UI summaries)."""
        return {
            "conversations": len(self.conversations),
            "turns": len(self.turns),
            "entities": len(self.entities),
            "facts": len(self.facts),
            "goals": len(self.goals),
            "strategies": len(self.strategies),
            "tool_invocations": len(self.tool_invocations),
            "pg_conversation_logs": len(self.pg_conversation_logs),
            "pg_tool_invocations": len(self.pg_tool_invocations),
        }

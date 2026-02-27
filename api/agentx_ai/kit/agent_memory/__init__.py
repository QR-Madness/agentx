"""
AgentX Memory System - Cognitive memory architecture for LLM-powered agents.

This package provides a comprehensive memory system using:
- Neo4j for graph-based and vector storage
- PostgreSQL with pgvector for relational and vector data
- Redis for working memory and caching
"""

from .models import Turn, Entity, Fact, Goal, Strategy, MemoryBundle
from .memory.interface import AgentMemory
from .memory.retrieval import RetrievalWeights, RetrievalMetrics
from .audit import MemoryAuditLogger, AuditLogLevel, OperationType, MemoryType
from .abc import MemoryStore, Embedder, Extractor, Reranker, ScoredResult, HealthStatus
from .events import (
    MemoryEventEmitter,
    EventPayload,
    TurnStoredPayload,
    FactLearnedPayload,
    EntityCreatedPayload,
    RetrievalCompletePayload,
)
from .query_utils import (
    CypherFilterBuilder,
    SQLFilterBuilder,
    convert_record_datetimes,
    neo4j_datetime_to_iso,
    validate_pagination,
    validate_time_window,
    COMMON_DATETIME_FIELDS,
)

__all__ = [
    # Core interface
    "AgentMemory",
    # Models
    "Turn",
    "Entity",
    "Fact",
    "Goal",
    "Strategy",
    "MemoryBundle",
    # Retrieval
    "RetrievalWeights",
    "RetrievalMetrics",
    # Audit
    "MemoryAuditLogger",
    "AuditLogLevel",
    "OperationType",
    "MemoryType",
    # ABCs for extensibility
    "MemoryStore",
    "Embedder",
    "Extractor",
    "Reranker",
    "ScoredResult",
    "HealthStatus",
    # Events
    "MemoryEventEmitter",
    "EventPayload",
    "TurnStoredPayload",
    "FactLearnedPayload",
    "EntityCreatedPayload",
    "RetrievalCompletePayload",
    # Query utilities
    "CypherFilterBuilder",
    "SQLFilterBuilder",
    "convert_record_datetimes",
    "neo4j_datetime_to_iso",
    "validate_pagination",
    "validate_time_window",
    "COMMON_DATETIME_FIELDS",
]

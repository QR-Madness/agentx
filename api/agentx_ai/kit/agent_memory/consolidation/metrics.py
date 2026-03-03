"""Consolidation metrics for tracking pipeline performance and costs."""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationMetrics:
    """
    Metrics collected during a consolidation run.

    Tracks turn processing, LLM usage, extraction results, and timing.
    Can be serialized to JSON for storage in Redis or audit logs.
    """

    # Identification
    job_id: str = ""
    conversation_id: str = ""
    user_id: str = ""
    channel: str = "_default"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Turn processing
    turns_total: int = 0
    turns_relevant: int = 0
    turns_skipped_heuristic: int = 0
    turns_skipped_llm: int = 0

    # LLM usage
    relevance_calls: int = 0
    correction_calls: int = 0
    extraction_calls: int = 0
    contradiction_calls: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0

    # Extraction results
    entities_extracted: int = 0
    facts_extracted: int = 0
    relationships_extracted: int = 0

    # Quality metrics
    duplicates_skipped: int = 0
    contradictions_found: int = 0
    contradictions_resolved: int = 0
    corrections_applied: int = 0

    # Storage results
    entities_stored: int = 0
    facts_stored: int = 0
    relationships_stored: int = 0
    storage_errors: int = 0

    # Timing (milliseconds)
    relevance_latency_ms: int = 0
    extraction_latency_ms: int = 0
    storage_latency_ms: int = 0
    total_latency_ms: int = 0

    # Error tracking
    errors: List[str] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        """Calculate the percentage of turns skipped by relevance filter."""
        if self.turns_total == 0:
            return 0.0
        skipped = self.turns_skipped_heuristic + self.turns_skipped_llm
        return skipped / self.turns_total

    @property
    def extraction_efficiency(self) -> float:
        """Calculate facts extracted per LLM call (higher is better)."""
        if self.extraction_calls == 0:
            return 0.0
        return self.facts_extracted / self.extraction_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetimes to ISO strings
        if d["started_at"]:
            d["started_at"] = d["started_at"].isoformat()
        if d["completed_at"]:
            d["completed_at"] = d["completed_at"].isoformat()
        # Add computed properties
        d["skip_rate"] = self.skip_rate
        d["extraction_efficiency"] = self.extraction_efficiency
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConsolidationMetrics":
        """Create from dictionary."""
        # Handle datetime conversion
        if d.get("started_at") and isinstance(d["started_at"], str):
            d["started_at"] = datetime.fromisoformat(d["started_at"])
        if d.get("completed_at") and isinstance(d["completed_at"], str):
            d["completed_at"] = datetime.fromisoformat(d["completed_at"])
        # Remove computed properties that aren't constructor args
        d.pop("skip_rate", None)
        d.pop("extraction_efficiency", None)
        return cls(**d)

    def log_summary(self) -> None:
        """Log a summary of the metrics."""
        logger.info(
            f"Consolidation metrics: "
            f"turns={self.turns_total} (skipped={self.turns_skipped_heuristic + self.turns_skipped_llm}, "
            f"rate={self.skip_rate:.1%}), "
            f"llm_calls={self.total_llm_calls}, tokens={self.total_tokens_used}, "
            f"extracted=[{self.entities_extracted}e, {self.facts_extracted}f, {self.relationships_extracted}r], "
            f"stored=[{self.entities_stored}e, {self.facts_stored}f, {self.relationships_stored}r], "
            f"latency={self.total_latency_ms}ms"
        )


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across multiple consolidation runs.

    Used for dashboard display and trend analysis.
    """

    period: str = ""  # e.g., "2024-03-01", "2024-03-01T14"
    runs: int = 0

    # Totals
    total_turns: int = 0
    total_turns_skipped: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_entities: int = 0
    total_facts: int = 0
    total_errors: int = 0

    # Averages (computed)
    avg_skip_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens_per_run: float = 0.0

    def add_run(self, metrics: ConsolidationMetrics) -> None:
        """Add a consolidation run to the aggregation."""
        self.runs += 1
        self.total_turns += metrics.turns_total
        self.total_turns_skipped += metrics.turns_skipped_heuristic + metrics.turns_skipped_llm
        self.total_llm_calls += metrics.total_llm_calls
        self.total_tokens += metrics.total_tokens_used
        self.total_entities += metrics.entities_stored
        self.total_facts += metrics.facts_stored
        self.total_errors += len(metrics.errors)

        # Update averages
        if self.total_turns > 0:
            self.avg_skip_rate = self.total_turns_skipped / self.total_turns
        if self.runs > 0:
            self.avg_tokens_per_run = self.total_tokens / self.runs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

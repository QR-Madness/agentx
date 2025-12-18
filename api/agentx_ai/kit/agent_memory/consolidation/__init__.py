"""Consolidation modules for background memory processing and decay."""

from .worker import ConsolidationWorker
from .jobs import (
    consolidate_episodic_to_semantic,
    detect_patterns,
    apply_memory_decay,
    cleanup_old_memories,
    trigger_reflection,
)

__all__ = [
    "ConsolidationWorker",
    "consolidate_episodic_to_semantic",
    "detect_patterns",
    "apply_memory_decay",
    "cleanup_old_memories",
    "trigger_reflection",
]

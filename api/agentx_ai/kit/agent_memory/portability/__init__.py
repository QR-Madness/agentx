"""Scriptable memory import/export (round-trippable JSON, stable IDs).

This package serializes the agent memory graph (episodic, semantic, procedural,
goals) plus the PostgreSQL audit mirror into a single versioned envelope, and
re-imports it idempotently by ``MERGE``-ing on the stable node ids. Embeddings
can be carried verbatim or stripped (``--no-embeddings``) and recomputed on
import — the latter yields a deterministic, diffable artifact.

Public surface:
    - ``SCHEMA_VERSION`` / ``MemoryExport`` (envelope)
    - ``MemoryExporter`` / ``MemoryImporter``
    - ``current_embedder_info`` (provider/model + dims for the envelope)
"""

from .schema import SCHEMA_VERSION, MemoryExport, EmbedderInfo, current_embedder_info
from .exporter import MemoryExporter
from .importer import MemoryImporter, ImportMode

__all__ = [
    "SCHEMA_VERSION",
    "MemoryExport",
    "EmbedderInfo",
    "current_embedder_info",
    "MemoryExporter",
    "MemoryImporter",
    "ImportMode",
]

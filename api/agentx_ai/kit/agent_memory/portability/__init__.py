"""Scriptable memory import/export (round-trippable JSON, stable IDs).

This package serializes the agent memory graph (episodic, semantic, procedural,
goals) plus the PostgreSQL audit mirror into a single versioned envelope, and
re-imports it idempotently by ``MERGE``-ing on the stable node ids. Exports are
text-only — embeddings are regenerated from text on import, so files are small,
deterministic, diffable, and portable across embedding models.

Public surface:
    - ``SCHEMA_VERSION`` / ``MemoryExport`` (envelope)
    - ``MemoryExporter`` / ``MemoryImporter``
    - ``current_embedder_info`` (provider/model + dims for the envelope)
    - ``cluster`` — cluster-wide snapshot/wipe/restore shared by the eval harnesses
"""

from . import cluster
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
    "cluster",
]

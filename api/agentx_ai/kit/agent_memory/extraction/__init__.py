"""Extraction modules for entities, facts, and relationships from text."""

from .entities import extract_entities
from .facts import extract_facts
from .relationships import extract_relationships
from .service import ExtractionService, ExtractionResult, get_extraction_service, reset_extraction_service

__all__ = [
    "extract_entities",
    "extract_facts",
    "extract_relationships",
    "ExtractionService",
    "ExtractionResult",
    "get_extraction_service",
    "reset_extraction_service",
]

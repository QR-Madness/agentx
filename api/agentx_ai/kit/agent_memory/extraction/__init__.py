"""Extraction modules for entities, facts, and relationships from text."""

from .entities import extract_entities
from .facts import extract_facts
from .relationships import extract_relationships

__all__ = [
    "extract_entities",
    "extract_facts",
    "extract_relationships",
]

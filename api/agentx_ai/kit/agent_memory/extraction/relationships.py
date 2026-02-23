"""Relationship extraction from text to build knowledge graph connections."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_relationships(
    text: str,
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities from text using LLM.

    Supports relationship types: works_at, knows, uses, prefers,
    created, located_in, part_of, related_to, mentioned_with.

    Args:
        text: Text to extract relationships from
        entities: List of entities found in the text

    Returns:
        List of relationship dictionaries with 'source', 'target',
        'type', 'confidence'
    """
    if not text or not entities:
        return []

    from .service import get_extraction_service

    try:
        service = get_extraction_service()
        result = service.extract_relationships(text, entities)

        # Validate relationships reference known entities
        entity_names = {str(e.get("name", "")).lower() for e in entities if e.get("name")}
        validated = []

        for rel in result:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            rel_type = rel.get("relation") or rel.get("type")

            # Check that both entities exist (case-insensitive match)
            if (source.lower() in entity_names and
                target.lower() in entity_names and
                rel_type):
                validated.append({
                    "source": source,
                    "target": target,
                    "type": str(rel_type).upper().replace(" ", "_").replace("-", "_"),
                    "confidence": float(rel.get("confidence", 0.7)),
                })

        logger.info(f"Extracted {len(validated)} relationships from text")
        return validated

    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        return []

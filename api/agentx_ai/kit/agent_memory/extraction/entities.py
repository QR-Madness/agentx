"""Entity extraction from text using LLM-based extraction."""

import asyncio
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using LLM.

    Uses the ExtractionService to perform LLM-based extraction
    and returns Entity-compatible dictionaries.

    Args:
        text: Text to extract entities from

    Returns:
        List of entity dictionaries with 'name', 'type', 'description', 'confidence'
    """
    if not text or len(text.strip()) < 20:
        return []

    from .service import get_extraction_service

    try:
        service = get_extraction_service()
        # Bridge async to sync for consolidation jobs
        result = asyncio.run(service.extract_entities(text))

        # Ensure required fields and validate
        validated = []
        for entity in result:
            if entity.get("name") and entity.get("type"):
                validated.append({
                    "name": str(entity["name"]).strip(),
                    "type": str(entity["type"]).strip(),
                    "description": entity.get("description"),
                    "confidence": float(entity.get("confidence", 0.7)),
                })

        logger.info(f"Extracted {len(validated)} entities from text")
        return validated

    except RuntimeError as e:
        # Handle case where event loop is already running
        if "cannot be called from a running event loop" in str(e):
            logger.warning("Cannot run async extraction from running event loop")
            return []
        raise
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []

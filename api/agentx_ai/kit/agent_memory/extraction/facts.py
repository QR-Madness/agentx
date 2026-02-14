"""Fact extraction from text using LLM-based extraction."""

import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_facts(text: str, source_turn_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract factual claims from text using LLM.

    Extracts claims, preferences, relationships, and stated goals
    from conversation text.

    Args:
        text: Text to extract facts from
        source_turn_id: Optional source turn ID for attribution

    Returns:
        List of fact dictionaries with 'claim', 'confidence', 'entity_names', 'source_turn_id'
    """
    if not text or len(text.strip()) < 20:
        return []

    from .service import get_extraction_service

    try:
        service = get_extraction_service()
        # Bridge async to sync for consolidation jobs
        result = asyncio.run(service.extract_facts(text))

        # Validate and add source attribution
        validated = []
        for fact in result:
            if fact.get("claim"):
                validated.append({
                    "claim": str(fact["claim"]).strip(),
                    "confidence": float(fact.get("confidence", 0.7)),
                    "entity_names": fact.get("entity_names", []),
                    "source_turn_id": source_turn_id,
                })

        logger.info(f"Extracted {len(validated)} facts from text")
        return validated

    except RuntimeError as e:
        # Handle case where event loop is already running
        if "cannot be called from a running event loop" in str(e):
            logger.warning("Cannot run async extraction from running event loop")
            return []
        raise
    except Exception as e:
        logger.error(f"Fact extraction failed: {e}")
        return []

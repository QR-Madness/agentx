"""Fact extraction from text using LLM-based extraction."""

import asyncio
import concurrent.futures
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _run_async_in_thread(coro):
    """
    Run an async coroutine safely from any context.

    Handles the case where we're already in an async event loop
    by running the coroutine in a separate thread with its own loop.
    """
    try:
        # Check if we're in an existing event loop
        asyncio.get_running_loop()
        # We're in an async context - run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running event loop - safe to use asyncio.run()
        return asyncio.run(coro)


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
        # Bridge async to sync safely for any context
        result = _run_async_in_thread(service.extract_facts(text))

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

    except Exception as e:
        logger.error(f"Fact extraction failed: {e}")
        return []

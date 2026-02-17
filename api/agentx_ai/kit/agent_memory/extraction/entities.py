"""Entity extraction from text using LLM-based extraction."""

import asyncio
import concurrent.futures
import logging
from typing import List, Dict, Any

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
        # Bridge async to sync safely for any context
        result = _run_async_in_thread(service.extract_entities(text))

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

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []

"""Relationship extraction from text to build knowledge graph connections."""

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
        # Bridge async to sync safely for any context
        result = _run_async_in_thread(service.extract_relationships(text, entities))

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

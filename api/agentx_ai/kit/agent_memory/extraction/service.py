"""
Extraction Service - LLM-based extraction for memory system.

Provides unified interface for entity, fact, and relationship extraction
using configured model providers.
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel

from ....providers.base import Message, MessageRole
from ....providers.registry import get_registry
from ....agent.output_parser import validate_json_output
from ..config import get_settings

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Result of an extraction operation."""
    entities: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0


class ExtractionService:
    """
    Service for LLM-based extraction from text.

    Uses configured model provider to extract entities, facts,
    and relationships from conversation text.
    """

    def __init__(self):
        self._registry = None
        self._settings = None

    @property
    def registry(self):
        """Lazy-load provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    @property
    def settings(self):
        """Lazy-load settings."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    async def extract_all(
        self,
        text: str,
        source_turn_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract entities, facts, and relationships in a single call.

        Combines extraction for efficiency (one API call instead of three).

        Args:
            text: The text to extract information from
            source_turn_id: Optional source turn ID for attribution

        Returns:
            ExtractionResult containing entities, facts, and relationships
        """
        if not self.settings.extraction_enabled:
            logger.debug("Extraction is disabled")
            return ExtractionResult(success=True)

        if not text or len(text.strip()) < 20:
            logger.debug("Text too short for extraction")
            return ExtractionResult(success=True)

        try:
            provider, model_id = self.registry.get_provider_for_model(
                self.settings.extraction_model
            )
        except ValueError as e:
            logger.warning(f"Extraction model not available: {e}")
            return ExtractionResult(success=False, error=str(e))

        prompt = self._build_combined_extraction_prompt(text)
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._get_system_prompt()),
            Message(role=MessageRole.USER, content=prompt),
        ]

        try:
            result = await provider.complete(
                messages,
                model_id,
                temperature=self.settings.extraction_temperature,
                max_tokens=self.settings.extraction_max_tokens,
            )

            parsed = self._parse_extraction_response(result.content)
            parsed.tokens_used = result.usage.get("total_tokens", 0) if result.usage else 0

            # Add source turn ID to facts
            if source_turn_id:
                for fact in parsed.facts:
                    fact["source_turn_id"] = source_turn_id

            logger.info(
                f"Extraction complete: {len(parsed.entities)} entities, "
                f"{len(parsed.facts)} facts, {len(parsed.relationships)} relationships"
            )

            return parsed

        except asyncio.TimeoutError:
            logger.error(f"Extraction timed out after {self.settings.extraction_timeout}s")
            return ExtractionResult(success=False, error="Extraction timeout")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))

    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract only entities from text."""
        result = await self.extract_all(text)
        return result.entities

    async def extract_facts(self, text: str) -> list[dict[str, Any]]:
        """Extract only facts from text."""
        result = await self.extract_all(text)
        return result.facts

    async def extract_relationships(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Extract relationships given existing entities.

        Uses a focused extraction call when entities are already known.

        Args:
            text: The text to extract relationships from
            entities: List of entities found in the text

        Returns:
            List of relationship dictionaries
        """
        if not entities:
            return []

        # For relationship extraction with known entities,
        # we still do combined extraction but focus on relationships
        result = await self.extract_all(text)
        return result.relationships

    def _get_system_prompt(self) -> str:
        """Get the system prompt for extraction."""
        entity_types = ", ".join(self.settings.entity_types)
        relationship_types = ", ".join(self.settings.relationship_types)

        return f"""You are an information extraction system. Extract structured information from conversation text.

Rules:
- Extract only information explicitly stated or strongly implied
- Assign confidence scores (0.0-1.0) based on how certain the information is
- Use canonical names (e.g., "John Smith" not "John" if full name is known)
- For entity types, use: {entity_types}
- For relationships, use: {relationship_types}
- Be conservative: prefer fewer high-confidence extractions over many low-confidence ones
- Facts should be atomic claims that can stand alone

Output ONLY valid JSON with no markdown formatting, code fences, or explanation."""

    def _build_combined_extraction_prompt(self, text: str) -> str:
        """Build the user prompt with JSON output schema."""
        # Truncate very long texts to avoid token limits
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"

        return f'''Extract entities, facts, and relationships from this conversation:

"""
{text}
"""

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "string", "type": "string", "description": "brief description", "confidence": 0.0-1.0}}
  ],
  "facts": [
    {{"claim": "atomic factual statement", "confidence": 0.0-1.0, "entity_names": ["related entity names"]}}
  ],
  "relationships": [
    {{"source": "entity_name", "relation": "relationship_type", "target": "entity_name", "confidence": 0.0-1.0}}
  ]
}}

If nothing to extract, return: {{"entities": [], "facts": [], "relationships": []}}'''

    def _parse_extraction_response(self, content: str) -> ExtractionResult:
        """
        Parse LLM response into ExtractionResult.

        Handles various response formats including markdown-wrapped JSON.

        Args:
            content: The raw LLM response content

        Returns:
            ExtractionResult with parsed entities, facts, and relationships
        """
        # First try the output_parser utility
        is_valid, parsed, error = validate_json_output(content)

        if is_valid and parsed:
            return ExtractionResult(
                entities=parsed.get("entities", []),
                facts=parsed.get("facts", []),
                relationships=parsed.get("relationships", []),
                success=True,
            )

        # Try to extract JSON directly from response
        # Handle cases where LLM adds extra text
        json_patterns = [
            r'```json\s*([\s\S]*?)```',  # Markdown code block
            r'```\s*([\s\S]*?)```',       # Generic code block
            r'(\{[\s\S]*\})',             # Raw JSON object
        ]

        for pattern in json_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    json_str = match.group(1).strip()
                    parsed = json.loads(json_str)
                    return ExtractionResult(
                        entities=parsed.get("entities", []),
                        facts=parsed.get("facts", []),
                        relationships=parsed.get("relationships", []),
                        success=True,
                    )
                except json.JSONDecodeError:
                    continue

        # If all parsing attempts fail, log and return empty result
        logger.warning(f"Failed to parse extraction response: {error}")
        logger.debug(f"Raw response: {content[:500]}...")
        return ExtractionResult(success=False, error=f"JSON parse error: {error}")


# Module-level singleton
_extraction_service: Optional[ExtractionService] = None


def get_extraction_service() -> ExtractionService:
    """Get the global extraction service instance."""
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = ExtractionService()
    return _extraction_service


def reset_extraction_service() -> None:
    """Reset the extraction service (useful for testing)."""
    global _extraction_service
    _extraction_service = None

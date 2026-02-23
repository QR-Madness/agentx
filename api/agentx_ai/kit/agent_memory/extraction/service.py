"""
Extraction Service - LLM-based extraction for memory system.

Provides unified interface for entity, fact, and relationship extraction
using configured model providers. Supports multiple consolidation stages:
- Relevance filtering (skip low-value turns)
- Fact/entity extraction with condensation
- Contradiction detection
- User correction handling
"""

import json
import logging
import re
from typing import Any, Optional, Tuple

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


class RelevanceResult(BaseModel):
    """Result of relevance check."""
    is_relevant: bool = False
    reason: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


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

    def _get_provider_for_stage(self, stage: str) -> Tuple[Any, str, float, int]:
        """
        Get provider, model, temperature, and max_tokens for a consolidation stage.

        Args:
            stage: One of 'extraction', 'relevance', 'contradiction', 'correction', 'entity_linking'

        Returns:
            Tuple of (provider, model_id, temperature, max_tokens)
        """
        s = self.settings

        stage_config = {
            'extraction': (s.extraction_model, s.extraction_temperature, s.extraction_max_tokens),
            'relevance': (s.relevance_filter_model, s.relevance_filter_temperature, s.relevance_filter_max_tokens),
            'contradiction': (s.contradiction_model, s.contradiction_temperature, s.contradiction_max_tokens),
            'correction': (s.correction_model, s.correction_temperature, s.correction_max_tokens),
            'entity_linking': (s.entity_linking_model, 0.2, 500),
        }

        if stage not in stage_config:
            raise ValueError(f"Unknown stage: {stage}")

        model, temperature, max_tokens = stage_config[stage]

        try:
            provider, model_id = self.registry.get_provider_for_model(model)
            return provider, model_id, temperature, max_tokens
        except ValueError as e:
            logger.warning(f"Provider for {stage} not available: {e}")
            raise

    def check_relevance(self, text: str) -> RelevanceResult:
        """
        Check if text contains memorable information worth extracting.

        Uses a fast LLM call to filter out low-value turns like "thanks", "ok", etc.

        Args:
            text: The text to check

        Returns:
            RelevanceResult indicating if extraction should proceed
        """
        if not self.settings.relevance_filter_enabled:
            return RelevanceResult(is_relevant=True, reason="filter_disabled")

        # Quick heuristic pre-filter (skip LLM for obvious cases)
        text_lower = text.strip().lower()
        skip_patterns = [
            "ok", "okay", "thanks", "thank you", "got it", "sure", "yes", "no",
            "yep", "nope", "alright", "sounds good", "perfect", "great", "cool",
            "understood", "i see", "ah", "oh", "hmm", "hm", "um", "uh",
        ]
        if text_lower in skip_patterns or len(text.strip()) < 10:
            return RelevanceResult(is_relevant=False, reason="heuristic_skip")

        try:
            provider, model_id, temperature, max_tokens = self._get_provider_for_stage('relevance')
        except ValueError as e:
            # If provider unavailable, default to allowing extraction
            return RelevanceResult(is_relevant=True, reason="provider_unavailable", error=str(e))

        prompt = f'''Does this text contain memorable information worth storing long-term?
Memorable = facts, preferences, personal details, goals, relationships, or specific claims.
NOT memorable = greetings, acknowledgments, filler words, or generic responses.

Text: "{text}"

Reply with only YES or NO.'''

        messages = [
            Message(role=MessageRole.USER, content=prompt),
        ]

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response = result.content.strip().upper()
            is_relevant = response.startswith("YES")

            logger.debug(f"Relevance check: '{text[:50]}...' -> {response}")
            return RelevanceResult(is_relevant=is_relevant, reason=f"llm_{response}")

        except Exception as e:
            logger.warning(f"Relevance check failed: {e}, defaulting to relevant")
            return RelevanceResult(is_relevant=True, reason="error_default", error=str(e))

    def extract_all(
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
            provider, model_id, temperature, max_tokens = self._get_provider_for_stage('extraction')
        except ValueError as e:
            logger.warning(f"Extraction model not available: {e}")
            return ExtractionResult(success=False, error=str(e))

        prompt = self._build_combined_extraction_prompt(text)
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._get_system_prompt()),
            Message(role=MessageRole.USER, content=prompt),
        ]

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=max_tokens,
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

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract only entities from text."""
        result = self.extract_all(text)
        return result.entities

    def extract_facts(self, text: str) -> list[dict[str, Any]]:
        """Extract only facts from text."""
        result = self.extract_all(text)
        return result.facts

    def extract_relationships(
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
        result = self.extract_all(text)
        return result.relationships

    def _get_system_prompt(self) -> str:
        """Get the system prompt for extraction."""
        entity_types = ", ".join(self.settings.entity_types)
        relationship_types = ", ".join(self.settings.relationship_types)
        condense_instruction = """
- CONDENSE verbose statements into atomic facts (one claim per fact)
  Example: "My birthday is March 15th and I was born in 1990" → TWO facts:
    1. "User's birthday is March 15th"
    2. "User was born in 1990" """ if self.settings.extraction_condense_facts else ""

        return f"""You are an information extraction system for a personal memory system.
Extract ONLY information that the USER explicitly stated about themselves, their preferences, or their world.

CRITICAL RULES:
- Extract ONLY what the user directly stated — not inferences or assumptions
- DO NOT extract information from assistant/AI responses
- DO NOT extract generic facts or common knowledge
- Facts should be personal, specific, and worth remembering long-term{condense_instruction}

CONFIDENCE SCORING:
- 0.9-1.0: User explicitly stated ("My birthday is March 15")
- 0.7-0.9: Clearly implied ("I'll be visiting Paris next month" → user is planning a Paris trip)
- 0.5-0.7: Reasonable inference with some uncertainty
- Below 0.5: Do not extract — too speculative

ENTITY TYPES: {entity_types}
RELATIONSHIP TYPES: {relationship_types}

Output ONLY valid JSON with no markdown formatting, code fences, or explanation."""

    def _build_combined_extraction_prompt(self, text: str) -> str:
        """Build the user prompt with JSON output schema."""
        # Truncate very long texts to avoid token limits
        max_chars = 8000
        if len(text) > max_chars:
            truncated_chars = len(text) - max_chars
            logger.warning(
                f"Text truncated for extraction: {truncated_chars} chars removed "
                f"({len(text)} -> {max_chars})"
            )
            text = text[:max_chars] + "\n...[truncated]"

        return f'''Extract personal information from what the USER stated in this text:

"""
{text}
"""

Remember:
- Only extract what the USER explicitly stated
- Condense into atomic facts (one claim each)
- Include confidence based on how explicit the statement was
- Facts should start with "User..." or reference specific entities

Return a JSON object:
{{
  "entities": [
    {{"name": "string", "type": "Person|Organization|Location|etc", "description": "brief description", "confidence": 0.0-1.0}}
  ],
  "facts": [
    {{"claim": "User's birthday is March 15th", "confidence": 0.95, "entity_names": []}}
  ],
  "relationships": [
    {{"source": "entity_name", "relation": "relationship_type", "target": "entity_name", "confidence": 0.0-1.0}}
  ]
}}

If nothing worth remembering, return: {{"entities": [], "facts": [], "relationships": []}}'''

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

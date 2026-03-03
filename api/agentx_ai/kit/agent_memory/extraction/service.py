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
from ....agent.output_parser import validate_json_output, parse_output, extract_yes_no_answer
from ....prompts.loader import get_prompt_loader
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


class CorrectionResult(BaseModel):
    """Result of user correction detection."""
    is_correction: bool = False
    original_claim: Optional[str] = None
    corrected_claim: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class ContradictionResult(BaseModel):
    """Result of contradiction check."""
    has_contradiction: bool = False
    contradicting_fact_id: Optional[str] = None
    reason: Optional[str] = None
    resolution: str = "flag_review"  # prefer_new, prefer_old, flag_review
    success: bool = True
    error: Optional[str] = None


class CombinedExtractionResult(BaseModel):
    """
    Result of combined relevance check + extraction in a single LLM call.

    This reduces LLM calls by ~75% compared to separate relevance + extraction.
    Uses a reasoning model (nvidia/nemotron-3-nano by default) for better quality.
    """
    is_relevant: bool = False
    reason: str = ""  # "heuristic_skip", "llm_not_relevant", "llm_extracted"
    entities: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


# Heuristic patterns that suggest user is making a correction
CORRECTION_PATTERNS = [
    r"^actually[,\s]",
    r"^no[,\s].*(?:i meant|that's|it's)",
    r"^sorry[,\s].*(?:i meant|misspoke|wrong)",
    r"^i meant\b",
    r"^correction[:\s]",
    r"^wait[,\s].*(?:i meant|that's wrong)",
    r"that'?s (?:not right|wrong|incorrect)",
    r"^let me correct",
    r"^i misspoke",
    r"not .+[,\s]+(?:but|it's|rather)",
]


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
            'combined': (s.combined_extraction_model, s.combined_extraction_temperature, s.combined_extraction_max_tokens),
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
        loader = get_prompt_loader()
        text_lower = text.strip().lower()
        skip_patterns = loader.get_list("constants.skip_patterns")
        if text_lower in skip_patterns or len(text.strip()) < 10:
            return RelevanceResult(is_relevant=False, reason="heuristic_skip")

        try:
            provider, model_id, temperature, max_tokens = self._get_provider_for_stage('relevance')
        except ValueError as e:
            # If provider unavailable, default to allowing extraction
            return RelevanceResult(is_relevant=True, reason="provider_unavailable", error=str(e))

        # Use custom prompt if configured, otherwise use default
        if self.settings.relevance_filter_prompt:
            prompt = self.settings.relevance_filter_prompt.replace("{text}", text)
        else:
            prompt = loader.get("extraction.relevance", text=text)

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

            # Log raw response for debugging
            logger.debug(f"Relevance raw response:\n{result.content}")

            # Parse to strip thinking tags
            parsed = parse_output(result.content)
            logger.debug(f"Relevance parsed content: {parsed.content}")
            logger.debug(f"Relevance has_thinking: {parsed.has_thinking}")

            # Use robust YES/NO extraction that handles reasoning models
            is_relevant = extract_yes_no_answer(result.content)
            logger.debug(f"Relevance extracted answer: {is_relevant}")

            # If extraction failed, default to relevant to avoid missing data
            if is_relevant is None:
                logger.warning("Could not extract YES/NO from response")
                is_relevant = True
                reason = "parse_failed_default_yes"
            else:
                reason = f"llm_{'YES' if is_relevant else 'NO'}"

            logger.debug(f"Relevance check: '{text[:50]}...' -> {reason}")
            return RelevanceResult(is_relevant=is_relevant, reason=reason)

        except Exception as e:
            logger.warning(f"Relevance check failed: {e}, defaulting to relevant")
            return RelevanceResult(is_relevant=True, reason="error_default", error=str(e))

    def check_correction(self, text: str) -> CorrectionResult:
        """
        Check if text contains a user correction of previously stated information.

        Uses heuristic patterns first, then LLM if a pattern matches.

        Args:
            text: The text to check for corrections

        Returns:
            CorrectionResult with original and corrected claims if found
        """
        if not self.settings.correction_detection_enabled:
            return CorrectionResult(is_correction=False)

        # Quick heuristic pre-filter
        text_lower = text.strip().lower()
        pattern_matched = False
        for pattern in CORRECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                pattern_matched = True
                break

        if not pattern_matched:
            return CorrectionResult(is_correction=False)

        # Pattern matched - use LLM to extract correction details
        try:
            provider, model_id, temperature, max_tokens = self._get_provider_for_stage('correction')
        except ValueError as e:
            logger.warning(f"Correction model not available: {e}")
            return CorrectionResult(success=False, error=str(e))

        loader = get_prompt_loader()
        prompt = loader.get("extraction.correction", text=text)

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

            # Parse response
            parsed = parse_output(result.content)
            content = parsed.content

            # Check for CORRECTION: YES/NO
            if "CORRECTION: NO" in content.upper():
                return CorrectionResult(is_correction=False)

            if "CORRECTION: YES" not in content.upper():
                logger.debug(f"Unclear correction response: {content[:100]}")
                return CorrectionResult(is_correction=False)

            # Extract ORIGINAL and CORRECTED
            original = None
            corrected = None

            original_match = re.search(r"ORIGINAL:\s*(.+?)(?:\n|CORRECTED:|$)", content, re.IGNORECASE | re.DOTALL)
            if original_match:
                original = original_match.group(1).strip()

            corrected_match = re.search(r"CORRECTED:\s*(.+?)(?:\n|$)", content, re.IGNORECASE | re.DOTALL)
            if corrected_match:
                corrected = corrected_match.group(1).strip()

            logger.info(f"Correction detected: '{original}' -> '{corrected}'")
            return CorrectionResult(
                is_correction=True,
                original_claim=original,
                corrected_claim=corrected,
                success=True,
            )

        except Exception as e:
            logger.warning(f"Correction check failed: {e}")
            return CorrectionResult(success=False, error=str(e))

    def check_contradictions(
        self,
        new_claim: str,
        existing_facts: list[dict[str, Any]],
    ) -> ContradictionResult:
        """
        Check if a new fact contradicts any existing facts.

        Args:
            new_claim: The new fact claim to check
            existing_facts: List of existing facts with 'id' and 'claim' keys

        Returns:
            ContradictionResult indicating if contradiction was found
        """
        if not self.settings.contradiction_detection_enabled:
            return ContradictionResult(has_contradiction=False)

        if not existing_facts:
            return ContradictionResult(has_contradiction=False)

        try:
            provider, model_id, temperature, max_tokens = self._get_provider_for_stage('contradiction')
        except ValueError as e:
            logger.warning(f"Contradiction model not available: {e}")
            return ContradictionResult(success=False, error=str(e))

        # Format existing facts as numbered list
        facts_text = "\n".join(
            f"[{f['id']}] {f['claim']}"
            for f in existing_facts
            if f.get('claim')
        )

        loader = get_prompt_loader()
        prompt = loader.get(
            "extraction.contradiction",
            new_claim=new_claim,
            existing_facts=facts_text,
        )

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

            # Parse response
            parsed = parse_output(result.content)
            content = parsed.content

            # Check for CONTRADICTION: YES/NO
            if "CONTRADICTION: NO" in content.upper():
                return ContradictionResult(has_contradiction=False)

            if "CONTRADICTION: YES" not in content.upper():
                logger.debug(f"Unclear contradiction response: {content[:100]}")
                return ContradictionResult(has_contradiction=False)

            # Extract FACT_ID, REASON, RESOLUTION
            fact_id = None
            reason = None
            resolution = "flag_review"

            fact_id_match = re.search(r"FACT_ID:\s*\[?([^\]\n]+)\]?", content, re.IGNORECASE)
            if fact_id_match:
                fact_id = fact_id_match.group(1).strip()

            reason_match = re.search(r"REASON:\s*(.+?)(?:\n|RESOLUTION:|$)", content, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()

            resolution_match = re.search(r"RESOLUTION:\s*(PREFER_NEW|PREFER_OLD|FLAG_REVIEW)", content, re.IGNORECASE)
            if resolution_match:
                resolution = resolution_match.group(1).lower()

            logger.info(f"Contradiction detected: fact_id={fact_id}, resolution={resolution}")
            return ContradictionResult(
                has_contradiction=True,
                contradicting_fact_id=fact_id,
                reason=reason,
                resolution=resolution,
                success=True,
            )

        except Exception as e:
            logger.warning(f"Contradiction check failed: {e}")
            return ContradictionResult(success=False, error=str(e))

    def _apply_confidence_calibration(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Map LLM certainty levels to calibrated confidence scores.

        The LLM provides a 'certainty' field (explicit/implied/inferred/uncertain)
        which we map to numeric confidence values based on settings.

        Args:
            facts: List of fact dictionaries with 'certainty' field

        Returns:
            Facts with 'certainty' replaced by calibrated 'confidence' score
        """
        mapping = {
            "explicit": self.settings.confidence_explicit,
            "implied": self.settings.confidence_implied,
            "inferred": self.settings.confidence_inferred,
            "uncertain": self.settings.confidence_uncertain,
        }

        for fact in facts:
            certainty = fact.pop("certainty", "inferred")
            fact["confidence"] = mapping.get(certainty.lower(), self.settings.confidence_inferred)

        return facts

    def check_relevance_and_extract(
        self,
        text: str,
        source_turn_id: Optional[str] = None,
    ) -> CombinedExtractionResult:
        """
        Combined relevance check and extraction in a single LLM call.

        This reduces LLM calls by ~75% compared to separate relevance + extraction.
        Uses a reasoning model (nvidia/nemotron-3-nano by default) for better quality
        on the two-step analysis task.

        Args:
            text: The text to check and extract from
            source_turn_id: Optional source turn ID for attribution

        Returns:
            CombinedExtractionResult with relevance and extracted data
        """
        # Quick heuristic pre-filter (skip LLM for obvious cases)
        loader = get_prompt_loader()
        text_lower = text.strip().lower()
        skip_patterns = loader.get_list("constants.skip_patterns")

        if text_lower in skip_patterns or len(text.strip()) < 10:
            return CombinedExtractionResult(
                is_relevant=False,
                reason="heuristic_skip",
                success=True,
            )

        # Get provider for combined extraction (uses reasoning model)
        try:
            provider, model_id, temperature, max_tokens = self._get_provider_for_stage('combined')
        except ValueError as e:
            logger.warning(f"Combined extraction provider unavailable: {e}")
            return CombinedExtractionResult(
                is_relevant=True,  # Default to relevant when provider unavailable
                reason="provider_unavailable",
                success=False,
                error=str(e),
            )

        # Build prompt using the combined relevance+extraction template
        prompt = loader.get("extraction.combined_with_relevance", text=text)

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

            tokens_used = result.usage.get("total_tokens", 0) if result.usage else 0

            # Parse the response
            parsed = self._parse_combined_response(result.content)

            if not parsed.success:
                logger.warning(f"Failed to parse combined response: {parsed.error}")
                return CombinedExtractionResult(
                    is_relevant=True,  # Default to relevant on parse error
                    reason="parse_error_default_relevant",
                    tokens_used=tokens_used,
                    success=False,
                    error=parsed.error,
                )

            # Apply confidence calibration to facts
            if parsed.facts:
                parsed.facts = self._apply_confidence_calibration(parsed.facts)

            # Add source turn ID to facts
            if source_turn_id:
                for fact in parsed.facts:
                    fact["source_turn_id"] = source_turn_id

            parsed.tokens_used = tokens_used
            parsed.reason = "llm_extracted" if parsed.is_relevant else "llm_not_relevant"

            logger.debug(
                f"Combined extraction: relevant={parsed.is_relevant}, "
                f"{len(parsed.entities)} entities, {len(parsed.facts)} facts"
            )

            return parsed

        except Exception as e:
            logger.error(f"Combined extraction failed: {e}")
            return CombinedExtractionResult(
                is_relevant=True,  # Default to relevant on error
                reason="error_default_relevant",
                success=False,
                error=str(e),
            )

    def _parse_combined_response(self, content: str) -> CombinedExtractionResult:
        """
        Parse combined relevance+extraction LLM response.

        Handles reasoning model output with thinking tags.

        Args:
            content: Raw LLM response

        Returns:
            CombinedExtractionResult with parsed data
        """
        # Strip thinking tags from reasoning models
        parsed_output = parse_output(content)
        cleaned_content = parsed_output.content

        if parsed_output.has_thinking:
            logger.debug("Stripped thinking tags from combined response")

        # Try to parse JSON
        is_valid, parsed, error = validate_json_output(cleaned_content)

        if is_valid and parsed:
            return CombinedExtractionResult(
                is_relevant=parsed.get("is_relevant", False),
                entities=parsed.get("entities", []),
                facts=parsed.get("facts", []),
                relationships=parsed.get("relationships", []),
                success=True,
            )

        # Try regex patterns for JSON extraction
        json_patterns = [
            r'```json\s*([\s\S]*?)```',
            r'```\s*([\s\S]*?)```',
            r'(\{[\s\S]*\})',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, cleaned_content)
            if match:
                try:
                    json_str = match.group(1).strip()
                    parsed = json.loads(json_str)
                    return CombinedExtractionResult(
                        is_relevant=parsed.get("is_relevant", False),
                        entities=parsed.get("entities", []),
                        facts=parsed.get("facts", []),
                        relationships=parsed.get("relationships", []),
                        success=True,
                    )
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Failed to parse combined response: {error}")
        logger.debug(f"Raw combined response: {cleaned_content[:500]}...")
        return CombinedExtractionResult(
            success=False,
            error=f"JSON parse error: {error}",
        )

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

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for extraction (used when no custom prompt set)."""
        loader = get_prompt_loader()
        entity_types = ", ".join(self.settings.entity_types)
        relationship_types = ", ".join(self.settings.relationship_types)
        condense_instruction = (
            loader.get("extraction.condense_instruction")
            if self.settings.extraction_condense_facts else ""
        )

        return loader.get(
            "extraction.system",
            entity_types=entity_types,
            relationship_types=relationship_types,
            condense_instruction=condense_instruction,
        )

    def _get_default_relevance_prompt(self) -> str:
        """Get the default relevance filter prompt (used when no custom prompt set)."""
        loader = get_prompt_loader()
        return loader.get("extraction.relevance")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for extraction."""
        # Use custom prompt if configured
        if self.settings.extraction_system_prompt:
            # Allow template variables in custom prompt
            return self.settings.extraction_system_prompt.replace(
                "{entity_types}", ", ".join(self.settings.entity_types)
            ).replace(
                "{relationship_types}", ", ".join(self.settings.relationship_types)
            )

        return self._get_default_system_prompt()

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

        loader = get_prompt_loader()
        return loader.get("extraction.combined", text=text)

    def _parse_extraction_response(self, content: str) -> ExtractionResult:
        """
        Parse LLM response into ExtractionResult.

        Handles various response formats including markdown-wrapped JSON.
        Strips thinking tags from reasoning models before parsing.

        Args:
            content: The raw LLM response content

        Returns:
            ExtractionResult with parsed entities, facts, and relationships
        """
        # Strip thinking tags from reasoning models (e.g., Nemotron)
        parsed_output = parse_output(content)
        cleaned_content = parsed_output.content

        if parsed_output.has_thinking:
            logger.debug("Stripped thinking tags from extraction response")

        # First try the output_parser utility
        is_valid, parsed, error = validate_json_output(cleaned_content)

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
            match = re.search(pattern, cleaned_content)
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
        logger.debug(f"Raw response: {cleaned_content[:500]}...")
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

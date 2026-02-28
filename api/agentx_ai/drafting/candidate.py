"""
Candidate Generation implementation.

Generates multiple candidate responses from one or more models,
then selects the best using various scoring/ranking methods.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry
from .base import DraftingConfig, DraftingStrategy, DraftResult, DraftStatus

logger = logging.getLogger(__name__)


class ScoringMethod(str, Enum):
    """Methods for scoring and selecting candidates."""
    MAJORITY_VOTE = "majority_vote"  # Most common answer wins
    LENGTH_PREFERENCE = "length_preference"  # Prefer longer/shorter responses
    CONFIDENCE = "confidence"  # Use model confidence/logprobs
    VERIFIER = "verifier"  # Use a separate model to score
    CUSTOM = "custom"  # Custom scoring function


@dataclass
class CandidateConfig:
    """Configuration for candidate generation."""
    name: str
    models: list[str]  # Models to generate candidates from
    candidates_per_model: int = 1  # How many candidates from each model
    scoring_method: ScoringMethod = ScoringMethod.MAJORITY_VOTE
    
    # For length preference
    prefer_longer: bool = True
    
    # For verifier scoring
    verifier_model: Optional[str] = None
    verifier_prompt: Optional[str] = None
    
    # For custom scoring
    custom_scorer: Optional[Callable[[str, list[str]], float]] = None
    
    # Temperature settings (higher = more diversity)
    base_temperature: float = 0.7
    temperature_increment: float = 0.1  # Increase temp for each candidate


@dataclass
class Candidate:
    """A single candidate response."""
    content: str
    model: str
    index: int
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CandidateGenerator(DraftingStrategy):
    """
    N-best candidate generation with scoring.
    
    Generates multiple candidates from one or more models,
    then selects the best using configurable scoring methods.
    
    Useful for:
    - Self-consistency: Generate multiple reasoning paths, majority vote
    - Ensemble: Combine outputs from different models
    - Quality improvement: Generate many, select best
    
    Example usage:
        generator = CandidateGenerator(CandidateConfig(
            name="consensus",
            models=["gpt-4", "claude-3-sonnet", "llama3"],
            candidates_per_model=2,
            scoring_method=ScoringMethod.MAJORITY_VOTE,
        ))
        result = await generator.generate(messages)
    """
    
    def __init__(self, config: CandidateConfig):
        self.cand_config = config
        super().__init__(DraftingConfig(
            name=config.name,
            strategy_type="candidate",
            extra={"cand_config": config},
        ))
        self._registry = None
    
    @property
    def name(self) -> str:
        return self.cand_config.name
    
    @property
    def strategy_type(self) -> str:
        return "candidate"
    
    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    async def generate(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> DraftResult:
        """
        Generate candidates and select the best one.
        """
        start_time = time.time()
        
        # Generate all candidates
        candidates = await self._generate_candidates(messages, **kwargs)
        
        if not candidates:
            return DraftResult(
                content="",
                strategy=self.name,
                status=DraftStatus.FAILED,
                candidates_generated=0,
            )
        
        # Score candidates
        scored_candidates = await self._score_candidates(candidates, messages)
        
        # Select best
        best = max(scored_candidates, key=lambda c: c.score)
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate totals
        total_input_tokens = sum(
            c.metadata.get("input_tokens", 0) for c in candidates
        )
        total_output_tokens = sum(
            c.metadata.get("output_tokens", 0) for c in candidates
        )
        
        return DraftResult(
            content=best.content,
            strategy=self.name,
            status=DraftStatus.COMPLETE,
            total_time_ms=total_time,
            models_used=list(set(c.model for c in candidates)),
            candidates_generated=len(candidates),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            raw_results=[
                {
                    "model": c.model,
                    "index": c.index,
                    "content": c.content[:200] + "..." if len(c.content) > 200 else c.content,
                    "score": c.score,
                    "selected": c == best,
                }
                for c in scored_candidates
            ],
        )
    
    async def _generate_candidates(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> list[Candidate]:
        """Generate candidates from all configured models."""
        tasks = []
        
        for model in self.cand_config.models:
            for i in range(self.cand_config.candidates_per_model):
                # Vary temperature slightly for diversity
                temp = (
                    self.cand_config.base_temperature +
                    i * self.cand_config.temperature_increment
                )
                tasks.append(self._generate_single(model, i, messages, temp, **kwargs))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        candidates = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Candidate generation failed: {result}")
            elif result is not None:
                candidates.append(result)
        
        return candidates
    
    async def _generate_single(
        self,
        model: str,
        index: int,
        messages: list[Message],
        temperature: float,
        **kwargs: Any,
    ) -> Optional[Candidate]:
        """Generate a single candidate."""
        try:
            provider, model_id = self.registry.get_provider_for_model(model)
            
            result = await provider.complete(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens"),
            )
            
            metadata = {"temperature": temperature}
            if result.usage:
                metadata["input_tokens"] = result.usage.get("prompt_tokens", 0)
                metadata["output_tokens"] = result.usage.get("completion_tokens", 0)
            
            return Candidate(
                content=result.content,
                model=model,
                index=index,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to generate candidate from {model}: {e}")
            return None
    
    async def _score_candidates(
        self,
        candidates: list[Candidate],
        messages: list[Message],
    ) -> list[Candidate]:
        """Score all candidates using the configured method."""
        method = self.cand_config.scoring_method
        
        if method == ScoringMethod.MAJORITY_VOTE:
            return self._score_majority_vote(candidates)
        elif method == ScoringMethod.LENGTH_PREFERENCE:
            return self._score_by_length(candidates)
        elif method == ScoringMethod.VERIFIER:
            return await self._score_with_verifier(candidates, messages)
        elif method == ScoringMethod.CUSTOM:
            return self._score_custom(candidates)
        else:
            # Default: equal scores
            for c in candidates:
                c.score = 1.0
            return candidates
    
    def _score_majority_vote(self, candidates: list[Candidate]) -> list[Candidate]:
        """Score by similarity to other candidates (majority vote approximation)."""
        # Simple word overlap as similarity proxy
        for i, c in enumerate(candidates):
            c_words = set(c.content.lower().split())
            
            similarity_sum = 0.0
            for j, other in enumerate(candidates):
                if i != j:
                    other_words = set(other.content.lower().split())
                    if c_words or other_words:
                        overlap = len(c_words & other_words)
                        union = len(c_words | other_words)
                        similarity_sum += overlap / union if union > 0 else 0
            
            # Average similarity to others
            c.score = similarity_sum / (len(candidates) - 1) if len(candidates) > 1 else 1.0
        
        return candidates
    
    def _score_by_length(self, candidates: list[Candidate]) -> list[Candidate]:
        """Score by response length."""
        lengths = [len(c.content) for c in candidates]
        max_len = max(lengths) if lengths else 1
        min_len = min(lengths) if lengths else 0
        range_len = max_len - min_len if max_len != min_len else 1
        
        for c in candidates:
            normalized = (len(c.content) - min_len) / range_len
            c.score = normalized if self.cand_config.prefer_longer else (1 - normalized)
        
        return candidates
    
    async def _score_with_verifier(
        self,
        candidates: list[Candidate],
        messages: list[Message],
    ) -> list[Candidate]:
        """Score using a verifier model."""
        if not self.cand_config.verifier_model:
            logger.warning("No verifier model configured, using equal scores")
            for c in candidates:
                c.score = 1.0
            return candidates
        
        try:
            provider, model_id = self.registry.get_provider_for_model(
                self.cand_config.verifier_model
            )
        except ValueError as e:
            logger.error(f"Verifier model not available: {e}")
            for c in candidates:
                c.score = 1.0
            return candidates
        
        # Score each candidate
        for c in candidates:
            prompt = self.cand_config.verifier_prompt or (
                "Rate the following response on a scale of 1-10 for quality, "
                "accuracy, and helpfulness. Respond with just the number."
            )
            
            verify_messages = [
                Message(role=MessageRole.SYSTEM, content=prompt),
                Message(role=MessageRole.USER, content=f"Response to evaluate:\n\n{c.content}"),
            ]
            
            try:
                result = await provider.complete(
                    verify_messages,
                    model_id,
                    temperature=0.0,
                    max_tokens=10,
                )
                
                # Parse score from response
                score_text = result.content.strip()
                try:
                    score = float(score_text.split()[0])
                    c.score = min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
                except (ValueError, IndexError):
                    c.score = 0.5
            except Exception as e:
                logger.error(f"Verifier scoring failed: {e}")
                c.score = 0.5
        
        return candidates
    
    def _score_custom(self, candidates: list[Candidate]) -> list[Candidate]:
        """Score using custom scoring function."""
        if not self.cand_config.custom_scorer:
            logger.warning("No custom scorer configured, using equal scores")
            for c in candidates:
                c.score = 1.0
            return candidates
        
        all_contents = [c.content for c in candidates]
        for c in candidates:
            try:
                c.score = self.cand_config.custom_scorer(c.content, all_contents)
            except Exception as e:
                logger.error(f"Custom scoring failed: {e}")
                c.score = 0.5
        
        return candidates
    
    async def validate(self) -> bool:
        """Validate that all models are accessible."""
        for model in self.cand_config.models:
            try:
                self.registry.get_provider_for_model(model)
            except ValueError as e:
                logger.error(f"Model {model} validation failed: {e}")
                return False
        
        if self.cand_config.verifier_model:
            try:
                self.registry.get_provider_for_model(self.cand_config.verifier_model)
            except ValueError as e:
                logger.error(f"Verifier model validation failed: {e}")
                return False
        
        return True
    
    def get_description(self) -> str:
        models_str = ", ".join(self.cand_config.models[:3])
        if len(self.cand_config.models) > 3:
            models_str += f" (+{len(self.cand_config.models) - 3} more)"
        
        return (
            f"Candidate generation: {self.cand_config.candidates_per_model} candidates "
            f"from [{models_str}], scored by {self.cand_config.scoring_method.value}"
        )

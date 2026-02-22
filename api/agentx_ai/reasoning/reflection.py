"""
Reflective Reasoning implementation.

Reflection enables self-critique and revision of responses,
improving quality through iterative refinement.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry
from .base import (
    ReasoningConfig,
    ReasoningResult,
    ReasoningStatus,
    ReasoningStrategy,
    ThoughtStep,
    ThoughtType,
)

logger = logging.getLogger(__name__)


@dataclass
class ReflectionConfig:
    """Configuration for reflective reasoning."""
    model: str
    
    # Reflection settings
    max_revisions: int = 3
    reflection_model: Optional[str] = None  # Use different model for critique
    
    # Prompts
    critique_prompt: str = (
        "Review the following response critically. "
        "Identify any errors, weaknesses, or areas for improvement. "
        "Be specific and constructive."
    )
    revision_prompt: str = (
        "Based on the critique, revise and improve the response. "
        "Address all the issues mentioned while maintaining what was good."
    )
    
    # Stopping criteria
    min_improvement_threshold: float = 0.1  # Stop if improvement is below this
    satisfaction_threshold: float = 0.9  # Stop if quality is above this


@dataclass
class Revision:
    """A single revision in the reflection process."""
    version: int
    content: str
    critique: str
    score: float
    improvements: list[str]


class ReflectiveReasoner(ReasoningStrategy):
    """
    Reflective reasoning strategy.
    
    Improves responses through iterative self-critique and revision:
    1. Generate initial response
    2. Critique the response (identify weaknesses)
    3. Revise based on critique
    4. Repeat until satisfied or max revisions reached
    
    Example usage:
        reasoner = ReflectiveReasoner(ReflectionConfig(
            model="gpt-4-turbo",
            max_revisions=3,
        ))
        result = await reasoner.reason("Write a compelling introduction for an essay about climate change")
    """
    
    def __init__(self, config: ReflectionConfig):
        self.ref_config = config
        super().__init__(ReasoningConfig(
            name="reflection",
            strategy_type="reflection",
            model=config.model,
            extra={"ref_config": config},
        ))
        self._registry = None
    
    @property
    def name(self) -> str:
        return "reflection"
    
    @property
    def strategy_type(self) -> str:
        return "reflection"
    
    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    def reason(
        self,
        task: str,
        context: Optional[list[Message]] = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Apply reflective reasoning to improve a response.
        """
        start_time = time.time()

        provider, model_id = self.registry.get_provider_for_model(
            self.ref_config.model
        )

        # Get reflection model (may be same as main model)
        reflection_model = self.ref_config.reflection_model or self.ref_config.model
        ref_provider, ref_model_id = self.registry.get_provider_for_model(
            reflection_model
        )

        logger.info(f"Reflective reasoning: {task[:50]}...")

        steps = []
        revisions = []
        total_tokens = 0
        step_num = 0

        # Step 1: Generate initial response
        initial_messages = self._build_initial_prompt(task, context)

        try:
            initial_result = provider.complete(
                initial_messages,
                model_id,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
            )
        except Exception as e:
            logger.error(f"Initial generation failed: {e}")
            return ReasoningResult(
                answer="",
                strategy=self.name,
                status=ReasoningStatus.FAILED,
            )

        if initial_result.usage:
            total_tokens += initial_result.usage.get("total_tokens", 0)

        current_response = initial_result.content

        step_num += 1
        steps.append(ThoughtStep(
            step_number=step_num,
            thought_type=ThoughtType.REASONING,
            content=f"Initial response:\n{current_response[:500]}...",
            model=self.ref_config.model,
        ))

        previous_score = 0.0

        # Iterate: critique and revise
        for revision_num in range(self.ref_config.max_revisions):
            # Step 2: Critique
            critique, critique_score, critique_tokens = self._critique(
                task, current_response, ref_provider, ref_model_id
            )
            total_tokens += critique_tokens

            step_num += 1
            steps.append(ThoughtStep(
                step_number=step_num,
                thought_type=ThoughtType.REFLECTION,
                content=f"Critique (score={critique_score:.2f}):\n{critique}",
                confidence=critique_score,
                model=reflection_model,
            ))

            # Check if we should stop
            if critique_score >= self.ref_config.satisfaction_threshold:
                logger.info(f"Satisfaction threshold reached at revision {revision_num}")
                break

            improvement = critique_score - previous_score
            if revision_num > 0 and improvement < self.ref_config.min_improvement_threshold:
                logger.info(f"Minimal improvement ({improvement:.2f}), stopping")
                break

            previous_score = critique_score

            # Step 3: Revise
            revised_response, revision_tokens = self._revise(
                task, current_response, critique, provider, model_id
            )
            total_tokens += revision_tokens

            step_num += 1
            steps.append(ThoughtStep(
                step_number=step_num,
                thought_type=ThoughtType.REASONING,
                content=f"Revision {revision_num + 1}:\n{revised_response[:500]}...",
                model=self.ref_config.model,
            ))

            # Track revision
            revisions.append(Revision(
                version=revision_num + 1,
                content=revised_response,
                critique=critique,
                score=critique_score,
                improvements=self._extract_improvements(critique),
            ))

            current_response = revised_response
        
        total_time = (time.time() - start_time) * 1000
        
        # Add conclusion step
        step_num += 1
        steps.append(ThoughtStep(
            step_number=step_num,
            thought_type=ThoughtType.CONCLUSION,
            content=f"Final response after {len(revisions)} revisions",
        ))
        
        models_used = [self.ref_config.model]
        if reflection_model != self.ref_config.model:
            models_used.append(reflection_model)
        
        return ReasoningResult(
            answer=current_response,
            strategy=self.name,
            status=ReasoningStatus.COMPLETE,
            steps=steps,
            total_steps=len(steps),
            revisions=len(revisions),
            total_tokens=total_tokens,
            total_time_ms=total_time,
            models_used=models_used,
            raw_trace=[
                {
                    "version": r.version,
                    "score": r.score,
                    "improvements": r.improvements,
                }
                for r in revisions
            ],
        )
    
    def _build_initial_prompt(
        self,
        task: str,
        context: Optional[list[Message]],
    ) -> list[Message]:
        """Build the initial generation prompt."""
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a helpful assistant. Provide a thorough, "
                    "well-structured response to the user's request."
                )
            ),
        ]
        
        if context:
            messages.extend(context)
        
        messages.append(Message(role=MessageRole.USER, content=task))
        
        return messages
    
    def _critique(
        self,
        task: str,
        response: str,
        provider: Any,
        model_id: str,
    ) -> tuple[str, float, int]:
        """Generate a critique of the response."""
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self.ref_config.critique_prompt,
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    f"Original task: {task}\n\n"
                    f"Response to critique:\n{response}\n\n"
                    "Provide your critique, then rate the response from 0.0 to 1.0."
                )
            ),
        ]

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=0.3,  # Lower temperature for more consistent critique
                max_tokens=500,
            )
        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return "Unable to critique", 0.5, 0

        tokens = result.usage.get("total_tokens", 0) if result.usage else 0

        # Extract score from response
        score = self._extract_score(result.content)

        return result.content, score, tokens

    def _revise(
        self,
        task: str,
        response: str,
        critique: str,
        provider: Any,
        model_id: str,
    ) -> tuple[str, int]:
        """Generate a revised response based on critique."""
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self.ref_config.revision_prompt,
            ),
            Message(
                role=MessageRole.USER,
                content=(
                    f"Original task: {task}\n\n"
                    f"Current response:\n{response}\n\n"
                    f"Critique:\n{critique}\n\n"
                    "Please provide an improved version."
                )
            ),
        ]

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=0.7,
                max_tokens=1000,
            )
        except Exception as e:
            logger.error(f"Revision failed: {e}")
            return response, 0  # Return original if revision fails

        tokens = result.usage.get("total_tokens", 0) if result.usage else 0

        return result.content, tokens
    
    def _extract_score(self, critique: str) -> float:
        """Extract a quality score from the critique."""
        import re
        
        # Look for explicit ratings
        patterns = [
            r"(?:rating|score)[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)",
            r"(\d+(?:\.\d+)?)\s*(?:/\s*1\.0|out of 1)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, critique, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1
                if score > 1:
                    score = score / 10.0
                return min(max(score, 0.0), 1.0)
        
        # Heuristic based on critique content
        negative_words = ["error", "wrong", "incorrect", "poor", "weak", "missing", "fail"]
        positive_words = ["good", "excellent", "well", "strong", "clear", "accurate"]
        
        critique_lower = critique.lower()
        negative_count = sum(1 for w in negative_words if w in critique_lower)
        positive_count = sum(1 for w in positive_words if w in critique_lower)
        
        if negative_count + positive_count == 0:
            return 0.5
        
        return positive_count / (negative_count + positive_count)
    
    def _extract_improvements(self, critique: str) -> list[str]:
        """Extract specific improvement suggestions from critique."""
        improvements = []
        
        # Look for numbered points
        import re
        points = re.findall(r'\d+[.)]\s*([^\n]+)', critique)
        improvements.extend(points[:5])  # Limit to 5
        
        # Look for bullet points
        bullets = re.findall(r'[-•]\s*([^\n]+)', critique)
        for bullet in bullets[:5]:
            if bullet not in improvements:
                improvements.append(bullet)
        
        return improvements
    
    def get_description(self) -> str:
        return (
            f"Reflective reasoning with up to {self.ref_config.max_revisions} revisions, "
            f"satisfaction threshold {self.ref_config.satisfaction_threshold}"
        )

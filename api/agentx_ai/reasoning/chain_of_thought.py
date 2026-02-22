"""
Chain-of-Thought (CoT) reasoning implementation.

CoT prompts the model to break down problems into steps,
showing its reasoning process before arriving at an answer.
"""

import logging
import re
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
class CoTConfig:
    """Configuration for Chain-of-Thought reasoning."""
    model: str
    mode: str = "zero_shot"  # "zero_shot", "few_shot", "auto"
    
    # Prompt customization
    thinking_prompt: str = "Let's think step by step."
    step_prefix: str = "Step"
    
    # Few-shot examples (for few_shot mode)
    examples: Optional[list[dict[str, str]]] = None
    
    # Auto-CoT settings
    auto_examples_count: int = 3
    
    # Extraction settings
    extract_steps: bool = True
    extract_confidence: bool = False


# Default few-shot examples for common task types
DEFAULT_EXAMPLES = {
    "math": [
        {
            "question": "If a store has 45 apples and sells 12, then receives a shipment of 30 more, how many apples does it have?",
            "reasoning": "Step 1: Start with 45 apples.\nStep 2: Subtract 12 sold: 45 - 12 = 33 apples.\nStep 3: Add 30 from shipment: 33 + 30 = 63 apples.",
            "answer": "The store has 63 apples."
        },
    ],
    "logic": [
        {
            "question": "All cats are mammals. Some mammals are pets. Can we conclude that some cats are pets?",
            "reasoning": "Step 1: 'All cats are mammals' means every cat belongs to the mammal category.\nStep 2: 'Some mammals are pets' means there exist mammals that are pets.\nStep 3: However, the mammals that are pets might not include any cats.\nStep 4: We cannot logically conclude that some cats are pets from these premises alone.",
            "answer": "No, we cannot conclude that some cats are pets. The logical connection is not guaranteed."
        },
    ],
}


class ChainOfThought(ReasoningStrategy):
    """
    Chain-of-Thought reasoning strategy.
    
    Implements step-by-step reasoning by prompting the model
    to show its work before providing an answer.
    
    Modes:
    - zero_shot: Add "Let's think step by step" to prompt
    - few_shot: Provide examples of step-by-step reasoning
    - auto: Automatically generate relevant examples
    
    Example usage:
        cot = ChainOfThought(CoTConfig(
            model="gpt-4-turbo",
            mode="zero_shot",
        ))
        result = await cot.reason("What is 15% of 80?")
    """
    
    def __init__(self, config: CoTConfig):
        self.cot_config = config
        super().__init__(ReasoningConfig(
            name=f"cot-{config.mode}",
            strategy_type="cot",
            model=config.model,
            extra={"cot_config": config},
        ))
        self._registry = None
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def strategy_type(self) -> str:
        return "cot"
    
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
        Apply Chain-of-Thought reasoning to a task.
        """
        start_time = time.time()

        provider, model_id = self.registry.get_provider_for_model(
            self.cot_config.model
        )

        # Build the CoT prompt
        messages = self._build_cot_prompt(task, context)

        logger.info(f"CoT reasoning ({self.cot_config.mode}): {task[:50]}...")

        try:
            result = provider.complete(
                messages,
                model_id,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
            )
        except Exception as e:
            logger.error(f"CoT reasoning failed: {e}")
            return ReasoningResult(
                answer="",
                strategy=self.name,
                status=ReasoningStatus.FAILED,
            )
        
        total_time = (time.time() - start_time) * 1000
        
        # Extract steps from the response
        steps = []
        if self.cot_config.extract_steps:
            steps = self._extract_steps(result.content)
        
        # Extract the final answer
        answer = self._extract_answer(result.content)
        
        tokens = 0
        if result.usage:
            tokens = result.usage.get("total_tokens", 0)
        
        return ReasoningResult(
            answer=answer,
            strategy=self.name,
            status=ReasoningStatus.COMPLETE,
            steps=steps,
            total_steps=len(steps),
            total_tokens=tokens,
            total_time_ms=total_time,
            models_used=[self.cot_config.model],
            raw_trace=[{"full_response": result.content}],
        )
    
    def _build_cot_prompt(
        self,
        task: str,
        context: Optional[list[Message]] = None,
    ) -> list[Message]:
        """Build the Chain-of-Thought prompt."""
        messages = []
        
        if self.cot_config.mode == "zero_shot":
            # Zero-shot: Just add the thinking prompt
            system = (
                "You are a helpful assistant that thinks through problems step by step. "
                "Show your reasoning process clearly before providing your final answer."
            )
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
            
            if context:
                messages.extend(context)
            
            prompt = f"{task}\n\n{self.cot_config.thinking_prompt}"
            messages.append(Message(role=MessageRole.USER, content=prompt))
            
        elif self.cot_config.mode == "few_shot":
            # Few-shot: Include examples
            system = (
                "You are a helpful assistant that solves problems step by step. "
                "Follow the format shown in the examples: show your reasoning, "
                "then provide the final answer."
            )
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
            
            # Add examples
            examples = self.cot_config.examples or DEFAULT_EXAMPLES.get("math", [])
            for ex in examples:
                messages.append(Message(
                    role=MessageRole.USER,
                    content=f"Question: {ex['question']}"
                ))
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=f"{ex['reasoning']}\n\nAnswer: {ex['answer']}"
                ))
            
            if context:
                messages.extend(context)
            
            messages.append(Message(
                role=MessageRole.USER,
                content=f"Question: {task}"
            ))
            
        else:  # auto mode
            # Auto: Generate examples dynamically (simplified version)
            system = (
                "You are a helpful assistant that thinks through problems carefully. "
                "Break down your reasoning into clear, numbered steps. "
                "After showing your work, provide a clear final answer."
            )
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
            
            if context:
                messages.extend(context)
            
            prompt = (
                f"{task}\n\n"
                f"{self.cot_config.thinking_prompt}\n"
                f"Show your reasoning as numbered steps, then give your final answer."
            )
            messages.append(Message(role=MessageRole.USER, content=prompt))
        
        return messages
    
    def _extract_steps(self, response: str) -> list[ThoughtStep]:
        """Extract reasoning steps from the response."""
        steps = []
        
        # Try to find numbered steps
        step_pattern = rf"{self.cot_config.step_prefix}\s*(\d+)[:\.]?\s*(.+?)(?=(?:{self.cot_config.step_prefix}\s*\d+|Answer:|Final|$))"
        matches = re.findall(step_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if matches:
            for num, content in matches:
                steps.append(ThoughtStep(
                    step_number=int(num),
                    thought_type=ThoughtType.REASONING,
                    content=content.strip(),
                    model=self.cot_config.model,
                ))
        else:
            # Fall back to splitting by newlines
            lines = [ln.strip() for ln in response.split('\n') if ln.strip()]
            for i, line in enumerate(lines):
                if line.lower().startswith(('answer:', 'final', 'therefore', 'thus', 'so,')):
                    break
                steps.append(ThoughtStep(
                    step_number=i + 1,
                    thought_type=ThoughtType.REASONING,
                    content=line,
                    model=self.cot_config.model,
                ))
        
        return steps
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        # Try common answer patterns
        patterns = [
            r"(?:Final\s+)?Answer[:\s]+(.+?)(?:\n\n|$)",
            r"Therefore[,:\s]+(.+?)(?:\n\n|$)",
            r"Thus[,:\s]+(.+?)(?:\n\n|$)",
            r"In conclusion[,:\s]+(.+?)(?:\n\n|$)",
            r"The answer is[:\s]+(.+?)(?:\n\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fall back to last paragraph
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]
        
        return response
    
    def get_description(self) -> str:
        return f"Chain-of-Thought ({self.cot_config.mode}) using {self.cot_config.model}"

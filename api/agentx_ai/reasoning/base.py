"""
Base classes for reasoning strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from ..providers.base import Message


class ReasoningStatus(str, Enum):
    """Status of a reasoning operation."""
    PENDING = "pending"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    FAILED = "failed"


class ThoughtType(str, Enum):
    """Type of thought in a reasoning trace."""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    HYPOTHESIS = "hypothesis"
    ACTION = "action"
    RESULT = "result"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"


class ThoughtStep(BaseModel):
    """A single step in a reasoning trace."""
    step_number: int
    thought_type: ThoughtType
    content: str
    confidence: float = 1.0
    
    # For branching (ToT)
    parent_step: Optional[int] = None
    branch_id: Optional[str] = None
    
    # For actions (ReAct)
    action_name: Optional[str] = None
    action_input: Optional[dict[str, Any]] = None
    action_output: Optional[str] = None
    
    # Metadata
    model: Optional[str] = None
    tokens_used: int = 0
    time_ms: float = 0.0


class ReasoningResult(BaseModel):
    """Result of a reasoning operation."""
    answer: str
    strategy: str
    status: ReasoningStatus = ReasoningStatus.COMPLETE
    
    # Reasoning trace
    steps: list[ThoughtStep] = []
    total_steps: int = 0
    
    # For ToT
    branches_explored: int = 0
    best_branch: Optional[str] = None
    
    # For ReAct
    actions_taken: int = 0
    
    # For Reflection
    revisions: int = 0
    
    # Metrics
    total_tokens: int = 0
    total_time_ms: float = 0.0
    models_used: list[str] = []
    
    # Raw data
    raw_trace: Optional[list[dict[str, Any]]] = None


@dataclass
class ReasoningConfig:
    """Configuration for a reasoning strategy."""
    name: str
    strategy_type: str  # "cot", "tot", "react", "reflection"
    model: str
    
    # Common settings
    temperature: float = 0.7
    max_steps: int = 10
    timeout_seconds: float = 120.0
    
    # Strategy-specific config
    extra: dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(ABC):
    """
    Abstract base class for reasoning strategies.
    
    Reasoning strategies implement different approaches to
    multi-step problem solving and decision making.
    """
    
    def __init__(self, config: ReasoningConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Return the type of strategy (cot, tot, react, reflection)."""
        pass
    
    @abstractmethod
    async def reason(
        self,
        task: str,
        context: Optional[list[Message]] = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Apply reasoning to solve a task.
        
        Args:
            task: The task or question to reason about
            context: Optional conversation context
            **kwargs: Additional parameters
            
        Returns:
            ReasoningResult with answer and reasoning trace
        """
        pass
    
    async def validate(self) -> bool:
        """Validate that the strategy is properly configured."""
        return True
    
    def get_description(self) -> str:
        """Get a human-readable description of this strategy."""
        return f"{self.strategy_type} strategy: {self.name}"

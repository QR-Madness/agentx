"""
Base classes for drafting strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from ..providers.base import Message


class DraftStatus(str, Enum):
    """Status of a drafting operation."""
    PENDING = "pending"
    DRAFTING = "drafting"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


class DraftResult(BaseModel):
    """Result of a drafting operation."""
    content: str
    strategy: str
    status: DraftStatus = DraftStatus.COMPLETE
    
    # Metrics
    draft_tokens: int = 0
    accepted_tokens: int = 0
    total_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Multi-model info
    models_used: list[str] = []
    stages_completed: int = 0
    candidates_generated: int = 0
    
    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    
    # Raw data
    raw_results: Optional[list[dict[str, Any]]] = None


@dataclass
class DraftingConfig:
    """Configuration for a drafting strategy."""
    name: str
    strategy_type: str  # "speculative", "pipeline", "candidate"
    
    # Common settings
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_seconds: float = 60.0
    
    # Strategy-specific config stored here
    extra: dict[str, Any] = field(default_factory=dict)


class DraftingStrategy(ABC):
    """
    Abstract base class for drafting strategies.
    
    Drafting strategies implement different approaches to multi-model
    generation, trading off between speed, quality, and cost.
    """
    
    def __init__(self, config: DraftingConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Return the type of strategy (speculative, pipeline, candidate)."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> DraftResult:
        """
        Generate content using this drafting strategy.
        
        Args:
            messages: The conversation messages
            **kwargs: Additional parameters
            
        Returns:
            DraftResult with generated content and metrics
        """
        pass
    
    async def validate(self) -> bool:
        """
        Validate that the strategy is properly configured.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def get_description(self) -> str:
        """Get a human-readable description of this strategy."""
        return f"{self.strategy_type} strategy: {self.name}"

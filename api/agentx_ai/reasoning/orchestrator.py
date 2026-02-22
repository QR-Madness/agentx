"""
Reasoning Orchestrator for selecting and combining strategies.

The orchestrator selects appropriate reasoning strategies based on
task characteristics and can combine multiple strategies for
complex problems.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..providers.base import Message
from .base import ReasoningResult, ReasoningStrategy
from .chain_of_thought import ChainOfThought, CoTConfig
from .tree_of_thought import TreeOfThought, ToTConfig
from .react import ReActAgent, ReActConfig, Tool
from .reflection import ReflectiveReasoner, ReflectionConfig

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks for strategy selection."""
    SIMPLE = "simple"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    RESEARCH = "research"
    PLANNING = "planning"
    CODE = "code"
    MATH = "math"
    UNKNOWN = "unknown"


@dataclass
class OrchestratorConfig:
    """Configuration for the reasoning orchestrator."""
    # Default to local LM Studio model
    default_model: str = "llama3.2"
    
    # Strategy preferences
    strategy_map: dict[TaskType, str] = field(default_factory=lambda: {
        TaskType.SIMPLE: "cot",
        TaskType.ANALYTICAL: "cot",
        TaskType.CREATIVE: "reflection",
        TaskType.RESEARCH: "react",
        TaskType.PLANNING: "tot",
        TaskType.CODE: "reflection",
        TaskType.MATH: "cot",
        TaskType.UNKNOWN: "cot",
    })
    
    # Model overrides per task type
    model_map: dict[TaskType, str] = field(default_factory=dict)
    
    # ReAct tools
    react_tools: list[Tool] = field(default_factory=list)
    
    # Fallback behavior
    fallback_strategy: str = "cot"
    enable_fallback: bool = True


class ReasoningOrchestrator:
    """
    Orchestrates reasoning strategies for optimal task solving.
    
    The orchestrator:
    1. Analyzes the task to determine its type
    2. Selects an appropriate reasoning strategy
    3. Executes the strategy
    4. Falls back to alternatives if needed
    
    Example usage:
        orchestrator = ReasoningOrchestrator(OrchestratorConfig(
            default_model="gpt-4-turbo",
        ))
        result = await orchestrator.reason("Plan a week-long vacation to Japan")
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._strategies: dict[str, ReasoningStrategy] = {}
    
    def _get_strategy(self, strategy_type: str, model: str) -> ReasoningStrategy:
        """Get or create a reasoning strategy."""
        key = f"{strategy_type}:{model}"
        
        if key not in self._strategies:
            if strategy_type == "cot":
                self._strategies[key] = ChainOfThought(CoTConfig(
                    model=model,
                    mode="zero_shot",
                ))
            elif strategy_type == "tot":
                self._strategies[key] = TreeOfThought(ToTConfig(
                    model=model,
                    branching_factor=3,
                    max_depth=3,
                    search_method="beam",
                ))
            elif strategy_type == "react":
                self._strategies[key] = ReActAgent(ReActConfig(
                    model=model,
                    tools=self.config.react_tools,
                    max_iterations=10,
                ))
            elif strategy_type == "reflection":
                self._strategies[key] = ReflectiveReasoner(ReflectionConfig(
                    model=model,
                    max_revisions=3,
                ))
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return self._strategies[key]
    
    def reason(
        self,
        task: str,
        context: Optional[list[Message]] = None,
        strategy: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Apply reasoning to solve a task.

        Args:
            task: The task or question to reason about
            context: Optional conversation context
            strategy: Override strategy selection (cot, tot, react, reflection)
            task_type: Override task type detection
            **kwargs: Additional parameters passed to the strategy

        Returns:
            ReasoningResult with answer and reasoning trace
        """
        # Determine task type if not provided
        if task_type is None:
            task_type = self._classify_task(task)

        logger.info(f"Task classified as: {task_type.value}")

        # Determine strategy
        if strategy is None:
            strategy = self.config.strategy_map.get(task_type, self.config.fallback_strategy)

        # Determine model
        model = self.config.model_map.get(task_type, self.config.default_model)

        logger.info(f"Using strategy: {strategy}, model: {model}")

        # Get and execute strategy
        try:
            reasoning_strategy = self._get_strategy(strategy, model)
            result = reasoning_strategy.reason(task, context, **kwargs)

            # Check for failure and fallback (safely handle status)
            status_value = getattr(result.status, 'value', str(result.status))
            if status_value == "failed" and self.config.enable_fallback:
                logger.warning(f"Strategy {strategy} failed, trying fallback")
                fallback = self._get_strategy(self.config.fallback_strategy, model)
                result = fallback.reason(task, context, **kwargs)

            return result

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")

            if self.config.enable_fallback:
                logger.info("Attempting fallback strategy")
                try:
                    fallback = self._get_strategy(self.config.fallback_strategy, model)
                    return fallback.reason(task, context, **kwargs)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            return ReasoningResult(
                answer="Unable to complete reasoning due to an error.",
                strategy=strategy,
                status="failed",
            )
    
    def _classify_task(self, task: str) -> TaskType:
        """Classify the task type based on keywords and patterns."""
        task_lower = task.lower()
        
        # Math indicators
        math_words = ["calculate", "compute", "solve", "equation", "formula", "sum", "product", "divide", "multiply"]
        if any(w in task_lower for w in math_words):
            return TaskType.MATH
        
        # Code indicators
        code_words = ["code", "program", "function", "implement", "algorithm", "debug", "fix the bug", "script"]
        if any(w in task_lower for w in code_words):
            return TaskType.CODE
        
        # Research indicators
        research_words = ["search", "find information", "look up", "research", "what is the latest", "current"]
        if any(w in task_lower for w in research_words):
            return TaskType.RESEARCH
        
        # Planning indicators
        planning_words = ["plan", "design", "strategy", "approach", "organize", "schedule", "roadmap"]
        if any(w in task_lower for w in planning_words):
            return TaskType.PLANNING
        
        # Creative indicators
        creative_words = ["write", "create", "compose", "draft", "story", "essay", "poem", "creative"]
        if any(w in task_lower for w in creative_words):
            return TaskType.CREATIVE
        
        # Analytical indicators
        analytical_words = ["analyze", "compare", "evaluate", "assess", "review", "examine", "explain why"]
        if any(w in task_lower for w in analytical_words):
            return TaskType.ANALYTICAL
        
        # Simple task indicators
        simple_words = ["what is", "who is", "when did", "where is", "define", "list"]
        if any(w in task_lower for w in simple_words):
            return TaskType.SIMPLE
        
        return TaskType.UNKNOWN
    
    def add_react_tool(self, tool: Tool) -> None:
        """Add a tool for ReAct reasoning."""
        self.config.react_tools.append(tool)
        
        # Update existing ReAct strategies
        for key, strategy in self._strategies.items():
            if isinstance(strategy, ReActAgent):
                strategy.add_tool(tool)
    
    def set_strategy_for_task_type(self, task_type: TaskType, strategy: str) -> None:
        """Set the strategy to use for a task type."""
        self.config.strategy_map[task_type] = strategy
    
    def set_model_for_task_type(self, task_type: TaskType, model: str) -> None:
        """Set the model to use for a task type."""
        self.config.model_map[task_type] = model
    
    def list_strategies(self) -> list[str]:
        """List available strategy types."""
        return ["cot", "tot", "react", "reflection"]
    
    def get_strategy_description(self, strategy_type: str) -> str:
        """Get description of a strategy type."""
        descriptions = {
            "cot": "Chain-of-Thought: Step-by-step reasoning",
            "tot": "Tree-of-Thought: Explores multiple reasoning paths",
            "react": "ReAct: Reasoning with tool use",
            "reflection": "Reflection: Self-critique and revision",
        }
        return descriptions.get(strategy_type, "Unknown strategy")

"""
Reasoning Orchestrator for selecting and combining strategies.

The orchestrator selects appropriate reasoning strategies based on
task characteristics and can combine multiple strategies for
complex problems.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..providers.base import Message
from .base import ReasoningResult, ReasoningStatus, ReasoningStrategy
from .chain_of_thought import ChainOfThought, CoTConfig
from .selection import TaskType, classify_task_type
from .tree_of_thought import TreeOfThought, ToTConfig
from .react import ReActAgent, ReActConfig, Tool
from .reflection import ReflectiveReasoner, ReflectionConfig

logger = logging.getLogger(__name__)

# Backward-compat re-export: TaskType moved to reasoning/selection.py (ONE
# brain for classification, shared with the chat thinking patterns).
__all__ = ["OrchestratorConfig", "ReasoningOrchestrator", "TaskType"]


@dataclass
class OrchestratorConfig:
    """Configuration for the reasoning orchestrator."""
    # Default to local LM Studio model (use provider:model format)
    default_model: str = "lmstudio:llama3.2"
    
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

    # Wall-clock budget per strategy execution (a hung provider call no longer
    # hangs the whole /agent/run request). Applied around each `reason()` call.
    timeout_seconds: float = 180.0


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
            default_model="anthropic:claude-3-5-sonnet-latest",
        ))
        result = await orchestrator.reason("Plan a week-long vacation to Japan")
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._strategies: dict[str, ReasoningStrategy] = {}
    
    # Chat-first pattern values mapped to their nearest offline kit strategy
    # (the chat path compiles these into the streaming loop instead; a profile
    # carrying one must still work on /agent/run).
    _OFFLINE_ALIASES: dict[str, str] = {
        "native": "cot",
        "step_back": "cot",
        "deep_reflection": "reflection",
        "self_consistency": "cot",
    }

    def _get_strategy(self, strategy_type: str, model: str) -> ReasoningStrategy:
        """Get or create a reasoning strategy."""
        strategy_type = self._OFFLINE_ALIASES.get(strategy_type, strategy_type)
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
    
    async def reason(
        self,
        task: str,
        context: list[Message] | None = None,
        strategy: str | None = None,
        task_type: TaskType | None = None,
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

        # Get and execute strategy (bounded — a hung provider call fails over
        # instead of hanging the whole /agent/run request).
        try:
            reasoning_strategy = self._get_strategy(strategy, model)
            result = await asyncio.wait_for(
                reasoning_strategy.reason(task, context, **kwargs),
                timeout=self.config.timeout_seconds,
            )

            # Check for failure and fallback (safely handle status)
            status_value = getattr(result.status, 'value', str(result.status))
            if status_value == "failed" and self.config.enable_fallback:
                logger.warning(f"Strategy {strategy} failed, trying fallback")
                fallback = self._get_strategy(self.config.fallback_strategy, model)
                result = await asyncio.wait_for(
                    fallback.reason(task, context, **kwargs),
                    timeout=self.config.timeout_seconds,
                )

            return result

        except Exception as e:
            # Top-level guard for genuine provider/runtime failures (incl.
            # TimeoutError from the wall-clock budget above).
            logger.error(f"Reasoning failed: {e}", exc_info=True)

            if self.config.enable_fallback:
                logger.info("Attempting fallback strategy")
                try:
                    fallback = self._get_strategy(self.config.fallback_strategy, model)
                    return await asyncio.wait_for(
                        fallback.reason(task, context, **kwargs),
                        timeout=self.config.timeout_seconds,
                    )
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            return ReasoningResult(
                answer="Unable to complete reasoning due to an error.",
                strategy=strategy,
                status=ReasoningStatus.FAILED,
            )

    def _classify_task(self, task: str) -> TaskType:
        """Classify the task type — delegates to the shared selection module
        (ONE definition for the chat thinking patterns AND this orchestrator)."""
        return classify_task_type(task)
    
    def add_react_tool(self, tool: Tool) -> None:
        """Add a tool for ReAct reasoning."""
        self.config.react_tools.append(tool)
        
        # Update existing ReAct strategies
        for _key, strategy in self._strategies.items():
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

"""
Task Planner for decomposing complex tasks.

The planner breaks down complex tasks into manageable subtasks
and determines the best approach for each.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Estimated complexity of a task."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class SubtaskType(str, Enum):
    """Type of subtask."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TOOL_USE = "tool_use"
    DECISION = "decision"
    VERIFICATION = "verification"


@dataclass
class Subtask:
    """A subtask in a task plan."""
    id: int
    description: str
    type: SubtaskType
    dependencies: list[int] = field(default_factory=list)
    estimated_complexity: TaskComplexity = TaskComplexity.SIMPLE
    tools_needed: list[str] = field(default_factory=list)
    completed: bool = False
    result: Optional[str] = None


@dataclass
class TaskPlan:
    """A plan for executing a task."""
    task: str
    complexity: TaskComplexity
    steps: list[Subtask]
    reasoning_strategy: str = "auto"
    estimated_tokens: int = 0
    
    def get_next_subtask(self) -> Optional[Subtask]:
        """Get the next subtask that can be executed."""
        for subtask in self.steps:
            if subtask.completed:
                continue
            
            # Check if dependencies are satisfied
            deps_satisfied = all(
                self.steps[d].completed
                for d in subtask.dependencies
                if d < len(self.steps)
            )
            
            if deps_satisfied:
                return subtask
        
        return None
    
    def mark_complete(self, subtask_id: int, result: str) -> None:
        """Mark a subtask as complete."""
        if 0 <= subtask_id < len(self.steps):
            self.steps[subtask_id].completed = True
            self.steps[subtask_id].result = result
    
    def is_complete(self) -> bool:
        """Check if all subtasks are complete."""
        return all(s.completed for s in self.steps)
    
    def get_progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if not self.steps:
            return 1.0
        return sum(1 for s in self.steps if s.completed) / len(self.steps)


class TaskPlanner:
    """
    Plans task execution by decomposing into subtasks.
    
    The planner:
    1. Analyzes task complexity
    2. Breaks down into subtasks
    3. Identifies dependencies
    4. Suggests reasoning strategies
    
    Example usage:
        planner = TaskPlanner("gpt-4-turbo")
        plan = await planner.plan("Build a web scraper for news articles")
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self._registry = None
    
    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    async def plan(
        self,
        task: str,
        context: Optional[list[Message]] = None,
    ) -> TaskPlan:
        """
        Create a plan for executing a task.
        
        Args:
            task: The task description
            context: Optional conversation context
            
        Returns:
            TaskPlan with subtasks and execution order
        """
        # First, assess complexity
        complexity = self._assess_complexity(task)
        
        if complexity == TaskComplexity.SIMPLE:
            # Simple tasks don't need decomposition
            return TaskPlan(
                task=task,
                complexity=complexity,
                steps=[
                    Subtask(
                        id=0,
                        description=task,
                        type=SubtaskType.GENERATION,
                        estimated_complexity=TaskComplexity.SIMPLE,
                    )
                ],
                reasoning_strategy="cot",
            )
        
        # For complex tasks, use LLM to decompose
        provider, model_id = self.registry.get_provider_for_model(self.model)
        
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="""You are a task planning assistant. Break down the given task into clear, sequential subtasks.

For each subtask, specify:
1. A clear description
2. The type: RESEARCH, ANALYSIS, GENERATION, TOOL_USE, DECISION, or VERIFICATION
3. Any dependencies on previous subtasks (by number)
4. Tools that might be needed

Format your response as:
SUBTASK 1: [description]
TYPE: [type]
DEPENDS: [comma-separated subtask numbers, or "none"]
TOOLS: [comma-separated tool names, or "none"]

Continue for each subtask."""
            ),
        ]
        
        if context:
            messages.extend(context)
        
        messages.append(Message(
            role=MessageRole.USER,
            content=f"Task to plan: {task}"
        ))
        
        try:
            result = await provider.complete(
                messages,
                model_id,
                temperature=0.3,
                max_tokens=1000,
            )
            
            steps = self._parse_plan(result.content)
            
            # Determine best reasoning strategy
            strategy = self._select_strategy(steps)
            
            return TaskPlan(
                task=task,
                complexity=complexity,
                steps=steps,
                reasoning_strategy=strategy,
                estimated_tokens=result.usage.get("total_tokens", 0) if result.usage else 0,
            )
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fall back to single-step plan
            return TaskPlan(
                task=task,
                complexity=complexity,
                steps=[
                    Subtask(
                        id=0,
                        description=task,
                        type=SubtaskType.GENERATION,
                    )
                ],
                reasoning_strategy="cot",
            )
    
    def _assess_complexity(self, task: str) -> TaskComplexity:
        """Assess the complexity of a task."""
        task_lower = task.lower()
        
        # Complex indicators
        complex_words = [
            "analyze", "design", "build", "create a system",
            "implement", "develop", "architecture", "compare multiple",
            "research and", "plan a", "comprehensive"
        ]
        if any(w in task_lower for w in complex_words):
            return TaskComplexity.COMPLEX
        
        # Moderate indicators
        moderate_words = [
            "explain how", "summarize", "write a", "draft",
            "evaluate", "describe in detail", "list and explain"
        ]
        if any(w in task_lower for w in moderate_words):
            return TaskComplexity.MODERATE
        
        # Check length as a proxy for complexity
        if len(task.split()) > 50:
            return TaskComplexity.COMPLEX
        elif len(task.split()) > 20:
            return TaskComplexity.MODERATE
        
        return TaskComplexity.SIMPLE
    
    def _parse_plan(self, response: str) -> list[Subtask]:
        """Parse the LLM response into subtasks."""
        steps = []
        
        # Pattern to match subtasks
        pattern = r"SUBTASK\s*(\d+)[:\s]+(.+?)(?=SUBTASK\s*\d+|$)"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            subtask_id = int(match[0]) - 1  # Convert to 0-indexed
            content = match[1].strip()
            
            # Extract description (first line)
            lines = content.split('\n')
            description = lines[0].strip()
            
            # Extract type
            type_match = re.search(r"TYPE[:\s]+(\w+)", content, re.IGNORECASE)
            subtask_type = SubtaskType.GENERATION
            if type_match:
                type_str = type_match.group(1).upper()
                try:
                    subtask_type = SubtaskType(type_str.lower())
                except ValueError:
                    pass
            
            # Extract dependencies
            deps_match = re.search(r"DEPENDS[:\s]+(.+?)(?:\n|$)", content, re.IGNORECASE)
            dependencies = []
            if deps_match:
                deps_str = deps_match.group(1).strip()
                if deps_str.lower() != "none":
                    try:
                        dependencies = [int(d.strip()) - 1 for d in deps_str.split(',')]
                    except ValueError:
                        pass
            
            # Extract tools
            tools_match = re.search(r"TOOLS[:\s]+(.+?)(?:\n|$)", content, re.IGNORECASE)
            tools = []
            if tools_match:
                tools_str = tools_match.group(1).strip()
                if tools_str.lower() != "none":
                    tools = [t.strip() for t in tools_str.split(',')]
            
            steps.append(Subtask(
                id=subtask_id,
                description=description,
                type=subtask_type,
                dependencies=dependencies,
                tools_needed=tools,
            ))
        
        # If parsing failed, create a single step
        if not steps:
            steps.append(Subtask(
                id=0,
                description="Execute the task",
                type=SubtaskType.GENERATION,
            ))
        
        return steps
    
    def _select_strategy(self, steps: list[Subtask]) -> str:
        """Select the best reasoning strategy based on subtasks."""
        # Count subtask types
        type_counts = {}
        for step in steps:
            type_counts[step.type] = type_counts.get(step.type, 0) + 1
        
        # If tool use is prominent, use ReAct
        if type_counts.get(SubtaskType.TOOL_USE, 0) >= len(steps) / 3:
            return "react"
        
        # If analysis/decision heavy, use ToT
        analysis_count = type_counts.get(SubtaskType.ANALYSIS, 0) + type_counts.get(SubtaskType.DECISION, 0)
        if analysis_count >= len(steps) / 2:
            return "tot"
        
        # If generation heavy with verification, use reflection
        if type_counts.get(SubtaskType.GENERATION, 0) > 0 and type_counts.get(SubtaskType.VERIFICATION, 0) > 0:
            return "reflection"
        
        # Default to CoT
        return "cot"

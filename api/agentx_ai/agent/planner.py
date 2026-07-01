"""
Task Planner for decomposing complex tasks.

The planner breaks down complex tasks into manageable subtasks
and determines the best approach for each.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry
from ..prompts.loader import get_prompt_loader

if TYPE_CHECKING:
    from ..kit.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


# Instruction appended to the *main agent's* assembled turn context to compose a
# plan with full conversation/memory awareness, returning structured JSON (no
# brittle regex format). The model itself decides whether to decompose.
_COMPOSE_INSTRUCTION = """Before answering, decide whether the user's latest request genuinely benefits from being broken into multiple sequential steps, or whether a single response handles it well.

Respond with ONLY a JSON object — no prose, no markdown fence — in one of these two forms:
- A single response suffices: {{"plan": null}}
- Multiple steps genuinely help: {{"plan": [{{"description": "<what this step accomplishes>", "type": "research|analysis|generation|tool_use|decision|verification", "depends": [<0-based indices of earlier steps this needs>], "tools": ["<tool_name>", ...]}}]}}

Guidelines:
- Use the FEWEST steps that genuinely help (at most {max}). Strongly prefer {{"plan": null}} unless the task has real multi-step structure.
- `depends` are 0-based indices into the plan array, pointing only at EARLIER steps.
- `tools` names the tools a step will need; use [] or omit when none.

The request to assess:
{task}"""


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extract a single JSON object from an LLM response.

    Handles raw JSON, ```json fenced blocks, and JSON embedded in prose (scans
    for the first balanced ``{...}``). Returns None when nothing usable parses.
    """
    if not text:
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Scan for the first balanced top-level object (braces in string values are
    # rare in our plan JSON; the fast path above covers clean output anyway).
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass
                    break
        start = s.find("{", start + 1)
    return None


def _coerce_subtask_type(v: object) -> SubtaskType:
    try:
        return SubtaskType(str(v).strip().lower())
    except Exception:
        return SubtaskType.GENERATION


def _coerce_int_list(v: object) -> list[int]:
    if not isinstance(v, (list, tuple)):
        return []
    out: list[int] = []
    for x in v:
        try:
            out.append(int(x))
        except (ValueError, TypeError):
            continue
    return out


def _coerce_str_list(v: object) -> list[str]:
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if not isinstance(v, (list, tuple)):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


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
    result: str | None = None
    goal_id: str | None = None  # For future subtask-level goal tracking

    def to_dict(self) -> dict:
        """Serialize for Redis storage.

        Includes the full structure (type, dependencies, tools_needed,
        complexity, goal_id) so a plan can be round-tripped and resumed
        after a process death — not just rendered.
        """
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type.value,
            "dependencies": self.dependencies,
            "estimated_complexity": self.estimated_complexity.value,
            "tools_needed": self.tools_needed,
            "completed": self.completed,
            "result": self.result,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Subtask:
        """Rehydrate a Subtask from its ``to_dict`` form (lenient about missing
        optional keys so older snapshots still load)."""
        return cls(
            id=int(data["id"]),
            description=data.get("description", ""),
            type=SubtaskType(data.get("type", SubtaskType.GENERATION.value)),
            dependencies=list(data.get("dependencies", [])),
            estimated_complexity=TaskComplexity(
                data.get("estimated_complexity", TaskComplexity.SIMPLE.value)
            ),
            tools_needed=list(data.get("tools_needed", [])),
            completed=bool(data.get("completed", False)),
            result=data.get("result"),
            goal_id=data.get("goal_id"),
        )


@dataclass
class TaskPlan:
    """A plan for executing a task."""
    task: str
    complexity: TaskComplexity
    steps: list[Subtask]
    reasoning_strategy: str = "auto"
    estimated_tokens: int = 0
    goal_id: str | None = None  # Linked goal in memory system

    def to_dict(self) -> dict:
        """Serialize for Redis storage."""
        return {
            "task": self.task,
            "complexity": self.complexity.value,
            "steps": [s.to_dict() for s in self.steps],
            "reasoning_strategy": self.reasoning_strategy,
            "estimated_tokens": self.estimated_tokens,
            "goal_id": self.goal_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskPlan:
        """Rehydrate a TaskPlan from its ``to_dict`` form (for plan resumption)."""
        return cls(
            task=data.get("task", ""),
            complexity=TaskComplexity(data.get("complexity", TaskComplexity.SIMPLE.value)),
            steps=[Subtask.from_dict(s) for s in data.get("steps", [])],
            reasoning_strategy=data.get("reasoning_strategy", "auto"),
            estimated_tokens=int(data.get("estimated_tokens", 0) or 0),
            goal_id=data.get("goal_id"),
        )

    def get_next_subtask(self) -> Subtask | None:
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
        planner = TaskPlanner("anthropic:claude-3-5-sonnet-latest")
        plan = await planner.plan("Build a web scraper for news articles")
    """

    def __init__(
        self,
        model: str = "anthropic:claude-3-5-sonnet-latest",
        *,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        prompt_override: str | None = None,
        max_subtasks: int = 6,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_override = (prompt_override or "").strip() or None
        self.max_subtasks = max(1, int(max_subtasks)) if max_subtasks else 0
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
        context: list[Message] | None = None,
        memory: AgentMemory | None = None,
    ) -> TaskPlan:
        """
        Create a plan for executing a task.

        Args:
            task: The task description
            context: Optional conversation context
            memory: Optional AgentMemory for goal tracking

        Returns:
            TaskPlan with subtasks and execution order
        """
        # First, assess complexity
        complexity = self._assess_complexity(task)

        if complexity == TaskComplexity.SIMPLE:
            # Simple tasks don't need decomposition
            plan = TaskPlan(
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
            return self._create_goal_for_plan(plan, memory)

        # For complex tasks, use LLM to decompose
        logger.info(f"Planner decomposing task with model={self.model}")
        provider, model_id, _ = self.registry.resolve_with_fallback(self.model)
        loader = get_prompt_loader()
        system_prompt = self.prompt_override or loader.get("planner.decompose")

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=system_prompt,
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
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            steps = self._parse_plan(result.content)

            # Determine best reasoning strategy
            strategy = self._select_strategy(steps)

            plan = TaskPlan(
                task=task,
                complexity=complexity,
                steps=steps,
                reasoning_strategy=strategy,
                estimated_tokens=result.usage.get("total_tokens", 0) if result.usage else 0,
            )
            return self._create_goal_for_plan(plan, memory)

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fall back to single-step plan
            plan = TaskPlan(
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
            return self._create_goal_for_plan(plan, memory)
    
    async def compose_with_model(
        self,
        provider,
        model_id: str,
        base_messages: list[Message],
        task: str,
        *,
        memory: AgentMemory | None = None,
    ) -> TaskPlan | None:
        """Decompose ``task`` using the *caller's* model + already-assembled context.

        Unlike :meth:`plan`, this reuses the **main agent's** provider/model and
        the full turn context (system prompt + memory bundle + history) rather
        than a separate, context-blind planner call, and takes a structured JSON
        plan back instead of regex-scraping ``SUBTASK`` prose. The model decides
        whether decomposition is warranted — it returns ``{"plan": null}`` (or a
        single step) when not.

        Returns a normalized :class:`TaskPlan` when the model proposes >1 step,
        else ``None`` (caller falls through to a normal single-pass turn).
        """
        instruction = _COMPOSE_INSTRUCTION.format(task=task, max=self.max_subtasks or 6)
        messages = list(base_messages) + [Message(role=MessageRole.USER, content=instruction)]

        try:
            result = await provider.complete(
                messages, model_id,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning(f"Model-composed planning failed ({e}); single-pass turn")
            return None

        data = _extract_json_object(result.content)
        if not data:
            logger.info("Planner pre-pass: no JSON plan in model output; single-pass turn")
            return None

        raw = data.get("plan")
        if not isinstance(raw, list) or len(raw) < 2:
            # Model judged one response enough ({"plan": null} or a single step).
            return None

        steps: list[Subtask] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description") or item.get("task") or "").strip()
            if not desc:
                continue
            steps.append(Subtask(
                id=i,
                description=desc[:500],
                type=_coerce_subtask_type(item.get("type")),
                dependencies=_coerce_int_list(item.get("depends") or item.get("dependencies")),
                tools_needed=_coerce_str_list(item.get("tools") or item.get("tools_needed")),
            ))

        if len(steps) < 2:
            return None

        steps = self._normalize_steps(steps)
        plan = TaskPlan(
            task=task,
            complexity=TaskComplexity.COMPLEX,
            steps=steps,
            reasoning_strategy=self._select_strategy(steps),
            estimated_tokens=result.usage.get("total_tokens", 0) if result.usage else 0,
        )
        logger.info(f"Planner pre-pass: main model composed {len(steps)} subtasks")
        return self._create_goal_for_plan(plan, memory)

    def _assess_complexity(self, task: str) -> TaskComplexity:
        """Assess the complexity of a task.

        Deliberately conservative: a lone keyword ("build", "write a", "summarize")
        does NOT make a task complex — most such requests are a single step. We only
        escalate when the task shows genuine multi-step *structure*: explicit sequence
        markers, several distinct imperatives/deliverables, or substantial length. This
        prevents the planner from decomposing simple chat requests into giant plans.
        """
        task_lower = task.lower()
        words = task.split()

        # Explicit multi-step / sequencing language — strong signal of real structure.
        sequence_markers = [
            "step by step", "step-by-step", "and then", " then ", "after that",
            "first,", "firstly", "secondly", "finally,", "followed by",
            "multiple steps", "each of the following", "for each",
        ]
        has_sequence = any(m in task_lower for m in sequence_markers)

        # Count distinct imperative/deliverable clauses (rough proxy for "several things").
        # Split on connectors/sentence boundaries and count clauses that look like asks.
        clauses = [c for c in re.split(r"[.;,\n]|\band\b|\bthen\b", task_lower) if c.strip()]
        action_words = (
            "analyze", "design", "build", "implement", "develop", "create",
            "compare", "research", "write", "draft", "summarize", "evaluate",
            "plan", "refactor", "test", "document", "review", "generate",
        )
        action_clauses = sum(1 for c in clauses if any(a in c for a in action_words))

        # Architectural / inherently-complex phrasing.
        complex_phrases = [
            "create a system", "design a system", "architecture", "comprehensive",
            "end-to-end", "compare multiple", "research and", "pipeline",
        ]
        if any(p in task_lower for p in complex_phrases) or len(words) > 60:
            return TaskComplexity.COMPLEX

        # Genuine multi-step structure → complex enough to decompose.
        if (has_sequence and action_clauses >= 2) or action_clauses >= 3:
            return TaskComplexity.COMPLEX

        # Some structure (a couple of distinct asks, or notable length) → moderate.
        if action_clauses == 2 or len(words) > 40:
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

        return self._normalize_steps(steps)

    def _normalize_steps(self, steps: list[Subtask]) -> list[Subtask]:
        """Cap subtasks and reindex ids to contiguous list positions.

        Every positional lookup in the plan machinery (``get_next_subtask``,
        ``mark_complete``, dependency injection/skip) assumes ``steps[i].id == i``.
        The LLM's ``SUBTASK`` numbering isn't guaranteed to be contiguous, ordered,
        or unique, so without this the executor can mark the wrong slot complete and
        loop forever on one step. We reindex by parse order and remap each dependency
        through the old→new map, dropping self/unknown/forward references.
        """
        if self.max_subtasks and len(steps) > self.max_subtasks:
            logger.info(
                f"Planner produced {len(steps)} subtasks; capping to {self.max_subtasks}"
            )
            steps = steps[: self.max_subtasks]

        old_to_new = {s.id: i for i, s in enumerate(steps)}
        for new_index, s in enumerate(steps):
            remapped: list[int] = []
            for d in s.dependencies:
                nd = old_to_new.get(d)
                # Keep only resolvable deps that point at an earlier step.
                if nd is not None and nd < new_index and nd not in remapped:
                    remapped.append(nd)
            s.id = new_index
            s.dependencies = remapped

        return steps

    def _create_goal_for_plan(
        self,
        plan: TaskPlan,
        memory: AgentMemory | None
    ) -> TaskPlan:
        """
        Create a goal in memory for the given plan.

        Args:
            plan: The task plan to link a goal to
            memory: Optional AgentMemory instance

        Returns:
            The plan with goal_id set if memory was available
        """
        if memory is None:
            return plan

        try:
            from ..kit.agent_memory.models import Goal

            goal = Goal(
                id=str(uuid4()),
                description=plan.task[:500],  # Truncate long tasks
                status="active",
                priority=3,
            )
            memory.add_goal(goal)
            plan.goal_id = goal.id
            logger.debug(f"Created goal {goal.id} for task plan")

            # Create child goals for each subtask (skip if plan has only one step;
            # the parent goal already represents that work).
            if len(plan.steps) > 1:
                for step in plan.steps:
                    try:
                        sub_goal = Goal(
                            id=str(uuid4()),
                            description=step.description[:500],
                            status="active",
                            priority=3,
                            parent_goal_id=plan.goal_id,
                        )
                        memory.add_goal(sub_goal)
                        step.goal_id = sub_goal.id
                    except Exception as sub_e:
                        logger.warning(
                            f"Failed to create subgoal for step {step.id}: {sub_e}"
                        )
        except Exception as e:
            logger.warning(f"Failed to create goal for plan: {e}")

        return plan

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

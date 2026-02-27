"""
ReAct (Reasoning + Acting) implementation.

ReAct interleaves reasoning with action execution,
allowing the agent to gather information and take actions
as part of its reasoning process.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry
from ..prompts.loader import get_prompt_loader
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
class Tool:
    """Definition of a tool available to the ReAct agent."""
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], Any]


@dataclass
class ReActConfig:
    """Configuration for ReAct reasoning."""
    model: str
    
    # Available tools
    tools: list[Tool] = field(default_factory=list)
    
    # Loop settings
    max_iterations: int = 10
    max_consecutive_failures: int = 3
    
    # Prompting
    thought_prefix: str = "Thought:"
    action_prefix: str = "Action:"
    observation_prefix: str = "Observation:"
    answer_prefix: str = "Final Answer:"
    
    # Termination
    stop_phrases: list[str] = field(default_factory=lambda: ["Final Answer:"])


# Built-in tools
def _search_tool(query: str) -> str:
    """Placeholder search tool."""
    return f"Search results for '{query}': (No search API configured)"


def _calculate_tool(expression: str) -> str:
    """Simple calculator tool using safe AST evaluation."""
    import ast
    import operator
    
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            if type(node.op) not in operators:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            # Prevent huge exponents
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large (max 1000)")
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations (-x, +x)
            if type(node.op) not in operators:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return operators[type(node.op)](safe_eval(node.operand))
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


DEFAULT_TOOLS = [
    Tool(
        name="search",
        description="Search for information on a topic",
        parameters={"query": "string"},
        execute=lambda args: _search_tool(args.get("query", "")),
    ),
    Tool(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={"expression": "string"},
        execute=lambda args: _calculate_tool(args.get("expression", "")),
    ),
]


class ReActAgent(ReasoningStrategy):
    """
    ReAct (Reasoning + Acting) agent.
    
    Implements the Thought → Action → Observation loop:
    1. Think about what to do next
    2. Take an action (call a tool)
    3. Observe the result
    4. Repeat until task is complete
    
    Example usage:
        agent = ReActAgent(ReActConfig(
            model="gpt-4-turbo",
            tools=[
                Tool(name="search", description="...", execute=search_fn),
            ],
            max_iterations=10,
        ))
        result = await agent.reason("What is the capital of France and its population?")
    """
    
    def __init__(self, config: ReActConfig):
        self.react_config = config
        super().__init__(ReasoningConfig(
            name="react",
            strategy_type="react",
            model=config.model,
            extra={"react_config": config},
        ))
        self._registry = None
        
        # Use provided tools or defaults
        self.tools = {t.name: t for t in (config.tools or DEFAULT_TOOLS)}
    
    @property
    def name(self) -> str:
        return "react"
    
    @property
    def strategy_type(self) -> str:
        return "react"
    
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
        Apply ReAct reasoning to a task.
        """
        start_time = time.time()

        provider, model_id = self.registry.get_provider_for_model(
            self.react_config.model
        )

        logger.info(f"ReAct reasoning: {task[:50]}...")

        # Build initial prompt with tool descriptions
        messages = self._build_initial_prompt(task, context)

        steps = []
        actions_taken = 0
        total_tokens = 0
        consecutive_failures = 0
        final_answer = ""

        for iteration in range(self.react_config.max_iterations):
            step_num = iteration * 3 + 1  # Each iteration has thought, action, observation

            # Get model response (thought + action)
            try:
                result = provider.complete(
                    messages,
                    model_id,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 500),
                    stop=self.react_config.stop_phrases,
                )
            except Exception as e:
                logger.error(f"ReAct iteration {iteration} failed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= self.react_config.max_consecutive_failures:
                    break
                continue

            if result.usage:
                total_tokens += result.usage.get("total_tokens", 0)

            response = result.content

            # Check for final answer
            if self.react_config.answer_prefix in response:
                final_answer = self._extract_final_answer(response)
                steps.append(ThoughtStep(
                    step_number=step_num,
                    thought_type=ThoughtType.CONCLUSION,
                    content=final_answer,
                    model=self.react_config.model,
                ))
                break

            # Parse thought
            thought = self._extract_thought(response)
            if thought:
                steps.append(ThoughtStep(
                    step_number=step_num,
                    thought_type=ThoughtType.REASONING,
                    content=thought,
                    model=self.react_config.model,
                ))

            # Parse and execute action
            action_name, action_input = self._extract_action(response)

            if action_name:
                steps.append(ThoughtStep(
                    step_number=step_num + 1,
                    thought_type=ThoughtType.ACTION,
                    content=f"{action_name}({action_input})",
                    action_name=action_name,
                    action_input=action_input,
                    model=self.react_config.model,
                ))

                # Execute the action
                observation = self._execute_action(action_name, action_input)
                actions_taken += 1

                steps.append(ThoughtStep(
                    step_number=step_num + 2,
                    thought_type=ThoughtType.RESULT,
                    content=observation,
                    action_output=observation,
                ))

                # Add to conversation
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=response,
                ))
                messages.append(Message(
                    role=MessageRole.USER,
                    content=f"{self.react_config.observation_prefix} {observation}",
                ))

                consecutive_failures = 0
            else:
                # No valid action found
                consecutive_failures += 1
                if consecutive_failures >= self.react_config.max_consecutive_failures:
                    logger.warning("Max consecutive failures reached, stopping")
                    break

                # Prompt for action
                messages.append(Message(
                    role=MessageRole.USER,
                    content="Please provide a valid action or final answer.",
                ))
        
        total_time = (time.time() - start_time) * 1000
        
        # If no final answer was found, use the last thought
        if not final_answer:
            thought_steps = [s for s in steps if s.thought_type == ThoughtType.REASONING]
            if thought_steps:
                final_answer = thought_steps[-1].content
            else:
                final_answer = "Unable to determine an answer."
        
        return ReasoningResult(
            answer=final_answer,
            strategy=self.name,
            status=ReasoningStatus.COMPLETE,
            steps=steps,
            total_steps=len(steps),
            actions_taken=actions_taken,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            models_used=[self.react_config.model],
            raw_trace=[{"messages": [m.content[:200] for m in messages]}],
        )
    
    def _build_initial_prompt(
        self,
        task: str,
        context: Optional[list[Message]],
    ) -> list[Message]:
        """Build the initial ReAct prompt with tool descriptions."""
        tool_descriptions = []
        for name, tool in self.tools.items():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            tool_descriptions.append(f"- {name}({params}): {tool.description}")

        tools_text = "\n".join(tool_descriptions)
        loader = get_prompt_loader()

        system_prompt = loader.get(
            "reasoning.react.system",
            tools_text=tools_text,
            thought_prefix=self.react_config.thought_prefix,
            action_prefix=self.react_config.action_prefix,
            answer_prefix=self.react_config.answer_prefix,
        )

        messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]

        if context:
            messages.extend(context)

        messages.append(Message(
            role=MessageRole.USER,
            content=f"Task: {task}",
        ))

        return messages
    
    def _extract_thought(self, response: str) -> Optional[str]:
        """Extract the thought from a response."""
        pattern = rf"{self.react_config.thought_prefix}\s*(.+?)(?={self.react_config.action_prefix}|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_action(self, response: str) -> tuple[Optional[str], dict[str, Any]]:
        """Extract the action from a response."""
        pattern = rf"{self.react_config.action_prefix}\s*(\w+)\s*\(([^)]*)\)"
        match = re.search(pattern, response, re.IGNORECASE)
        
        if not match:
            return None, {}
        
        action_name = match.group(1).lower()
        args_str = match.group(2)
        
        # Parse arguments
        args = {}
        if args_str:
            # Try to parse key=value pairs
            arg_pattern = r'(\w+)\s*=\s*["\']?([^"\'=,]+)["\']?'
            for key, value in re.findall(arg_pattern, args_str):
                args[key] = value.strip()
            
            # If no key=value pairs, use first argument as default param
            if not args and action_name in self.tools:
                first_param = list(self.tools[action_name].parameters.keys())[0]
                args[first_param] = args_str.strip().strip('"\'')
        
        return action_name, args
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a response."""
        pattern = rf"{self.react_config.answer_prefix}\s*(.+)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response
    
    def _execute_action(
        self,
        action_name: str,
        action_input: dict[str, Any],
    ) -> str:
        """Execute an action and return the observation."""
        if action_name not in self.tools:
            return f"Error: Unknown tool '{action_name}'. Available tools: {list(self.tools.keys())}"

        tool = self.tools[action_name]

        try:
            # Execute the tool
            result = tool.execute(action_input)
            return str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error executing {action_name}: {e}"
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool
    
    def remove_tool(self, name: str) -> None:
        """Remove a tool from the agent."""
        self.tools.pop(name, None)
    
    def get_description(self) -> str:
        tools_list = ", ".join(self.tools.keys())
        return f"ReAct agent with tools: [{tools_list}]"

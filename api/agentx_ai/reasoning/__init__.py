"""
Reasoning Framework for AgentX.

This module implements flexible reasoning patterns for complex tasks:
- Chain-of-Thought (CoT): Step-by-step reasoning
- Tree-of-Thought (ToT): Exploring multiple reasoning paths
- ReAct: Reasoning + Acting with tool use
- Reflection: Self-critique and revision
"""

from .base import ReasoningStrategy, ReasoningResult, ThoughtStep
from .chain_of_thought import ChainOfThought
from .tree_of_thought import TreeOfThought
from .react import ReActAgent
from .reflection import ReflectiveReasoner
from .orchestrator import ReasoningOrchestrator

__all__ = [
    "ReasoningStrategy",
    "ReasoningResult",
    "ThoughtStep",
    "ChainOfThought",
    "TreeOfThought",
    "ReActAgent",
    "ReflectiveReasoner",
    "ReasoningOrchestrator",
]

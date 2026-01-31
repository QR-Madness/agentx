"""
Core Agent implementation.

The Agent class orchestrates all AgentX capabilities:
- Planning and task decomposition
- Reasoning strategy selection
- Tool execution via MCP
- Context and memory management
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry, ProviderRegistry
from ..reasoning import ReasoningOrchestrator
from ..reasoning.orchestrator import OrchestratorConfig
from ..reasoning.react import Tool
from ..drafting import DraftingStrategy
from ..drafting.speculative import SpeculativeDecoder, SpeculativeConfig

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of an agent task."""
    IDLE = "idle"
    PLANNING = "planning"
    REASONING = "reasoning"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentResult(BaseModel):
    """Result of an agent task."""
    task_id: str
    status: AgentStatus
    answer: str
    
    # Execution details
    plan_steps: int = 0
    reasoning_steps: int = 0
    tools_used: list[str] = []
    models_used: list[str] = []
    
    # Metrics
    total_tokens: int = 0
    total_time_ms: float = 0.0
    
    # Trace for debugging
    trace: Optional[list[dict[str, Any]]] = None


@dataclass
class AgentConfig:
    """Configuration for an Agent instance."""
    # Identity
    name: str = "agentx"
    user_id: Optional[str] = None
    
    # Model settings
    default_model: str = "gpt-4-turbo"
    reasoning_model: Optional[str] = None
    drafting_model: Optional[str] = None
    
    # Behavior settings
    max_iterations: int = 20
    timeout_seconds: float = 300.0
    
    # Capabilities
    enable_planning: bool = True
    enable_reasoning: bool = True
    enable_drafting: bool = False
    enable_memory: bool = True
    enable_tools: bool = True
    
    # Reasoning settings
    default_reasoning_strategy: str = "auto"  # "auto", "cot", "tot", "react", "reflection"
    
    # Tool settings
    allowed_tools: Optional[list[str]] = None
    blocked_tools: Optional[list[str]] = None
    
    # Context settings
    max_context_tokens: int = 8000
    summarize_threshold: int = 6000


class Agent:
    """
    AgentX Core Agent.
    
    The Agent class is the central orchestrator that combines all AgentX
    capabilities to solve complex tasks through planning, reasoning, and action.
    
    Example usage:
        agent = Agent(AgentConfig(
            default_model="gpt-4-turbo",
            enable_tools=True,
        ))
        
        result = await agent.run("Analyze the codebase and suggest improvements")
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.IDLE
        
        # Core components (lazy-loaded)
        self._registry: Optional[ProviderRegistry] = None
        self._reasoning: Optional[ReasoningOrchestrator] = None
        self._drafting: Optional[DraftingStrategy] = None
        self._context_manager = None
        self._session_manager = None
        self._memory = None
        self._mcp_client = None
        
        # Runtime state
        self._current_task_id: Optional[str] = None
        self._cancel_requested = False
        
        # Tool registry
        self._tools: dict[str, Tool] = {}
    
    @property
    def registry(self) -> ProviderRegistry:
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    @property
    def reasoning(self) -> ReasoningOrchestrator:
        """Lazy-load the reasoning orchestrator."""
        if self._reasoning is None:
            model = self.config.reasoning_model or self.config.default_model
            
            # Configure orchestrator with available tools
            tools = list(self._tools.values()) if self.config.enable_tools else []
            
            self._reasoning = ReasoningOrchestrator(OrchestratorConfig(
                default_model=model,
                react_tools=tools,
            ))
        return self._reasoning
    
    @property
    def drafting(self) -> Optional[DraftingStrategy]:
        """Lazy-load the drafting strategy."""
        if self._drafting is None and self.config.enable_drafting:
            model = self.config.drafting_model or self.config.default_model
            # Default to speculative decoding with a fast/accurate pair
            self._drafting = SpeculativeDecoder(SpeculativeConfig(
                draft_model="gpt-3.5-turbo",
                target_model=model,
            ))
        return self._drafting
    
    async def run(
        self,
        task: str,
        context: Optional[list[Message]] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Execute a task using the full agent pipeline.
        
        Args:
            task: The task description
            context: Optional conversation context
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with the answer and execution details
        """
        task_id = str(uuid.uuid4())[:8]
        self._current_task_id = task_id
        self._cancel_requested = False
        
        start_time = time.time()
        trace = []
        
        logger.info(f"Agent task {task_id}: {task[:50]}...")
        
        try:
            self.status = AgentStatus.PLANNING
            
            # Step 1: Planning (if enabled)
            plan = None
            if self.config.enable_planning:
                from .planner import TaskPlanner
                planner = TaskPlanner(self.config.default_model)
                plan = await planner.plan(task, context)
                trace.append({
                    "phase": "planning",
                    "steps": len(plan.steps) if plan else 0,
                })
            
            if self._cancel_requested:
                return self._cancelled_result(task_id, start_time)
            
            # Step 2: Reasoning
            self.status = AgentStatus.REASONING
            
            reasoning_result = None
            if self.config.enable_reasoning:
                strategy = kwargs.get("reasoning_strategy", self.config.default_reasoning_strategy)
                
                if strategy == "auto":
                    reasoning_result = await self.reasoning.reason(task, context)
                else:
                    reasoning_result = await self.reasoning.reason(
                        task, context, strategy=strategy
                    )
                
                trace.append({
                    "phase": "reasoning",
                    "strategy": reasoning_result.strategy if reasoning_result else None,
                    "steps": reasoning_result.total_steps if reasoning_result else 0,
                })
            
            if self._cancel_requested:
                return self._cancelled_result(task_id, start_time)
            
            # Step 3: Generate final answer
            self.status = AgentStatus.EXECUTING
            
            if reasoning_result:
                answer = reasoning_result.answer
                total_tokens = reasoning_result.total_tokens
                models_used = reasoning_result.models_used
                reasoning_steps = reasoning_result.total_steps
                tools_used = [
                    s.action_name for s in reasoning_result.steps
                    if s.action_name
                ]
            else:
                # Direct completion without reasoning
                provider, model_id = self.registry.get_provider_for_model(
                    self.config.default_model
                )
                
                messages = [Message(role=MessageRole.USER, content=task)]
                if context:
                    messages = context + messages
                
                result = await provider.complete(
                    messages,
                    model_id,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 2000),
                )
                
                answer = result.content
                total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0
                models_used = [self.config.default_model]
                reasoning_steps = 0
                tools_used = []
            
            total_time = (time.time() - start_time) * 1000
            self.status = AgentStatus.COMPLETE
            
            return AgentResult(
                task_id=task_id,
                status=AgentStatus.COMPLETE,
                answer=answer,
                plan_steps=len(plan.steps) if plan else 0,
                reasoning_steps=reasoning_steps,
                tools_used=list(set(tools_used)),
                models_used=list(set(models_used)),
                total_tokens=total_tokens,
                total_time_ms=total_time,
                trace=trace,
            )
            
        except Exception as e:
            logger.error(f"Agent task {task_id} failed: {e}")
            self.status = AgentStatus.FAILED
            
            return AgentResult(
                task_id=task_id,
                status=AgentStatus.FAILED,
                answer=f"Task failed: {str(e)}",
                trace=trace,
            )
        finally:
            self._current_task_id = None
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Handle a conversational message.
        
        Maintains context across messages within a session.
        
        Args:
            message: The user message
            session_id: Optional session ID for context
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with the response
        """
        # Get or create session
        from .session import SessionManager
        if self._session_manager is None:
            self._session_manager = SessionManager()
        
        session = self._session_manager.get_or_create(session_id)
        
        # Add user message to session
        session.add_message(Message(role=MessageRole.USER, content=message))
        
        # Get context from session
        context = session.get_messages()
        
        # Run the task with context
        result = await self.run(message, context=context[:-1], **kwargs)  # Exclude current message
        
        # Add assistant response to session
        session.add_message(Message(role=MessageRole.ASSISTANT, content=result.answer))
        
        return result
    
    def cancel(self) -> bool:
        """
        Request cancellation of the current task.
        
        Returns:
            True if a task was running and cancellation was requested
        """
        if self._current_task_id and self.status not in (AgentStatus.IDLE, AgentStatus.COMPLETE):
            self._cancel_requested = True
            return True
        return False
    
    def _cancelled_result(self, task_id: str, start_time: float) -> AgentResult:
        """Create a cancelled result."""
        self.status = AgentStatus.CANCELLED
        return AgentResult(
            task_id=task_id,
            status=AgentStatus.CANCELLED,
            answer="Task was cancelled.",
            total_time_ms=(time.time() - start_time) * 1000,
        )
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self._tools[tool.name] = tool
        
        # Update reasoning orchestrator if already initialized
        if self._reasoning:
            self._reasoning.add_react_tool(tool)
    
    def remove_tool(self, name: str) -> None:
        """Remove a tool from the agent."""
        self._tools.pop(name, None)
    
    def get_status(self) -> dict[str, Any]:
        """Get the current agent status."""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "current_task": self._current_task_id,
            "tools_available": list(self._tools.keys()),
            "config": {
                "default_model": self.config.default_model,
                "enable_planning": self.config.enable_planning,
                "enable_reasoning": self.config.enable_reasoning,
                "enable_drafting": self.config.enable_drafting,
                "enable_memory": self.config.enable_memory,
                "enable_tools": self.config.enable_tools,
            },
        }

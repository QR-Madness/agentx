"""
Core Agent implementation.

The Agent class orchestrates all AgentX capabilities:
- Planning and task decomposition
- Reasoning strategy selection
- Tool execution via MCP
- Context and memory management
"""

import logging
import time
import uuid
from dataclasses import dataclass
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
from ..kit.memory_utils import get_agent_memory
from ..kit.agent_memory.models import Turn

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
    
    # Parsed output components
    thinking: Optional[str] = None  # Extracted thinking/reasoning content
    has_thinking: bool = False
    
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
    
    # Model settings - offline-first, default to local Ollama
    default_model: str = "llama3.2"
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
    
    # Prompt settings
    prompt_profile_id: Optional[str] = None  # Use default if None
    
    # Memory settings
    memory_channel: str = "_global"  # Channel for memory scoping
    memory_top_k: int = 10  # Number of memories to retrieve
    memory_time_window_hours: Optional[int] = None  # Time window filter for retrieval


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
    
    @property
    def memory(self):
        """
        Lazy-load the agent memory system.
        
        Returns None if memory is disabled or databases are unavailable.
        """
        if self._memory is None and self.config.enable_memory:
            user_id = self.config.user_id or "default"
            self._memory = get_agent_memory(
                user_id=user_id,
                channel=self.config.memory_channel,
            )
            if self._memory is None:
                logger.warning("Memory system unavailable, agent will operate without persistent memory")
        return self._memory
    
    @property
    def mcp_client(self):
        """Lazy-load the MCP client manager."""
        if self._mcp_client is None and self.config.enable_tools:
            from ..mcp import MCPClientManager
            self._mcp_client = MCPClientManager()
            
            # Wire memory-based tool usage recording if memory is available
            if self.memory:
                def record_tool_usage(
                    tool_name: str,
                    tool_input: dict,
                    tool_output,
                    success: bool,
                    latency_ms: int,
                    error_message: str | None,
                ) -> None:
                    try:
                        self.memory.record_tool_usage(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_output=tool_output,
                            success=success,
                            latency_ms=latency_ms,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to record tool usage in memory: {e}")
                
                self._mcp_client.tool_executor.set_usage_recorder(record_tool_usage)
        
        return self._mcp_client
    
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
        
        # Retrieve relevant memories for task context
        memory_bundle = None
        if self.memory:
            try:
                memory_bundle = self.memory.remember(
                    query=task,
                    top_k=self.config.memory_top_k,
                    time_window_hours=self.config.memory_time_window_hours,
                )
                trace.append({
                    "phase": "memory_retrieval",
                    "turns": len(memory_bundle.relevant_turns) if memory_bundle else 0,
                    "facts": len(memory_bundle.facts) if memory_bundle else 0,
                })
            except Exception as e:
                logger.warning(f"Failed to retrieve memories for task: {e}")
        
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
                
                # Inject memories into context if available
                reasoning_context = context
                if memory_bundle and context:
                    from .context import ContextManager, ContextConfig
                    if self._context_manager is None:
                        self._context_manager = ContextManager(ContextConfig(
                            max_tokens=self.config.max_context_tokens,
                            summarize_threshold=self.config.summarize_threshold,
                        ))
                    reasoning_context = self._context_manager.inject_memory(context, memory_bundle)
                
                if strategy == "auto":
                    reasoning_result = await self.reasoning.reason(task, reasoning_context)
                else:
                    reasoning_result = await self.reasoning.reason(
                        task, reasoning_context, strategy=strategy
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
                
                # Inject memories if available
                if memory_bundle:
                    from .context import ContextManager, ContextConfig
                    if self._context_manager is None:
                        self._context_manager = ContextManager(ContextConfig(
                            max_tokens=self.config.max_context_tokens,
                            summarize_threshold=self.config.summarize_threshold,
                        ))
                    messages = self._context_manager.inject_memory(messages, memory_bundle)
                
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
            
            # Trigger memory reflection after task completion
            if self.memory:
                try:
                    self.memory.reflect({
                        "task_id": task_id,
                        "task": task[:200],
                        "status": "complete",
                        "total_tokens": total_tokens,
                        "total_time_ms": total_time,
                        "reasoning_steps": reasoning_steps,
                        "tools_used": tools_used,
                    })
                except Exception as e:
                    logger.warning(f"Failed to trigger memory reflection: {e}")
            
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
            
            # Trigger reflection for failed tasks too
            if self.memory:
                try:
                    self.memory.reflect({
                        "task_id": task_id,
                        "task": task[:200],
                        "status": "failed",
                        "error": str(e),
                    })
                except Exception as reflect_error:
                    logger.warning(f"Failed to trigger memory reflection: {reflect_error}")
            
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
        simple_mode: bool = True,
        profile_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Handle a conversational message.
        
        Maintains context across messages within a session.
        
        Args:
            message: The user message
            session_id: Optional session ID for context
            simple_mode: If True, use direct completion without reasoning (default)
            profile_id: Optional prompt profile ID to use
            **kwargs: Additional parameters
            
        Returns:
            AgentResult with the response
        """
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Get or create session
        from .session import SessionManager
        if self._session_manager is None:
            self._session_manager = SessionManager()
        
        session = self._session_manager.get_or_create(session_id)
        conversation_id = session_id or session.id
        
        # Update memory with conversation context if available
        if self.memory:
            self.memory.conversation_id = conversation_id
        
        # Add user message to session
        user_message = Message(role=MessageRole.USER, content=message)
        session.add_message(user_message)
        
        # Store user turn in memory
        if self.memory:
            user_turn = Turn(
                conversation_id=conversation_id,
                index=len(session.get_messages()) - 1,
                role="user",
                content=message,
            )
            try:
                self.memory.store_turn(user_turn)
            except Exception as e:
                logger.warning(f"Failed to store user turn in memory: {e}")
        
        # Get context from session (excluding current message for the prompt)
        context = session.get_messages()[:-1]
        
        # Retrieve relevant memories for context injection
        memory_bundle = None
        if self.memory:
            try:
                memory_bundle = self.memory.remember(
                    query=message,
                    top_k=self.config.memory_top_k,
                    time_window_hours=self.config.memory_time_window_hours,
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")
        
        try:
            if simple_mode:
                # Direct completion without planning/reasoning - best for chat
                provider, model_id = self.registry.get_provider_for_model(
                    self.config.default_model
                )
                
                # Get system prompt from prompt manager
                from ..prompts import get_prompt_manager
                prompt_manager = get_prompt_manager()
                system_prompt = prompt_manager.get_system_prompt(
                    profile_id=profile_id or self.config.prompt_profile_id
                )
                
                # Build messages with composed system prompt
                messages = [
                    Message(
                        role=MessageRole.SYSTEM,
                        content=system_prompt or "You are a helpful AI assistant."
                    )
                ]
                if context:
                    messages.extend(context)
                messages.append(Message(role=MessageRole.USER, content=message))
                
                # Inject memories into context
                if memory_bundle:
                    from .context import ContextManager, ContextConfig
                    if self._context_manager is None:
                        self._context_manager = ContextManager(ContextConfig(
                            max_tokens=self.config.max_context_tokens,
                            summarize_threshold=self.config.summarize_threshold,
                        ))
                    messages = self._context_manager.inject_memory(messages, memory_bundle)
                
                logger.info(f"Agent chat {task_id} using {model_id}")
                
                result = await provider.complete(
                    messages,
                    model_id,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 2000),
                )
                
                # Parse output to extract thinking tags
                from .output_parser import parse_output
                parsed = parse_output(result.content)
                
                answer = parsed.content
                thinking = parsed.thinking
                total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0
                
                # Add assistant response to session (store parsed content)
                session.add_message(Message(role=MessageRole.ASSISTANT, content=answer))
                
                total_time = (time.time() - start_time) * 1000
                
                # Store assistant turn in memory
                if self.memory:
                    assistant_turn = Turn(
                        conversation_id=conversation_id,
                        index=len(session.get_messages()) - 1,
                        role="assistant",
                        content=answer,
                        token_count=total_tokens,
                        metadata={
                            "model": model_id,
                            "latency_ms": total_time,
                            "task_id": task_id,
                        }
                    )
                    try:
                        self.memory.store_turn(assistant_turn)
                    except Exception as e:
                        logger.warning(f"Failed to store assistant turn in memory: {e}")
                
                return AgentResult(
                    task_id=task_id,
                    status=AgentStatus.COMPLETE,
                    answer=answer,
                    thinking=thinking,
                    has_thinking=parsed.has_thinking,
                    models_used=[self.config.default_model],
                    total_tokens=total_tokens,
                    total_time_ms=total_time,
                )
            else:
                # Full agent pipeline with reasoning
                result = await self.run(message, context=context, **kwargs)
                
                # Add assistant response to session
                session.add_message(Message(role=MessageRole.ASSISTANT, content=result.answer))
                
                # Store assistant turn in memory
                if self.memory:
                    assistant_turn = Turn(
                        conversation_id=conversation_id,
                        index=len(session.get_messages()) - 1,
                        role="assistant",
                        content=result.answer,
                        token_count=result.total_tokens,
                        metadata={
                            "model": result.models_used[0] if result.models_used else self.config.default_model,
                            "latency_ms": result.total_time_ms,
                            "task_id": task_id,
                            "reasoning_steps": result.reasoning_steps,
                            "tools_used": result.tools_used,
                        }
                    )
                    try:
                        self.memory.store_turn(assistant_turn)
                    except Exception as e:
                        logger.warning(f"Failed to store assistant turn in memory: {e}")
                
                return result
                
        except Exception as e:
            logger.error(f"Agent chat {task_id} failed: {e}")
            return AgentResult(
                task_id=task_id,
                status=AgentStatus.FAILED,
                answer=f"Sorry, I encountered an error: {str(e)}",
                total_time_ms=(time.time() - start_time) * 1000,
            )
    
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

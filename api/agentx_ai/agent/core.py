"""
Core Agent implementation.

The Agent class orchestrates all AgentX capabilities:
- Planning and task decomposition
- Reasoning strategy selection
- Tool execution via MCP
- Context and memory management
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from ..providers.base import CompletionResult, Message, MessageRole, ToolCall
from ..providers.registry import get_registry, ProviderRegistry
from ..reasoning import ReasoningOrchestrator
from ..reasoning.orchestrator import OrchestratorConfig
from ..reasoning.react import Tool
from ..drafting import DraftingStrategy
from ..drafting.speculative import SpeculativeDecoder, SpeculativeConfig
from ..kit.memory_utils import get_agent_memory
from ..kit.agent_memory.models import Turn
from .tool_output_compressor import get_compressor
from .tool_output_storage import store_tool_output

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
    
    # Model settings - default to local LM Studio
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
    max_tool_rounds: int = 10  # Max tool-call ↔ result round-trips per request
    max_tool_result_chars: int = 12000  # Threshold for storing oversized results in Redis
    store_oversized_results: bool = True  # Store large results in Redis instead of truncating
    tool_output_ttl_seconds: int = 3600  # TTL for stored tool outputs (1 hour default)
    compress_tool_outputs: bool = True  # Use LLM compression for oversized tool outputs
    
    # Context settings
    max_context_tokens: int = 8000
    summarize_threshold: int = 6000
    
    # Prompt settings
    prompt_profile_id: Optional[str] = None  # Use default if None
    
    # Memory settings
    memory_channel: str = "_default"  # Channel for memory scoping (use _default, not _global)
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
        """Lazy-load the MCP client manager using the global singleton."""
        if self._mcp_client is None and self.config.enable_tools:
            from ..mcp import get_mcp_manager
            self._mcp_client = get_mcp_manager()
            
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
    
    # ──────────────────────────────────────────────
    #  Tool helpers
    # ──────────────────────────────────────────────
    
    def _get_tools_for_provider(self) -> list[dict[str, Any]] | None:
        """
        Convert MCP tools to OpenAI function-calling format.
        
        Returns None when tools are disabled or no tools are available,
        which tells providers to skip tool-calling entirely.
        """
        if not self.config.enable_tools or not self.mcp_client:
            return None
        
        mcp_tools = self.mcp_client.list_tools()
        logger.debug(
            f"MCP tools for provider: {len(mcp_tools)} tools from "
            f"{len(self.mcp_client.list_connections())} connections"
        )
        if not mcp_tools:
            return None
        
        tools = []
        for t in mcp_tools:
            # Apply allow/block filters
            if self.config.allowed_tools and t.name not in self.config.allowed_tools:
                continue
            if self.config.blocked_tools and t.name in self.config.blocked_tools:
                continue
            
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema or {"type": "object"},
                },
            })
        
        return tools if tools else None
    
    def _execute_tool_calls(self, tool_calls: list[ToolCall], task_context: str = "") -> list[Message]:
        """
        Execute tool calls via MCP and return tool-result messages.
        
        Each tool call produces a Message with role=TOOL containing the result
        text, keyed by tool_call_id so the provider can match call → result.
        """
        results: list[Message] = []
        
        for tc in tool_calls:
            # Find which server owns this tool
            tool_info = self.mcp_client.tool_executor.find_tool(tc.name)
            if not tool_info:
                results.append(Message(
                    role=MessageRole.TOOL,
                    content=json.dumps({"error": f"Tool '{tc.name}' not found"}),
                    tool_call_id=tc.id,
                    name=tc.name,
                ))
                continue
            
            try:
                tool_result = self.mcp_client.call_tool_sync(
                    tool_info.server_name,
                    tc.name,
                    tc.arguments,
                )
                # Always use the text content - even errors from MCP tools are in content
                content = tool_result.text
                if not content:
                    content = tool_result.error or "Tool execution failed (no output)"
                if not tool_result.success:
                    logger.warning(f"Tool '{tc.name}' returned error: {content[:200]}")
            except Exception as e:
                logger.error(f"Tool execution error for '{tc.name}': {e}")
                content = json.dumps({"error": str(e)})

            # Handle large tool results - store in Redis or truncate
            # Skip for retrieval tools (they access already-stored content, re-storing causes loops)
            from ..mcp.internal_tools import is_retrieval_tool
            max_chars = self.config.max_tool_result_chars
            original_len = len(content)
            if original_len > max_chars and not is_retrieval_tool(tc.name):
                stored = False
                if self.config.store_oversized_results:
                    # Try to store full output in Redis
                    storage_key = store_tool_output(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=content,
                        ttl_seconds=self.config.tool_output_ttl_seconds,
                    )
                    if storage_key:
                        # Try LLM compression, fall back to dumb preview
                        compressed_preview = None
                        if self.config.compress_tool_outputs:
                            try:
                                compressor = get_compressor()
                                cr = compressor.compress_sync(
                                    tool_name=tc.name,
                                    tool_output=content,
                                    task_context=task_context,
                                )
                                if cr.success and cr.compressed_text:
                                    compressed_preview = cr.compressed_text
                                    logger.info(
                                        f"Compressed tool output for '{tc.name}': "
                                        f"{original_len:,} -> {cr.compressed_chars:,} chars "
                                        f"({cr.tokens_used} tokens)"
                                    )
                            except Exception as e:
                                logger.warning(f"Compression failed for '{tc.name}', using preview: {e}")

                        tool_hint = (
                            f"Retrieval tools for key=\"{storage_key}\":\n"
                            f"- read_stored_output(key, offset=0, limit=10000) — paginated raw content\n"
                            f"- tool_output_query(key, query) — semantic search (best for finding specific info)\n"
                            f"- tool_output_section(key) — list/access named sections\n"
                            f"- tool_output_path(key, jsonpath) — JSON path query\n"
                            f"TIP: Prefer query/section/path over reading raw content."
                        )
                        if compressed_preview:
                            content = (
                                f"{compressed_preview}\n\n"
                                f"[COMPRESSED SUMMARY - {original_len:,} chars total]\n"
                                f"{tool_hint}"
                            )
                        else:
                            preview = content[:1000] + "..." if len(content) > 1000 else content
                            content = (
                                f"{preview}\n\n"
                                f"[OUTPUT STORED - {original_len:,} chars total]\n"
                                f"{tool_hint}"
                            )
                        logger.info(f"Stored tool result for '{tc.name}' in Redis: {storage_key} ({original_len:,} chars)")
                        stored = True

                if not stored:
                    # Fall back to truncation (Redis unavailable or storage disabled)
                    truncated_content = content[:max_chars]
                    content = f"{truncated_content}\n\n[OUTPUT TRUNCATED - {original_len:,} chars total, showing first {max_chars:,}]"
                    logger.info(f"Truncated tool result for '{tc.name}': {original_len:,} -> {max_chars:,} chars")

            results.append(Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tc.id,
                name=tc.name,
            ))
        
        return results
    
    def _complete_with_tools(
        self,
        provider,
        model_id: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> tuple[CompletionResult, list[str]]:
        """
        Call provider.complete() in a tool-use loop.
        
        If the model returns tool_calls, execute them, append results to
        messages, and call again — up to max_tool_rounds iterations.
        
        Returns:
            (final CompletionResult, list of tool names used)
        """
        tools_used: list[str] = []
        
        for _ in range(self.config.max_tool_rounds):
            result = provider.complete(
                messages,
                model_id,
                tools=tools,
                tool_choice="auto" if tools else None,
                **kwargs,
            )
            
            if not result.tool_calls:
                return result, tools_used
            
            # Model wants to call tools — build assistant + tool messages
            tools_used.extend(tc.name for tc in result.tool_calls)
            
            # Add the assistant message with tool_calls attached
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=result.content or "",
                tool_calls=[
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in result.tool_calls
                ],
            ))
            
            # Extract task context from last user message for compression
            task_context = ""
            for msg in reversed(messages):
                if msg.role == MessageRole.USER:
                    task_context = msg.content[:500]
                    break

            # Execute and append results
            tool_messages = self._execute_tool_calls(result.tool_calls, task_context=task_context)
            messages.extend(tool_messages)

            logger.info(
                f"Tool round: executed {len(result.tool_calls)} tools "
                f"({', '.join(tc.name for tc in result.tool_calls)})"
            )

            # Trajectory compression: consolidate older rounds if context is growing large
            from ..streaming.trajectory_compression import compress_trajectory
            if compress_trajectory(messages, self.config.max_context_tokens, task_context):
                logger.info("Trajectory compressed, continuing with reduced context")

        # Exhausted rounds — do one final call without tools
        logger.warning(f"Reached max tool rounds ({self.config.max_tool_rounds})")
        result = provider.complete(messages, model_id, **kwargs)
        return result, tools_used

    def run(
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
                plan = planner.plan(task, context, memory=self.memory)
                trace.append({
                    "phase": "planning",
                    "steps": len(plan.steps) if plan else 0,
                    "goal_id": plan.goal_id if plan else None,
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
                    reasoning_result = self.reasoning.reason(task, reasoning_context)
                else:
                    reasoning_result = self.reasoning.reason(
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

                # Get MCP tools for function calling
                tools = self._get_tools_for_provider()

                result, tools_used = self._complete_with_tools(
                    provider,
                    model_id,
                    messages,
                    tools,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 2000),
                )

                answer = result.content
                total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0
                models_used = [self.config.default_model]
                reasoning_steps = 0
            
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

            # Complete goal in memory if one was created
            if self.memory and plan and plan.goal_id:
                try:
                    self.memory.complete_goal(
                        plan.goal_id,
                        status="completed",
                        result=answer[:500] if answer else None,
                    )
                except Exception as e:
                    logger.warning(f"Failed to complete goal: {e}")
            
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

            # Mark goal as abandoned on failure
            if self.memory and plan and plan.goal_id:
                try:
                    self.memory.complete_goal(
                        plan.goal_id,
                        status="abandoned",
                        result=f"Task failed: {str(e)[:400]}",
                    )
                except Exception as goal_error:
                    logger.warning(f"Failed to update goal status: {goal_error}")
            
            return AgentResult(
                task_id=task_id,
                status=AgentStatus.FAILED,
                answer=f"Task failed: {str(e)}",
                trace=trace,
            )
        finally:
            self._current_task_id = None
    
    def chat(
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

                # Get MCP tools for function calling
                tools = self._get_tools_for_provider()

                result, tools_used = self._complete_with_tools(
                    provider,
                    model_id,
                    messages,
                    tools,
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
                    tools_used=tools_used,
                    models_used=[self.config.default_model],
                    total_tokens=total_tokens,
                    total_time_ms=total_time,
                )
            else:
                # Full agent pipeline with reasoning
                result = self.run(message, context=context, **kwargs)

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

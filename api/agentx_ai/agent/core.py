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
from typing import Any

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
from .hooks import AgentHooks, MemoryRecorder, TaskOutcome
from ..utils.async_bridge import run_coro_sync
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
    thinking: str | None = None  # Extracted thinking/reasoning content
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
    trace: list[dict[str, Any]] | None = None


@dataclass
class AgentConfig:
    """Configuration for an Agent instance."""
    # Identity
    name: str = "agentx"
    user_id: str | None = None
    
    # Model settings - default to local LM Studio (use provider:model format)
    default_model: str = "lmstudio:llama3.2"
    reasoning_model: str | None = None
    drafting_model: str | None = None
    
    # Behavior settings
    max_iterations: int = 20
    timeout_seconds: float = 300.0
    
    # Capabilities
    enable_planning: bool = True
    enable_reasoning: bool = True
    enable_drafting: bool = False
    enable_memory: bool = True
    enable_tools: bool = True
    # Direct mode: send the model only the user message — no system prompt, no memory,
    # no tools. Set from the agent profile (and auto-forced for image-only models in the
    # streaming chat path). See AgentProfile.direct_mode.
    direct_mode: bool = False

    # Reasoning settings
    default_reasoning_strategy: str = "auto"  # "auto", "cot", "tot", "react", "reflection"
    
    # Tool settings
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] | None = None
    max_tool_rounds: int = 10  # Max tool-call ↔ result round-trips per request
    max_tool_result_chars: int = 12000  # Threshold for storing oversized results in Redis
    store_oversized_results: bool = True  # Store large results in Redis instead of truncating
    tool_output_ttl_seconds: int = 3600  # TTL for stored tool outputs (1 hour default)
    compress_tool_outputs: bool = True  # Use LLM compression for oversized tool outputs

    # User message caching settings
    user_message_cache_threshold: int = 5000  # Chars threshold for caching user messages
    user_message_cache_ttl: int = 3600  # TTL for cached user messages (1 hour default)
    user_message_preview_chars: int = 2000  # Chars to keep in context after caching

    # Context settings
    max_context_tokens: int = 8000
    summarize_threshold: int = 6000
    
    # Prompt settings
    prompt_profile_id: str | None = None  # Use default if None
    agent_id: str | None = None  # Human-friendly agent identifier for self-memory channel

    # Memory settings
    memory_channel: str = "_default"  # Channel for memory scoping (use _default, not _global)
    memory_top_k: int = 10  # Number of memories to retrieve
    memory_time_window_hours: int | None = None  # Time window filter for retrieval
    memory_recall_turn_chars: int = 2000  # Per-turn char budget when formatting recall context
    memory_recall_max_turns: int = 10  # Max recalled turns surfaced in the prompt


class Agent:
    """
    AgentX Core Agent.
    
    The Agent class is the central orchestrator that combines all AgentX
    capabilities to solve complex tasks through planning, reasoning, and action.
    
    Example usage:
        agent = Agent(AgentConfig(
            default_model="anthropic:claude-3-5-sonnet-latest",
            enable_tools=True,
        ))
        
        result = await agent.run("Analyze the codebase and suggest improvements")
    """
    
    def __init__(self, config: AgentConfig, *, registry: ProviderRegistry | None = None):
        self.config = config
        self.status = AgentStatus.IDLE

        # Core components (lazy-loaded). An injected registry takes precedence;
        # the `registry` property falls back to get_registry() when None.
        self._registry: ProviderRegistry | None = registry
        self._reasoning: ReasoningOrchestrator | None = None
        self._drafting: DraftingStrategy | None = None
        self._context_manager = None
        self._session_manager = None
        self._memory = None
        self._mcp_client = None
        # Lifecycle subscribers (built lazily once memory resolves)
        self._hooks: list[AgentHooks] | None = None

        # Runtime state
        self._current_task_id: str | None = None
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
    def drafting(self) -> DraftingStrategy | None:
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
                agent_id=self.config.agent_id,
            )
            if self._memory is None:
                logger.warning("Memory system unavailable, agent will operate without persistent memory")
        return self._memory

    @property
    def hooks(self) -> list[AgentHooks]:
        """Agent lifecycle subscribers. Built once; registers a MemoryRecorder
        when memory is available so write-backs route through the hook seam."""
        if self._hooks is None:
            self._hooks = []
            mem = self.memory
            if mem is not None:
                self._hooks.append(MemoryRecorder(mem))
        return self._hooks

    def _dispatch(self, event: str, *args: Any) -> None:
        """Fire a lifecycle event to all hooks, isolating subscriber failures."""
        for hook in self.hooks:
            try:
                getattr(hook, event)(*args)
            except Exception as e:
                logger.warning(f"Agent hook {type(hook).__name__}.{event} failed: {e}")

    @property
    def mcp_client(self):
        """Lazy-load the MCP client manager using the global singleton."""
        if self._mcp_client is None and self.config.enable_tools:
            from ..mcp import get_mcp_manager
            self._mcp_client = get_mcp_manager()
            
            # Route tool-usage recording through the hook seam (MemoryRecorder
            # subscribes when memory is available).
            if self.hooks:
                self._mcp_client.tool_executor.set_usage_recorder(
                    lambda *a: self._dispatch("on_tool_use", *a)
                )

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
        
        # Phase 18.2: server-side per-agent whitelist (allowed_agent_ids on ServerConfig).
        # Build server_name -> bool gate once per call.
        # Security: when a whitelist is set but the caller's agent_id is unknown,
        # default to DENY rather than allow — otherwise a misrouted request silently
        # bypasses the whitelist.
        agent_id = getattr(self.config, "agent_id", None)
        server_gate: dict[str, bool] = {}
        try:
            for cfg in self.mcp_client.registry.list():
                allowed = cfg.allowed_agent_ids
                if allowed is None:
                    server_gate[cfg.name] = True
                elif agent_id is None:
                    server_gate[cfg.name] = False
                    logger.warning(
                        f"Server '{cfg.name}' has whitelist but agent_id is unknown — denying"
                    )
                else:
                    server_gate[cfg.name] = agent_id in allowed
            logger.info(
                f"[tool-gate] agent_id={agent_id!r} gate={server_gate}"
            )
        except Exception as e:
            logger.warning(f"Failed to build server gate: {e}")

        from ..mcp.internal_tools import legacy_names_for

        tools = []
        for t in mcp_tools:
            # Server-side whitelist (default-allow only for servers not in registry,
            # e.g. internal tools with server_name="_internal").
            server_name = getattr(t, "server_name", None) or getattr(t, "server", None)
            if server_name in server_gate and server_gate[server_name] is False:
                continue
            # Phase 18.9.x: allow/block filters keyed by fully-qualified
            # `{server_name}.{tool_name}`. Two MCP servers may legitimately
            # expose a same-named tool; matching on bare names would gate them
            # together. Internal tools resolve to `_internal.<name>`. Tools
            # without a server attribution (legacy paths) fall back to the
            # bare name so old configs keep working until they're migrated.
            # Renamed internal tools (e.g. workspace_search → project_search)
            # also match their legacy names so existing profile lists hold.
            names = {t.name, *legacy_names_for(t.name)}
            candidates = (
                {f"{server_name}.{n}" for n in names} if server_name else names
            )
            if self.config.allowed_tools and candidates.isdisjoint(self.config.allowed_tools):
                continue
            if self.config.blocked_tools and not candidates.isdisjoint(self.config.blocked_tools):
                continue

            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema or {"type": "object"},
                },
            })

        if not tools:
            return None

        # Don't send tools to a model that can't use them — OpenRouter 404s
        # ("no endpoints support tool use") rather than ignoring them. Gate on the
        # resolved model's capability, but only disable when *confirmed* tool-less:
        # a provider with a lazy model catalog (OpenRouter) reports tools=False for an
        # uncached model, so warm the catalog once and re-check before stripping tools.
        if not self._model_supports_tools():
            logger.info(
                f"[tool-gate] model {self.config.default_model!r} has no tool support — sending no tools"
            )
            return None
        return tools

    def _model_supports_tools(self) -> bool:
        """Whether the agent's chat model supports tool/function calling. Defaults to
        True on any uncertainty (never strip tools from a capable model on a probe miss)."""
        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(self.config.default_model)
            caps = provider.get_capabilities(model_id)
            if caps.supports_tools:
                return True
            # Possibly a cold lazy catalog → warm it once, then re-check authoritatively.
            warm = getattr(provider, "fetch_models", None)
            if warm is not None:
                from ..utils.async_bridge import run_coro_sync
                run_coro_sync(warm(), timeout=15.0)
                caps = provider.get_capabilities(model_id)
            return bool(caps.supports_tools)
        except Exception as e:  # noqa: BLE001 — never block tools on a capability-probe failure
            logger.debug(f"[tool-gate] capability probe failed, assuming tools ok: {e}")
            return True
    
    def _execute_tool_calls(self, tool_calls: list[ToolCall], task_context: str = "") -> list[Message]:
        """
        Execute tool calls via MCP and return tool-result messages.
        
        Each tool call produces a Message with role=TOOL containing the result
        text, keyed by tool_call_id so the provider can match call → result.
        """
        results: list[Message] = []

        mcp = self.mcp_client
        if mcp is None:
            # Tools are disabled; surface a result per call rather than crashing.
            return [
                Message(
                    role=MessageRole.TOOL,
                    content=json.dumps({"error": "Tools are not enabled"}),
                    tool_call_id=tc.id,
                    name=tc.name,
                )
                for tc in tool_calls
            ]

        for tc in tool_calls:
            # Find which server owns this tool
            tool_info = mcp.tool_executor.find_tool(tc.name)
            if not tool_info:
                results.append(Message(
                    role=MessageRole.TOOL,
                    content=json.dumps({"error": f"Tool '{tc.name}' not found"}),
                    tool_call_id=tc.id,
                    name=tc.name,
                ))
                continue
            
            try:
                tool_result = mcp.call_tool_sync(
                    tool_info.server_name,
                    tc.name,
                    tc.arguments,
                )
                # Flatten content blocks — text-only results are byte-identical to the
                # legacy `.text`; image/audio blocks get stored + surfaced as served-blob
                # refs (see mcp.media_passthrough). Errors from MCP tools ride content too.
                from ..mcp.media_passthrough import flatten_result_content

                content = flatten_result_content(
                    tool_result.content,
                    tool_name=tc.name,
                    server_name=tool_info.server_name,
                )
                if not content:
                    content = tool_result.error or "Tool execution failed (no output)"
                if not tool_result.success:
                    logger.warning(f"Tool '{tc.name}' returned error: {content[:200]}")
            except Exception as e:
                logger.error(f"Tool execution error for '{tc.name}': {e}")
                content = json.dumps({"error": str(e)})

            content = self.handle_oversized_tool_output(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=content,
                task_context=task_context,
            )

            results.append(Message(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id=tc.id,
                name=tc.name,
            ))

        return results

    def handle_oversized_tool_output(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        content: str,
        task_context: str = "",
    ) -> str:
        """
        Apply storage + compression to a tool result if it exceeds the
        size threshold, returning the (possibly summarized) content with
        a retrieval hint pointing to the stored key.

        Used by both ``_execute_tool_calls`` (regular tool path) and the
        Agent Alloy delegation path so the supervisor sees the same
        ``read_stored_output``/``tool_output_query`` affordances when a
        specialist returns a large response.
        """
        from ..mcp.internal_tools import is_retrieval_tool

        max_chars = self.config.max_tool_result_chars
        original_len = len(content)
        if original_len <= max_chars or is_retrieval_tool(tool_name):
            return content

        if self.config.store_oversized_results:
            storage_key = store_tool_output(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=content,
                ttl_seconds=self.config.tool_output_ttl_seconds,
            )
            if storage_key:
                compressed_preview = None
                if self.config.compress_tool_outputs:
                    try:
                        compressor = get_compressor()
                        cr = compressor.compress_sync(
                            tool_name=tool_name,
                            tool_output=content,
                            task_context=task_context,
                            preferred_fallback=self.config.default_model,
                        )
                        if cr.success and cr.compressed_text:
                            compressed_preview = cr.compressed_text
                            logger.info(
                                f"Compressed tool output for '{tool_name}': "
                                f"{original_len:,} -> {cr.compressed_chars:,} chars "
                                f"({cr.tokens_used} tokens)"
                            )
                    except Exception as e:
                        logger.warning(f"Compression failed for '{tool_name}', using preview: {e}")

                tool_hint = (
                    f"Retrieval tools for key=\"{storage_key}\":\n"
                    f"- read_stored_output(key, offset=0, limit=12000) — paginated raw content\n"
                    f"- tool_output_query(key, query) — semantic search (best for finding specific info)\n"
                    f"- tool_output_section(key) — list/access named sections\n"
                    f"- tool_output_path(key, jsonpath) — JSON path query\n"
                    f"TIP: Prefer query/section/path over reading raw content."
                )
                if compressed_preview:
                    body = (
                        f"{compressed_preview}\n\n"
                        f"[COMPRESSED SUMMARY - {original_len:,} chars total]\n"
                        f"{tool_hint}"
                    )
                else:
                    preview = content[:1000] + "..." if len(content) > 1000 else content
                    body = (
                        f"{preview}\n\n"
                        f"[OUTPUT STORED - {original_len:,} chars total]\n"
                        f"{tool_hint}"
                    )
                logger.info(
                    f"Stored tool result for '{tool_name}' in Redis: {storage_key} ({original_len:,} chars)"
                )
                return body

        # Fall back to truncation (Redis unavailable or storage disabled)
        truncated_content = content[:max_chars]
        logger.info(f"Truncated tool result for '{tool_name}': {original_len:,} -> {max_chars:,} chars")
        return (
            f"{truncated_content}\n\n"
            f"[OUTPUT TRUNCATED - {original_len:,} chars total, showing first {max_chars:,}]"
        )

    def _complete_with_tools(
        self,
        provider,
        model_id: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        *,
        max_context_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[CompletionResult, list[str]]:
        """
        Call provider.complete() in a tool-use loop.

        If the model returns tool_calls, execute them, append results to
        messages, and call again — up to max_tool_rounds iterations.
        ``max_context_tokens`` is the in-turn trajectory-compression ceiling;
        callers that resolved the model's real window pass it (``chat()``),
        others fall back to the conservative ``config.max_context_tokens``.

        Returns:
            (final CompletionResult, list of tool names used)
        """
        tools_used: list[str] = []
        trajectory_limit = max_context_tokens or self.config.max_context_tokens
        
        for _ in range(self.config.max_tool_rounds):
            # Provider.complete is async; bridge it to this sync path (used by
            # the non-streaming chat + background worker). 10 min covers slow
            # local models pulling tools.
            result = run_coro_sync(
                provider.complete(
                    messages,
                    model_id,
                    tools=tools,
                    tool_choice="auto" if tools else None,
                    **kwargs,
                ),
                timeout=600.0,
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
            if compress_trajectory(
                messages, trajectory_limit, task_context,
                active_model=self.config.default_model,
            ):
                logger.info("Trajectory compressed, continuing with reduced context")

        # Exhausted rounds — do one final call without tools
        logger.warning(f"Reached max tool rounds ({self.config.max_tool_rounds})")
        result = run_coro_sync(provider.complete(messages, model_id, **kwargs), timeout=600.0)
        return result, tools_used

    def run(
        self,
        task: str,
        context: list[Message] | None = None,
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

        # Bound before the try so the except handler can reference it safely
        # even if planning never ran.
        plan = None
        try:
            self.status = AgentStatus.PLANNING

            # Step 1: Planning (if enabled)
            from ..config import get_config_manager
            _pcfg = get_config_manager()
            if self.config.enable_planning and _pcfg.get("planner.enabled", True):
                from .planner import TaskPlanner
                planner = TaskPlanner(
                    _pcfg.get("planner.model") or self.config.default_model,
                    temperature=_pcfg.get("planner.temperature", 0.3),
                    max_tokens=_pcfg.get("planner.max_tokens", 1000),
                    prompt_override=_pcfg.get("planner.prompt_override", ""),
                )
                # plan() is async — bridge to sync context
                plan = run_coro_sync(planner.plan(task, context, memory=self.memory))
                trace.append({
                    "phase": "planning",
                    "steps": len(plan.steps) if plan else 0,
                    "goal_id": plan.goal_id if plan else None,
                })

            if self._cancel_requested:
                return self._cancelled_result(task_id, start_time)

            # Step 1b: Plan execution for non-trivial plans
            if plan and plan.complexity.value != "simple" and len(plan.steps) > 1:
                from .plan_state import PlanStateStore
                from .plan_executor import PlanExecutor

                self.status = AgentStatus.EXECUTING
                state_store = PlanStateStore(kwargs.get("session_id", "default"))
                executor = PlanExecutor(self, state_store)
                answer = executor.execute(plan, context)
                tools_used = []
                total_tokens = 0
                models_used = [self.config.default_model]
                total_time = (time.time() - start_time) * 1000
                self.status = AgentStatus.COMPLETE

                trace.append({"phase": "plan_execution", "steps": len(plan.steps)})

                self._dispatch("on_task_complete", TaskOutcome(
                    task_id=task_id,
                    task=task,
                    status="complete",
                    answer=answer,
                    total_tokens=total_tokens,
                    total_time_ms=total_time,
                    reasoning_steps=0,
                    tools_used=tools_used,
                    goal_id=plan.goal_id,
                ))

                return AgentResult(
                    task_id=task_id,
                    status=AgentStatus.COMPLETE,
                    answer=answer,
                    plan_steps=len(plan.steps),
                    reasoning_steps=0,
                    tools_used=list(set(tools_used)),
                    models_used=list(set(models_used)),
                    total_tokens=total_tokens,
                    total_time_ms=total_time,
                    trace=trace,
                )

            # Step 2: Reasoning (simple tasks / planning disabled)
            self.status = AgentStatus.REASONING

            reasoning_result = None
            if self.config.enable_reasoning:
                strategy = kwargs.get("reasoning_strategy", self.config.default_reasoning_strategy)

                # Inject memories into context if available
                reasoning_context = context
                if memory_bundle and context:
                    from .context import ContextManager, ContextConfig
                    if self._context_manager is None:
                        # ContextManager here is used only for inject_memory; the
                        # legacy token-budget knobs were retired in Foundation #6.
                        self._context_manager = ContextManager(ContextConfig())
                    reasoning_context = self._context_manager.inject_memory(context, memory_bundle)

                if strategy == "auto":
                    reasoning_result = run_coro_sync(
                        self.reasoning.reason(task, reasoning_context)
                    )
                else:
                    reasoning_result = run_coro_sync(
                        self.reasoning.reason(
                            task, reasoning_context, strategy=strategy
                        )
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
                provider, model_id, _ = self.registry.resolve_with_fallback(
                    self.config.default_model
                )

                messages = [Message(role=MessageRole.USER, content=task)]
                if context:
                    messages = context + messages

                # Inject memories if available
                if memory_bundle:
                    from .context import ContextManager, ContextConfig
                    if self._context_manager is None:
                        # ContextManager here is used only for inject_memory; the
                        # legacy token-budget knobs were retired in Foundation #6.
                        self._context_manager = ContextManager(ContextConfig())
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
            
            self._dispatch("on_task_complete", TaskOutcome(
                task_id=task_id,
                task=task,
                status="complete",
                answer=answer,
                total_tokens=total_tokens,
                total_time_ms=total_time,
                reasoning_steps=reasoning_steps,
                tools_used=tools_used,
                goal_id=plan.goal_id if plan else None,
            ))

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

            self._dispatch("on_task_error", TaskOutcome(
                task_id=task_id,
                task=task,
                status="failed",
                error=str(e),
                goal_id=plan.goal_id if plan else None,
            ))

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
        session_id: str | None = None,
        simple_mode: bool = True,
        profile_id: str | None = None,
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

        # Get or create session — use the process-wide singleton so history
        # survives across per-request Agent instances.
        from .session import get_session_manager
        self._session_manager = get_session_manager()

        session = self._session_manager.get_or_create(session_id)
        conversation_id = session_id or session.id

        # Rehydrate a cold session (new process / evicted / a queued background
        # job picking up an existing conversation) from durable history before
        # the new turn is appended. Idempotent — a no-op on an already-live or
        # already-hydrated session — so it never clobbers in-process state. This
        # is what makes the background-chat path resume warm (the interactive
        # stream hydrates separately in views.py).
        try:
            from .conversation_history import hydrate_session_from_history
            hydrate_session_from_history(session, conversation_id, token_budget=10_000_000)
        except Exception as _hy_err:  # pragma: no cover - DB offline
            logger.debug(f"Session rehydration skipped: {_hy_err}")

        # Update memory with conversation context if available
        if self.memory:
            self.memory.conversation_id = conversation_id

        # Cache large user messages in Redis to save context
        message_for_context = message  # Version sent to LLM (may be truncated)
        if len(message) > self.config.user_message_cache_threshold:
            from .user_message_storage import store_user_message

            message_id = str(uuid.uuid4())[:8]
            cached_message_key = store_user_message(
                message_id=message_id,
                content=message,
                session_id=session_id,
                ttl_seconds=self.config.user_message_cache_ttl,
            )

            if cached_message_key:
                # Create truncated version with cache hint for context
                preview_chars = self.config.user_message_preview_chars
                message_for_context = (
                    f"{message[:preview_chars]}\n\n"
                    f"[USER MESSAGE CACHED - key: {cached_message_key}]\n"
                    f"Full message ({len(message):,} chars) stored in cache. "
                    f"Use read_user_message(key=\"{cached_message_key}\") to retrieve full content."
                )
                logger.info(f"Cached large user message: {cached_message_key} ({len(message):,} chars)")

        # Add user message to session (full message for history)
        user_message = Message(role=MessageRole.USER, content=message)
        session.add_message(user_message)

        # Store user turn in memory (full message)
        if self.hooks:
            user_turn = Turn(
                conversation_id=conversation_id,
                index=len(session.get_messages()) - 1,
                role="user",
                content=message,
            )
            self._dispatch("on_turn", user_turn)

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
                provider, model_id, _ = self.registry.resolve_with_fallback(
                    self.config.default_model
                )

                # Get system prompt from prompt manager
                from ..prompts import get_prompt_manager
                prompt_manager = get_prompt_manager()
                system_prompt = prompt_manager.get_system_prompt(
                    profile_id=profile_id or self.config.prompt_profile_id
                )

                # Budget-fit assembly (window-aware): this path previously dumped
                # the ENTIRE rehydrated session into the prompt — a long resumed
                # conversation could exceed the provider window outright. Resolve
                # the model's real window (per-model overrides win, like the
                # streaming path) and fit the recent verbatim tail; persisted
                # compaction coverage (state digest / legacy summary) rides as a
                # system block so aged-out turns stay represented.
                from ..config import get_config_manager, get_context_limit_overrides

                caps = provider.get_capabilities(model_id)
                overrides = get_context_limit_overrides(model_id, provider.name)
                context_window = overrides.get("context_window") or caps.context_window
                max_tokens = kwargs.get("max_tokens", 2000)

                system_blocks = [Message(
                    role=MessageRole.SYSTEM,
                    content=system_prompt or "You are a helpful AI assistant.",
                )]
                try:
                    from .conversation_state_storage import render_state_block
                    coverage = render_state_block(conversation_id)
                    if coverage:
                        system_blocks.append(
                            Message(role=MessageRole.SYSTEM, content=coverage)
                        )
                except Exception as st_err:  # pragma: no cover — Redis offline
                    logger.debug(f"State block skipped (chat path): {st_err}")
                if session.summary:
                    system_blocks.append(Message(
                        role=MessageRole.SYSTEM,
                        content=f"Earlier conversation summary: {session.summary}",
                    ))

                from .context import ContextManager, ContextConfig
                if self._context_manager is None:
                    self._context_manager = ContextManager(ContextConfig())
                cfg_mgr = get_config_manager()
                messages = self._context_manager.assemble_turn_context(
                    system_blocks=system_blocks,
                    history=context,
                    new_message=Message(role=MessageRole.USER, content=message_for_context),
                    context_window=context_window,
                    reserved_tokens=max_tokens + 2000,
                    verbatim_ratio=float(cfg_mgr.get("context.verbatim_budget_ratio", 0.9)),
                    recent_floor=int(cfg_mgr.get("context.recent_floor", 4)),
                )

                # Inject memories into context
                if memory_bundle:
                    messages = self._context_manager.inject_memory(messages, memory_bundle)

                logger.info(f"Agent chat {task_id} using {model_id}")

                # Get MCP tools for function calling
                tools = self._get_tools_for_provider()

                result, tools_used = self._complete_with_tools(
                    provider,
                    model_id,
                    messages,
                    tools,
                    max_context_tokens=max(context_window - max_tokens - 2000, 4096),
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=max_tokens,
                )

                # Parse output to extract thinking tags
                from .output_parser import parse_output
                parsed = parse_output(result.content)

                answer = parsed.content
                thinking = parsed.thinking
                total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0

                # Add assistant response to session (store parsed content)
                session.add_message(Message(role=MessageRole.ASSISTANT, content=answer))

                # Post-turn compaction pre-warm — the SAME target gate as the
                # streaming path — so background/queued conversations keep their
                # digest fresh instead of growing until a later fit drops turns.
                try:
                    from .session import compaction_uses_state
                    input_budget = min(
                        int(context_window * float(
                            cfg_mgr.get("context.verbatim_budget_ratio", 0.9))),
                        context_window - (max_tokens + 2000),
                    )
                    threshold = int(input_budget * float(
                        cfg_mgr.get("context.summary_trigger_ratio", 0.85)))
                    recent_floor = int(cfg_mgr.get("context.recent_floor", 4))
                    if threshold > 0:
                        if compaction_uses_state(cfg_mgr):
                            run_coro_sync(self._session_manager.maybe_compact_to_state(
                                session.id, token_threshold=threshold,
                                recent_floor=recent_floor,
                            ))
                        else:
                            run_coro_sync(self._session_manager.maybe_update_summary(
                                session.id, token_threshold=threshold,
                                recent_floor=recent_floor,
                            ))
                except Exception as pw_err:  # noqa: BLE001 — never fail the turn
                    logger.debug(f"Compaction pre-warm skipped (chat path): {pw_err}")

                total_time = (time.time() - start_time) * 1000

                # Store assistant turn in memory
                if self.hooks:
                    assistant_turn = Turn(
                        conversation_id=conversation_id,
                        index=len(session.get_messages()) - 1,
                        role="assistant",
                        content=answer,
                        token_count=total_tokens,
                        agent_id=self.config.agent_id,
                        metadata={
                            "model": model_id,
                            "latency_ms": total_time,
                            "task_id": task_id,
                            "agent_name": self.config.name,
                        }
                    )
                    self._dispatch("on_turn", assistant_turn)

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
                if self.hooks:
                    assistant_turn = Turn(
                        conversation_id=conversation_id,
                        index=len(session.get_messages()) - 1,
                        role="assistant",
                        content=result.answer,
                        token_count=result.total_tokens,
                        agent_id=self.config.agent_id,
                        metadata={
                            "model": result.models_used[0] if result.models_used else self.config.default_model,
                            "latency_ms": result.total_time_ms,
                            "task_id": task_id,
                            "reasoning_steps": result.reasoning_steps,
                            "tools_used": result.tools_used,
                            "agent_name": self.config.name,
                        }
                    )
                    self._dispatch("on_turn", assistant_turn)

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

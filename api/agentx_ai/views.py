import json
import logging
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from .kit.memory_utils import check_memory_health
from .mcp import get_mcp_manager
from .providers import get_registry
from .streaming import (
    CONTEXT_BUFFER_TOKENS,
    CONTEXT_WARNING_THRESHOLD,
    DEFAULT_MAX_TOOL_ROUNDS,
    MAX_INPUT_TOKENS,
    MIN_OUTPUT_TOKENS,
    STREAM_CLOSE_DELAY,
    estimate_tokens,
    resolve_with_priority,
    truncate_tool_messages,
)
from .utils.decorators import lazy_singleton
from .utils.responses import (
    json_error,
    json_success,
    parse_json_body,
    require_methods,
)

logger = logging.getLogger(__name__)


@lazy_singleton
def get_translation_kit():
    """Get or create TranslationKit instance lazily."""
    from .kit.translation import TranslationKit
    logger.info("Initializing TranslationKit (loading models)...")
    kit = TranslationKit()
    logger.info("TranslationKit initialized successfully")
    return kit


def index(request):
    return JsonResponse({'message': 'Hello, AgentX AI!'})


def health(request):
    """
    Health check endpoint for all services.

    Returns status of:
    - API server (always healthy if responding)
    - Translation models (loaded or not)
    - Memory system connections (neo4j, postgres, redis)
    - Storage metrics (postgres size, neo4j size, redis memory) when include_storage=true
    """
    # Check translation kit (don't initialize just for health check)
    kit = get_translation_kit.get_if_initialized()
    translation_status = {
        "status": "healthy" if kit else "not_loaded",
        "models": {
            "language_detection": kit.language_detection_model_name if kit else None,
            "translation": kit.level_ii_translation_model_name if kit else None,
        }
    }

    # Check memory system (lazy - only if explicitly requested)
    include_memory = request.GET.get('include_memory', 'false').lower() == 'true'
    include_storage = request.GET.get('include_storage', 'false').lower() == 'true'
    memory_status = None
    if include_memory:
        memory_status = check_memory_health()

    response = {
        "status": "healthy",
        "api": {"status": "healthy"},
        "translation": translation_status,
    }

    if memory_status:
        response["memory"] = memory_status
        # Overall status is unhealthy if any memory component is unhealthy
        if any(v["status"] == "unhealthy" for v in memory_status.values()):
            response["status"] = "degraded"

    # Get storage metrics if requested
    if include_storage:
        response["storage"] = _get_storage_metrics()

    return JsonResponse(response)


def _get_storage_metrics() -> dict:
    """Get storage size metrics for all databases."""
    from .kit.agent_memory.connections import PostgresConnection, Neo4jConnection, RedisConnection
    from sqlalchemy import text

    storage = {
        "postgres_size_mb": None,
        "neo4j_size_mb": None,
        "redis_memory_mb": None,
    }

    # PostgreSQL database size
    try:
        engine = PostgresConnection.get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT pg_database_size(current_database()) / 1024.0 / 1024.0 AS size_mb"))
            row = result.fetchone()
            if row:
                storage["postgres_size_mb"] = round(row[0], 2)
    except Exception:
        pass

    # Neo4j store size (use dbms.queryJmx or estimate from count)
    try:
        with Neo4jConnection.session() as session:
            # Get approximate size by counting nodes/relationships
            result = session.run("""
                CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store sizes')
                YIELD attributes
                RETURN attributes.TotalStoreSize.value AS total_bytes
            """)
            record = result.single()
            if record and record["total_bytes"]:
                storage["neo4j_size_mb"] = round(record["total_bytes"] / 1024 / 1024, 2)
    except Exception:
        # Fallback: estimate from node/relationship counts
        try:
            with Neo4jConnection.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as nodes")
                record = result.single()
                # Rough estimate: ~500 bytes per node
                if record:
                    storage["neo4j_size_mb"] = round(record["nodes"] * 500 / 1024 / 1024, 2)
        except Exception:
            pass

    # Redis memory usage
    try:
        redis_client = RedisConnection.get_client()
        info = redis_client.info("memory")
        if "used_memory" in info:
            storage["redis_memory_mb"] = round(info["used_memory"] / 1024 / 1024, 2)
    except Exception:
        pass

    return storage


@csrf_exempt
@require_methods("POST")
def translate(request):
    """Translate text to a target language."""
    data, error = parse_json_body(request)
    if error:
        return error
    assert data is not None  # Type narrowing for pyright

    logger.info(f"Translation request received, body size: {len(request.body)}")

    text = data.get("text")
    # Accept both camelCase and snake_case for compatibility
    target_language = data.get("targetLanguage") or data.get("target_language")

    if not text:
        return json_error("Missing required field: text")
    if not target_language:
        return json_error("Missing required field: targetLanguage or target_language")

    logger.debug(f"Translation request: target={target_language}, text_length={len(text)}")

    try:
        translated_text = get_translation_kit().translate_text(text, target_language, target_language_level=2)
    except ValueError as e:
        return json_error(str(e))
    except Exception as e:
        logger.exception("Translation error")
        return json_error(f"Translation failed: {str(e)}", status=500)

    return json_success({
        "original": text,
        "translatedText": str(translated_text),
    })


@csrf_exempt
def language_detect(request):
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    # Accept both GET (for backwards compatibility) and POST
    if request.method == 'POST':
        try:
            content = request.body.decode('utf-8')
            if not content:
                return JsonResponse({'error': 'No content provided'}, status=400)
            data = json.loads(content)
            unclassified_text = data.get("text")
            if not unclassified_text:
                return JsonResponse({'error': 'Missing required field: text'}, status=400)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)
    elif request.method == 'GET':
        # Backwards compatibility: use query param or default text
        unclassified_text = request.GET.get('text', "Hello, AgentX AI, this is a language test to detect the spoken language!")
    else:
        return JsonResponse({'error': 'Only GET or POST requests allowed'}, status=405)

    logger.info(f"Language detection request, text_length: {len(unclassified_text)}")

    detected_language, confidence = get_translation_kit().detect_language_level_i(unclassified_text)

    return JsonResponse({
        'original': unclassified_text,
        'detected_language': detected_language,
        'confidence': confidence
    })


def mcp_servers(request):
    """List configured MCP servers and their connection status."""
    manager = get_mcp_manager()
    
    servers = []
    for config in manager.registry.list():
        connection = manager.get_connection(config.name)
        server_data = {
            "name": config.name,
            "transport": config.transport.value,
            "status": "connected" if connection else "disconnected",
            "tools": [t.name for t in connection.tools] if connection else [],
            "tools_count": len(connection.tools) if connection else 0,
            "resources_count": len(connection.resources) if connection else 0,
        }
        servers.append(server_data)
    
    return JsonResponse({"servers": servers})


def mcp_tools(request):
    """List available tools from connected MCP servers."""
    manager = get_mcp_manager()
    server_name = request.GET.get('server')
    
    tools = manager.list_tools(server_name)
    
    return JsonResponse({
        "tools": [tool.to_dict() for tool in tools],
        "count": len(tools),
    })


def mcp_resources(request):
    """List available resources from connected MCP servers."""
    manager = get_mcp_manager()
    server_name = request.GET.get('server')
    
    resources = manager.list_resources(server_name)
    
    return JsonResponse({
        "resources": [res.to_dict() for res in resources],
        "count": len(resources),
    })


@csrf_exempt
def mcp_connect(request):
    """Connect to one or all configured MCP servers."""
    if request.method != "POST":
        return json_error("Method not allowed", status=405)
    
    data, error = parse_json_body(request)
    if error:
        return error
    
    manager = get_mcp_manager()
    server_name = data.get("server")
    connect_all = data.get("all", False)
    
    if connect_all:
        results = manager.connect_all()
        return JsonResponse({"results": results})
    
    if not server_name:
        return json_error("Provide 'server' name or 'all': true", status=400)
    
    try:
        connection = manager.connect(server_name)
        return JsonResponse({
            "status": "connected",
            "server": server_name,
            "tools_count": len(connection.tools),
            "resources_count": len(connection.resources),
        })
    except ValueError as e:
        return json_error(str(e), status=404)
    except Exception as e:
        logger.error(f"Failed to connect to '{server_name}': {e}")
        return json_error(f"Connection failed: {e}", status=500)


@csrf_exempt
def mcp_disconnect(request):
    """Disconnect from an MCP server."""
    if request.method != "POST":
        return json_error("Method not allowed", status=405)
    
    data, error = parse_json_body(request)
    if error:
        return error
    
    manager = get_mcp_manager()
    server_name = data.get("server")
    disconnect_all = data.get("all", False)
    
    if disconnect_all:
        manager.disconnect_all()
        return JsonResponse({"status": "disconnected_all"})
    
    if not server_name:
        return json_error("Provide 'server' name or 'all': true", status=400)
    
    disconnected = manager.disconnect(server_name)
    if not disconnected:
        return json_error(f"Server '{server_name}' is not connected", status=404)
    
    return JsonResponse({"status": "disconnected", "server": server_name})


# ============== Provider Endpoints ==============

def providers_list(request):
    """List available model providers and their status."""
    registry = get_registry()
    
    providers_info = []
    for name in registry.list_providers():
        try:
            provider = registry.get_provider(name)
            providers_info.append({
                "name": name,
                "status": "configured",
                "models": provider.list_models()[:5],  # First 5 for brevity
            })
        except ValueError as e:
            providers_info.append({
                "name": name,
                "status": "not_configured",
                "error": str(e),
            })
    
    return JsonResponse({
        "providers": providers_info,
    })


async def providers_models(request):
    """List all available models across providers."""
    registry = get_registry()
    provider_filter = request.GET.get('provider')

    models = []

    for provider_name in registry.list_providers():
        if provider_filter and provider_name != provider_filter:
            continue

        try:
            provider = registry.get_provider(provider_name)

            # For providers that need to fetch models dynamically (like LM Studio),
            # call fetch_models first to populate the available models list
            if hasattr(provider, 'fetch_models'):
                await provider.fetch_models()

            for model_name in provider.list_models():
                caps = provider.get_capabilities(model_name)
                models.append({
                    "id": model_name,
                    "name": model_name,
                    "provider": provider_name,
                    "context_length": caps.context_window,
                    "context_window": caps.context_window,  # Legacy field
                    "supports_tools": caps.supports_tools,
                    "supports_vision": caps.supports_vision,
                    "supports_streaming": caps.supports_streaming,
                    "cost_per_1k_input": caps.cost_per_1k_input,
                    "cost_per_1k_output": caps.cost_per_1k_output,
                })
        except ValueError:
            # Provider not configured, skip
            continue
        except Exception as e:
            # Log but don't fail - other providers may still work
            logger.warning(f"Failed to fetch models from {provider_name}: {e}")
            continue

    return JsonResponse({
        "models": models,
        "count": len(models),
    })


async def providers_health(request):
    """Check health of all configured providers."""
    registry = get_registry()

    results = await registry.health_check()

    overall_status = "healthy"
    if any(r.get("status") != "healthy" for r in results.values()):
        overall_status = "degraded"

    return JsonResponse({
        "status": overall_status,
        "providers": results,
    })


# ============== Agent Endpoints ==============


@lazy_singleton
def get_agent():
    """Get or create Agent instance lazily."""
    import os
    from .agent import Agent, AgentConfig

    # Offline-first: default to local Ollama model
    default_model = os.environ.get("DEFAULT_MODEL", "llama3.2")

    agent = Agent(AgentConfig(
        default_model=default_model,
        enable_planning=True,
        enable_reasoning=True,
    ))
    logger.info(f"Agent initialized with model: {default_model}")
    return agent


@csrf_exempt
@require_methods("POST")
def agent_run(request):
    """Execute a task using the agent."""
    data, error = parse_json_body(request)
    if error:
        return error
    assert data is not None  # Type narrowing for pyright

    task = data.get("task")
    if not task:
        return json_error("Missing required field: task")

    reasoning_strategy = data.get("reasoning_strategy")

    agent = get_agent()

    kwargs = {}
    if reasoning_strategy:
        kwargs["reasoning_strategy"] = reasoning_strategy
    result = agent.run(task, **kwargs)

    return json_success({
        "task_id": result.task_id,
        "status": result.status.value,
        "answer": result.answer,
        "plan_steps": result.plan_steps,
        "reasoning_steps": result.reasoning_steps,
        "tools_used": result.tools_used,
        "models_used": result.models_used,
        "total_tokens": result.total_tokens,
        "total_time_ms": result.total_time_ms,
    })


@csrf_exempt
def agent_chat(request):
    """Handle a conversational message with the agent."""
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get("message")
        if not message:
            return JsonResponse({'error': 'Missing required field: message'}, status=400)

        session_id = data.get("session_id")
        model = data.get("model")
        profile_id = data.get("profile_id")  # Prompt profile to use
        _temperature = data.get("temperature", 0.7)  # Reserved for future Agent config
        use_memory = data.get("use_memory", True)

    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    # Get agent, optionally with custom model and settings
    from .agent import Agent, AgentConfig
    config_kwargs = {"enable_memory": use_memory}
    if model:
        config_kwargs["default_model"] = model
        logger.info(f"Using custom model for chat: {model}")
    # Note: temperature (_temperature) is reserved for future implementation
    agent = Agent(AgentConfig(**config_kwargs))

    result = agent.chat(message, session_id=session_id, profile_id=profile_id)

    return JsonResponse({
        "task_id": result.task_id,
        "status": result.status.value,
        "response": result.answer,  # Alias for UI compatibility
        "answer": result.answer,
        "thinking": result.thinking,  # Extracted thinking content
        "has_thinking": result.has_thinking,
        "session_id": session_id or result.task_id,  # Return session ID for continuity
        "reasoning_trace": result.reasoning_steps,
        "reasoning_steps": result.reasoning_steps,
        "tokens_used": result.total_tokens,
        "total_tokens": result.total_tokens,
        "total_time_ms": result.total_time_ms,
    })


@csrf_exempt
async def agent_chat_stream(request):
    """
    Handle a streaming conversational message with the agent.

    Returns Server-Sent Events (SSE) for real-time token streaming.
    """
    if request.method == 'OPTIONS':
        response = JsonResponse({}, status=200)
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        message = data.get("message")
        if not message:
            return JsonResponse({'error': 'Missing required field: message'}, status=400)

        session_id = data.get("session_id")
        model = data.get("model")
        profile_id = data.get("profile_id")  # Prompt profile
        agent_profile_id = data.get("agent_profile_id")  # Agent profile
        temperature = data.get("temperature")  # None if not specified
        use_memory = data.get("use_memory", True)

    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    async def generate_sse():
        """Async generator that yields SSE events."""
        import time
        import uuid
        from .agent import Agent, AgentConfig
        from .agent.session import SessionManager
        from .agent.output_parser import parse_output
        from .agent.profiles import get_profile_manager
        from .prompts import get_prompt_manager
        from .providers.base import Message, MessageRole

        task_id = str(uuid.uuid4())[:8]  # Short ID for UI display
        full_conversation_id = str(uuid.uuid4())  # Full UUID for database storage
        start_time = time.time()

        # Look up agent profile early to apply all settings
        logger.debug(f"Stream chat request: agent_profile_id={agent_profile_id}, model={model}")
        profile_manager = get_profile_manager()
        agent_profile = None
        if agent_profile_id:
            agent_profile = profile_manager.get_profile(agent_profile_id)
            if agent_profile:
                logger.debug(f"Found agent profile: {agent_profile.name} (model: {agent_profile.default_model})")
            else:
                logger.warning(f"Agent profile not found: {agent_profile_id}")
        else:
            logger.debug("No agent_profile_id in request")

        # Build agent config from profile + request overrides
        # Priority: request params > agent profile > defaults
        config_kwargs = {
            "enable_memory": use_memory,
        }

        # Model: request > profile > None (use agent default)
        resolved_model = resolve_with_priority(
            model,
            agent_profile.default_model if agent_profile else None,
        )
        if resolved_model:
            config_kwargs["default_model"] = resolved_model

        # Prompt profile: agent profile setting (request override via profile_id param)
        if agent_profile and agent_profile.prompt_profile_id:
            config_kwargs["prompt_profile_id"] = agent_profile.prompt_profile_id

        # Temperature: request > profile > default
        effective_temperature = resolve_with_priority(
            temperature,
            agent_profile.temperature if agent_profile else None,
            0.7,
        )

        logger.debug(f"Agent config_kwargs: {config_kwargs}")
        agent = Agent(AgentConfig(**config_kwargs))
        logger.debug(f"Agent created with default_model: {agent.config.default_model}")

        # Session management
        if agent._session_manager is None:
            agent._session_manager = SessionManager()
        session = agent._session_manager.get_or_create(session_id)
        session.add_message(Message(role=MessageRole.USER, content=message))
        context = session.get_messages()[:-1]

        # Resolve conversation ID early so tool usage recording has it
        conv_id = session_id or full_conversation_id
        if use_memory and agent.memory:
            agent.memory.conversation_id = conv_id

        try:
            # Get provider and model
            provider, model_id = agent.registry.get_provider_for_model(
                agent.config.default_model
            )

            # Build messages with system prompt
            prompt_manager = get_prompt_manager()

            # Get agent name from profile (already loaded above)
            agent_name = agent_profile.name if agent_profile else None

            system_prompt = prompt_manager.get_system_prompt(
                profile_id=profile_id or agent.config.prompt_profile_id,
                agent_name=agent_name,
            )

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=system_prompt or "You are a helpful AI assistant."
                )
            ]

            # Retrieve relevant memories and inject into context
            memory_bundle = None
            if use_memory and agent.memory:
                try:
                    memory_bundle = agent.memory.remember(
                        query=message,
                        top_k=agent.config.memory_top_k,
                        time_window_hours=agent.config.memory_time_window_hours,
                    )
                    if memory_bundle:
                        memory_context = memory_bundle.to_context_string()
                        if memory_context:
                            messages.append(Message(
                                role=MessageRole.SYSTEM,
                                content=f"Relevant information from memory:\n{memory_context}"
                            ))
                            logger.debug(f"Injected memory context: {len(memory_bundle.facts)} facts, {len(memory_bundle.entities)} entities")
                except Exception as mem_err:
                    logger.warning(f"Failed to retrieve memories: {mem_err}")

            if context:
                messages.extend(context)
            messages.append(Message(role=MessageRole.USER, content=message))

            # Get MCP tools for function calling
            tools = agent._get_tools_for_provider()
            logger.info(f"Stream chat: {len(tools) if tools else 0} MCP tools available")

            # Resolve prompt profile name for metadata
            prompt_profile = prompt_manager.get_profile(profile_id) if profile_id else None
            prompt_profile_name = prompt_profile.name if prompt_profile else None

            # Get context limits: provider capabilities as primary, config as override
            from .config import get_context_limit_overrides
            caps = provider.get_capabilities(model_id)
            overrides = get_context_limit_overrides(model_id, provider.name)
            context_window = overrides.get("context_window") or caps.context_window
            max_output_tokens = overrides.get("max_output_tokens") or caps.max_output_tokens or 4096

            logger.info(
                f"Using context limits for {provider.name}/{model_id}: "
                f"window={context_window:,}, max_output={max_output_tokens:,}"
            )

            # Send start event with enhanced metadata
            start_data = {
                'task_id': task_id,
                'model': model_id,
                'model_display_name': model_id,
                'profile_name': prompt_profile_name,
                'agent_name': agent_name or 'AgentX',
                'context_window': context_window,
                'max_output_tokens': max_output_tokens,
            }
            yield f"event: start\ndata: {json.dumps(start_data)}\n\n"

            # Emit memory_context event if memories were retrieved
            if memory_bundle and (memory_bundle.facts or memory_bundle.entities or memory_bundle.relevant_turns):
                logger.debug(f"Emitting memory_context: {len(memory_bundle.facts)} facts, {len(memory_bundle.entities)} entities, {len(memory_bundle.relevant_turns)} turns")
                memory_event = {
                    'facts': [
                        {'claim': f.get('claim', '') if isinstance(f, dict) else f.claim,
                         'confidence': f.get('confidence', 0) if isinstance(f, dict) else f.confidence,
                         'source': f.get('source_turn_id') if isinstance(f, dict) else getattr(f, 'source_turn_id', None)}
                        for f in memory_bundle.facts
                    ],
                    'entities': [
                        {'name': e.get('name', '') if isinstance(e, dict) else e.name,
                         'type': e.get('entity_type', e.get('type', 'unknown')) if isinstance(e, dict) else getattr(e, 'entity_type', 'unknown')}
                        for e in memory_bundle.entities
                    ],
                    'relevant_turns': [
                        {
                            'timestamp': t.get('timestamp').isoformat() if hasattr(t.get('timestamp'), 'isoformat') else str(t.get('timestamp', '')),
                            'role': t.get('role', 'unknown'),
                            'content': (t.get('content') or '')[:200],
                        }
                        for t in memory_bundle.relevant_turns[:5]
                    ],
                    'query': message,
                }
                yield f"event: memory_context\ndata: {json.dumps(memory_event)}\n\n"
            else:
                logger.debug(f"No memory context to emit (bundle: {memory_bundle is not None})")

            # Calculate adaptive max_tokens based on context usage
            estimated_input = estimate_tokens(messages)
            available_for_output = context_window - estimated_input - CONTEXT_BUFFER_TOKENS

            # Use the smaller of: configured max_output or available space (but at least minimum)
            adaptive_max_tokens = max(
                min(max_output_tokens, available_for_output),
                MIN_OUTPUT_TOKENS
            )

            logger.debug(
                f"Adaptive max_tokens: {adaptive_max_tokens} "
                f"(context_window={context_window}, estimated_input={estimated_input}, "
                f"max_output={max_output_tokens}, available={available_for_output})"
            )

            # Stream tokens with tool-use loop
            full_content = ""
            tools_used = []
            tool_turns_data = []  # Collect tool call/result data for DB persistence
            max_tool_rounds = DEFAULT_MAX_TOOL_ROUNDS
            total_tokens_input = 0
            total_tokens_output = 0

            # Hard limit for context to prevent corruption (leave room for output)
            max_context_tokens = min(context_window - adaptive_max_tokens - 1000, MAX_INPUT_TOKENS)

            for tool_round in range(max_tool_rounds + 1):
                round_tool_calls = []

                # Check context size and truncate if needed
                estimated_context_tokens = estimate_tokens(messages)
                logger.info(
                    f"Tool round {tool_round + 1}: {len(messages)} messages, "
                    f"~{estimated_context_tokens:,} tokens, limit={max_context_tokens}"
                )

                if estimated_context_tokens > max_context_tokens:
                    logger.warning(f"Context exceeds limit, truncating tool messages")
                    truncate_tool_messages(messages, estimated_context_tokens, max_context_tokens)
                    estimated_context_tokens = estimate_tokens(messages)

                if estimated_context_tokens > context_window * CONTEXT_WARNING_THRESHOLD:
                    logger.warning(
                        f"Context usage high: {estimated_context_tokens:,} / {context_window:,} tokens "
                        f"({100 * estimated_context_tokens / context_window:.1f}%)"
                    )

                async for chunk in provider.stream(
                    messages, model_id,
                    temperature=effective_temperature,
                    max_tokens=adaptive_max_tokens,
                    tools=tools if tool_round < max_tool_rounds else None,
                    tool_choice="auto" if tools and tool_round < max_tool_rounds else None,
                ):
                    # Collect tool calls from the stream
                    if chunk.tool_calls:
                        round_tool_calls.extend(chunk.tool_calls)

                    # Accumulate token usage across rounds
                    if chunk.usage:
                        total_tokens_input += chunk.usage.get("prompt_tokens", 0)
                        total_tokens_output += chunk.usage.get("completion_tokens", 0)

                    if chunk.content:
                        full_content += chunk.content
                        yield f"event: chunk\ndata: {json.dumps({'content': chunk.content})}\n\n"

                # If no tool calls, we're done
                if not round_tool_calls:
                    logger.debug(f"Stream loop complete after {tool_round + 1} round(s), no more tool calls")
                    break

                # Execute tool calls and build follow-up messages
                for tc in round_tool_calls:
                    tools_used.append(tc.name)
                    yield f"event: tool_call\ndata: {json.dumps({'tool': tc.name, 'tool_call_id': tc.id, 'arguments': tc.arguments})}\n\n"

                # Add assistant message with tool_calls to conversation
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in round_tool_calls
                    ],
                ))

                # Execute tools and append results with timing
                tool_start_time = time.perf_counter()
                tool_messages = agent._execute_tool_calls(round_tool_calls)
                tool_total_time = (time.perf_counter() - tool_start_time) * 1000
                tool_avg_time = tool_total_time / len(tool_messages) if tool_messages else 0

                for tm in tool_messages:
                    # Detect success based on content (errors typically contain "error" key)
                    is_error = tm.content.startswith('{"error"') or tm.content.startswith("Error:")
                    tool_result_data = {
                        'tool': tm.name,
                        'tool_call_id': tm.tool_call_id,
                        'content': tm.content[:500],
                        'success': not is_error,
                        'duration_ms': round(tool_avg_time, 2),
                    }
                    yield f"event: tool_result\ndata: {json.dumps(tool_result_data)}\n\n"

                    # Capture for DB persistence
                    tool_turns_data.append({
                        'type': 'tool_call',
                        'tool': tm.name,
                        'tool_call_id': tm.tool_call_id,
                        'arguments': next(
                            (tc.arguments for tc in round_tool_calls if tc.id == tm.tool_call_id),
                            {}
                        ),
                    })
                    tool_turns_data.append({
                        'type': 'tool_result',
                        'tool': tm.name,
                        'tool_call_id': tm.tool_call_id,
                        'content': tm.content[:2000],
                        'success': not is_error,
                        'duration_ms': round(tool_avg_time, 2),
                    })
                messages.extend(tool_messages)

                logger.info(
                    f"Stream tool round {tool_round + 1}: executed "
                    f"{', '.join(tc.name for tc in round_tool_calls)}"
                )

            # Parse output for thinking tags
            logger.debug("Stream complete, parsing output...")
            parsed = parse_output(full_content)

            # Add to session
            session.add_message(Message(role=MessageRole.ASSISTANT, content=parsed.content))

            total_time = (time.time() - start_time) * 1000
            logger.debug(f"Stream total time: {total_time:.0f}ms, sending done event...")

            # Calculate final context usage (use actual if available, otherwise estimate)
            final_context_chars = sum(len(m.content or '') for m in messages)
            if total_tokens_input > 0:
                # Use actual token counts from provider
                final_context_tokens = total_tokens_input + total_tokens_output
                estimated_input = total_tokens_input
                estimated_output = total_tokens_output
            else:
                # Fallback: estimate tokens from character count
                final_context_tokens = final_context_chars // 4
                estimated_input = final_context_tokens
                estimated_output = len(full_content) // 4

            logger.info(
                f"Final context: {final_context_tokens:,} tokens "
                f"(in={estimated_input:,}, out={estimated_output:,}, chars={final_context_chars:,})"
            )

            # Send completion event with enhanced metadata
            done_data = {
                'task_id': task_id,
                'thinking': parsed.thinking,
                'has_thinking': parsed.has_thinking,
                'total_time_ms': total_time,
                'session_id': conv_id,
                'profile_name': prompt_profile_name,
                'agent_name': agent_name or 'AgentX',
                # Token counts (actual from provider, or estimated)
                'tokens_input': estimated_input,
                'tokens_output': estimated_output,
                # Context window info for UI display
                'context_window': context_window,
                'context_used': final_context_tokens,
            }
            done_json = json.dumps(done_data)
            logger.debug(f"Done event JSON length: {len(done_json)} chars")
            yield f"event: done\ndata: {done_json}\n\n"
            logger.debug("Done event sent")

            # Small delay to ensure done event is flushed before close
            import asyncio
            await asyncio.sleep(STREAM_CLOSE_DELAY)

            # Explicit stream close signal for frontend
            yield f"event: close\ndata: {{}}\n\n"
            logger.debug("Close event sent, stream should terminate now")

            # Store turns in memory in a background thread so the response closes immediately
            if use_memory and agent.memory:
                import threading

                user_turn_id = f"{conv_id}-{uuid.uuid4().hex[:8]}-user"
                asst_turn_id = f"{conv_id}-{uuid.uuid4().hex[:8]}-asst"

                def _store_turns():
                    try:
                        from .kit.agent_memory.models import Turn
                        from .kit.agent_memory.connections import get_postgres_session
                        from sqlalchemy import text as sa_text

                        # Query max existing turn_index for this conversation to avoid collisions
                        next_index = 0
                        try:
                            with get_postgres_session() as pg:
                                row = pg.execute(
                                    sa_text("SELECT COALESCE(MAX(turn_index), -1) FROM conversation_logs WHERE conversation_id = :cid"),
                                    {"cid": conv_id},
                                ).scalar()
                                next_index = (row or 0) + 1
                        except Exception:
                            pass  # Fallback to 0 if DB query fails

                        idx = next_index

                        user_turn = Turn(
                            id=user_turn_id,
                            conversation_id=conv_id,
                            role="user",
                            content=message,
                            index=idx,
                        )
                        agent.memory.store_turn(user_turn)
                        idx += 1

                        # Store tool call/result turns
                        for td in tool_turns_data:
                            turn_id = f"{conv_id}-{uuid.uuid4().hex[:8]}-{td['type']}"
                            if td['type'] == 'tool_call':
                                turn = Turn(
                                    id=turn_id,
                                    conversation_id=conv_id,
                                    role="tool_call",
                                    content=json.dumps(td.get('arguments', {})),
                                    index=idx,
                                    metadata={
                                        "tool": td['tool'],
                                        "tool_call_id": td['tool_call_id'],
                                    },
                                )
                            else:  # tool_result
                                turn = Turn(
                                    id=turn_id,
                                    conversation_id=conv_id,
                                    role="tool_result",
                                    content=td.get('content', ''),
                                    index=idx,
                                    metadata={
                                        "tool": td['tool'],
                                        "tool_call_id": td['tool_call_id'],
                                        "success": td.get('success', True),
                                        "duration_ms": td.get('duration_ms'),
                                    },
                                )
                            agent.memory.store_turn(turn)
                            idx += 1

                        # Store assistant turn with thinking in metadata
                        asst_metadata = {"model": model_id, "latency_ms": total_time}
                        if parsed.thinking:
                            asst_metadata["thinking"] = parsed.thinking

                        assistant_turn = Turn(
                            id=asst_turn_id,
                            conversation_id=conv_id,
                            role="assistant",
                            content=parsed.content,
                            index=idx,
                            metadata=asst_metadata,
                        )
                        agent.memory.store_turn(assistant_turn)
                        logger.debug(f"Stored {2 + len(tool_turns_data)} turns in memory for conversation {conv_id}")
                    except Exception as mem_err:
                        logger.warning(f"Failed to store turns in memory: {mem_err}")

                threading.Thread(target=_store_turns, daemon=True).start()
                logger.debug("Background thread started for memory storage")

            logger.debug("Generator returning, stream should close")
            return  # Close the stream immediately

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            return  # Ensure generator terminates after error

    response = StreamingHttpResponse(
        generate_sse(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    return response


def agent_status(request):
    """Get the current agent status."""
    agent = get_agent()
    return JsonResponse(agent.get_status())


# ============== Prompt Management Endpoints ==============

def prompts_profiles(request):
    """List all prompt profiles."""
    from .prompts import get_prompt_manager
    manager = get_prompt_manager()
    
    profiles = manager.list_profiles()
    return JsonResponse({
        "profiles": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "is_default": p.is_default,
                "sections_count": len(p.sections),
                "enabled_sections": len(p.get_enabled_sections()),
            }
            for p in profiles
        ],
    })


def prompts_profile_detail(request, profile_id):
    """Get a specific prompt profile with full details."""
    from .prompts import get_prompt_manager
    manager = get_prompt_manager()
    
    profile = manager.get_profile(profile_id)
    if not profile:
        return JsonResponse({"error": "Profile not found"}, status=404)
    
    return JsonResponse({
        "profile": {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "is_default": profile.is_default,
            "sections": [
                {
                    "id": s.id,
                    "name": s.name,
                    "type": s.type,
                    "content": s.content,
                    "enabled": s.enabled,
                    "order": s.order,
                }
                for s in profile.sections
            ],
        },
        "composed_prompt": profile.compose(),
    })


def prompts_global(request):
    """Get the global prompt."""
    from .prompts import get_prompt_manager
    manager = get_prompt_manager()
    
    global_prompt = manager.get_global_prompt()
    return JsonResponse({
        "global_prompt": {
            "content": global_prompt.content,
            "enabled": global_prompt.enabled,
        },
    })


@csrf_exempt
def prompts_global_update(request):
    """Update the global prompt."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        content = data.get("content")
        enabled = data.get("enabled", True)
        
        if content is None:
            return JsonResponse({'error': 'Missing required field: content'}, status=400)
        
    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)
    
    from .prompts import get_prompt_manager
    manager = get_prompt_manager()
    
    global_prompt = manager.set_global_prompt(content, enabled)
    return JsonResponse({
        "global_prompt": {
            "content": global_prompt.content,
            "enabled": global_prompt.enabled,
        },
    })


def prompts_sections(request):
    """List all available prompt sections."""
    from .prompts import get_prompt_manager
    manager = get_prompt_manager()
    
    sections = manager.list_sections()
    return JsonResponse({
        "sections": [
            {
                "id": s.id,
                "name": s.name,
                "type": s.type,
                "content": s.content,
                "enabled": s.enabled,
                "order": s.order,
            }
            for s in sections
        ],
    })


def prompts_compose(request):
    """Compose a full system prompt for preview."""
    from .prompts import get_prompt_manager
    manager = get_prompt_manager()
    
    profile_id = request.GET.get('profile_id')
    
    system_prompt = manager.get_system_prompt(profile_id=profile_id)
    
    return JsonResponse({
        "system_prompt": system_prompt,
        "profile_id": profile_id,
    })


def prompts_mcp_tools(request):
    """Get the auto-generated MCP tools prompt."""
    from .prompts import generate_mcp_tools_prompt

    # Get tools from MCP manager
    mcp_manager = get_mcp_manager()
    tools = mcp_manager.list_tools()

    tools_data = [tool.to_dict() for tool in tools]
    mcp_prompt = generate_mcp_tools_prompt(tools_data)

    return JsonResponse({
        "mcp_tools_prompt": mcp_prompt,
        "tools_count": len(tools_data),
    })


# ============== Prompt Template Endpoints ==============


@csrf_exempt
def prompts_templates_list(request):
    """
    List or create prompt templates.

    GET: List all templates with optional filtering
        Query params:
        - type: Filter by template type (system, user, snippet)
        - tag: Filter by tag
        - search: Search in name, description, content

    POST: Create a new template
    """
    from .prompts import get_template_manager, PromptTemplate, TemplateType

    if request.method == 'GET':
        manager = get_template_manager()

        # Parse filter params
        type_filter = request.GET.get('type')
        tag_filter = request.GET.get('tag')
        search = request.GET.get('search')

        # Convert type string to enum if provided
        if type_filter:
            try:
                type_filter = TemplateType(type_filter)
            except ValueError:
                return JsonResponse({'error': f'Invalid type: {type_filter}'}, status=400)

        templates = manager.list_templates(
            type_filter=type_filter,
            tag_filter=tag_filter,
            search=search,
        )

        return JsonResponse({
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "content": t.content,
                    "default_content": t.default_content,
                    "tags": t.tags,
                    "placeholders": t.placeholders,
                    "type": t.type if isinstance(t.type, str) else t.type.value,
                    "is_builtin": t.is_builtin,
                    "description": t.description,
                    "has_modifications": t.has_modifications(),
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                }
                for t in templates
            ],
            "total": len(templates),
        })

    elif request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

        # Validate required fields
        name = data.get('name')
        content = data.get('content')

        if not name:
            return JsonResponse({'error': 'Missing required field: name'}, status=400)
        if not content:
            return JsonResponse({'error': 'Missing required field: content'}, status=400)

        # Generate ID from name
        template_id = name.lower().replace(' ', '_').replace('-', '_')

        # Parse optional fields
        tags = data.get('tags', [])
        placeholders = data.get('placeholders', [])
        template_type = data.get('type', 'snippet')
        description = data.get('description')

        try:
            template_type_enum = TemplateType(template_type)
        except ValueError:
            return JsonResponse({'error': f'Invalid type: {template_type}'}, status=400)

        manager = get_template_manager()

        # Check if template already exists
        if manager.get_template(template_id):
            return JsonResponse({'error': f'Template with ID "{template_id}" already exists'}, status=409)

        template = PromptTemplate(
            id=template_id,
            name=name,
            content=content,
            default_content=content,  # New templates start with content as default
            tags=tags,
            placeholders=placeholders,
            type=template_type_enum,
            is_builtin=False,
            description=description,
        )

        created = manager.create_template(template)

        return JsonResponse({
            "template": {
                "id": created.id,
                "name": created.name,
                "content": created.content,
                "default_content": created.default_content,
                "tags": created.tags,
                "placeholders": created.placeholders,
                "type": created.type if isinstance(created.type, str) else created.type.value,
                "is_builtin": created.is_builtin,
                "description": created.description,
                "has_modifications": created.has_modifications(),
                "created_at": created.created_at.isoformat() if created.created_at else None,
                "updated_at": created.updated_at.isoformat() if created.updated_at else None,
            },
            "message": "Template created successfully",
        }, status=201)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def prompts_template_detail(request, template_id):
    """
    Get, update, or delete a specific prompt template.

    GET: Get template details
    PUT: Update template content/metadata
    DELETE: Delete template (fails for builtin)
    """
    from .prompts import get_template_manager

    manager = get_template_manager()
    template = manager.get_template(template_id)

    if not template:
        return JsonResponse({'error': 'Template not found'}, status=404)

    if request.method == 'GET':
        return JsonResponse({
            "template": {
                "id": template.id,
                "name": template.name,
                "content": template.content,
                "default_content": template.default_content,
                "tags": template.tags,
                "placeholders": template.placeholders,
                "type": template.type if isinstance(template.type, str) else template.type.value,
                "is_builtin": template.is_builtin,
                "description": template.description,
                "has_modifications": template.has_modifications(),
                "created_at": template.created_at.isoformat() if template.created_at else None,
                "updated_at": template.updated_at.isoformat() if template.updated_at else None,
            },
        })

    elif request.method == 'PUT':
        try:
            data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

        # Update allowed fields
        updates = {}
        if 'name' in data:
            updates['name'] = data['name']
        if 'content' in data:
            updates['content'] = data['content']
        if 'tags' in data:
            updates['tags'] = data['tags']
        if 'placeholders' in data:
            updates['placeholders'] = data['placeholders']
        if 'description' in data:
            updates['description'] = data['description']

        updated = manager.update_template(template_id, updates)

        return JsonResponse({
            "template": {
                "id": updated.id,
                "name": updated.name,
                "content": updated.content,
                "default_content": updated.default_content,
                "tags": updated.tags,
                "placeholders": updated.placeholders,
                "type": updated.type if isinstance(updated.type, str) else updated.type.value,
                "is_builtin": updated.is_builtin,
                "description": updated.description,
                "has_modifications": updated.has_modifications(),
                "created_at": updated.created_at.isoformat() if updated.created_at else None,
                "updated_at": updated.updated_at.isoformat() if updated.updated_at else None,
            },
            "message": "Template updated successfully",
        })

    elif request.method == 'DELETE':
        try:
            deleted = manager.delete_template(template_id)
            if deleted:
                return JsonResponse({"deleted": True, "message": "Template deleted successfully"})
            return JsonResponse({'error': 'Template not found'}, status=404)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def prompts_template_reset(request, template_id):
    """Reset a template's content to its default_content."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    from .prompts import get_template_manager

    manager = get_template_manager()
    template = manager.reset_to_default(template_id)

    if not template:
        return JsonResponse({'error': 'Template not found'}, status=404)

    return JsonResponse({
        "template": {
            "id": template.id,
            "name": template.name,
            "content": template.content,
            "default_content": template.default_content,
            "tags": template.tags,
            "placeholders": template.placeholders,
            "type": template.type if isinstance(template.type, str) else template.type.value,
            "is_builtin": template.is_builtin,
            "description": template.description,
            "has_modifications": template.has_modifications(),
            "created_at": template.created_at.isoformat() if template.created_at else None,
            "updated_at": template.updated_at.isoformat() if template.updated_at else None,
        },
        "message": "Template reset to default",
    })


def prompts_templates_tags(request):
    """Get all unique tags with counts."""
    from .prompts import get_template_manager

    manager = get_template_manager()
    tag_counts = manager.get_tag_counts()

    return JsonResponse({
        "tags": [
            {"name": tag, "count": count}
            for tag, count in tag_counts.items()
        ],
        "total": len(tag_counts),
    })


# ============== Memory Channel Endpoints ==============

@csrf_exempt
def memory_channels(request):
    """
    List or create memory channels.

    GET: List all channels with item counts
    POST: Create a new named channel
    """
    from .kit.agent_memory.connections import Neo4jConnection

    if request.method == 'GET':
        # List all channels with item counts
        try:
            channels = []

            with Neo4jConnection.session() as session:
                # Get all distinct channels and their counts
                result = session.run("""
                    // Get all channels from different node types
                    MATCH (n)
                    WHERE n.channel IS NOT NULL
                    WITH DISTINCT n.channel AS channel

                    // Count items per channel per type
                    OPTIONAL MATCH (t:Turn {channel: channel})
                    WITH channel, count(DISTINCT t) AS turn_count

                    OPTIONAL MATCH (e:Entity {channel: channel})
                    WITH channel, turn_count, count(DISTINCT e) AS entity_count

                    OPTIONAL MATCH (f:Fact {channel: channel})
                    WITH channel, turn_count, entity_count, count(DISTINCT f) AS fact_count

                    OPTIONAL MATCH (s:Strategy {channel: channel})
                    WITH channel, turn_count, entity_count, fact_count, count(DISTINCT s) AS strategy_count

                    OPTIONAL MATCH (g:Goal {channel: channel})

                    RETURN channel,
                           turn_count AS turns,
                           entity_count AS entities,
                           fact_count AS facts,
                           strategy_count AS strategies,
                           count(DISTINCT g) AS goals
                    ORDER BY channel
                """)

                for record in result:
                    channels.append({
                        "name": record["channel"],
                        "is_default": record["channel"] == "_global",
                        "item_counts": {
                            "turns": record["turns"],
                            "entities": record["entities"],
                            "facts": record["facts"],
                            "strategies": record["strategies"],
                            "goals": record["goals"],
                        }
                    })

            # Ensure _global channel always exists in the list
            if not any(c["name"] == "_global" for c in channels):
                channels.insert(0, {
                    "name": "_global",
                    "is_default": True,
                    "item_counts": {
                        "turns": 0,
                        "entities": 0,
                        "facts": 0,
                        "strategies": 0,
                        "goals": 0,
                    }
                })

            return JsonResponse({"channels": channels})

        except Exception as e:
            logger.error(f"Error listing memory channels: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    elif request.method == 'POST':
        # Create a new channel
        try:
            data = json.loads(request.body.decode('utf-8'))
            channel_name = data.get("name")

            if not channel_name:
                return JsonResponse({"error": "Channel name is required"}, status=400)

            # Validate channel name (alphanumeric, hyphens, underscores)
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', channel_name):
                return JsonResponse({
                    "error": "Channel name must contain only alphanumeric characters, hyphens, and underscores"
                }, status=400)

            # Check if channel already exists
            with Neo4jConnection.session() as session:
                result = session.run("""
                    MATCH (n {channel: $channel})
                    RETURN count(n) AS count
                """, channel=channel_name)
                record = result.single()
                if record and record["count"] > 0:
                    return JsonResponse({
                        "error": f"Channel '{channel_name}' already exists"
                    }, status=409)

            # Channel is created implicitly when first item is added
            # For now, just confirm it's valid and doesn't exist
            from datetime import datetime, timezone
            return JsonResponse({
                "name": channel_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": "Channel created successfully"
            }, status=201)

        except json.JSONDecodeError as e:
            return JsonResponse({"error": f"Invalid JSON: {str(e)}"}, status=400)
        except Exception as e:
            logger.error(f"Error creating memory channel: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    elif request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
@require_methods("GET")
def conversations_list(request):
    """
    GET /api/conversations - List past conversations from the database.

    Query params:
        limit  - Max results (default 50, max 100)
        offset - Skip N results (default 0)
        channel - Filter by memory channel (default: all)
    """
    from .kit.agent_memory.connections import PostgresConnection

    try:
        limit = min(100, max(1, int(request.GET.get("limit", "50"))))
    except (ValueError, TypeError):
        limit = 50
    try:
        offset = max(0, int(request.GET.get("offset", "0")))
    except (ValueError, TypeError):
        offset = 0
    channel = request.GET.get("channel")

    try:
        pg_conn = PostgresConnection.get_engine().raw_connection()
        try:
            with pg_conn.cursor() as cursor:
                # Build query: group by conversation_id, get summary fields
                channel_filter = ""
                params: list = []
                if channel:
                    channel_filter = "WHERE cl.channel = %s"
                    params.append(channel)

                cursor.execute(f"""
                    SELECT
                        cl.conversation_id::text,
                        MIN(cl.timestamp) AS created_at,
                        MAX(cl.timestamp) AS last_message_at,
                        COUNT(*) AS message_count,
                        MAX(cl.channel) AS channel,
                        (SELECT content FROM conversation_logs sub
                         WHERE sub.conversation_id = cl.conversation_id
                           AND sub.role = 'user'
                         ORDER BY sub.turn_index ASC LIMIT 1) AS first_user_message,
                        (SELECT content FROM conversation_logs sub
                         WHERE sub.conversation_id = cl.conversation_id
                         ORDER BY sub.turn_index DESC LIMIT 1) AS last_message
                    FROM conversation_logs cl
                    {channel_filter}
                    GROUP BY cl.conversation_id
                    ORDER BY MAX(cl.timestamp) DESC
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])

                rows = cursor.fetchall()

                # Get total count
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT conversation_id)
                    FROM conversation_logs
                    {"WHERE channel = %s" if channel else ""}
                """, [channel] if channel else [])
                total = cursor.fetchone()[0]

                conversations = []
                for row in rows:
                    conv_id, created_at, last_message_at, count, ch, first_msg, last_msg = row
                    # Title from first user message (truncated)
                    title = "New Conversation"
                    if first_msg:
                        title = first_msg[:80].strip()
                        if len(first_msg) > 80:
                            title += "…"
                    # Preview from last message
                    preview = ""
                    if last_msg:
                        preview = last_msg[:120].strip()
                        if len(last_msg) > 120:
                            preview += "…"

                    conversations.append({
                        "conversation_id": conv_id,
                        "title": title,
                        "preview": preview,
                        "message_count": count,
                        "channel": ch or "_global",
                        "created_at": created_at.isoformat() if created_at else None,
                        "last_message_at": last_message_at.isoformat() if last_message_at else None,
                    })

                return JsonResponse({
                    "conversations": conversations,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                })
        finally:
            pg_conn.close()

    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return json_error(str(e), status=500)


@csrf_exempt
@require_methods("GET")
def conversations_messages(request, conversation_id):
    """
    GET /api/conversations/<id>/messages - Fetch all messages for a conversation.

    Returns messages ordered by turn_index.
    """
    from .kit.agent_memory.connections import PostgresConnection

    try:
        pg_conn = PostgresConnection.get_engine().raw_connection()
        try:
            with pg_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT role, content, timestamp, turn_index, metadata, model
                    FROM conversation_logs
                    WHERE conversation_id = %s
                    ORDER BY turn_index ASC
                """, (conversation_id,))

                rows = cursor.fetchall()

                if not rows:
                    return json_error("Conversation not found", status=404)

                messages = []
                for row in rows:
                    role, content, timestamp, turn_index, metadata, model = row
                    msg = {
                        "role": role,
                        "content": content,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                        "turn_index": turn_index,
                    }
                    if metadata:
                        msg["metadata"] = metadata if isinstance(metadata, dict) else {}
                    if model:
                        msg["metadata"] = msg.get("metadata", {})
                        msg["metadata"]["model"] = model
                    messages.append(msg)

                return JsonResponse({
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "message_count": len(messages),
                })
        finally:
            pg_conn.close()

    except Exception as e:
        logger.error(f"Error fetching conversation messages: {e}")
        return json_error(str(e), status=500)


@csrf_exempt
def memory_conversation_delete(request, conversation_id):
    """
    Delete a single conversation and its associated data.

    DELETE: Remove the conversation's turns from Neo4j, PostgreSQL, and Redis.
    """
    from .kit.agent_memory.connections import Neo4jConnection, PostgresConnection, RedisConnection

    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'DELETE':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        deleted_counts = {
            "turns": 0,
            "conversation": 0,
            "postgres_rows": 0,
            "redis_keys": 0,
        }

        # Delete from Neo4j: conversation node and its turns
        try:
            with Neo4jConnection.session() as session:
                # Delete turns linked to this conversation
                result = session.run("""
                    MATCH (c:Conversation {id: $conv_id})-[:HAS_TURN]->(t:Turn)
                    WITH t, count(t) AS cnt
                    DETACH DELETE t
                    RETURN cnt
                """, conv_id=conversation_id)
                record = result.single()
                if record:
                    deleted_counts["turns"] = record["cnt"] or 0

                # Delete the conversation node itself
                result = session.run("""
                    MATCH (c:Conversation {id: $conv_id})
                    DETACH DELETE c
                    RETURN count(c) AS cnt
                """, conv_id=conversation_id)
                record = result.single()
                if record:
                    deleted_counts["conversation"] = record["cnt"] or 0
        except Exception as e:
            logger.warning(f"Error deleting conversation from Neo4j: {e}")

        # Delete from PostgreSQL
        try:
            pg_conn = PostgresConnection.get_engine().raw_connection()
            try:
                with pg_conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM conversation_logs WHERE conversation_id = %s",
                        (conversation_id,)
                    )
                    deleted_counts["postgres_rows"] += cursor.rowcount
                    pg_conn.commit()
            finally:
                pg_conn.close()
        except Exception as e:
            logger.warning(f"Error deleting conversation from PostgreSQL: {e}")

        # Delete from Redis: working memory for this conversation
        try:
            redis_client = RedisConnection.get_client()
            pattern = f"working:*:*:{conversation_id}"
            keys = redis_client.keys(pattern)
            if keys:
                deleted_counts["redis_keys"] = len(keys)
                redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Error deleting conversation from Redis: {e}")

        return JsonResponse({
            "message": f"Conversation '{conversation_id}' deleted successfully",
            "deleted": deleted_counts
        }, status=200)

    except Exception as e:
        logger.error(f"Error deleting conversation '{conversation_id}': {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_channel_delete(request, name):
    """
    Delete a memory channel and all its data.

    DELETE: Remove the channel and all associated data
    """
    from .kit.agent_memory.connections import Neo4jConnection, PostgresConnection, RedisConnection

    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'DELETE':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    # Prevent deleting the _global channel
    if name == "_global":
        return JsonResponse({
            "error": "Cannot delete the _global channel"
        }, status=400)

    try:
        deleted_counts = {
            "turns": 0,
            "entities": 0,
            "facts": 0,
            "strategies": 0,
            "goals": 0,
            "conversations": 0,
            "postgres_rows": 0,
            "redis_keys": 0,
        }

        # Delete from Neo4j
        with Neo4jConnection.session() as session:
            # Delete turns
            result = session.run("""
                MATCH (t:Turn {channel: $channel})
                WITH t, count(t) AS cnt
                DETACH DELETE t
                RETURN cnt
            """, channel=name)
            record = result.single()
            if record:
                deleted_counts["turns"] = record["cnt"] or 0

            # Delete entities
            result = session.run("""
                MATCH (e:Entity {channel: $channel})
                WITH e, count(e) AS cnt
                DETACH DELETE e
                RETURN cnt
            """, channel=name)
            record = result.single()
            if record:
                deleted_counts["entities"] = record["cnt"] or 0

            # Delete facts
            result = session.run("""
                MATCH (f:Fact {channel: $channel})
                WITH f, count(f) AS cnt
                DETACH DELETE f
                RETURN cnt
            """, channel=name)
            record = result.single()
            if record:
                deleted_counts["facts"] = record["cnt"] or 0

            # Delete strategies
            result = session.run("""
                MATCH (s:Strategy {channel: $channel})
                WITH s, count(s) AS cnt
                DETACH DELETE s
                RETURN cnt
            """, channel=name)
            record = result.single()
            if record:
                deleted_counts["strategies"] = record["cnt"] or 0

            # Delete goals
            result = session.run("""
                MATCH (g:Goal {channel: $channel})
                WITH g, count(g) AS cnt
                DETACH DELETE g
                RETURN cnt
            """, channel=name)
            record = result.single()
            if record:
                deleted_counts["goals"] = record["cnt"] or 0

            # Delete conversations
            result = session.run("""
                MATCH (c:Conversation {channel: $channel})
                WITH c, count(c) AS cnt
                DETACH DELETE c
                RETURN cnt
            """, channel=name)
            record = result.single()
            if record:
                deleted_counts["conversations"] = record["cnt"] or 0

        # Delete from PostgreSQL
        try:
            pg_conn = PostgresConnection.get_engine().raw_connection()
            try:
                with pg_conn.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM conversation_logs WHERE channel = %s",
                        (name,)
                    )
                    deleted_counts["postgres_rows"] += cursor.rowcount

                    cursor.execute(
                        "DELETE FROM tool_invocations WHERE channel = %s",
                        (name,)
                    )
                    deleted_counts["postgres_rows"] += cursor.rowcount

                    cursor.execute(
                        "DELETE FROM memory_timeline WHERE channel = %s",
                        (name,)
                    )
                    deleted_counts["postgres_rows"] += cursor.rowcount

                    pg_conn.commit()
            finally:
                pg_conn.close()
        except Exception as e:
            logger.warning(f"Error deleting from PostgreSQL: {e}")

        # Delete from Redis
        try:
            redis_client = RedisConnection.get_client()
            pattern = f"working:*:{name}:*"
            keys = redis_client.keys(pattern)
            if keys:
                deleted_counts["redis_keys"] = len(keys)
                redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Error deleting from Redis: {e}")

        return JsonResponse({
            "message": f"Channel '{name}' deleted successfully",
            "deleted": deleted_counts
        }, status=200)

    except Exception as e:
        logger.error(f"Error deleting memory channel '{name}': {e}")
        return JsonResponse({"error": str(e)}, status=500)


# ============== Memory Explorer Endpoints ==============

DEFAULT_USER_ID = "default"  # TODO: Replace with actual auth when multi-user is implemented


@csrf_exempt
def memory_entities(request):
    """
    GET /api/memory/entities - List entities with pagination and filtering.

    Query params:
        - channel: Filter by channel (default: "_global")
        - page: Page number, 1-indexed (default: 1)
        - limit: Items per page, max 100 (default: 20)
        - search: Text search on entity name
        - type: Filter by entity type
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        channel = request.GET.get('channel', '_all')
        page = max(1, int(request.GET.get('page', 1)))
        limit = min(100, max(1, int(request.GET.get('limit', 20))))
        search = request.GET.get('search', None)
        entity_type = request.GET.get('type', None)
        offset = (page - 1) * limit

        # Get semantic memory instance
        from .kit.agent_memory.memory.semantic import SemanticMemory
        semantic = SemanticMemory()

        entities, total = semantic.list_entities(
            user_id=DEFAULT_USER_ID,
            channel=channel,
            offset=offset,
            limit=limit,
            search=search,
            entity_type=entity_type
        )

        return JsonResponse({
            "entities": entities,
            "total": total,
            "page": page,
            "limit": limit,
            "has_next": (page * limit) < total
        })

    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_entity_graph(request, entity_id):
    """
    GET /api/memory/entities/{id}/graph - Get entity subgraph.

    Returns the entity with its connected facts and relationships.

    Query params:
        - depth: Traversal depth, max 3 (default: 2)
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        depth = min(3, max(1, int(request.GET.get('depth', 2))))

        from .kit.agent_memory.memory.semantic import SemanticMemory
        semantic = SemanticMemory()

        result = semantic.get_entity_facts_and_relationships(
            entity_id=entity_id,
            user_id=DEFAULT_USER_ID,
            depth=depth
        )

        if not result.get("entity"):
            return JsonResponse({"error": "Entity not found"}, status=404)

        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error getting entity graph: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_facts(request):
    """
    GET /api/memory/facts - List facts with pagination and filtering.

    Query params:
        - channel: Filter by channel (default: "_global")
        - page: Page number, 1-indexed (default: 1)
        - limit: Items per page, max 100 (default: 20)
        - min_confidence: Minimum confidence threshold 0.0-1.0 (default: 0.0)
        - search: Text search on fact claim
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        channel = request.GET.get('channel', '_all')
        page = max(1, int(request.GET.get('page', 1)))
        limit = min(100, max(1, int(request.GET.get('limit', 20))))
        min_confidence = min(1.0, max(0.0, float(request.GET.get('min_confidence', 0.0))))
        search = request.GET.get('search', None)
        offset = (page - 1) * limit

        from .kit.agent_memory.memory.semantic import SemanticMemory
        semantic = SemanticMemory()

        facts, total = semantic.list_facts(
            user_id=DEFAULT_USER_ID,
            channel=channel,
            offset=offset,
            limit=limit,
            min_confidence=min_confidence,
            search=search
        )

        return JsonResponse({
            "facts": facts,
            "total": total,
            "page": page,
            "limit": limit,
            "has_next": (page * limit) < total
        })

    except Exception as e:
        logger.error(f"Error listing facts: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_strategies(request):
    """
    GET /api/memory/strategies - List strategies with pagination.

    Query params:
        - channel: Filter by channel (default: "_global")
        - page: Page number, 1-indexed (default: 1)
        - limit: Items per page, max 100 (default: 20)
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        channel = request.GET.get('channel', '_all')
        page = max(1, int(request.GET.get('page', 1)))
        limit = min(100, max(1, int(request.GET.get('limit', 20))))
        offset = (page - 1) * limit

        from .kit.agent_memory.memory.procedural import ProceduralMemory
        procedural = ProceduralMemory()

        strategies, total = procedural.list_strategies(
            user_id=DEFAULT_USER_ID,
            channel=channel,
            offset=offset,
            limit=limit
        )

        return JsonResponse({
            "strategies": strategies,
            "total": total,
            "page": page,
            "limit": limit,
            "has_next": (page * limit) < total
        })

    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_stats(request):
    """
    GET /api/memory/stats - Get memory statistics.

    Returns total counts and per-channel breakdowns for entities, facts,
    strategies, and turns.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        from .kit.agent_memory.connections import Neo4jConnection

        with Neo4jConnection.session() as session:
            # Get totals - query each type separately since they have different structures
            # Entity, Fact, Strategy have user_id property
            # Turn is linked via User -> Conversation -> Turn
            totals = {"entities": 0, "facts": 0, "strategies": 0, "turns": 0}

            entity_count = session.run("""
                MATCH (e:Entity {user_id: $user_id})
                RETURN count(e) AS cnt
            """, user_id=DEFAULT_USER_ID).single()
            totals["entities"] = entity_count["cnt"] if entity_count else 0

            fact_count = session.run("""
                MATCH (f:Fact {user_id: $user_id})
                RETURN count(f) AS cnt
            """, user_id=DEFAULT_USER_ID).single()
            totals["facts"] = fact_count["cnt"] if fact_count else 0

            strategy_count = session.run("""
                MATCH (s:Strategy {user_id: $user_id})
                RETURN count(s) AS cnt
            """, user_id=DEFAULT_USER_ID).single()
            totals["strategies"] = strategy_count["cnt"] if strategy_count else 0

            # Turn is linked through Conversation, not directly to user_id property
            turn_count = session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)-[:HAS_TURN]->(t:Turn)
                RETURN count(t) AS cnt
            """, user_id=DEFAULT_USER_ID).single()
            totals["turns"] = turn_count["cnt"] if turn_count else 0

            # Get per-channel breakdown
            by_channel = {}

            # Query each type separately for accurate per-channel counts
            entity_result = session.run("""
                MATCH (e:Entity {user_id: $user_id})
                RETURN e.channel AS channel, count(e) AS cnt
            """, user_id=DEFAULT_USER_ID)
            for record in entity_result:
                ch = record["channel"] or "_global"
                if ch not in by_channel:
                    by_channel[ch] = {"entities": 0, "facts": 0, "strategies": 0, "turns": 0}
                by_channel[ch]["entities"] = record["cnt"]

            fact_result = session.run("""
                MATCH (f:Fact {user_id: $user_id})
                RETURN f.channel AS channel, count(f) AS cnt
            """, user_id=DEFAULT_USER_ID)
            for record in fact_result:
                ch = record["channel"] or "_global"
                if ch not in by_channel:
                    by_channel[ch] = {"entities": 0, "facts": 0, "strategies": 0, "turns": 0}
                by_channel[ch]["facts"] = record["cnt"]

            strategy_result = session.run("""
                MATCH (s:Strategy {user_id: $user_id})
                RETURN s.channel AS channel, count(s) AS cnt
            """, user_id=DEFAULT_USER_ID)
            for record in strategy_result:
                ch = record["channel"] or "_global"
                if ch not in by_channel:
                    by_channel[ch] = {"entities": 0, "facts": 0, "strategies": 0, "turns": 0}
                by_channel[ch]["strategies"] = record["cnt"]

            # Turn linked through Conversation
            turn_result = session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)-[:HAS_TURN]->(t:Turn)
                RETURN t.channel AS channel, count(t) AS cnt
            """, user_id=DEFAULT_USER_ID)
            for record in turn_result:
                ch = record["channel"] or "_global"
                if ch not in by_channel:
                    by_channel[ch] = {"entities": 0, "facts": 0, "strategies": 0, "turns": 0}
                by_channel[ch]["turns"] = record["cnt"]

        return JsonResponse({
            "totals": totals,
            "by_channel": by_channel
        })

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_settings(request):
    """
    GET /api/memory/settings - Get consolidation settings.
    POST /api/memory/settings - Update consolidation settings.

    Returns/accepts settings for extraction, relevance filter, entity linking,
    quality thresholds, and job scheduling.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method == 'GET':
        try:
            from .kit.agent_memory.config import get_consolidation_settings

            settings = get_consolidation_settings()

            # Also include the default prompts for display
            # (so UI can show what the defaults are when custom is empty)
            from .kit.agent_memory.extraction.service import ExtractionService
            service = ExtractionService()
            settings["default_extraction_prompt"] = service._get_default_system_prompt()
            settings["default_relevance_prompt"] = service._get_default_relevance_prompt()

            return JsonResponse(settings)
        except Exception as e:
            logger.error(f"Error getting memory settings: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    elif request.method == 'POST':
        try:
            from .kit.agent_memory.config import save_memory_settings, get_consolidation_settings

            data = json.loads(request.body.decode('utf-8'))

            # Validate and filter to only allowed settings
            allowed_keys = set(get_consolidation_settings().keys())
            # Remove read-only keys
            allowed_keys -= {"entity_types", "relationship_types",
                            "default_extraction_prompt", "default_relevance_prompt"}

            filtered = {k: v for k, v in data.items() if k in allowed_keys}

            if not filtered:
                return JsonResponse({"error": "No valid settings provided"}, status=400)

            save_memory_settings(filtered)

            return JsonResponse({
                "success": True,
                "updated": list(filtered.keys())
            })
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error updating memory settings: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


@csrf_exempt
def recall_settings(request):
    """
    GET /api/memory/recall-settings - Get RecallLayer settings.
    POST /api/memory/recall-settings - Update RecallLayer settings.

    Returns/accepts settings for enhanced retrieval techniques:
    hybrid search, entity-centric, query expansion, HyDE, self-query.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method == 'GET':
        try:
            from .kit.agent_memory.config import get_recall_settings

            settings = get_recall_settings()
            return JsonResponse(settings)
        except Exception as e:
            logger.error(f"Error getting recall settings: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    elif request.method == 'POST':
        try:
            from .kit.agent_memory.config import save_memory_settings, get_recall_settings

            data = json.loads(request.body.decode('utf-8'))

            # Validate and filter to only allowed settings
            allowed_keys = set(get_recall_settings().keys())

            filtered = {k: v for k, v in data.items() if k in allowed_keys}

            if not filtered:
                return JsonResponse({"error": "No valid settings provided"}, status=400)

            save_memory_settings(filtered)

            return JsonResponse({
                "success": True,
                "updated": list(filtered.keys())
            })
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error updating recall settings: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)


# =============================================================================
# Job Monitoring Endpoints
# =============================================================================

@csrf_exempt
def jobs_list(request):
    """
    GET /api/jobs - List all consolidation jobs with status.

    Returns list of jobs with their current status, metrics, and configuration.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        from .kit.agent_memory.consolidation import JobRegistry
        from dataclasses import asdict

        registry = JobRegistry.get_instance()
        jobs = registry.list_jobs()
        worker = registry.get_worker_status()

        return JsonResponse({
            "jobs": [asdict(job) for job in jobs],
            "worker": worker
        })

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def job_detail(request, job_name):
    """
    GET /api/jobs/{name} - Get job details with history.

    Returns job status and recent execution history.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        from .kit.agent_memory.consolidation import JobRegistry
        from dataclasses import asdict

        registry = JobRegistry.get_instance()
        job = registry.get_job(job_name)

        if not job:
            return JsonResponse({"error": f"Unknown job: {job_name}"}, status=404)

        history = registry.get_job_history(job_name, limit=10)

        return JsonResponse({
            "job": asdict(job),
            "history": history
        })

    except Exception as e:
        logger.error(f"Error getting job detail: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def job_run(request, job_name):
    """
    POST /api/jobs/{name}/run - Manually trigger a job.

    Returns execution result with metrics.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        from .kit.agent_memory.consolidation import JobRegistry

        registry = JobRegistry.get_instance()
        result = registry.run_job(job_name)

        status_code = 200 if result.get("success") else 400
        return JsonResponse(result, status=status_code)

    except Exception as e:
        logger.error(f"Error running job: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def job_toggle(request, job_name):
    """
    POST /api/jobs/{name}/toggle - Enable or disable a job.

    Request body: {"enabled": true/false}
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        from .kit.agent_memory.consolidation import JobRegistry

        data = json.loads(request.body) if request.body else {}
        enabled = data.get("enabled", True)

        registry = JobRegistry.get_instance()

        if enabled:
            success = registry.enable_job(job_name)
        else:
            success = registry.disable_job(job_name)

        if not success:
            return JsonResponse({"error": f"Unknown job: {job_name}"}, status=404)

        return JsonResponse({"enabled": enabled, "job": job_name})

    except Exception as e:
        logger.error(f"Error toggling job: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
async def memory_consolidate(request):
    """
    POST /api/memory/consolidate - Run consolidation pipeline manually.

    Request body (optional):
        {"jobs": ["consolidate", "patterns", "promote"]}

    If no jobs specified, runs: consolidate, patterns, promote.

    Returns combined results from all jobs.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        from .kit.agent_memory.consolidation import JobRegistry

        data = json.loads(request.body) if request.body else {}
        jobs = data.get("jobs")  # None means default set

        registry = JobRegistry.get_instance()
        result = await registry.run_consolidation_pipeline(jobs)

        status_code = 200 if result.get("success") else 400
        if not result.get("success"):
            logger.warning(f"Consolidation failed: {result}")
        return JsonResponse(result, status=status_code)

    except Exception as e:
        logger.error(f"Error running consolidation: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def memory_reset(request):
    """
    POST /api/memory/reset - Reset consolidation for all conversations.

    This clears the consolidated timestamp from all conversations,
    allowing them to be reprocessed by the consolidation job.

    Request body (optional):
        {"delete_memories": true}  - Also delete all entities, facts, strategies

    Useful when extraction logic has changed or to rebuild semantic memory.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        from .kit.agent_memory.consolidation.jobs import reset_consolidation

        data = json.loads(request.body) if request.body else {}
        delete_memories = data.get('delete_memories', False)

        result = reset_consolidation(delete_memories=delete_memories)
        return JsonResponse(result)

    except Exception as e:
        logger.error(f"Error resetting consolidation: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def jobs_clear_stuck(request):
    """
    POST /api/jobs/clear-stuck - Clear any jobs stuck in 'running' state.

    Useful when a job crashed and left its status as 'running',
    blocking future runs.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        from .kit.agent_memory.consolidation import JobRegistry

        registry = JobRegistry.get_instance()
        cleared = registry.clear_stuck_jobs()

        return JsonResponse({
            "success": True,
            "cleared_jobs": cleared,
            "message": f"Cleared {len(cleared)} stuck job(s)" if cleared else "No stuck jobs found"
        })

    except Exception as e:
        logger.error(f"Error clearing stuck jobs: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# ============== Config Management Endpoint ==============

@csrf_exempt
def config_update(request):
    """
    POST /api/config/update - Update runtime configuration.

    Accepts partial updates and persists to data/config.json.
    Hot-reloads providers after update.

    Security: POST-only, no GET endpoint to prevent config exposure.
    """
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    from .config import get_config_manager
    config = get_config_manager()

    updated_keys = []

    # Update providers
    providers = data.get("providers", {})
    for provider, settings in providers.items():
        if provider not in ("lmstudio", "anthropic", "openai"):
            continue  # Skip unknown providers
        for key, value in settings.items():
            if value is not None:
                config.set(f"providers.{provider}.{key}", value)
                updated_keys.append(f"providers.{provider}.{key}")

    # Update preferences
    preferences = data.get("preferences", {})
    for key, value in preferences.items():
        if value is not None:
            config.set(f"preferences.{key}", value)
            updated_keys.append(f"preferences.{key}")

    # Update LLM settings
    llm_settings = data.get("llm_settings", {})
    for key, value in llm_settings.items():
        if value is not None:
            config.set(f"llm_settings.{key}", value)
            updated_keys.append(f"llm_settings.{key}")

    # Update context limits
    context_limits = data.get("context_limits", {})
    for provider_or_model, settings in context_limits.items():
        if isinstance(settings, dict):
            for key, value in settings.items():
                if value is not None:
                    config.set(f"context_limits.{provider_or_model}.{key}", value)
                    updated_keys.append(f"context_limits.{provider_or_model}.{key}")

    # Persist to disk
    if not config.save():
        return JsonResponse({
            'error': 'Failed to save config to disk'
        }, status=500)

    # Hot-reload providers
    try:
        registry = get_registry()
        registry.reload()
        logger.info(f"Config updated and providers reloaded: {updated_keys}")
    except Exception as e:
        logger.error(f"Failed to reload providers: {e}")
        return JsonResponse({
            'status': 'partial',
            'message': f'Config saved but provider reload failed: {e}',
            'updated': updated_keys,
        })

    return JsonResponse({
        'status': 'ok',
        'message': 'Config updated and applied',
        'updated': updated_keys,
    })


@csrf_exempt
def context_limits(request):
    """
    GET /api/config/context-limits - Get context limit settings.
    POST /api/config/context-limits - Update context limit settings.

    Returns/accepts:
    {
        "lmstudio": {"context_window": 32768, "max_output_tokens": 8192},
        "anthropic": {"context_window": 200000, "max_output_tokens": 8192},
        "openai": {"context_window": 128000, "max_output_tokens": 4096},
        "models": {
            "model-id": {"context_window": 1000000, "max_output_tokens": 32000}
        }
    }
    """
    from .config import get_config_manager, DEFAULT_CONFIG

    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    config = get_config_manager()

    if request.method == 'GET':
        # Return current context limits (merged with defaults)
        limits = config.get("context_limits") or {}

        # Ensure all providers have defaults
        defaults = DEFAULT_CONFIG.get("context_limits", {})
        result = {
            "lmstudio": {
                "context_window": limits.get("lmstudio", {}).get("context_window")
                    or defaults.get("lmstudio", {}).get("context_window", 32768),
                "max_output_tokens": limits.get("lmstudio", {}).get("max_output_tokens")
                    or defaults.get("lmstudio", {}).get("max_output_tokens", 8192),
            },
            "anthropic": {
                "context_window": limits.get("anthropic", {}).get("context_window")
                    or defaults.get("anthropic", {}).get("context_window", 200000),
                "max_output_tokens": limits.get("anthropic", {}).get("max_output_tokens")
                    or defaults.get("anthropic", {}).get("max_output_tokens", 8192),
            },
            "openai": {
                "context_window": limits.get("openai", {}).get("context_window")
                    or defaults.get("openai", {}).get("context_window", 128000),
                "max_output_tokens": limits.get("openai", {}).get("max_output_tokens")
                    or defaults.get("openai", {}).get("max_output_tokens", 4096),
            },
            "models": limits.get("models", {}),
        }
        return JsonResponse(result)

    elif request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

        updated = []

        # Update provider limits
        for provider in ("lmstudio", "anthropic", "openai"):
            if provider in data and isinstance(data[provider], dict):
                for key in ("context_window", "max_output_tokens"):
                    if key in data[provider]:
                        config.set(f"context_limits.{provider}.{key}", data[provider][key])
                        updated.append(f"context_limits.{provider}.{key}")

        # Update model-specific limits
        if "models" in data and isinstance(data["models"], dict):
            for model_id, settings in data["models"].items():
                if isinstance(settings, dict):
                    for key in ("context_window", "max_output_tokens"):
                        if key in settings:
                            config.set(f"context_limits.models.{model_id}.{key}", settings[key])
                            updated.append(f"context_limits.models.{model_id}.{key}")

        if not config.save():
            return JsonResponse({'error': 'Failed to save config'}, status=500)

        logger.info(f"Context limits updated: {updated}")
        return JsonResponse({
            'status': 'ok',
            'updated': updated,
        })

    return JsonResponse({'error': 'Method not allowed'}, status=405)


# ============== Agent Profile Endpoints ==============


@csrf_exempt
def agent_profiles_list(request):
    """
    GET /api/agent/profiles - List all agent profiles.
    POST /api/agent/profiles - Create a new agent profile.
    """
    from .agent.profiles import get_profile_manager

    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    manager = get_profile_manager()

    if request.method == 'GET':
        profiles = manager.list_profiles()
        return JsonResponse({
            "profiles": [
                {
                    "id": p.id,
                    "name": p.name,
                    "avatar": p.avatar,
                    "description": p.description,
                    "default_model": p.default_model,
                    "temperature": p.temperature,
                    "prompt_profile_id": p.prompt_profile_id,
                    "system_prompt": p.system_prompt,
                    "reasoning_strategy": p.reasoning_strategy,
                    "enable_memory": p.enable_memory,
                    "memory_channel": p.memory_channel,
                    "enable_tools": p.enable_tools,
                    "is_default": p.is_default,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                }
                for p in profiles
            ],
        })

    if request.method == 'POST':
        data, error = parse_json_body(request)
        if error:
            return error

        # Validate required fields
        if not data.get("name"):
            return json_error("Missing required field: name")

        # Generate ID if not provided
        if not data.get("id"):
            import uuid
            data["id"] = str(uuid.uuid4())[:8]

        from .agent.models import AgentProfile
        try:
            profile = AgentProfile(**data)
            created = manager.create_profile(profile)
            return JsonResponse({
                "profile": {
                    "id": created.id,
                    "name": created.name,
                    "avatar": created.avatar,
                    "description": created.description,
                    "default_model": created.default_model,
                    "temperature": created.temperature,
                    "prompt_profile_id": created.prompt_profile_id,
                    "system_prompt": created.system_prompt,
                    "reasoning_strategy": created.reasoning_strategy,
                    "enable_memory": created.enable_memory,
                    "memory_channel": created.memory_channel,
                    "enable_tools": created.enable_tools,
                    "is_default": created.is_default,
                    "created_at": created.created_at.isoformat() if created.created_at else None,
                    "updated_at": created.updated_at.isoformat() if created.updated_at else None,
                },
            }, status=201)
        except Exception as e:
            return json_error(str(e), status=400)

    return json_error("Method not allowed", status=405)


@csrf_exempt
def agent_profile_detail(request, profile_id):
    """
    GET /api/agent/profiles/{id} - Get a specific profile.
    PUT /api/agent/profiles/{id} - Update a profile.
    DELETE /api/agent/profiles/{id} - Delete a profile.
    """
    from .agent.profiles import get_profile_manager

    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    manager = get_profile_manager()

    if request.method == 'GET':
        profile = manager.get_profile(profile_id)
        if not profile:
            return json_error("Profile not found", status=404)

        return JsonResponse({
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "avatar": profile.avatar,
                "description": profile.description,
                "default_model": profile.default_model,
                "temperature": profile.temperature,
                "prompt_profile_id": profile.prompt_profile_id,
                "system_prompt": profile.system_prompt,
                "reasoning_strategy": profile.reasoning_strategy,
                "enable_memory": profile.enable_memory,
                "memory_channel": profile.memory_channel,
                "enable_tools": profile.enable_tools,
                "is_default": profile.is_default,
                "created_at": profile.created_at.isoformat() if profile.created_at else None,
                "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
            },
        })

    if request.method == 'PUT':
        data, error = parse_json_body(request)
        if error:
            return error

        updated = manager.update_profile(profile_id, data)
        if not updated:
            return json_error("Profile not found", status=404)

        return JsonResponse({
            "profile": {
                "id": updated.id,
                "name": updated.name,
                "avatar": updated.avatar,
                "description": updated.description,
                "default_model": updated.default_model,
                "temperature": updated.temperature,
                "prompt_profile_id": updated.prompt_profile_id,
                "system_prompt": updated.system_prompt,
                "reasoning_strategy": updated.reasoning_strategy,
                "enable_memory": updated.enable_memory,
                "memory_channel": updated.memory_channel,
                "enable_tools": updated.enable_tools,
                "is_default": updated.is_default,
                "created_at": updated.created_at.isoformat() if updated.created_at else None,
                "updated_at": updated.updated_at.isoformat() if updated.updated_at else None,
            },
        })

    if request.method == 'DELETE':
        try:
            if not manager.delete_profile(profile_id):
                return json_error("Profile not found", status=404)
            return JsonResponse({"deleted": True})
        except ValueError as e:
            return json_error(str(e), status=400)

    return json_error("Method not allowed", status=405)


@csrf_exempt
def agent_profile_set_default(request, profile_id):
    """
    POST /api/agent/profiles/{id}/set-default - Set a profile as the default.
    """
    from .agent.profiles import get_profile_manager

    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return json_error("Method not allowed", status=405)

    manager = get_profile_manager()
    if not manager.set_default_profile(profile_id):
        return json_error("Profile not found", status=404)

    return JsonResponse({"default_profile_id": profile_id})

import json
import logging
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

from .kit.memory_utils import check_memory_health
from .mcp import MCPClientManager, ServerRegistry
from .providers import get_registry

logger = logging.getLogger(__name__)

# Lazy-loaded MCP manager
_mcp_manager = None


def get_mcp_manager():
    """Get or create MCPClientManager instance lazily."""
    global _mcp_manager
    if _mcp_manager is None:
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent / "mcp_servers.json"
        registry = ServerRegistry(config_path) if config_path.exists() else ServerRegistry()
        _mcp_manager = MCPClientManager(registry)
        logger.info(f"MCPClientManager initialized with {len(registry.list())} configured servers")
    return _mcp_manager

# Lazy-loaded translation kit to avoid import-time model loading
_translation_kit = None


def get_translation_kit():
    """Get or create TranslationKit instance lazily."""
    global _translation_kit
    if _translation_kit is None:
        from .kit.translation import TranslationKit
        logger.info("Initializing TranslationKit (loading models)...")
        _translation_kit = TranslationKit()
        logger.info("TranslationKit initialized successfully")
    return _translation_kit


def index(request):
    return JsonResponse({'message': 'Hello, AgentX AI!'})


def health(request):
    """
    Health check endpoint for all services.
    
    Returns status of:
    - API server (always healthy if responding)
    - Translation models (loaded or not)
    - Memory system connections (neo4j, postgres, redis)
    """
    # Check translation kit (don't initialize just for health check)
    kit = _translation_kit  # Check without triggering lazy load
    translation_status = {
        "status": "healthy" if kit else "not_loaded",
        "models": {
            "language_detection": kit.language_detection_model_name if kit else None,
            "translation": kit.level_ii_translation_model_name if kit else None,
        }
    }
    
    # Check memory system (lazy - only if explicitly requested)
    include_memory = request.GET.get('include_memory', 'false').lower() == 'true'
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
    
    return JsonResponse(response)


@csrf_exempt
def translate(request):
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    # Only accept POST requests
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    content = request.body.decode('utf-8')
    logger.info(f"Translation request received, body size: {len(request.body)}")

    if not content:
        return JsonResponse({'error': 'No content provided'}, status=400)

    try:
        data = json.loads(content)
        text = data.get("text")
        # Accept both camelCase and snake_case for compatibility
        target_language = data.get("targetLanguage") or data.get("target_language")

        if not text:
            return JsonResponse({'error': 'Missing required field: text'}, status=400)
        if not target_language:
            return JsonResponse({'error': 'Missing required field: targetLanguage or target_language'}, status=400)

        logger.debug(f"Translation request: target={target_language}, text_length={len(text)}")
    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    try:
        translated_text = get_translation_kit().translate_text(text, target_language, target_language_level=2)
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.exception("Translation error")
        return JsonResponse({'error': f'Translation failed: {str(e)}'}, status=500)

    return JsonResponse({
        'original': text,
        'translatedText': str(translated_text),
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
    """List configured MCP servers and their status."""
    manager = get_mcp_manager()
    
    # Get configured servers from registry
    configured = [
        {
            "name": config.name,
            "transport": config.transport.value,
            "command": config.command,
            "url": config.url,
            "connected": manager.is_connected(config.name),
        }
        for config in manager.registry.list()
    ]
    
    # Get active connections
    active = [
        conn.to_dict()
        for conn in manager.list_connections()
    ]
    
    return JsonResponse({
        "configured_servers": configured,
        "active_connections": active,
    })


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


def providers_models(request):
    """List all available models across providers."""
    registry = get_registry()
    provider_filter = request.GET.get('provider')
    
    models = []
    
    for provider_name in registry.list_providers():
        if provider_filter and provider_name != provider_filter:
            continue
        
        try:
            provider = registry.get_provider(provider_name)
            for model_name in provider.list_models():
                caps = provider.get_capabilities(model_name)
                models.append({
                    "name": model_name,
                    "provider": provider_name,
                    "context_window": caps.context_window,
                    "supports_tools": caps.supports_tools,
                    "supports_vision": caps.supports_vision,
                    "supports_streaming": caps.supports_streaming,
                    "cost_per_1k_input": caps.cost_per_1k_input,
                    "cost_per_1k_output": caps.cost_per_1k_output,
                })
        except ValueError:
            # Provider not configured, skip
            continue
    
    return JsonResponse({
        "models": models,
        "count": len(models),
    })


def providers_health(request):
    """Check health of all configured providers."""
    registry = get_registry()

    results = registry.health_check()

    overall_status = "healthy"
    if any(r.get("status") != "healthy" for r in results.values()):
        overall_status = "degraded"

    return JsonResponse({
        "status": overall_status,
        "providers": results,
    })


# ============== Agent Endpoints ==============

# Lazy-loaded agent instance
_agent = None


def get_agent():
    """Get or create Agent instance lazily."""
    global _agent
    if _agent is None:
        import os
        from .agent import Agent, AgentConfig
        
        # Offline-first: default to local Ollama model
        default_model = os.environ.get("DEFAULT_MODEL", "llama3.2")
        
        _agent = Agent(AgentConfig(
            default_model=default_model,
            enable_planning=True,
            enable_reasoning=True,
        ))
        logger.info(f"Agent initialized with model: {default_model}")
    return _agent


@csrf_exempt
def agent_run(request):
    """Execute a task using the agent."""
    if request.method == 'OPTIONS':
        return JsonResponse({}, status=200)

    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
        task = data.get("task")
        if not task:
            return JsonResponse({'error': 'Missing required field: task'}, status=400)

        reasoning_strategy = data.get("reasoning_strategy")

    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    agent = get_agent()

    kwargs = {}
    if reasoning_strategy:
        kwargs["reasoning_strategy"] = reasoning_strategy
    result = agent.run(task, **kwargs)

    return JsonResponse({
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
def agent_chat_stream(request):
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
        profile_id = data.get("profile_id")
        temperature = data.get("temperature", 0.7)
        use_memory = data.get("use_memory", True)

    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    def generate_sse():
        """Generator that yields SSE events."""
        import time
        import uuid
        from .agent import Agent, AgentConfig
        from .agent.session import SessionManager
        from .agent.output_parser import parse_output
        from .prompts import get_prompt_manager
        from .providers.base import Message, MessageRole

        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Get or create agent with settings
        config_kwargs = {"enable_memory": use_memory}
        if model:
            config_kwargs["default_model"] = model
        agent = Agent(AgentConfig(**config_kwargs))

        # Session management
        if agent._session_manager is None:
            agent._session_manager = SessionManager()
        session = agent._session_manager.get_or_create(session_id)
        session.add_message(Message(role=MessageRole.USER, content=message))
        context = session.get_messages()[:-1]

        try:
            # Get provider and model
            provider, model_id = agent.registry.get_provider_for_model(
                agent.config.default_model
            )

            # Build messages with system prompt
            prompt_manager = get_prompt_manager()
            system_prompt = prompt_manager.get_system_prompt(
                profile_id=profile_id or agent.config.prompt_profile_id
            )

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content=system_prompt or "You are a helpful AI assistant."
                )
            ]
            if context:
                messages.extend(context)
            messages.append(Message(role=MessageRole.USER, content=message))

            # Send start event
            yield f"event: start\ndata: {json.dumps({'task_id': task_id, 'model': model_id})}\n\n"

            # Stream tokens using sync generator
            full_content = ""

            for chunk in provider.stream(messages, model_id, temperature=temperature, max_tokens=2000):
                if chunk.content:
                    full_content += chunk.content
                    yield f"event: chunk\ndata: {json.dumps({'content': chunk.content})}\n\n"

            # Parse output for thinking tags
            parsed = parse_output(full_content)

            # Add to session
            session.add_message(Message(role=MessageRole.ASSISTANT, content=parsed.content))

            total_time = (time.time() - start_time) * 1000

            # Store turns in memory if enabled
            if use_memory and agent.memory:
                try:
                    from .kit.agent_memory.models import Turn
                    conv_id = session_id or task_id

                    # Store user turn
                    user_turn = Turn(
                        id=f"{conv_id}-{len(session.get_messages())-2}",
                        conversation_id=conv_id,
                        role="user",
                        content=message,
                        index=len(session.get_messages()) - 2,
                    )
                    agent.memory.store_turn(user_turn)

                    # Store assistant turn
                    assistant_turn = Turn(
                        id=f"{conv_id}-{len(session.get_messages())-1}",
                        conversation_id=conv_id,
                        role="assistant",
                        content=parsed.content,
                        index=len(session.get_messages()) - 1,
                        metadata={"model": model_id, "latency_ms": total_time},
                    )
                    agent.memory.store_turn(assistant_turn)
                    logger.debug(f"Stored turns in memory for conversation {conv_id}")
                except Exception as mem_err:
                    logger.warning(f"Failed to store turns in memory: {mem_err}")

            # Send completion event
            yield f"event: done\ndata: {json.dumps({'task_id': task_id, 'thinking': parsed.thinking, 'has_thinking': parsed.has_thinking, 'total_time_ms': total_time, 'session_id': session_id or task_id})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

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
            pg_conn = PostgresConnection.get_connection()
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

DEFAULT_USER_ID = "default_user"  # TODO: Replace with actual auth when multi-user is implemented


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
        channel = request.GET.get('channel', '_global')
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
        channel = request.GET.get('channel', '_global')
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
        channel = request.GET.get('channel', '_global')
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
def memory_consolidate(request):
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
        result = registry.run_consolidation_pipeline(jobs)

        status_code = 200 if result.get("success") else 400
        return JsonResponse(result, status=status_code)

    except Exception as e:
        logger.error(f"Error running consolidation: {e}")
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

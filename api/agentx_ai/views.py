import asyncio
import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync

from .kit.memory_utils import check_memory_health
from .mcp import MCPClientManager, ServerRegistry, ServerConfig
from .providers import get_provider, get_model_config, get_registry

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

    translated_text = get_translation_kit().translate_text(text, target_language, target_language_level=2)

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
    
    # Run health checks using async_to_sync for proper Django integration
    results = async_to_sync(registry.health_check)()
    
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
        from .agent import Agent, AgentConfig
        _agent = Agent(AgentConfig(
            default_model="gpt-4-turbo",
            enable_planning=True,
            enable_reasoning=True,
        ))
        logger.info("Agent initialized")
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
    
    # Run the agent task using async_to_sync
    async def run_task():
        kwargs = {}
        if reasoning_strategy:
            kwargs["reasoning_strategy"] = reasoning_strategy
        return await agent.run(task, **kwargs)
    
    result = async_to_sync(run_task)()
    
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
        
    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)
    
    agent = get_agent()
    
    result = async_to_sync(agent.chat)(message, session_id=session_id)
    
    return JsonResponse({
        "task_id": result.task_id,
        "status": result.status.value,
        "answer": result.answer,
        "reasoning_steps": result.reasoning_steps,
        "total_tokens": result.total_tokens,
        "total_time_ms": result.total_time_ms,
    })


def agent_status(request):
    """Get the current agent status."""
    agent = get_agent()
    return JsonResponse(agent.get_status())

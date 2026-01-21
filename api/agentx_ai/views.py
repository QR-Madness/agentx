import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .kit.memory_utils import check_memory_health

logger = logging.getLogger(__name__)

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

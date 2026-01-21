import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .kit.translation import TranslationKit

logger = logging.getLogger(__name__)

translation_kit = TranslationKit()


def index(request):
    return JsonResponse({'message': 'Hello, AgentX AI!'})


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

    translated_text = translation_kit.translate_text(text, target_language, target_language_level=2)

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

    detected_language, confidence = translation_kit.detect_language_level_i(unclassified_text)

    return JsonResponse({
        'original': unclassified_text,
        'detected_language': detected_language,
        'confidence': confidence
    })

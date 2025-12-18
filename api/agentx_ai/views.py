import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .kit.translation import TranslationKit

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
    size = len(request.body)
    print(f"Got translation request; body size: {size}")
    print(f"JSON content: {content}")

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

        print(f"JSON data: {data}")
    except json.JSONDecodeError as e:
        return JsonResponse({'error': f'Invalid JSON: {str(e)}'}, status=400)

    print(f"Got translation request to {target_language}; text size: {len(text)}")

    translated_text = translation_kit.translate_text(text, target_language, target_language_level=2)

    return JsonResponse({
        'original': text,
        'translatedText': str(translated_text),
    })


@csrf_exempt
def language_detect(request):
    unclassified_text = "Hello, AgentX AI, this is a language test to detect the spoken language!"

    detected_language, confidence = translation_kit.detect_language_level_i(unclassified_text)

    return JsonResponse({
        'original': unclassified_text,
        'detected_language': detected_language,
        'confidence': confidence
    })

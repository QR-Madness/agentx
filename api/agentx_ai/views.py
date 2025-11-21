import json
from django.http import JsonResponse

from api.agentx_ai.kit.translation import TranslationKit

translation_kit = TranslationKit()


def index(request):
    return JsonResponse({'message': 'Hello, AgentX AI!'})


def translate(request):
    data = json.loads(request.body.decode())
    text = data["text"]
    target_language = data["target_language"]

    print(f"Got translation request  to {target_language}; text size: {len(text)}")

    translated_text = translation_kit.translate_level_i(text, target_language)

    return JsonResponse({
        'original': text,
        'translated': str(translated_text),
    })


def language_detect(request):
    unclassified_text = "Hello, AgentX AI, this is a language test to detect the spoken language!"

    detected_language, confidence = translation_kit.detect_language_level_i(unclassified_text)

    return JsonResponse({
        'original': unclassified_text,
        'detected_language': detected_language,
        'confidence': confidence
    })

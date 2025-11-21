from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from api.agentx_ai.kit.translation import TranslationKit

# Remove the hardcoded LANGUAGE_LABELS dictionary
# LANGUAGE_LABELS = { ... }

translation_kit = TranslationKit()


def index(request):
    return JsonResponse({'message': 'Hello, AgentX AI!'})


def language_detect(request):
    unclassified_text = "Hello, AgentX AI, this is a language test to detect the spoken language!"

    detected_language, confidence = translation_kit.detect_language(unclassified_text)

    return JsonResponse({
        'original': unclassified_text,
        'detected_language': detected_language,
        'confidence': confidence
    })

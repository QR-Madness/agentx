import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TranslationKit:
    """Translation models for language detection and translation."""

    language_detection_model_name = "eleldar/language-detection"

    def __init__(self):
        self.language_detection_tokenizer = AutoTokenizer.from_pretrained(self.language_detection_model_name)
        self.language_detection_model = AutoModelForSequenceClassification.from_pretrained(self.language_detection_model_name)

    def detect_language(self, unclassified_text) -> tuple:
        """Detect the spoken language in the text in a tuple of (language, confidence)."""
        inputs = self.language_detection_tokenizer(unclassified_text, return_tensors="pt")
        outputs = self.language_detection_model(**inputs)

        # Get predicted class
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_label_id = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_label_id].item()

        # Use the model's internal config to get the correct label
        detected_language = self.language_detection_model.config.id2label[predicted_label_id]

        return detected_language, round(confidence * 100, 2)
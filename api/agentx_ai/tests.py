from django.test import TestCase, Client

from agentx_ai.kit.translation import LanguageLexicon

"""

TODO 0) Integrate FAISS for vector database
TODO 1) Implement Django ORM for AI settings, basic storage
TODO 2) Implement neo4j for structured information storage and relationship analysis
 
"""


# Create your tests here.
class TranslationKitTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_language_detect_post(self):
        """Test that the language detection API works with POST."""
        response = self.client.post(
            "/api/tools/language-detect-20",
            data={"text": "Bonjour, comment allez-vous?"},
            content_type="application/json"
        )
        data = response.json()
        print(data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('detected_language', data)
        self.assertIn('confidence', data)
        self.assertEqual(data['detected_language'], 'fr')

    def test_language_detect_get(self):
        """Test backwards compatibility with GET request."""
        response = self.client.get("/api/tools/language-detect-20")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('detected_language', data)
        # Default text is English
        self.assertEqual(data['detected_language'], 'en')

    def test_lexicon_convert_level_i_to_level_ii(self):
        """Test that the lexicon converts level I language codes to level II language codes."""
        lexicon = LanguageLexicon(verbose=True)
        level_i_language = "en"
        level_ii_language = lexicon.convert_level_i_detection_to_level_ii(level_i_language)
        self.assertEqual(level_ii_language, "eng_Latn")
        self.assertTrue(level_ii_language in lexicon.level_ii_languages)
        print(f'Got correct level II language: {level_ii_language}')

    def test_translate_to_french(self):
        """Test that the translation API works."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello, AgentX AI!", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        print(response.json())
        self.assertEqual(response.status_code, 200)

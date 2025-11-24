from unittest import SkipTest

from django.test import TestCase, Client

from api.agentx_ai.kit.translation import LanguageLexicon

"""

TODO 0) Integrate FAISS for vector database
TODO 1) Implement Django ORM for AI settings, basic storage
TODO 2) Implement neo4j for structured information storage and relationship analysis
 
"""


# Create your tests here.
class TranslationKitTest(TestCase):
    def setUp(self):
        self.client = Client()

    @SkipTest
    def test_language_detect(self):
        """Test that the language detection API works."""
        response = self.client.get("/api/language-detect")
        print(response.json())
        self.assertEqual(response.status_code, 200)

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
            "/api/translate",
            data={"text": "Hello, AgentX AI!", "target_language": "fr"},
            content_type="application/json"
        )
        print(response.json())
        self.assertEqual(response.status_code, 200)

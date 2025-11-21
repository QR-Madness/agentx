from django.test import TestCase, Client


# Create your tests here.
class Test(TestCase):
    def test_language_detect(self):
        client = Client()
        response = client.get("/api/language-detect")
        print(response.json())
        self.assertEqual(response.status_code, 200)

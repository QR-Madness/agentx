# Testing

Guide to writing and running tests for AgentX.

## Running Tests

```bash
# Run all tests
task test

# Run specific test class
uv run python api/manage.py test agentx_ai.TranslationKitTest

# Run single test method
uv run python api/manage.py test agentx_ai.TranslationKitTest.test_translate_to_french
```

## Test Structure

Tests are located in `api/agentx_ai/tests.py`.

## Writing Tests

```python
from django.test import TestCase

class MyTestCase(TestCase):
    def test_example(self):
        self.assertEqual(1 + 1, 2)
```

See Django testing documentation for more information.

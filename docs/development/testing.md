# Testing

Guide to writing and running tests for AgentX.

## Running Tests

### All Tests

```bash
# Run all tests
task test

# Or directly with Django
uv run python api/manage.py test agentx_ai
```

### Specific Tests

```bash
# Run a specific test class
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest

# Run a single test method
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest.test_translate_to_french

# Run health check tests
uv run python api/manage.py test agentx_ai.tests.HealthCheckTest
```

### Verbose Output

```bash
# Show detailed output
uv run python api/manage.py test agentx_ai --verbosity=2
```

## Test Structure

Tests are located in `api/agentx_ai/tests.py` and organized by feature:

```
api/agentx_ai/tests.py
├── TranslationKitTest      # Translation system tests
├── HealthCheckTest         # API health endpoint tests
└── LanguageDetectionTest   # Language detection tests
```

## Writing Tests

### Basic Test Case

```python
from django.test import TestCase

class MyFeatureTest(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def test_basic_functionality(self):
        """Test description here."""
        result = my_function()
        self.assertEqual(result, expected_value)
    
    def tearDown(self):
        """Clean up after tests."""
        pass
```

### API Endpoint Tests

```python
from django.test import TestCase, Client

class APIEndpointTest(TestCase):
    def setUp(self):
        self.client = Client()
    
    def test_health_endpoint(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_translation_endpoint(self):
        response = self.client.post(
            '/api/tools/translate',
            data={'text': 'Hello', 'targetLanguage': 'fra_Latn'},
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
```

### Skipping Tests

Use `@unittest.skip` for tests that aren't ready:

```python
import unittest

class MyTest(TestCase):
    @unittest.skip("Requires model download - slow")
    def test_slow_model_operation(self):
        pass
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation:

- Translation kit methods
- Language lexicon conversions
- Memory utilities

### Integration Tests

Test component interactions:

- API endpoint flows
- Database connections
- MCP client operations

### End-to-End Tests

Test complete user flows (future):

- Client → API → Model → Response
- Full translation workflow

## Database Tests

For tests requiring database services:

```python
from django.test import TestCase
from agentx_ai.kit.memory_utils import check_memory_health

class MemoryTest(TestCase):
    def test_memory_health(self):
        """Requires Docker services running."""
        health = check_memory_health()
        self.assertIn('neo4j', health)
        self.assertIn('postgres', health)
        self.assertIn('redis', health)
```

!!! note "Docker Services"
    Run `task runners` before executing database tests.

## Coverage

To check test coverage (requires `coverage` package):

```bash
# Run with coverage
uv run coverage run --source=agentx_ai api/manage.py test agentx_ai

# View report
uv run coverage report

# Generate HTML report
uv run coverage html
open htmlcov/index.html
```

## Best Practices

1. **Descriptive names**: `test_translate_english_to_french` not `test_1`
2. **One assertion focus**: Each test should verify one behavior
3. **Independent tests**: Tests shouldn't depend on each other
4. **Fast tests**: Mock external services when possible
5. **Clean up**: Use `tearDown()` to reset state

## See Also

- [Django Testing Documentation](https://docs.djangoproject.com/en/5.2/topics/testing/)
- [Contributing Guide](contributing.md)

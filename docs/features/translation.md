# Translation System

Multi-level language detection and translation system.

## Overview

Two-level architecture:

- **Level I**: Fast detection (~20 languages)
- **Level II**: Comprehensive translation (200+ languages)

## Models

- Detection: `eleldar/language-detection`
- Translation: `facebook/m2m100_418M` or `facebook/nllb-200-distilled-600M`

## API Usage

```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "target_language": "fr"}'
```

See [API Endpoints](../api/endpoints.md) for details.

# API Endpoints

REST API endpoint reference.

## Base URL

```
http://localhost:12319/api/
```

## Endpoints

### Health Check

```
GET /api/index
```

Returns API status.

### Language Detection

```
GET /api/language-detect
```

Detects language from test text.

### Translation

```
POST /api/translate
Content-Type: application/json

{
  "text": "Hello, world!",
  "target_language": "fr"
}
```

Translates text to target language using ISO 639-1 codes.

**Response:**
```json
{
  "original": "Hello, world!",
  "translated": "Bonjour le monde!",
  "source_language": "en",
  "target_language": "fr"
}
```

## Authentication

Currently no authentication required (development mode).

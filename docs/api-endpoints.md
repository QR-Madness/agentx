# AgentX API Endpoints Reference

## Base URL

```
http://localhost:12319/api/
```

All API endpoints are prefixed with `/api/`.

## Endpoints

### 1. Health Check

**GET** `/api/index`

Check if the API server is running.

**Request**: None

**Response**:
```json
{
    "message": "Hello, AgentX AI!"
}
```

**Status Codes**:
- `200 OK`: Server is running

**Example**:
```bash
curl http://localhost:12319/api/index
```

---

### 2. Language Detection

**GET** `/api/language-detect`

Detect the language of a hardcoded test string using Level I detection.

**Note**: This endpoint uses a hardcoded test string. For production use, this should be modified to accept text input.

**Request**: None

**Response**:
```json
{
    "original": "Hello, AgentX AI, this is a language test to detect the spoken language!",
    "detected_language": "english",
    "confidence": 99.87
}
```

**Response Fields**:
- `original` (string): The text that was analyzed
- `detected_language` (string): Detected language name (lowercase)
- `confidence` (number): Confidence score (0-100)

**Status Codes**:
- `200 OK`: Detection successful

**Supported Languages** (~20):
- Arabic, Bulgarian, German, Greek, English, Spanish
- French, Hindi, Italian, Japanese, Dutch, Polish
- Portuguese, Russian, Swahili, Thai, Turkish
- Urdu, Vietnamese, Chinese

**Example**:
```bash
curl http://localhost:12319/api/language-detect
```

**Implementation**: `api/agentx_ai/views.py:57-66`

---

### 3. Translate Text

**POST** `/api/translate`

Translate text to a target language using the NLLB-200 translation model.

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
    "text": "Hello, how are you?",
    "targetLanguage": "fra_Latn"
}
```

**Request Fields**:
- `text` (string, required): Text to translate
- `targetLanguage` (string, required): NLLB-200 language code (Level II format)
  - Format: `{iso639-3}_{Script}`
  - Examples: `fra_Latn` (French), `spa_Latn` (Spanish), `jpn_Jpan` (Japanese)
  - See [translation-system.md](./translation-system.md) for complete list of 204 languages

**Response**:
```json
{
    "original": "Hello, how are you?",
    "translatedText": "Bonjour, comment allez-vous?"
}
```

**Response Fields**:
- `original` (string): Original input text
- `translatedText` (string): Translated text in target language

**Status Codes**:
- `200 OK`: Translation successful
- `400 Bad Request`: Missing required fields or invalid JSON
- `405 Method Not Allowed`: Non-POST request
- `500 Internal Server Error`: Translation model error

**Error Response**:
```json
{
    "error": "Missing required field: text"
}
```

**Examples**:

1. **English to French**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "targetLanguage": "fra_Latn"
  }'
```

Response:
```json
{
    "original": "Hello, world!",
    "translatedText": "Bonjour le monde!"
}
```

2. **English to Japanese**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning",
    "targetLanguage": "jpn_Jpan"
  }'
```

3. **English to Arabic**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to AgentX",
    "targetLanguage": "arb_Arab"
  }'
```

**Implementation**: `api/agentx_ai/views.py:15-54`

**Notes**:
- No source language detection required - model auto-detects
- Supports 204 languages across multiple scripts
- Translation quality depends on model (currently using M2M100-418M temporarily)
- Long texts may take several seconds to process

---

## CORS Support

The API includes CORS (Cross-Origin Resource Sharing) support for development.

**Supported Origins**:
```python
# api/agentx_api/settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:1420",  # Vite dev server
    "https://tauri.localhost",  # Tauri app
]
```

**CORS Headers**:
- `Access-Control-Allow-Origin`
- `Access-Control-Allow-Methods`: GET, POST, OPTIONS
- `Access-Control-Allow-Headers`: Content-Type

**Preflight Requests**:

The `/api/translate` endpoint handles OPTIONS requests for CORS preflight:

```bash
curl -X OPTIONS http://localhost:12319/api/translate \
  -H "Origin: http://localhost:1420" \
  -H "Access-Control-Request-Method: POST"
```

---

## Authentication

**Current Status**: No authentication required

**Future Considerations**:
- API key authentication for production
- JWT tokens for user sessions
- Rate limiting per user/IP

---

## Error Handling

### Common Errors

| Error | Status | Cause | Solution |
|-------|--------|-------|----------|
| `Only POST requests allowed` | 405 | Wrong HTTP method | Use POST instead of GET |
| `No content provided` | 400 | Empty request body | Include JSON body |
| `Invalid JSON: ...` | 400 | Malformed JSON | Validate JSON syntax |
| `Missing required field: text` | 400 | Missing `text` field | Include `text` in request |
| `Missing required field: targetLanguage` | 400 | Missing `targetLanguage` | Include `targetLanguage` in request |

### Error Response Format

```json
{
    "error": "Description of what went wrong"
}
```

### Model Errors

If the translation model fails, you may see:
- PyTorch CUDA errors (GPU issues)
- Tokenizer errors (invalid language code)
- Out of memory errors (text too long)

---

## Rate Limiting

**Current Status**: No rate limiting implemented

**Recommendations for Production**:
- Limit: 60 requests per minute per IP
- Burst: Allow 10 requests in quick succession
- Implement using Django middleware or nginx

---

## Request/Response Examples

### Successful Translation (Multiple Languages)

**German**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "targetLanguage": "deu_Latn"}'
# Response: {"original": "Hello", "translatedText": "Hallo"}
```

**Spanish**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "targetLanguage": "spa_Latn"}'
# Response: {"original": "Hello", "translatedText": "Hola"}
```

**Chinese (Simplified)**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "targetLanguage": "zho_Hans"}'
# Response: {"original": "Hello", "translatedText": "你好"}
```

**Russian**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "targetLanguage": "rus_Cyrl"}'
# Response: {"original": "Hello", "translatedText": "Здравствуйте"}
```

### Error Cases

**Missing Field**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello"}'
# Response: {"error": "Missing required field: targetLanguage or target_language"}
```

**Invalid JSON**:
```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{text: "Hello"}'  # Missing quotes around key
# Response: {"error": "Invalid JSON: ..."}
```

**Wrong Method**:
```bash
curl -X GET http://localhost:12319/api/translate
# Response: {"error": "Only POST requests allowed"}
```

---

## Client Integration

### TypeScript/JavaScript Example

```typescript
import { api } from "../lib/api";

export interface TranslationRequest {
    text: string;
    targetLanguage: string;
}

export interface TranslationResponse {
    translatedText: string;
}

export const postTranslation = async (
    request: TranslationRequest
): Promise<TranslationResponse> => {
    const response = await fetch(`${api.baseUrl}/tools/translate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(request)
    });

    if (!response.ok) {
        throw new Error(`Translation failed: ${response.statusText}`);
    }

    return await response.json();
};

// Usage
const result = await postTranslation({
    text: "Hello, world!",
    targetLanguage: "fra_Latn"
});
console.log(result.translatedText); // "Bonjour le monde!"
```

### React Hook Example

```typescript
import { useState } from 'react';
import { postTranslation } from '../models/translation';

export const useTranslation = () => {
    const [isTranslating, setIsTranslating] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const translate = async (text: string, targetLanguage: string) => {
        setIsTranslating(true);
        setError(null);

        try {
            const response = await postTranslation({ text, targetLanguage });
            return response.translatedText;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setIsTranslating(false);
        }
    };

    return { translate, isTranslating, error };
};
```

---

## URL Routing

API endpoints are defined in:

**Main URL Configuration**: `api/agentx_api/urls.py`
```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('agentx_ai.urls')),
]
```

**App URL Configuration**: `api/agentx_ai/urls.py`
```python
urlpatterns = [
    path('index', views.index, name='index'),
    path('language-detect', views.language_detect, name='language_detect'),
    path('translate', views.translate, name='translate'),
]
```

---

## Development & Testing

### Start API Server

```bash
# Using Task (recommended)
task api:runserver

# Or directly
cd api && python manage.py runserver --port 12319
```

### Test Endpoints

```bash
# Health check
curl http://localhost:12319/api/index

# Language detection
curl http://localhost:12319/api/language-detect

# Translation
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "targetLanguage": "spa_Latn"}'
```

### View Request Logs

Django logs all requests to stdout when running the dev server:

```
Got translation request; body size: 58
JSON content: {"text":"Hello","targetLanguage":"spa_Latn"}
JSON data: {'text': 'Hello', 'targetLanguage': 'spa_Latn'}
Got translation request to spa_Latn; text size: 5
[25/Nov/2025 10:30:45] "POST /api/translate HTTP/1.1" 200 65
```

---

## Future Endpoints (Planned)

### Chat/Conversation

```
POST /api/chat
{
    "message": "Tell me about Paris",
    "conversationId": "uuid-1234"
}
```

### Language Detection (with input)

```
POST /api/language-detect
{
    "text": "Bonjour le monde"
}
```

### Batch Translation

```
POST /api/translate/batch
{
    "texts": ["Hello", "Goodbye", "Thank you"],
    "targetLanguage": "fra_Latn"
}
```

### Translation History

```
GET /api/translate/history?limit=10
```

---

**Last Updated**: 2025-11-25
**API Version**: v1.0
**Django Version**: 5.2.8

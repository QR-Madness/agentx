# API Models

Data models and schemas used by the API.

## Translation Request

```json
{
  "text": "string (required)",
  "target_language": "string (required, ISO 639-1)",
  "source_language": "string (optional, auto-detect)"
}
```

## Translation Response

```json
{
  "original": "string",
  "translated": "string",
  "source_language": "string",
  "target_language": "string",
  "confidence": "float (optional)"
}
```

## Language Detection

```json
{
  "text": "string",
  "detected_language": "string (ISO 639-1)",
  "confidence": "float"
}
```

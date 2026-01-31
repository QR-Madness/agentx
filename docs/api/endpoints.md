# API Endpoints

REST API endpoint reference.

## Base URL

```
http://localhost:12319/api/
```

## Core Endpoints

### Index / Hello

```
GET /api/index
```

Returns a simple hello message.

### Health Check

```
GET /api/health
GET /api/health?include_memory=true
```

Returns API and service health status. Add `include_memory=true` to check database connections.

**Response:**
```json
{
  "status": "healthy",
  "api": {"status": "healthy"},
  "translation": {"status": "healthy", "models": {...}},
  "memory": {
    "neo4j": {"status": "healthy"},
    "postgres": {"status": "healthy"},
    "redis": {"status": "healthy"}
  }
}
```

## Translation Endpoints

### Language Detection

```
GET  /api/tools/language-detect-20
POST /api/tools/language-detect-20
Content-Type: application/json

{"text": "Bonjour, comment allez-vous?"}
```

Detects language from text (supports ~20 languages for fast detection).

**Response:**
```json
{
  "original": "Bonjour, comment allez-vous?",
  "detected_language": "fr",
  "confidence": 98.5
}
```

### Translation

```
POST /api/tools/translate
Content-Type: application/json

{
  "text": "Hello, world!",
  "targetLanguage": "fra_Latn"
}
```

Translates text to target language using NLLB-200 language codes (e.g., `eng_Latn`, `fra_Latn`, `deu_Latn`).

**Response:**
```json
{
  "original": "Hello, world!",
  "translatedText": "Bonjour, monde!",
  "targetLanguage": "fra_Latn"
}
```

## MCP Endpoints

### List MCP Servers

```
GET /api/mcp/servers
```

Returns configured MCP server connections.

### List MCP Tools

```
GET /api/mcp/tools
```

Returns available tools from connected MCP servers.

### List MCP Resources

```
GET /api/mcp/resources
```

Returns available resources from connected MCP servers.

## Provider Endpoints

### List Providers

```
GET /api/providers
```

Returns configured model providers (OpenAI, Anthropic, Ollama).

### List Models

```
GET /api/providers/models
GET /api/providers/models?provider=openai
```

Returns available models. Filter by provider with query parameter.

### Provider Health

```
GET /api/providers/health
```

Returns health status of all configured providers.

## Agent Endpoints

### Run Task

```
POST /api/agent/run
Content-Type: application/json

{
  "task": "Analyze the sentiment of this text: I love this product!",
  "reasoning_strategy": "chain_of_thought"
}
```

Executes a task using the agent with optional reasoning strategy.

### Chat

```
POST /api/agent/chat
Content-Type: application/json

{
  "message": "What can you help me with?",
  "session_id": "optional-session-uuid"
}
```

Conversational interaction with session management.

### Agent Status

```
GET /api/agent/status
```

Returns current agent status (idle, running, etc.).

## Authentication

Currently no authentication required (development mode).

!!! warning "Production Security"
    Enable authentication before deploying to production. See [Security Guide](../deployment/docker.md#production-considerations).

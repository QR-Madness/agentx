# API Endpoints

Base URL: `http://localhost:12319/api/`

All endpoints return JSON. POST endpoints accept `application/json` bodies. All endpoints handle CORS preflight (`OPTIONS`) automatically.

---

## Core

### Index

```
GET /api/index
```

Returns `{"message": "Hello, AgentX AI!"}`.

### Health Check

```
GET /api/health
GET /api/health?include_memory=true
```

Returns API, translation, and (optionally) memory system health.

**Response:**
```json
{
  "status": "healthy",
  "api": {"status": "healthy"},
  "translation": {
    "status": "healthy",
    "models": {
      "language_detection": "eleldar/language-detection",
      "translation": "facebook/nllb-200-distilled-600M"
    }
  },
  "memory": {
    "neo4j": {"status": "healthy"},
    "postgres": {"status": "healthy"},
    "redis": {"status": "healthy"}
  }
}
```

Translation status is `"not_loaded"` if models haven't been initialized yet. Memory section only appears when `include_memory=true`.

---

## Translation

### Language Detection

```
GET  /api/tools/language-detect-20
POST /api/tools/language-detect-20
```

**Request (POST):**
```json
{"text": "Bonjour, comment allez-vous?"}
```

**Response:**
```json
{
  "original": "Bonjour, comment allez-vous?",
  "detected_language": "fr",
  "confidence": 98.5
}
```

GET accepts `?text=...` query parameter.

### Translation

```
POST /api/tools/translate
```

**Request:**
```json
{
  "text": "Hello, world!",
  "targetLanguage": "fra_Latn"
}
```

Uses NLLB-200 language codes (e.g., `eng_Latn`, `fra_Latn`, `deu_Latn`, `spa_Latn`).

**Response:**
```json
{
  "original": "Hello, world!",
  "translatedText": "Bonjour, monde!",
  "targetLanguage": "fra_Latn"
}
```

---

## MCP (Model Context Protocol)

### List Servers

```
GET /api/mcp/servers
```

Returns configured MCP servers and connection status.

**Response:**
```json
{
  "servers": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "status": "connected",
      "tools": ["read_file", "write_file", "list_directory"],
      "tools_count": 3,
      "resources_count": 0
    }
  ]
}
```

### List Tools

```
GET /api/mcp/tools
GET /api/mcp/tools?server=filesystem
```

Returns available tools from connected servers. Filter by server name with `?server=`.

### List Resources

```
GET /api/mcp/resources
GET /api/mcp/resources?server=filesystem
```

Returns available resources from connected servers. Filter by server name with `?server=`.

### Connect

```
POST /api/mcp/connect
```

Connect to one or all configured MCP servers.

**Request (single):**
```json
{"server": "filesystem"}
```

**Response:**
```json
{
  "status": "connected",
  "server": "filesystem",
  "tools_count": 3,
  "resources_count": 0
}
```

**Request (all):**
```json
{"all": true}
```

**Response:**
```json
{
  "results": {
    "filesystem": {"status": "connected", "tools_count": 3},
    "github": {"status": "error", "error": "..."}
  }
}
```

### Disconnect

```
POST /api/mcp/disconnect
```

**Request (single):** `{"server": "filesystem"}`
**Request (all):** `{"all": true}`

---

## Providers

### List Providers

```
GET /api/providers
```

Returns configured model providers and their status.

**Response:**
```json
{
  "providers": [
    {
      "name": "lmstudio",
      "status": "configured",
      "models": ["llama3.2", "deepseek-r1"]
    },
    {
      "name": "anthropic",
      "status": "not_configured",
      "error": "API key not set"
    }
  ]
}
```

### List Models

```
GET /api/providers/models
GET /api/providers/models?provider=openai
```

Returns all available models with capabilities. Filter by provider with `?provider=`.

**Response:**
```json
{
  "models": [
    {
      "name": "claude-3-5-sonnet-latest",
      "provider": "anthropic",
      "context_window": 200000,
      "supports_tools": true,
      "supports_vision": true,
      "supports_streaming": true,
      "cost_per_1k_input": 0.003,
      "cost_per_1k_output": 0.015
    }
  ],
  "count": 5
}
```

### Provider Health

```
GET /api/providers/health
```

Async health check of all configured providers.

**Response:**
```json
{
  "status": "healthy",
  "providers": {
    "lmstudio": {"status": "healthy"},
    "anthropic": {"status": "healthy"},
    "openai": {"status": "unhealthy", "error": "..."}
  }
}
```

`status` is `"healthy"` if all pass, `"degraded"` if any fail.

---

## Agent

### Run Task

```
POST /api/agent/run
```

Execute a task using the full agent pipeline (planning + reasoning).

**Request:**
```json
{
  "task": "Analyze the sentiment of this text: I love this product!",
  "reasoning_strategy": "chain_of_thought"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | yes | Task description |
| `reasoning_strategy` | string | no | `"auto"`, `"cot"`, `"tot"`, `"react"`, `"reflection"` |

**Response:**
```json
{
  "task_id": "a1b2c3d4",
  "status": "complete",
  "answer": "The sentiment is positive...",
  "plan_steps": 2,
  "reasoning_steps": 3,
  "tools_used": [],
  "models_used": ["llama3.2"],
  "total_tokens": 450,
  "total_time_ms": 1234.5
}
```

### Chat

```
POST /api/agent/chat
```

Conversational interaction with session management.

**Request:**
```json
{
  "message": "What can you help me with?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "claude-3-5-sonnet-latest",
  "profile_id": "default",
  "temperature": 0.7,
  "use_memory": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | yes | — | User message |
| `session_id` | string | no | auto-generated | Session ID for continuity |
| `model` | string | no | from config | Model to use |
| `profile_id` | string | no | `"default"` | Prompt profile ID |
| `temperature` | float | no | `0.7` | Sampling temperature |
| `use_memory` | bool | no | `true` | Enable memory recall/storage |

**Response:**
```json
{
  "task_id": "a1b2c3d4",
  "status": "complete",
  "response": "I can help you with...",
  "answer": "I can help you with...",
  "thinking": "Let me consider what this user needs...",
  "has_thinking": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reasoning_trace": 0,
  "reasoning_steps": 0,
  "tokens_used": 150,
  "total_tokens": 150,
  "total_time_ms": 892.3
}
```

### Chat (Streaming)

```
POST /api/agent/chat/stream
```

Same request body as `/agent/chat`. Returns Server-Sent Events (SSE).

**Response:** `Content-Type: text/event-stream`

SSE events in order:

| Event | Data | When |
|-------|------|------|
| `start` | `{"task_id": "...", "model": "..."}` | Stream begins |
| `chunk` | `{"content": "token text"}` | Each token |
| `tool_call` | `{"tool": "name", "arguments": {...}}` | Tool invocation starts |
| `tool_result` | `{"tool": "name", "content": "..."}` | Tool result (truncated to 500 chars) |
| `done` | `{"task_id": "...", "thinking": "...", "has_thinking": bool, "total_time_ms": float, "session_id": "..."}` | Stream complete |
| `error` | `{"error": "message"}` | On failure |

The streaming endpoint supports the same tool-use loop as the non-streaming endpoint (up to 10 rounds). Memory storage happens in a background thread after the stream closes.

**Example with curl:**
```bash
curl -N -X POST http://localhost:12319/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "llama3.2"}'
```

### Agent Status

```
GET /api/agent/status
```

Returns current agent status (`idle`, `planning`, `reasoning`, `executing`, `complete`, `failed`, `cancelled`).

---

## Prompts

### List Profiles

```
GET /api/prompts/profiles
```

**Response:**
```json
{
  "profiles": [
    {
      "id": "default",
      "name": "Default Assistant",
      "description": "General-purpose AI assistant",
      "is_default": true,
      "sections_count": 3,
      "enabled_sections": 2
    }
  ]
}
```

### Profile Detail

```
GET /api/prompts/profiles/{profile_id}
```

Returns the full profile with all sections and the composed prompt preview.

**Response:**
```json
{
  "profile": {
    "id": "default",
    "name": "Default Assistant",
    "description": "General-purpose AI assistant",
    "is_default": true,
    "sections": [
      {
        "id": "identity",
        "name": "Identity",
        "type": "system",
        "content": "You are a helpful AI assistant.",
        "enabled": true,
        "order": 0
      }
    ]
  },
  "composed_prompt": "You are a helpful AI assistant.\n\n..."
}
```

### Get Global Prompt

```
GET /api/prompts/global
```

**Response:**
```json
{
  "global_prompt": {
    "content": "Always be helpful and concise.",
    "enabled": true
  }
}
```

### Update Global Prompt

```
POST /api/prompts/global/update
```

**Request:**
```json
{
  "content": "Always be helpful and concise.",
  "enabled": true
}
```

### List Sections

```
GET /api/prompts/sections
```

Returns all available prompt sections with their content and ordering.

### Compose Preview

```
GET /api/prompts/compose
GET /api/prompts/compose?profile_id=default
```

Returns the fully composed system prompt that would be sent to the model.

**Response:**
```json
{
  "system_prompt": "You are a helpful AI assistant.\n\nAlways be helpful and concise.\n\n...",
  "profile_id": "default"
}
```

### MCP Tools Prompt

```
GET /api/prompts/mcp-tools
```

Returns the auto-generated prompt describing available MCP tools for injection into the system prompt.

**Response:**
```json
{
  "mcp_tools_prompt": "You have access to the following tools:\n\n- read_file: Read a file...",
  "tools_count": 5
}
```

---

## Memory

### Channels

```
GET  /api/memory/channels
POST /api/memory/channels
```

**GET** — List all channels with item counts per type.

**Response:**
```json
{
  "channels": [
    {
      "name": "_global",
      "is_default": true,
      "item_counts": {
        "turns": 42,
        "entities": 15,
        "facts": 28,
        "strategies": 3,
        "goals": 2
      }
    }
  ]
}
```

**POST** — Create a new channel. Name must be alphanumeric with hyphens/underscores.

**Request:** `{"name": "my-project"}`

### Delete Channel

```
DELETE /api/memory/channels/{name}
```

Deletes a channel and all associated data across Neo4j, PostgreSQL, and Redis. Cannot delete `_global`.

**Response:**
```json
{
  "message": "Channel 'my-project' deleted successfully",
  "deleted": {
    "turns": 10, "entities": 5, "facts": 8,
    "strategies": 1, "goals": 0, "conversations": 2,
    "postgres_rows": 15, "redis_keys": 3
  }
}
```

### Delete Conversation

```
DELETE /api/memory/conversations/{conversation_id}
```

Deletes a single conversation and its turns from all databases.

### List Entities

```
GET /api/memory/entities
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |
| `search` | string | — | Text search on entity name |
| `type` | string | — | Filter by entity type |

**Response:**
```json
{
  "entities": [...],
  "total": 45,
  "page": 1,
  "limit": 20,
  "has_next": true
}
```

### Entity Graph

```
GET /api/memory/entities/{entity_id}/graph
GET /api/memory/entities/{entity_id}/graph?depth=3
```

Returns the entity with connected facts and relationships.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `depth` | int | 2 | Traversal depth (max 3) |

Returns 404 if entity not found.

### List Facts

```
GET /api/memory/facts
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |
| `min_confidence` | float | 0.0 | Minimum confidence (0.0–1.0) |
| `search` | string | — | Text search on fact claim |

### List Strategies

```
GET /api/memory/strategies
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |

### Memory Stats

```
GET /api/memory/stats
```

Returns total counts and per-channel breakdowns.

**Response:**
```json
{
  "totals": {"entities": 45, "facts": 120, "strategies": 8, "turns": 300},
  "by_channel": {
    "_global": {"entities": 30, "facts": 80, "strategies": 5, "turns": 200},
    "my-project": {"entities": 15, "facts": 40, "strategies": 3, "turns": 100}
  }
}
```

### Consolidation Settings

```
GET  /api/memory/settings
POST /api/memory/settings
```

**GET** — Returns consolidation settings (extraction, relevance filter, entity linking, quality thresholds) plus default prompts.

**POST** — Update consolidation settings. Accepts partial updates.

### Recall Settings

```
GET  /api/memory/recall-settings
POST /api/memory/recall-settings
```

**GET** — Returns recall layer settings (hybrid search, entity-centric, query expansion, HyDE, self-query technique toggles and parameters).

**POST** — Update recall settings. Accepts partial updates.

### Run Consolidation

```
POST /api/memory/consolidate
```

Manually triggers the consolidation pipeline.

**Request (optional):**
```json
{"jobs": ["consolidate", "patterns", "promote"]}
```

If no jobs specified, runs the default set: consolidate, patterns, promote.

### Reset Consolidation

```
POST /api/memory/reset
```

Clears consolidated timestamps from all conversations, allowing reprocessing.

**Request (optional):**
```json
{"delete_memories": true}
```

When `delete_memories` is true, also deletes all entities, facts, and strategies. Useful when extraction logic has changed.

---

## Jobs

### List Jobs

```
GET /api/jobs
```

Returns all registered consolidation jobs with their status and worker info.

**Response:**
```json
{
  "jobs": [
    {
      "name": "consolidate",
      "status": "idle",
      "enabled": true,
      "last_run": "2026-03-10T14:30:00Z",
      "last_duration_ms": 1500,
      "run_count": 42
    }
  ],
  "worker": {
    "status": "running",
    "uptime_seconds": 3600
  }
}
```

### Job Detail

```
GET /api/jobs/{job_name}
```

Returns job details with recent execution history (last 10 runs).

**Response:**
```json
{
  "job": {...},
  "history": [
    {"timestamp": "...", "duration_ms": 1200, "success": true, "metrics": {...}}
  ]
}
```

### Run Job

```
POST /api/jobs/{job_name}/run
```

Manually triggers a specific job. Returns execution result with metrics.

### Toggle Job

```
POST /api/jobs/{job_name}/toggle
```

Enable or disable a scheduled job.

**Request:** `{"enabled": false}`

### Clear Stuck Jobs

```
POST /api/jobs/clear-stuck
```

Clears jobs stuck in `running` state (e.g., after a crash).

**Response:**
```json
{
  "success": true,
  "cleared_jobs": ["consolidate"],
  "message": "Cleared 1 stuck job(s)"
}
```

---

## Config

### Update Config

```
POST /api/config/update
```

Updates runtime configuration. Persists to `data/config.json` and hot-reloads providers.

**Request:**
```json
{
  "providers": {
    "anthropic": {"api_key": "sk-ant-..."},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "timeout": 300}
  },
  "preferences": {
    "default_model": "claude-3-5-sonnet-latest",
    "enable_memory_by_default": true
  },
  "llm_settings": {
    "default_temperature": 0.7,
    "default_max_tokens": 4096
  }
}
```

All fields are optional — only provided fields are updated.

**Response:**
```json
{
  "status": "ok",
  "message": "Config updated and applied",
  "updated": ["providers.anthropic.api_key", "preferences.default_model"]
}
```

No GET endpoint exists for security — config contains API keys.

---

## Authentication

Currently no authentication required (single-user development mode).

# Quick Start

Get AgentX running and make your first API calls.

## Start the Stack

```bash
# Full stack: Docker services + Django API + Tauri client
task dev
```

This starts Neo4j, PostgreSQL, Redis, the Django API on port 12319, and the Tauri desktop app. All support hot reload.

For API-only development:

```bash
task db:up        # Start database services
task dev:api      # Start Django API (port 12319)
```

## Health Check

Verify the API is running:

```bash
curl http://localhost:12319/api/health
```

```json
{"status": "ok", "version": "0.1.0"}
```

Include database status:

```bash
curl http://localhost:12319/api/health?include_memory=true
```

---

## Chat (Simple Completion)

Send a message and get a response:

```bash
curl -X POST http://localhost:12319/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "model": "llama3.2"
  }'
```

```json
{
  "status": "success",
  "response": "The capital of France is Paris.",
  "session_id": "abc123",
  "model": "llama3.2"
}
```

### Streaming Chat

Stream responses via Server-Sent Events:

```bash
curl -N -X POST http://localhost:12319/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing briefly",
    "model": "llama3.2"
  }'
```

Events arrive as SSE:

```
event: start
data: {"task_id": "t_abc123", "model": "llama3.2"}

event: chunk
data: {"content": "Quantum computing uses "}

event: chunk
data: {"content": "quantum mechanical phenomena..."}

event: done
data: {"task_id": "t_abc123", "total_time_ms": 1423.5, "session_id": "s_def456"}
```

### Session Continuity

Pass `session_id` to maintain conversation context:

```bash
curl -X POST http://localhost:12319/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about its population?",
    "session_id": "abc123"
  }'
```

---

## Translation

### Detect Language

```bash
curl -X POST http://localhost:12319/api/tools/language-detect-20 \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde"}'
```

```json
{
  "status": "success",
  "language": "fr",
  "confidence": 0.99,
  "language_name": "French"
}
```

### Translate Text

```bash
curl -X POST http://localhost:12319/api/tools/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "targetLanguage": "fra_Latn"
  }'
```

```json
{
  "status": "success",
  "translated_text": "Bonjour le monde!",
  "source_language": "eng_Latn",
  "target_language": "fra_Latn"
}
```

Target languages use NLLB-200 codes (e.g., `fra_Latn`, `deu_Latn`, `spa_Latn`). See [Translation](../features/translation.md) for the full language code reference.

---

## MCP Tools

### Connect a Server

```bash
curl -X POST http://localhost:12319/api/mcp/connect \
  -H "Content-Type: application/json" \
  -d '{"server": "filesystem"}'
```

Connect all configured servers:

```bash
curl -X POST http://localhost:12319/api/mcp/connect \
  -H "Content-Type: application/json" \
  -d '{"all": true}'
```

### List Available Tools

```bash
curl http://localhost:12319/api/mcp/tools
```

```json
{
  "status": "success",
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file",
      "server": "filesystem"
    }
  ]
}
```

Once tools are connected, the agent can use them automatically during chat. See [MCP](../features/mcp.md) for server configuration.

---

## Prompt Profiles

### List Profiles

```bash
curl http://localhost:12319/api/prompts/profiles
```

### Use a Profile in Chat

```bash
curl -X POST http://localhost:12319/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me write a poem",
    "profile_id": "creative"
  }'
```

See [Prompts](../features/prompts.md) for profile management.

---

## Memory

### Recall Memories

```bash
curl -X POST http://localhost:12319/api/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 5}'
```

### View Memory Stats

```bash
curl http://localhost:12319/api/memory/stats
```

Memory is automatically populated during chat when `enable_memory` is true (default). See [Memory](../features/memory.md) for the full memory system reference.

---

## Database Access

```bash
task db:shell:postgres    # psql shell
task db:shell:redis       # redis-cli
task db:shell:neo4j       # cypher-shell
```

Neo4j web browser: [http://localhost:7474](http://localhost:7474)

---

## Next Steps

- [Configuration](configuration.md) — Environment variables and config files
- [API Endpoints](../api/endpoints.md) — Full API reference
- [Architecture Overview](../architecture/overview.md) — System design
- [Chat](../features/chat.md) — Chat modes, streaming, and tool-use loops

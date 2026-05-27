# Configuration

AgentX uses four configuration layers: environment variables (`.env`), runtime config (`data/config.json`), MCP server config (`mcp_servers.json`), and prompt config (`data/system_prompts.yaml`).

## Environment Variables

Copy `.env.example` to `.env` in the project root. Variables are grouped by subsystem.

### Default Model

```bash
DEFAULT_MODEL=llama3.2              # Model used for agent chat/reasoning
```

### LM Studio (Local Models)

```bash
LMSTUDIO_BASE_URL=http://localhost:1234/v1   # OpenAI-compatible endpoint
LMSTUDIO_TIMEOUT=600                         # Request timeout (seconds)
HF_TOKEN=                                    # HuggingFace token (gated models)
```

### Cloud Providers

```bash
ANTHROPIC_API_KEY=                  # Anthropic (Claude models)
OPENAI_API_KEY=                     # OpenAI (GPT models)
TOGETHER_API_KEY=                   # Together.ai (hosted open models)
```

All three can also be configured at runtime via the Settings UI or `POST /api/config/update`.

### Database Credentials

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme

# PostgreSQL (with pgvector)
POSTGRES_USER=agent
POSTGRES_PASSWORD=changeme
POSTGRES_DB=agent_memory
POSTGRES_URI=postgresql://agent:changeme@localhost:5432/agent_memory

# Redis
REDIS_URI=redis://localhost:6379
```

These must match the values in `docker-compose.yml`.

### Embeddings

```bash
EMBEDDING_PROVIDER=local                          # "local" or "openai"
EMBEDDING_MODEL=text-embedding-3-small            # OpenAI model (if provider=openai)
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3                    # Local model (if provider=local)
```

Switching providers changes vector dimensions. If you switch after data exists, reset memory schemas (`task db:init:schemas`) or use `POST /api/memory/reset`.

### Django / Application

```bash
DJANGO_SECRET_KEY=your-secret-key-here   # Generate for production
DJANGO_DEBUG=true                        # false in production
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
API_PORT=12319
```

### Security

```bash
AGENTX_MAX_TEXT_LENGTH=100000        # Max input length (translation)
AGENTX_MAX_CHAT_LENGTH=10000         # Max input length (chat)
AGENTX_RATE_LIMIT_ENABLED=false      # Enable rate limiting
AGENTX_RATE_LIMIT_DEFAULT=100/m      # Default rate limit
```

### Authentication (Phase 17)

```bash
AGENTX_AUTH_ENABLED=false            # Enable session-based authentication
AGENTX_SESSION_TTL=86400             # Session TTL in seconds (default: 24h)
```

To set up authentication:
1. Set `AGENTX_AUTH_ENABLED=true`
2. Run `task auth:setup` to create the root password
3. Login via the client UI or POST to `/api/auth/login`

Client IP is tracked on all requests via `request.agentx_client_ip` for rate-limiting and auditing.

### Client

```bash
VITE_API_URL=http://localhost:12319/api   # API URL for Tauri client
```

---

## Runtime Config

`data/config.json` stores settings that can be changed without restarting the server. Managed by `ConfigManager` (singleton).

### Structure

```json
{
  "providers": {
    "lmstudio": { "base_url": null, "timeout": 300 },
    "anthropic": { "api_key": null, "base_url": null },
    "openai": { "api_key": null, "base_url": null }
  },
  "models": {
    "defaults": { "chat": null, "reasoning": null, "extraction": null },
    "overrides": {}
  },
  "llm_settings": {
    "default_temperature": 0.7,
    "default_max_tokens": 4096,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  },
  "preferences": {
    "default_model": null,
    "default_reasoning_strategy": "auto",
    "enable_memory_by_default": true
  }
}
```

### Priority

Runtime config takes priority over environment variables. The `ConfigManager.get_provider_value()` method checks: config file value → env var → default.

### API

```bash
# Read current config
curl http://localhost:12319/api/config

# Update a value (dot-notation keys)
curl -X POST http://localhost:12319/api/config/update \
  -H "Content-Type: application/json" \
  -d '{"key": "providers.lmstudio.base_url", "value": "http://localhost:1234/v1"}'
```

---

## MCP Server Config

`mcp_servers.json` defines external MCP tool servers. See `mcp_servers.json.example` for the full format.

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "transport": "stdio",
      "timeout": 30.0,
      "auto_reconnect": true
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      },
      "transport": "stdio"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Executable for stdio transport |
| `args` | list | Command arguments |
| `env` | dict | Environment variables (`${VAR}` syntax resolved from process env) |
| `transport` | string | `"stdio"`, `"sse"`, or `"streamable_http"` |
| `url` | string | Server URL (for SSE/HTTP transports) |
| `headers` | dict | HTTP headers (env vars resolved in values) |
| `timeout` | float | Connection timeout in seconds |
| `auto_reconnect` | bool | Reconnect on connection loss |

See [MCP Feature Guide](../features/mcp.md) for details on connection modes and tool execution.

---

## Prompt Config

`data/system_prompts.yaml` stores prompt profiles and the global prompt. Managed by `PromptManager` (singleton).

```yaml
global_prompt:
  content: "You are a helpful AI assistant."
  enabled: true

profiles:
  - id: default
    name: Default Assistant
    description: General-purpose AI assistant
    is_default: true
    sections:
      - id: identity
        name: Identity
        type: persona
        content: "You are a helpful AI assistant."
        enabled: true
        order: 0
```

Profiles can be managed via the prompts API endpoints. See [Prompts Feature Guide](../features/prompts.md) for the composition pipeline.

---

## Database Configuration

### Neo4j

Configured in `docker-compose.yml`:

```yaml
services:
  neo4j:
    environment:
      - NEO4J_AUTH=neo4j/your_secure_password
      - NEO4J_server_memory_heap_max__size=2G
      - NEO4J_server_memory_pagecache_size=1G
```

Web browser available at `http://localhost:7474`.

### PostgreSQL

```yaml
services:
  postgres:
    environment:
      - POSTGRES_USER=agent
      - POSTGRES_PASSWORD=your_secure_password
      - POSTGRES_DB=agent_memory
```

The pgvector extension is installed automatically. Django ORM uses a separate SQLite database (`api/db.sqlite3`).

### Redis

```yaml
services:
  redis:
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

---

## Related

- [Development Setup](../development/setup.md) — First-time setup walkthrough
- [MCP](../features/mcp.md) — MCP server configuration details
- [Prompts](../features/prompts.md) — Prompt profile system
- [Providers](../features/providers.md) — Model provider setup

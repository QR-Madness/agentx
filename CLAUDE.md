# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentX is an AI Agent Platform combining:
- **Backend**: Django REST API (`api/`) on port 12319 — translation, agent memory, MCP client, model providers, drafting, reasoning
- **Frontend**: Tauri v2 desktop app (`client/`) with React 19, TypeScript, Vite
- **Data Layer**: Neo4j (graphs), PostgreSQL + pgvector (vectors), Redis (cache) — all via Docker
- **Ignore**: `client-old/` contains a previous Electron implementation and should not be modified

## Architecture

```
Tauri Client (React 19 + Vite)          Django API (port 12319)
  TabBar → Dashboard, Agent,              Agent Core (planner, session, context)
           Translation, Chat,             ├── MCP Client (consume external tool servers)
           Tools, Settings                ├── Reasoning (CoT, ToT, ReAct, Reflection)
                                          ├── Drafting (speculative, pipeline, candidate)
     ↕ HTTP                               ├── Model Providers (OpenAI, Anthropic, Ollama)
                                          ├── Translation Kit (NLLB-200, 200+ languages)
                                          └── Agent Memory (episodic, semantic, procedural, working)
                                                ↕
                                          Neo4j │ PostgreSQL (pgvector) │ Redis
```

### Key Backend Modules (`api/agentx_ai/`)

- `kit/translation.py` — `TranslationKit` (NLLB-200 translation) and `LanguageLexicon` (ISO 639 code bridging between Level I detection codes and Level II translation codes)
- `kit/agent_memory/` — Full memory system with lazy-loaded connections (`interface.py` → `connections.py` → memory implementations)
- `mcp/` — MCP client manager, server registry, tool executor, transports (stdio, SSE). Configure via `mcp_servers.json`
- `providers/` — Abstract `ModelProvider` with OpenAI, Anthropic, Ollama implementations. Models defined in `models.yaml`
- `drafting/` — Speculative decoding, multi-stage pipelines, N-best candidate generation. Strategies in `drafting_strategies.yaml`
- `reasoning/` — CoT, ToT (BFS/DFS/beam), ReAct, Reflection. `orchestrator.py` selects strategy by task type
- `agent/` — `Agent` class orchestrating reasoning + drafting + tools. `TaskPlanner` for decomposition (with goal tracking), `SessionManager` for conversations

### Key Client Patterns (`client/src/`)

- All tabs are always mounted in DOM; visibility toggled via CSS `display` property to preserve state
- Multi-server support: per-server settings in localStorage (`agentx:servers`, `agentx:server:{id}:meta`, `agentx:activeServer`)
- `ServerContext` provides app-wide server state; `lib/api.ts` is the typed API client; `lib/hooks.ts` has React data hooks
- Cosmic dark theme with glassmorphism effects and Lucide-react icons

## Development Commands

All commands use [Task](https://taskfile.dev/) (see `Taskfile.yaml`). Run `task --list-all` for the complete list.

### Setup & Development

```bash
task setup              # First-time: install deps, init DB dirs, verify env
task check              # Verify environment is ready
task dev                # Start Docker + API + Client concurrently (full stack)
task dev:api            # API server only (assumes Docker running)
task dev:client         # Tauri client only (assumes API running)
task dev:web            # Client in browser mode (port 1420, no Tauri)
task install            # Install all deps (uv sync + bun install)
```

### Database Services (Docker)

```bash
task db:up              # Start Neo4j, PostgreSQL, Redis (aliases: runners)
task db:down            # Stop services (aliases: teardown)
task db:status          # Show container status
task db:init            # Create local data directories (data/neo4j, data/postgres, data/redis)
task db:init:schemas    # Initialize memory system schemas (Neo4j indexes, PostgreSQL tables)
task db:verify:schemas  # Verify schemas exist (read-only check)
task db:shell:postgres  # psql shell into agent-postgres
task db:shell:redis     # redis-cli into agent-redis
task db:shell:neo4j     # cypher-shell into agent-neo4j
```

### Testing

```bash
task test               # Run all backend tests (slow — loads translation models)
task test:quick         # Run tests that don't require model loading (HealthCheck, MCP tests)

# Run a specific test class or method:
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest -v2
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest.test_translate_to_french -v2
```

Test categories in `api/agentx_ai/tests.py`:
- `TranslationKitTest` — Requires HuggingFace models to be downloaded (slow first run)
- `HealthCheckTest` — API structure tests; `test_health_with_memory_check` auto-skips if Docker services aren't running
- `MCPClientTest`, `MCPServerRegistryTest` — Unit tests for MCP infrastructure (no external dependencies)

The `DJANGO_SETTINGS_MODULE` env var is set automatically by the Taskfile (`agentx_api.settings`).

### Linting & Formatting

```bash
task lint               # Run all linters (Python + Client)
task lint:python        # Lint Python with ruff: uv run ruff check api/
task lint:python:fix    # Auto-fix Python lint issues
task format             # Format all code
task format:python      # Format Python with ruff: uv run ruff format api/
```

### Django Commands

```bash
task api:run            # Run dev server (aliases: api:runserver)
task api:migrate        # Apply migrations
task api:makemigrations # Create new migrations
task api:shell          # Django interactive shell
```

### Build & Release

```bash
task client:build       # Build Tauri app for production
task client:build:web   # Build web assets only (TypeScript check + Vite build)
task release:check      # Verify release readiness (clean tree, tests, TS compile)
task models:download    # Pre-download HuggingFace models (NLLB-200, language detection)
```

## API Endpoints

Base URL: `http://localhost:12319/api/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (add `?include_memory=true` for DB status) |
| `/api/tools/language-detect-20` | GET/POST | Detect language of text |
| `/api/tools/translate` | POST | Translate text (`{"text": "...", "targetLanguage": "fra_Latn"}`) |
| `/api/mcp/servers` | GET | List configured MCP servers |
| `/api/mcp/tools` | GET | List available MCP tools |
| `/api/mcp/resources` | GET | List available MCP resources |
| `/api/providers` | GET | List configured model providers |
| `/api/providers/models` | GET | List available models (filter: `?provider=openai`) |
| `/api/providers/health` | GET | Check health of all providers |
| `/api/agent/run` | POST | Execute a task with the agent |
| `/api/agent/chat` | POST | Conversational interaction with session |
| `/api/agent/status` | GET | Get current agent status |

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:

```bash
NEO4J_PASSWORD=...          # Used by docker-compose and agent_memory
POSTGRES_PASSWORD=...       # Used by docker-compose and agent_memory
OPENAI_API_KEY=sk-...       # For OpenAI provider
ANTHROPIC_API_KEY=sk-ant-...# For Anthropic provider
EMBEDDING_PROVIDER=local    # "openai" or "local" (sentence-transformers)
```

MCP servers are configured in `mcp_servers.json` (see `mcp_servers.json.example`).

## Important Technical Details

- **Translation models load eagerly** at `TranslationKit` init, not lazily. First request downloads models from HuggingFace (~600MB for NLLB-200).
- **Memory system is lazy**: Database connections (Neo4j, PostgreSQL, Redis) are created on first use, not at import time. Config via pydantic-settings from `.env`.
- **Docker data is bind-mounted** to `./data/` (not Docker volumes). Run `task db:init` to create the directory structure.
- **Tauri dev server** runs Vite on port 1420 with HMR on port 1421. Main window config in `client/src-tauri/tauri.conf.json`.
- **Python managed by uv**, client packages by **bun**. The `task dev` command uses globally-installed `concurrently` (installed via `task install`).

## Agent Memory Interface

The `AgentMemory` class (`kit/agent_memory/memory/interface.py`) provides a unified API for memory operations:

### Core Methods
| Method | Description |
|--------|-------------|
| `store_turn(turn)` | Store conversation turn in episodic + working memory |
| `remember(query, top_k)` | Retrieve relevant memories (turns, facts, entities, strategies) |
| `learn_fact(claim, source, confidence)` | Store factual knowledge in semantic memory |
| `upsert_entity(entity)` | Create/update entity in semantic memory |
| `record_tool_usage(...)` | Record tool invocation for procedural learning |
| `reflect(outcome)` | Trigger async consolidation job |

### Goal Tracking Methods
| Method | Description |
|--------|-------------|
| `add_goal(goal)` | Create a goal linked to user |
| `get_goal(goal_id)` | Retrieve goal by ID |
| `complete_goal(goal_id, status, result)` | Update goal status ('completed', 'abandoned', 'blocked') |
| `get_active_goals()` | Get all active goals for user |

### TaskPlanner Integration
`TaskPlanner.plan()` accepts an optional `memory` parameter. When provided:
- Creates a `Goal` for the main task on plan creation
- `TaskPlan.goal_id` stores the linked goal ID
- `Agent.run()` calls `complete_goal()` on task success/failure

## Project Status

Phases 1-10 complete. Phase 11 (Memory System) in progress — 11.1-11.2 complete. See `Todo.md` for detailed tracking.

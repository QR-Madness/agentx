# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentX is an AI Agent Platform combining:
- **Backend**: Django REST API (`api/`) on port 12319 — translation, agent memory, MCP client, model providers, drafting, reasoning
- **Frontend**: Tauri v2 desktop app (`client/`) with React 19, TypeScript, Vite
- **Data Layer**: Neo4j (graphs), PostgreSQL + pgvector (vectors), Redis (cache) — all via Docker

## Terminology

| Term | Meaning |
|------|---------|
| **Agent Profile** / **Profile** | Configuration that defines an agent's identity, behavior, and settings. These are the primary configuration entity — they produce "agents" when used. |
| **Global Settings** | Settings that apply across all agents (e.g., server connections, API keys, MCP tools) |
| **Profile Settings** | Per-agent settings (e.g., model, temperature, system prompt, reasoning strategy, memory channel) |

Agent profiles configure: name, avatar, default model, temperature, system prompt, reasoning strategy, memory enable/channel, and tool enable. The global prompt has no agent name — agent names are injected from the profile during prompt composition.

## Architecture

```
Tauri Client (React 19 + Vite)          Django API (port 12319)
  TopBar → Start, Dashboard, AgentX       Agent Core (planner, session, context)
  ConversationTabs (browser-style)        ├── MCP Client (consume external tool servers)
  Drawers: Settings, Memory, Tools        ├── Reasoning (CoT, ToT, ReAct, Reflection)
  Modals: Translation, Prompt Library     ├── Drafting (speculative, pipeline, candidate)
     ↕ HTTP                               ├── Model Providers (LM Studio, Anthropic, OpenAI)
                                          ├── Context Gating (compression, chunking, retrieval)
                                          ├── Translation Kit (NLLB-200, 200+ languages)
                                          └── Agent Memory (episodic, semantic, procedural, working)
                                                ↕
                                          Neo4j │ PostgreSQL (pgvector) │ Redis
```

### Key Backend Modules (`api/agentx_ai/`)

- `kit/translation.py` — `TranslationKit` (NLLB-200 translation) and `LanguageLexicon` (ISO 639 code bridging between Level I detection codes and Level II translation codes)
- `kit/agent_memory/` — Full memory system with lazy-loaded connections (`interface.py` → `connections.py` → memory implementations). `RecallLayer` provides 5 retrieval techniques (hybrid, entity-centric, query expansion, HyDE, self-query)
- `mcp/` — MCP client manager, server registry, tool executor, transports (stdio, SSE). Configure via `mcp_servers.json`
- `providers/` — Abstract `ModelProvider` with LM Studio, Anthropic, OpenAI implementations. Models defined in `models.yaml`
- `prompts/` — `PromptManager` singleton for profile-based prompt composition. `models.py` defines PromptProfile, PromptSection, GlobalPrompt. Config in `data/system_prompts.yaml`
- `config.py` — `ConfigManager` singleton for runtime settings. Persists to `data/config.json` with dot-notation access and env var fallback
- `drafting/` — Speculative decoding, multi-stage pipelines, N-best candidate generation. Strategies in `drafting_strategies.yaml`
- `reasoning/` — CoT, ToT (BFS/DFS/beam), ReAct, Reflection. `orchestrator.py` selects strategy by task type
- `agent/` — `Agent` class orchestrating reasoning + drafting + tools. `TaskPlanner` for decomposition (with goal tracking), `SessionManager` for conversations
- `agent/profiles.py` — `ProfileManager` for agent profile CRUD. Profiles stored in `data/agent_profiles.yaml`. Each profile has a Docker-style `agent_id` (e.g., "bold-cosmic-falcon") for identity and a `self_channel` (`_self_{agent_id}`) for self-knowledge
- `agent/tool_output_compressor.py` — LLM-based task-aware compression for oversized tool outputs
- `agent/tool_output_chunker.py` — Section detection, JSON path queries, semantic search over stored tool outputs
- `streaming/trajectory_compression.py` — Focus-style intra-trajectory compression for multi-round tool loops

### Key Client Patterns (`client/src/`)

- 3-page layout: Start, Dashboard, AgentX — routed via `RootLayout` + `TopBar`
- Browser-style conversation tabs via `ConversationContext` (add, close, switch, rename, reorder)
- Former tabs (Settings, Memory, Tools) → right-side drawer panels from toolbar icons
- Multi-server support: per-server settings in localStorage (`agentx:servers`, `agentx:server:{id}:meta`, `agentx:activeServer`)
- `ServerContext` provides app-wide server state; `lib/api.ts` is the typed API client; `lib/hooks.ts` has React data hooks
- `AgentProfileContext` manages agent profiles (name, model, temperature, reasoning strategy, memory channel)
- Cosmic dark theme with glassmorphism effects and Lucide-react icons

## Development Commands

All commands use [Task](https://taskfile.dev/) (see `Taskfile.yml`). Run `task --list-all` for the complete list.

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

Test files:
- `tests.py` — Core tests (TranslationKit, HealthCheck, MCP, Extraction)
- `tests_memory.py` — Phase 11.8+ memory system tests (80 tests covering security, integration, edge cases)

Test categories:
- `TranslationKitTest` — Requires HuggingFace models (slow first run)
- `HealthCheckTest` — API structure tests; auto-skips if Docker not running
- `MCPClientTest`, `MCPServerRegistryTest` — MCP infrastructure (no external dependencies)
- `ExtractionPipelineTest` — Memory extraction; skips without API keys
- `ToolOutputCompressorTest`, `ToolOutputChunkerTest` — Context gating (no external dependencies)
- `IntentAwareRetrievalTest`, `TrajectoryCompressionTest`, `ContextGateTest` — Compression pipeline
- `AgentSelfMemoryTest` — Agent identity, self-channel, confidence calibration
- `FactVerificationPipelineTest` — Three-layer fact verification (temporal progression, config, metrics)
- Memory integration tests — Skip gracefully when Docker not running or embedding dimensions mismatch

The `DJANGO_SETTINGS_MODULE` env var is set automatically by the Taskfile (`agentx_api.settings`).

### Linting & Formatting

```bash
task lint               # Run all linters (Python + Client)
task lint:python        # Lint Python with ruff: uv run ruff check api/
task lint:python:fix    # Auto-fix Python lint issues
task format             # Format all code
task format:python      # Format Python with ruff: uv run ruff format api/
```

### Static Analysis & Type Checking

```bash
task check:static       # Run all static analysis (lint + types + build)
task check:types        # Run all type checkers (Python + TypeScript)
task check:types:python # Type check Python with pyright
task check:types:client # Type check TypeScript
task check:build        # Verify both API and client build successfully
task check:build:api    # Verify Python imports and Django configuration
task check:build:client # Build client web assets (TypeScript + Vite)
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
| `/api/mcp/servers` | GET | List configured MCP servers and connection status |
| `/api/mcp/tools` | GET | List available MCP tools (filter: `?server=name`) |
| `/api/mcp/resources` | GET | List available MCP resources (filter: `?server=name`) |
| `/api/mcp/connect` | POST | Connect to MCP server (`{"server": "name"}` or `{"all": true}`) |
| `/api/mcp/disconnect` | POST | Disconnect MCP server (`{"server": "name"}` or `{"all": true}`) |
| `/api/providers` | GET | List configured model providers |
| `/api/providers/models` | GET | List available models (filter: `?provider=openai`) |
| `/api/providers/health` | GET | Check health of all providers |
| `/api/agent/run` | POST | Execute a task with the agent |
| `/api/agent/chat` | POST | Conversational interaction with session |
| `/api/agent/chat/stream` | POST | Streaming chat via SSE (events: start, chunk, tool_call, tool_result, done, error) |
| `/api/agent/status` | GET | Get current agent status |
| `/api/prompts/profiles` | GET | List all prompt profiles |
| `/api/prompts/profiles/{id}` | GET | Get profile detail with composed preview |
| `/api/prompts/global` | GET | Get global prompt |
| `/api/prompts/global/update` | POST | Update global prompt |
| `/api/prompts/sections` | GET | List all prompt sections |
| `/api/prompts/compose` | GET | Preview composed system prompt |
| `/api/prompts/mcp-tools` | GET | Get auto-generated MCP tools prompt |
| `/api/memory/channels` | GET | List memory channels |
| `/api/memory/entities` | GET | List entities (filter: `?channel=`, `?type=`) |
| `/api/memory/entities/graph` | GET | Get entity relationship graph |
| `/api/memory/facts` | GET | List facts (filter: `?channel=`, `?entity_id=`) |
| `/api/memory/strategies` | GET | List procedural strategies |
| `/api/memory/stats` | GET | Memory system statistics |
| `/api/memory/recall` | POST | Recall relevant memories for a query |
| `/api/memory/recall-settings` | GET/POST | Get or update recall layer settings |
| `/api/memory/consolidate` | POST | Trigger manual consolidation |
| `/api/memory/reset` | POST | Reset memory data (with confirmation) |
| `/api/memory/settings` | GET/POST | Get or update memory settings |
| `/api/jobs` | GET | List background jobs |
| `/api/jobs/clear-stuck` | POST | Clear stuck jobs |
| `/api/jobs/{id}` | GET | Get job detail |
| `/api/jobs/{id}/run` | POST | Manually run a job |
| `/api/jobs/{id}/toggle` | POST | Enable/disable a job |
| `/api/config` | GET | Get runtime configuration |
| `/api/config/update` | POST | Update runtime config (`{"key": "dot.path", "value": ...}`) |

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

### Extraction Pipeline
The `ExtractionService` (`kit/agent_memory/extraction/service.py`) provides LLM-based extraction:
- `check_relevance_and_extract(text)` — Combined relevance check + extraction in single LLM call (~75% fewer calls)
- `check_relevance_and_extract_assistant(text)` — Self-knowledge extraction from agent's own responses (certainty: definitive/analytical/speculative)
- `check_contradictions(claim, facts)` — Three-layer fact verification: hash gate → semantic duplicate → entity-scoped candidates → LLM adjudication
- Uses `nvidia/nemotron-3-nano` by default for extraction (configurable via `combined_extraction_model` setting)
- Consolidation jobs (`consolidation/jobs.py`) call extractors every 15 minutes
- Gracefully degrades to empty results when no provider is configured

### Context Gating
Large tool outputs (>12K chars) are compressed and stored for retrieval:
- `ToolOutputCompressor` — task-aware LLM compression with structure indexing
- `tool_output_chunker.py` — section detection, JSON path queries, semantic search
- Internal MCP tools: `read_stored_output`, `tool_output_query`, `tool_output_section`, `tool_output_path`
- Trajectory compression — Focus-style intra-trajectory compression at 75% context threshold
- Retrieval tool bypass prevents re-storage loops

### Agent Identity
Each agent profile has a Docker-style `agent_id` (e.g., "bold-cosmic-falcon"):
- Auto-generated, immutable, adjective-adjective-noun format (~83K combinations)
- `self_channel` property: `_self_{agent_id}` for agent's self-knowledge
- Recall searches: `[active_channel, _self_{agent_id}, _global]`
- Agent self-extraction during consolidation stores knowledge to self-channel

## Project Status

Phases 1-14 complete. Phase 15 (Plan Execution + Memory Tuning) and Phase 16 (Multi-Agent Conversations) not started. See `Todo.md` for detailed tracking.

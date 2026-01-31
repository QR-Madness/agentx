# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentX is an AI Agent Platform combining:
- **Backend**: Django REST API providing AI-powered language translation, detection, and agent services
- **Frontend**: Tauri desktop application with React/TypeScript UI and Vite build system
- **AI Features**: Multi-level language detection/translation, MCP client integration, drafting models, reasoning framework
- **Memory System**: Persistent agent memory with Neo4j, PostgreSQL (pgvector), and Redis

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentX Desktop App                          │
├─────────────────────────────────────────────────────────────────────┤
│  Tauri Client (React 19 + Vite)     │    Django API (port 12319)   │
│  - Dashboard, Translation, Chat     │    - Translation Kit          │
│  - Tools tabs                       │    - Agent Memory System      │
│                                     │    - MCP Client (planned)     │
├─────────────────────────────────────────────────────────────────────┤
│                           Data Layer                                 │
│   Neo4j (graphs)  │  PostgreSQL (pgvector)  │  Redis (cache)        │
└─────────────────────────────────────────────────────────────────────┘
```

1. **API Layer** (`api/` directory)
   - Django 5.2.8 application running on port 12319
   - Main app: `agentx_ai` with endpoints for translation, language detection, and health
   - Agent memory system with episodic, semantic, procedural, and working memory
   - MCP client for consuming external tool servers (planned)
   - Drafting and reasoning frameworks for multi-model AI workflows (planned)

2. **Client Layer** (`client/` directory)
   - Tauri v2 desktop app with Rust backend
   - React 19 with TypeScript for UI
   - Vite build system for fast development
   - Tab-based navigation: Dashboard, Translation, Chat, Tools
   - Communicates with Django API

3. **Data Layer** (Docker services via `docker-compose.yml`)
   - Neo4j: Graph database for knowledge graphs and relationships
   - PostgreSQL + pgvector: Relational storage with vector search
   - Redis: Caching and working memory
   - Start with: `task runners` (or `docker-compose up -d`)

**Note**: The `client-old/` directory contains the previous Electron implementation and can be ignored.

### Translation System Architecture

The translation system implements a two-level approach:

**Level I**: Fast language detection (~20 languages)
- Model: `eleldar/language-detection`
- Used for: Initial language detection with confidence scores
- Returns ISO 639-1 codes (e.g., "en", "fr")

**Level II**: Comprehensive translation (200+ languages)
- Model: `facebook/nllb-200-distilled-600M`
- Used for: Multi-language translation via NLLB-200 architecture
- Uses extended ISO 639 codes with script info (e.g., "eng_Latn", "fra_Latn")

The `LanguageLexicon` class bridges Level I and Level II by converting between ISO 639 code formats using the `python-iso639` library.

### Key Components

**API Kit System** (`api/agentx_ai/kit/`)
- `translation.py`: Contains `TranslationKit` and `LanguageLexicon` classes
- `memory_utils.py`: Lazy-loading memory initialization and health checks
- `conversation.py`: Placeholder for conversation management
- `agent_memory/`: Full memory system package
  - `interface.py`: Main `AgentMemory` class
  - `connections.py`: Neo4j, PostgreSQL, Redis connection managers (lazy-loaded)
  - `models.py`: Pydantic models for Turn, Entity, Fact, Goal, Strategy
  - `memory/`: Episodic, semantic, procedural, working memory implementations

**MCP Client** (`api/agentx_ai/mcp/`)
- `client.py`: `MCPClientManager` for managing server connections
- `server_registry.py`: `ServerRegistry` and `ServerConfig` for configuration
- `tool_executor.py`: `ToolExecutor` for executing tools on connected servers
- `transports/`: Transport implementations (stdio, SSE)
- Configure servers in `mcp_servers.json` (see `mcp_servers.json.example`)

**Model Providers** (`api/agentx_ai/providers/`)
- `base.py`: Abstract `ModelProvider` interface with `Message`, `CompletionResult`, `StreamChunk`
- `openai_provider.py`: OpenAI API provider (GPT-4, GPT-4-turbo, GPT-3.5)
- `anthropic_provider.py`: Anthropic API provider (Claude 3 Opus/Sonnet/Haiku)
- `ollama_provider.py`: Ollama provider for local models (Llama, Mistral, etc.)
- `registry.py`: `ProviderRegistry` for managing providers and model configurations
- `models.yaml`: Model definitions and capabilities

**Drafting Framework** (`api/agentx_ai/drafting/`)
- `base.py`: `DraftingStrategy`, `DraftResult` base classes
- `speculative.py`: Speculative decoding (fast draft + accurate verify)
- `pipeline.py`: Multi-stage model pipelines
- `candidate.py`: N-best candidate generation with scoring
- `drafting_strategies.yaml`: Pre-configured strategy definitions

**Reasoning Framework** (`api/agentx_ai/reasoning/`)
- `base.py`: `ReasoningStrategy`, `ThoughtStep` base classes
- `chain_of_thought.py`: CoT reasoning (zero-shot, few-shot)
- `tree_of_thought.py`: ToT with BFS/DFS/beam search
- `react.py`: ReAct pattern (Thought → Action → Observation)
- `reflection.py`: Self-critique and revision
- `orchestrator.py`: Strategy selection based on task type

**Agent Core** (`api/agentx_ai/agent/`)
- `core.py`: Main `Agent` class orchestrating all capabilities
- `planner.py`: `TaskPlanner` for task decomposition
- `session.py`: `Session` and `SessionManager` for conversation state
- `context.py`: `ContextManager` for context window management

**Client Tabs** (`client/src/components/tabs/`)
- Each tab is a separate React component (DashboardTab, TranslationTab, ChatTab, ToolsTab)
- Tab switching handled by `App.tsx` state management
- All tabs remain mounted to preserve state (visibility controlled via CSS)

## Development Commands

### Running the Application

Use Task (Taskfile.yaml) for all development operations:

```bash
# Start Docker services + API + Client in development mode
task dev

# Start only Docker services (Neo4j, PostgreSQL, Redis)
task runners

# Stop Docker services
task teardown

# API only
task api:runserver          # Starts Django server on port 12319

# Client only (Tauri dev mode)
task client:dev             # Or: cd client && bun run tauri dev

# Install all dependencies
task install                # Runs: uv sync && bun install
```

### Django API Commands

```bash
# Database operations
task api:migrate
task api:makemigrations

# Django shell
task api:shell

# Direct command (if needed)
cd api && uv run python manage.py runserver 127.0.0.1:12319
```

### Tauri Client Commands

```bash
# Development (starts Vite dev server + Tauri window)
cd client && bun run tauri dev

# Build distributable packages
cd client && bun run tauri build

# Vite-only development (browser preview, no Tauri)
cd client && bun run dev          # Runs on localhost:1420
cd client && bun run build        # TypeScript check + Vite build
cd client && bun run preview      # Preview production build
```

### Testing

```bash
# Run all tests
task test

# Run specific test class
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest

# Run specific test
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest.test_translate_to_french

# Run health check tests
uv run python api/manage.py test agentx_ai.tests.HealthCheckTest
```

## Important Technical Details

### API Endpoints

Base URL: `http://localhost:12319/api/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/index` | GET | Simple hello message |
| `/api/health` | GET | Health check (add `?include_memory=true` for DB status) |
| `/api/tools/language-detect-20` | GET/POST | Detect language of text |
| `/api/tools/translate` | POST | Translate text to target language |
| `/api/mcp/servers` | GET | List configured MCP servers |
| `/api/mcp/tools` | GET | List available MCP tools |
| `/api/mcp/resources` | GET | List available MCP resources |
| `/api/providers` | GET | List configured model providers |
| `/api/providers/models` | GET | List all available models (add `?provider=openai` to filter) |
| `/api/providers/health` | GET | Check health of all providers |
| `/api/agent/run` | POST | Execute a task with the agent |
| `/api/agent/chat` | POST | Conversational interaction with session |
| `/api/agent/status` | GET | Get current agent status |

**Language Detection:**
```json
POST /api/tools/language-detect-20
{"text": "Bonjour, comment allez-vous?"}

Response: {"original": "...", "detected_language": "fr", "confidence": 98.5}
```

**Translation:**
```json
POST /api/tools/translate
{"text": "Hello, world!", "targetLanguage": "fra_Latn"}

Response: {"original": "...", "translatedText": "Bonjour, monde!"}
```

**Health Check:**
```json
GET /api/health?include_memory=true

Response: {
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

### Translation Model Loading

The `TranslationKit` class in `api/agentx_ai/kit/translation.py`:
- Loads models at initialization (not lazy-loaded)
- Uses NLLB-200-distilled-600M for translation
- First request may be slow while models download from HuggingFace

### Memory System

The agent memory system in `api/agentx_ai/kit/agent_memory/`:
- **Lazy initialization**: Connections created on first use, not at import time
- **Health checks**: `check_memory_health()` in `memory_utils.py`
- **Configuration**: `agent_memory/config.py` using pydantic-settings (loads from `.env`)

Database connections are managed by:
- `Neo4jConnection`: Graph database for entities and relationships
- `PostgresConnection`: Relational storage with pgvector
- `RedisConnection`: Working memory and caching

### Environment Configuration

Copy `.env.example` to `.env` and configure:
```bash
# Database credentials
NEO4J_PASSWORD=your_password
POSTGRES_PASSWORD=your_password

# API keys for model providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Embedding provider: "openai" or "local"
EMBEDDING_PROVIDER=local
```

### Tauri Configuration

- Main window config: `client/src-tauri/tauri.conf.json`
  - Window dimensions: 800x600
  - Dev URL: http://localhost:1420
  - Frontend build output: `../dist`
- Rust dependencies: `client/src-tauri/Cargo.toml`
  - tauri v2 with opener plugin
  - serde for serialization
- Vite config: `client/vite.config.ts`
  - Dev server port: 1420 (strict)
  - HMR port: 1421

### Client Architecture

**Tab Management Pattern:**
- All tabs are always mounted in the DOM
- Only one tab visible at a time (controlled by `display` CSS property)
- This preserves component state when switching between tabs
- State management done in `App.tsx` via `useState` hook

**File Structure:**
- `client/src/App.tsx` - Main app component with tab routing
- `client/src/components/TabBar.tsx` - Tab navigation UI
- `client/src/components/tabs/*.tsx` - Individual tab components
- `client/src-tauri/` - Rust/Tauri backend code

## Python Dependencies

Managed via `pyproject.toml` with uv:

**Core:**
- django>=5.2.8, django-cors-headers>=4.6.0

**AI/ML:**
- torch>=2.9.1, transformers>=4.57.1, sentencepiece>=0.2.1
- sentence-transformers>=2.2.0 (local embeddings)

**Databases:**
- neo4j>=6.0.3, sqlalchemy>=2.0.0, psycopg2-binary>=2.9.0
- redis>=5.0.0, faiss-cpu>=1.13.0

**MCP & Model Providers:**
- mcp>=1.0.0 (Model Context Protocol client)
- openai>=1.0.0, anthropic>=0.20.0, httpx>=0.27.0

**Utilities:**
- pydantic>=2.12.5, pydantic-settings>=2.12.0
- python-iso639>=2025.11.16, networkx>=3.0

## Project Roadmap

See `Todo.md` for detailed task tracking. Current status:
- ✅ Phase 1: Critical fixes (dependencies, Taskfile, OpenAPI)
- ✅ Phase 2: Wire up existing code (health endpoint, lazy loading)
- ✅ Phase 3: MCP Client integration
- 🔲 Phase 4: Model provider abstraction
- 🔲 Phase 5: Drafting models framework
- 🔲 Phase 6: Reasoning framework
- 🔲 Phase 7: Agent core

## Known Issues & TODOs

From `Todo.md` and `api/agentx_ai/tests.py`:
- MCP client integration (Phase 3)
- Model provider abstraction for OpenAI/Anthropic/Ollama (Phase 4)
- Drafting models: speculative decoding, pipelines, candidates (Phase 5)
- Reasoning: CoT, ToT, ReAct, reflection patterns (Phase 6)

## Migration Notes

The project recently migrated from Electron to Tauri:
- Old Electron code is in `client-old/` directory (can be ignored)
- New Tauri implementation is in `client/` directory
- Tauri provides smaller binary sizes and better security model
- Frontend stack remains React + TypeScript, but build system changed from Webpack to Vite
- Package manager changed from npm to bun

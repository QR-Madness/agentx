# AgentX Development Todo

**Project**: AgentX - AI Agent Platform with MCP Client, Drafting Models & Reasoning Framework  
**Status**: Pre-prototype  
**Last Updated**: 2026-01-21

---

## Overview

AgentX is an AI agent platform that:
1. **Consumes MCP tools** from external servers (filesystem, GitHub, databases, custom servers)
2. **Experiments with drafting models** (speculative decoding, multi-model pipelines, candidate generation)
3. **Implements flexible reasoning patterns** (CoT, ToT, ReAct, reflection/self-critique)
4. **Provides memory-augmented AI** with persistent knowledge graphs

### Current State
- ✅ Translation system functional (200+ languages via NLLB-200)
- ✅ Tauri client with React UI operational
- ✅ Docker services configured (Neo4j, PostgreSQL, Redis)
- ⚠️ Agent memory system designed but not connected
- ⚠️ Several dependency and configuration issues
- ❌ MCP client not yet implemented
- ❌ Drafting/reasoning framework not yet implemented
- ❌ Chat/agent functionality not implemented

---

## Phase 1: Critical Fixes ✅

> **Priority**: HIGH  
> **Goal**: Get the codebase into a working baseline state  
> **Status**: COMPLETE

### 1.1 Missing Dependencies
- [x] Add `redis>=5.0.0` to pyproject.toml
- [x] Add `sqlalchemy>=2.0.0` to pyproject.toml
- [x] Add `mcp>=1.0.0` to pyproject.toml (for Phase 3)
- [x] Add `openai>=1.0.0`, `anthropic>=0.20.0`, `httpx>=0.27.0` for model providers
- [x] Add `networkx>=3.0` for reasoning graphs
- [x] Run `uv sync` to regenerate lock file
- [x] Verify all imports resolve without errors

### 1.2 Taskfile Corrections
- [x] Fix `client:build`: change `bunx next build` → `bun run build`
- [x] Fix `client:start`: change `bunx next start` → `bun run preview`  
- [x] Fix `client:dev`: change `bunx tauri run dev` → `bun run tauri dev`
- [x] Remove Next.js references (project uses Vite)
- [ ] Test full `task dev` workflow

### 1.3 OpenAPI Specification
- [x] Update server URL port: `8000` → `12319`
- [x] Document `POST /tools/translate` endpoint with request/response schemas
- [x] Fix language-detect path: `/language-detect` → `/tools/language-detect-20`
- [x] Add proper response schemas for all endpoints
- [x] Add Error component schema

### 1.4 Environment Configuration
- [x] Create `.env.example` with all required variables
- [x] Update `docker-compose.yml` to use environment variables
- [x] Ensure `.env` is in `.gitignore`
- [x] Add redis-commander to optional `tools` profile

### 1.5 Code Quality (Bonus)
- [x] Replace `print()` statements with `logging` in views.py
- [x] Fix `language_detect` to accept POST with text body (backwards compatible with GET)
- [x] Fix test import paths
- [x] Fix test URL paths to match new API structure
- [x] Update CLAUDE.md with correct API endpoints

---

## Phase 2: Wire Up Existing Code ✅

> **Priority**: HIGH  
> **Goal**: Connect the scaffolded code that already exists  
> **Status**: COMPLETE

### 2.1 Language Detection API Fix
- [x] Modify `views.language_detect()` to accept POST (done in Phase 1)
- [x] Remove hardcoded test string (done in Phase 1)
- [x] Add input validation (done in Phase 1)
- [x] Return structured response with language code and confidence

### 2.2 Agent Memory System Connection
- [x] Renamed `agent_memory.py` → `memory_utils.py` to avoid naming conflict with `agent_memory/` package
- [x] Add database connection health check endpoint: `GET /api/health`
- [x] Create lazy-loading `get_agent_memory()` function
- [x] Make PostgreSQL connection lazy (PostgresConnection class)
- [x] Add `check_memory_health()` for connection status

### 2.3 Code Quality Improvements
- [x] Replace all `print()` statements with `logging`
- [x] Add logging configuration to Django settings
- [x] Add `psycopg2-binary` for PostgreSQL connectivity
- [x] Add `sentence-transformers` for local embeddings
- [x] Add health check tests

---

## Phase 3: MCP Client Integration ✅

> **Priority**: HIGH  
> **Goal**: Enable AgentX to consume tools from external MCP servers  
> **Status**: COMPLETE

### 3.1 MCP Client Infrastructure
- [x] Create module structure:
  ```
  api/agentx_ai/mcp/
  ├── __init__.py
  ├── client.py           # MCP client manager
  ├── server_registry.py  # Track connected MCP servers
  ├── tool_executor.py    # Execute tools on remote servers
  └── transports/
      ├── __init__.py
      ├── stdio.py        # stdio transport (subprocess)
      └── sse.py          # SSE transport (HTTP)
  ```
- [x] Implement `MCPClientManager` class:
  - [x] `connect_server(server_config)` → async context manager with ServerConnection
  - [x] `list_tools(server_name?)` → available tools
  - [x] `call_tool(server_name, tool_name, args)` → ToolResult
  - [x] `list_resources(server_name?)` → available resources
  - [x] `read_resource(server_name, uri)` → content

### 3.2 Server Configuration
- [x] Create `mcp_servers.json.example` configuration format
- [x] Add Taskfile commands: `task mcp:list-servers`, `task mcp:list-tools`
- [x] Add API endpoints: `/api/mcp/servers`, `/api/mcp/tools`, `/api/mcp/resources`
- [ ] Add UI for managing MCP server connections (deferred to Phase 8)

### 3.3 Standard MCP Server Support
- [ ] Test with `@modelcontextprotocol/server-filesystem` (configured in example)
- [ ] Test with `@modelcontextprotocol/server-github` (configured in example)
- [ ] Test with `@modelcontextprotocol/server-postgres` (configured in example)
- [ ] Test with `@modelcontextprotocol/server-brave-search` (configured in example)
- [ ] Document tested/supported servers (deferred)

### 3.4 Custom MCP Server Development
- [ ] Create template for custom MCP servers (deferred)
- [ ] Build example: AgentX Memory MCP Server (deferred)
- [ ] Build example: Translation MCP Server (deferred)

### 3.5 Tool Discovery & Caching
- [x] Cache tool schemas on connection (in ToolExecutor)
- [x] Handle server disconnection gracefully (via context managers)
- [ ] Refresh tool list on demand
- [ ] Add tool search/filter functionality

---

## Phase 4: Model Provider Abstraction ✅

> **Priority**: HIGH  
> **Goal**: Support multiple LLM backends with unified interface
> **Status**: COMPLETE

### 4.1 Provider Interface
- [x] Create abstract `ModelProvider` interface:
  ```python
  class ModelProvider(ABC):
      @abstractmethod
      async def complete(self, messages, **kwargs) -> CompletionResult
      
      @abstractmethod
      async def stream(self, messages, **kwargs) -> AsyncIterator[StreamChunk]
      
      @abstractmethod
      def get_capabilities(self) -> ModelCapabilities
  ```

### 4.2 Provider Implementations
- [x] `OpenAIProvider` - GPT-4, GPT-4-turbo, GPT-3.5
- [x] `AnthropicProvider` - Claude 3 Opus/Sonnet/Haiku
- [x] `OllamaProvider` - Local models (Llama, Mistral, etc.)
- [ ] `TogetherProvider` - Together.ai API (deferred)
- [ ] `LocalTransformersProvider` - HuggingFace transformers (deferred)

### 4.3 Model Registry
- [x] Create `models.yaml` configuration
- [x] Model selection based on task requirements (via ProviderRegistry)
- [ ] Cost tracking and budgeting (deferred)

---

## Phase 5: Drafting Models Framework ✅

> **Priority**: HIGH  
> **Goal**: Implement multi-model generation strategies
> **Status**: COMPLETE

### 5.1 Drafting Infrastructure
- [x] Create `drafting/` module:
  ```
  api/agentx_ai/drafting/
  ├── __init__.py
  ├── base.py            # DraftingStrategy, DraftResult
  ├── speculative.py     # Speculative decoding
  ├── pipeline.py        # Multi-model pipelines
  ├── candidate.py       # N-best candidate generation
  └── drafting_strategies.yaml
  ```

### 5.2 Speculative Decoding
- [x] Implement `SpeculativeDecoder`:
  - [x] Draft model generates N tokens quickly
  - [x] Target model verifies/corrects
  - [x] Configurable draft length and acceptance threshold
- [x] Support draft/target model pairs (configurable)

### 5.3 Multi-Model Pipeline
- [x] Implement `ModelPipeline`:
  - [x] Define pipeline stages (analyze → draft → refine → validate)
  - [x] Route to different models per stage
  - [x] Pass context between stages
- [x] Predefined stage roles (ANALYZE, DRAFT, REVIEW, REFINE, etc.)

### 5.4 Candidate Generation
- [x] Implement `CandidateGenerator`:
  - [x] Generate N candidates (same or different models)
  - [x] Scoring/ranking strategies:
    - [x] Self-consistency (majority vote)
    - [x] Verifier model scoring
    - [x] Heuristic scoring (length preference)
  - [x] Best-of-N selection

### 5.5 Drafting Configuration
- [x] Create `drafting_strategies.yaml` with pre-configured strategies

---

## Phase 6: Reasoning Framework ✅

> **Priority**: HIGH  
> **Goal**: Implement flexible reasoning patterns for complex tasks
> **Status**: COMPLETE

### 6.1 Reasoning Infrastructure
- [x] Create `reasoning/` module:
  ```
  api/agentx_ai/reasoning/
  ├── __init__.py
  ├── base.py              # ReasoningStrategy, ThoughtStep
  ├── chain_of_thought.py
  ├── tree_of_thought.py
  ├── react.py             # ReAct pattern
  ├── reflection.py        # Self-critique
  └── orchestrator.py      # Combines strategies
  ```

### 6.2 Chain-of-Thought (CoT)
- [x] Implement `ChainOfThought`:
  - [x] Zero-shot CoT ("Let's think step by step...")
  - [x] Few-shot CoT (with examples)
  - [x] Auto-CoT mode
- [x] Step extraction and validation
- [x] Answer extraction

### 6.3 Tree-of-Thought (ToT)
- [x] Implement `TreeOfThought`:
  - [x] BFS exploration (breadth-first)
  - [x] DFS exploration (depth-first)
  - [x] Beam search (top-k branches)
- [x] State evaluation heuristics
- [x] Pruning strategies
- [x] Tree serialization for traces

### 6.4 ReAct (Reasoning + Acting)
- [x] Implement `ReActAgent`:
  - [x] Thought → Action → Observation loop
  - [x] Tool execution framework
  - [x] Max iterations / termination conditions
- [x] Action parsing and execution
- [x] Built-in tools (search, calculate)

### 6.5 Reflection & Self-Critique
- [x] Implement `ReflectiveReasoner`:
  - [x] Generate initial response
  - [x] Self-critique: identify weaknesses
  - [x] Revise based on critique
  - [x] Configurable reflection depth
- [ ] Implement `DebateReasoner` (deferred)

### 6.6 Reasoning Orchestrator
- [x] Implement `ReasoningOrchestrator`:
  - [x] Select reasoning strategy based on task
  - [x] Task type classification
  - [x] Fallback strategies on failure
- [x] Metrics: steps taken, tokens used, time elapsed

---

## Phase 7: Agent Core ✅

> **Priority**: HIGH  
> **Goal**: Unify MCP, drafting, and reasoning into coherent agent
> **Status**: COMPLETE

### 7.1 Agent Architecture
- [x] Create `agent/` module:
  ```
  api/agentx_ai/agent/
  ├── __init__.py
  ├── core.py            # Main Agent class
  ├── planner.py         # Task decomposition
  ├── context.py         # Context management
  └── session.py         # Conversation session
  ```

### 7.2 Agent Core Implementation
- [x] Implement `Agent` class with:
  - [x] Integration with reasoning orchestrator
  - [x] Integration with drafting strategies
  - [x] Tool registration
  - [x] Status tracking
  - [x] Task cancellation

### 7.3 Task Planning
- [x] Implement `TaskPlanner`:
  - [x] Decompose complex tasks into subtasks
  - [x] Identify subtask types (research, analysis, generation, etc.)
  - [x] Estimate complexity
  - [x] Select reasoning strategy based on plan

### 7.4 Context Management
- [x] Implement `ContextManager`:
  - [x] Token estimation
  - [x] Sliding window for long conversations
  - [x] Summarization of old context
  - [x] Memory injection support

### 7.5 Session Management
- [x] Implement `Session` and `SessionManager`:
  - [x] Message history tracking
  - [x] Session creation/retrieval
  - [x] Session cleanup for inactive sessions

### 7.6 Agent API Endpoints
- [x] `POST /api/agent/run` - Execute a task
- [x] `POST /api/agent/chat` - Conversational interaction
- [x] `GET /api/agent/status` - Check agent status

---

## Phase 8: Client Updates ✅

> **Priority**: MEDIUM  
> **Goal**: Update UI to support new agent capabilities  
> **Status**: COMPLETE

### 8.1 Agent Tab (New)
- [x] Create AgentTab component
- [x] Task input with natural language
- [x] Real-time reasoning trace display
- [x] Tool usage visualization
- [x] Cancel/pause controls

### 8.2 Dashboard Updates
- [x] Connected MCP servers status
- [x] Model provider status (API keys valid, local models loaded)
- [x] Agent status widget
- [x] System health overview

### 8.3 Settings Tab (New)
- [x] Multi-server configuration (per-server settings storage)
- [x] Model provider API key management
- [x] Drafting strategy selection
- [x] Reasoning preferences
- [x] Memory/storage info

### 8.4 Tools Tab Updates
- [x] MCP tool browser (all available tools from connected servers)
- [x] Tool search functionality
- [x] Tool testing interface (placeholder)
- [x] Server status display

### 8.5 Design System Updates
- [x] Dark cosmic theme (nebula gradients, star accents)
- [x] Lucide-react icons throughout
- [x] Glassmorphism effects
- [x] Glow animations
- [x] Updated color palette (purple/cyan/pink cosmic theme)

### 8.6 Infrastructure
- [x] Multi-server storage system (localStorage)
- [x] ServerContext for app-wide server state
- [x] Typed API client with React hooks
- [x] Per-server metadata and preferences

---

## Phase 9: Security & Infrastructure

> **Priority**: MEDIUM  
> **Goal**: Secure and stabilize the platform

### 9.1 Tauri Security
- [ ] Configure Content Security Policy in `tauri.conf.json`
- [ ] Review and restrict Tauri capabilities
- [ ] Disable devtools in production builds (keep them on in 'Preview' builds which will be private builds for testers)

### 9.2 API Security
- [ ] Add rate limiting middleware
- [ ] API key authentication for external access
- [ ] Input validation and sanitization
- [ ] CORS configuration for production

### 9.3 Secrets Management
- [ ] Secure storage for API keys (Keyring/OS keychain)
- [ ] Encryption for sensitive config
- [ ] Audit logging for sensitive operations

---

## Phase 10: Testing

> **Priority**: MEDIUM  
> **Goal**: Ensure reliability through automated testing

### 10.1 Backend Tests
- [ ] Fix and unskip `test_language_detect`
- [ ] Add translation tests:
  - [ ] Test multiple language pairs
  - [ ] Test error handling (invalid language codes)
  - [ ] Test long text handling
- [ ] Add memory system tests:
  - [ ] Test fact storage and retrieval
  - [ ] Test goal lifecycle
  - [ ] Test memory relevance ranking
- [ ] Add MCP tool tests:
  - [ ] Test each tool with valid inputs
  - [ ] Test error handling
- [ ] Add reasoning framework tests:
  - [ ] Test CoT step extraction
  - [ ] Test ToT branching and pruning
  - [ ] Test ReAct action loop
- [ ] Add drafting strategy tests:
  - [ ] Test speculative decoding acceptance
  - [ ] Test pipeline stage transitions
  - [ ] Test candidate ranking

### 10.2 Integration Tests
- [ ] Test full translation flow (client → API → model → response)
- [ ] Test memory persistence across restarts
- [ ] Test Docker service dependencies

### 10.3 Client Tests (Optional for Prototype)
- [ ] Add React component tests
- [ ] Add E2E tests with Playwright/Cypress

---

## Phase 11: Documentation

> **Priority**: LOW  
> **Goal**: Comprehensive documentation for users and developers

### 11.1 User Documentation
- [ ] Update README.md with:
  - Quick start guide
  - Feature overview
  - Screenshots
- [ ] Add installation guide for each platform
- [ ] Document MCP setup for AI assistants

### 11.2 Developer Documentation  
- [ ] Update CLAUDE.md with MCP details
- [ ] Add API documentation (auto-generate from OpenAPI)
- [ ] Add architecture diagrams
- [ ] Document contribution guidelines

### 11.3 Inline Documentation
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout Python code
- [ ] Document complex algorithms/flows

---

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

- [ ] GPU acceleration for translation models
- [ ] Lazy model loading with progress indicator
- [ ] Multiple user support
- [ ] Cloud sync for memories
- [ ] Plugin system for additional tools
- [ ] Voice input/output
- [ ] Mobile companion app
- [ ] Offline mode with cached models

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Critical Fixes | ✅ Complete | 100% |
| Phase 2: Wire Up Code | ✅ Complete | 100% |
| Phase 3: MCP Client | ✅ Complete | 100% |
| Phase 4: Model Providers | ✅ Complete | 100% |
| Phase 5: Drafting Framework | ✅ Complete | 100% |
| Phase 6: Reasoning Framework | ✅ Complete | 100% |
| Phase 7: Agent Core | ✅ Complete | 100% |
| Phase 8: Client Updates | ✅ Complete | 100% |
| Phase 9: Security | Not Started | 0% |
| Phase 10: Testing | Not Started | 0% |
| Phase 11: Documentation | Not Started | 0% |

---

## Notes

### Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-21 | MCP Client (not server) | AgentX consumes tools from external MCP servers |
| 2026-01-21 | Multi-strategy drafting | Support speculative, pipeline, and candidate generation |
| 2026-01-21 | Flexible reasoning | Support CoT, ToT, ReAct, Reflection patterns |
| 2026-01-21 | Keep agent_memory architecture | Well-designed, integrates with reasoning |
| 2026-01-21 | Use Todo.md for tracking | Simple, version controlled |

### Blockers
_None currently_

### Questions to Resolve
- [x] Which LLM providers to prioritize? (OpenAI / Anthropic / Ollama / Together) → OpenAI, Anthropic, Ollama implemented
- [ ] Should agent memory require authentication?
- [ ] Target platforms for distribution? (Windows / macOS / Linux)
- [ ] Reasoning trace storage format? (JSON / SQLite / Neo4j)

---

## Dependencies to Add

```toml
# Add to pyproject.toml for new capabilities
dependencies = [
    # ... existing ...
    
    # MCP Client
    "mcp>=1.0.0",
    
    # Missing from current pyproject.toml
    "redis>=5.0.0",
    "sqlalchemy>=2.0.0",
    
    # Model Providers
    "openai>=1.0.0",
    "anthropic>=0.20.0",
    "httpx>=0.27.0",           # For async HTTP (Ollama, etc.)
    "aiohttp>=3.9.0",          # Alternative async HTTP
    
    # Async Support
    "asyncio>=3.4.3",
    
    # Reasoning/Drafting
    "networkx>=3.0",           # For ToT graph structures
    "numpy>=1.26.0",           # For embeddings/scoring
]
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentX Desktop App                          │
├─────────────────────────────────────────────────────────────────────┤
│                           Tauri Client                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Dashboard │ │Translate │ │  Agent   │ │  Tools   │ │ Settings │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         Django API Layer                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        Agent Core                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │   Planner   │  │  Executor   │  │  Context Manager    │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │  MCP Client    │  │   Reasoning    │  │    Drafting        │     │
│  │  ┌──────────┐  │  │  ┌──────────┐  │  │  ┌──────────────┐  │     │
│  │  │FS Server │  │  │  │   CoT    │  │  │  │ Speculative  │  │     │
│  │  │GH Server │  │  │  │   ToT    │  │  │  │ Pipeline     │  │     │
│  │  │DB Server │  │  │  │  ReAct   │  │  │  │ Candidate    │  │     │
│  │  │Custom... │  │  │  │Reflection│  │  │  └──────────────┘  │     │
│  │  └──────────┘  │  │  └──────────┘  │  └────────────────────┘     │
│  └────────────────┘  └────────────────┘                              │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │Model Providers │  │ Agent Memory   │  │  Translation Kit   │     │
│  │ OpenAI        │  │  Episodic      │  │  NLLB-200          │     │
│  │ Anthropic     │  │  Semantic      │  │  Language Detect   │     │
│  │ Ollama        │  │  Procedural    │  └────────────────────┘     │
│  │ Together      │  │  Working       │                              │
│  └────────────────┘  └────────────────┘                              │
├─────────────────────────────────────────────────────────────────────┤
│                        Data Layer                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Neo4j   │  │  PostgreSQL  │  │  Redis   │  │ Local Models  │   │
│  │ (Graph)  │  │  (pgvector)  │  │ (Cache)  │  │  (HuggingFace)│   │
│  └──────────┘  └──────────────┘  └──────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

                    External MCP Servers
┌──────────────────────────────────────────────────────────────────────┐
│  @mcp/filesystem  │  @mcp/github  │  @mcp/postgres  │  Custom MCP   │
└──────────────────────────────────────────────────────────────────────┘
```

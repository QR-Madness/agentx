# AgentX Development Roadmap

This document tracks the development history and future direction of AgentX.

**Current release:** v0.21.23 — the v0.20 milestone, "Mobile-Ready Alpha".

## Progress Overview

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Critical Fixes | Complete | Codebase baseline and dependency fixes |
| Phase 2: Wire Up Code | Complete | Connect scaffolded components |
| Phase 3: MCP Client | Complete | External tool server integration |
| Phase 4: Model Providers | Complete | Multi-provider LLM abstraction |
| Phase 5: Drafting Framework | Complete | Speculative decoding and pipelines |
| Phase 6: Reasoning Framework | Complete | CoT, ToT, ReAct, Reflection |
| Phase 7: Agent Core | Complete | Unified agent architecture |
| Phase 8: Client Updates | Complete | Tauri UI with cosmic theme |
| Phase 9: Security | Complete | Foundation security measures |
| Phase 10: Testing (Core) | Complete | 50 tests covering core functionality |
| Phase 11: Memory System | **Complete** | Persistent knowledge graphs |
| Phase 12: Documentation | **Partial** | Comprehensive docs refresh |
| Phase 13: UX Overhaul | Complete | Immersive 3-page app with conversation tabs |
| Phase 14: Context Gating | Complete | Large tool output compression + retrieval |
| Phase 15: Plan Execution | **Core Complete** | Plan state store, executor, streamed progress, cancellation (parallelism deferred) |
| Phase 16: Multi-Agent | **In Progress (~65%)** | Agent Alloy v1 + routing/delegation/@-mentions shipped; Factory UI + ambassador deferred |
| Phase 17: Server Management | **Complete** | Auth, Docker production stack, multi-cluster, version matching |
| Phase 18: UX + Memory Tuning | **In Progress (~93%)** | Toolkit, Relay, model metadata, metrics, extraction tuning, detached runs |

---

## Completed Phases

### Phase 1: Critical Fixes

> **Goal**: Get the codebase into a working baseline state

#### 1.1 Missing Dependencies
- Added `redis>=5.0.0`, `sqlalchemy>=2.0.0`, `mcp>=1.0.0` to pyproject.toml
- Added `openai>=1.0.0`, `anthropic>=0.20.0`, `httpx>=0.27.0` for model providers
- Added `networkx>=3.0` for reasoning graphs
- Ran `uv sync` to regenerate lock file and verified imports

#### 1.2 Taskfile Corrections
- Fixed `client:build`: `bunx next build` → `bun run build`
- Fixed `client:start`: `bunx next start` → `bun run preview`
- Fixed `client:dev`: `bunx tauri run dev` → `bun run tauri dev`
- Removed Next.js references (project uses Vite)

#### 1.3 OpenAPI Specification
- Updated server URL port: `8000` → `12319`
- Documented `POST /tools/translate` endpoint with request/response schemas
- Fixed language-detect path: `/language-detect` → `/tools/language-detect-20`
- Added proper response schemas and Error component schema

#### 1.4 Environment Configuration
- Created `.env.example` with all required variables
- Updated `docker-compose.yml` to use environment variables
- Ensured `.env` is in `.gitignore`
- Added redis-commander to optional `tools` profile

#### 1.5 Code Quality
- Replaced `print()` statements with `logging` in views.py
- Fixed `language_detect` to accept POST with text body (backwards compatible)
- Fixed test import paths and URL paths to match new API structure
- Updated CLAUDE.md with correct API endpoints

---

### Phase 2: Wire Up Existing Code

> **Goal**: Connect the scaffolded code that already exists

#### 2.1 Language Detection API Fix
- Modified `views.language_detect()` to accept POST
- Removed hardcoded test string
- Added input validation with structured response (language code + confidence)

#### 2.2 Agent Memory System Connection
- Renamed `agent_memory.py` → `memory_utils.py` to avoid naming conflict
- Added database connection health check endpoint: `GET /api/health`
- Created lazy-loading `get_agent_memory()` function
- Made PostgreSQL connection lazy (PostgresConnection class)
- Added `check_memory_health()` for connection status

#### 2.3 Code Quality Improvements
- Replaced all `print()` statements with `logging`
- Added logging configuration to Django settings
- Added `psycopg2-binary` for PostgreSQL connectivity
- Added `sentence-transformers` for local embeddings
- Added health check tests

---

### Phase 3: MCP Client Integration

> **Goal**: Enable AgentX to consume tools from external MCP servers

#### 3.1 MCP Client Infrastructure
Created module structure:
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

Implemented `MCPClientManager` class:
- `connect_server(server_config)` → async context manager with ServerConnection
- `list_tools(server_name?)` → available tools
- `call_tool(server_name, tool_name, args)` → ToolResult
- `list_resources(server_name?)` → available resources
- `read_resource(server_name, uri)` → content

#### 3.2 Server Configuration
- Created `mcp_servers.json.example` configuration format
- Added Taskfile commands: `task mcp:list-servers`, `task mcp:list-tools`
- Added API endpoints: `/api/mcp/servers`, `/api/mcp/tools`, `/api/mcp/resources`

#### 3.3 Tool Discovery & Caching
- Cache tool schemas on connection (in ToolExecutor)
- Handle server disconnection gracefully (via context managers)

---

### Phase 4: Model Provider Abstraction

> **Goal**: Support multiple LLM backends with unified interface

#### 4.1 Provider Interface
Created abstract `ModelProvider` interface:
```python
class ModelProvider(ABC):
    @abstractmethod
    async def complete(self, messages, **kwargs) -> CompletionResult

    @abstractmethod
    async def stream(self, messages, **kwargs) -> AsyncIterator[StreamChunk]

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities
```

#### 4.2 Provider Implementations
- `OpenAIProvider` - GPT-4, GPT-4-turbo, GPT-3.5
- `AnthropicProvider` - Claude 3 Opus/Sonnet/Haiku
- `OllamaProvider` - Local models (Llama, Mistral, etc.)

#### 4.3 Model Registry
- Created `models.yaml` configuration
- Model selection based on task requirements (via ProviderRegistry)

---

### Phase 5: Drafting Models Framework

> **Goal**: Implement multi-model generation strategies

#### 5.1 Drafting Infrastructure
Created module structure:
```
api/agentx_ai/drafting/
├── __init__.py
├── base.py            # DraftingStrategy, DraftResult
├── speculative.py     # Speculative decoding
├── pipeline.py        # Multi-model pipelines
├── candidate.py       # N-best candidate generation
└── drafting_strategies.yaml
```

#### 5.2 Speculative Decoding
Implemented `SpeculativeDecoder`:
- Draft model generates N tokens quickly
- Target model verifies/corrects
- Configurable draft length and acceptance threshold
- Support draft/target model pairs (configurable)

#### 5.3 Multi-Model Pipeline
Implemented `ModelPipeline`:
- Define pipeline stages (analyze → draft → refine → validate)
- Route to different models per stage
- Pass context between stages
- Predefined stage roles (ANALYZE, DRAFT, REVIEW, REFINE, etc.)

#### 5.4 Candidate Generation
Implemented `CandidateGenerator`:
- Generate N candidates (same or different models)
- Scoring/ranking strategies:
  - Self-consistency (majority vote)
  - Verifier model scoring
  - Heuristic scoring (length preference)
- Best-of-N selection

#### 5.5 Drafting Configuration
- Created `drafting_strategies.yaml` with pre-configured strategies

---

### Phase 6: Reasoning Framework

> **Goal**: Implement flexible reasoning patterns for complex tasks

#### 6.1 Reasoning Infrastructure
Created module structure:
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

#### 6.2 Chain-of-Thought (CoT)
Implemented `ChainOfThought`:
- Zero-shot CoT ("Let's think step by step...")
- Few-shot CoT (with examples)
- Auto-CoT mode
- Step extraction and validation
- Answer extraction

#### 6.3 Tree-of-Thought (ToT)
Implemented `TreeOfThought`:
- BFS exploration (breadth-first)
- DFS exploration (depth-first)
- Beam search (top-k branches)
- State evaluation heuristics
- Pruning strategies
- Tree serialization for traces

#### 6.4 ReAct (Reasoning + Acting)
Implemented `ReActAgent`:
- Thought → Action → Observation loop
- Tool execution framework
- Max iterations / termination conditions
- Action parsing and execution
- Built-in tools (search, calculate)

#### 6.5 Reflection & Self-Critique
Implemented `ReflectiveReasoner`:
- Generate initial response
- Self-critique: identify weaknesses
- Revise based on critique
- Configurable reflection depth

#### 6.6 Reasoning Orchestrator
Implemented `ReasoningOrchestrator`:
- Select reasoning strategy based on task
- Task type classification
- Fallback strategies on failure
- Metrics: steps taken, tokens used, time elapsed

---

### Phase 7: Agent Core

> **Goal**: Unify MCP, drafting, and reasoning into coherent agent

#### 7.1 Agent Architecture
Created module structure:
```
api/agentx_ai/agent/
├── __init__.py
├── core.py            # Main Agent class
├── planner.py         # Task decomposition
├── context.py         # Context management
└── session.py         # Conversation session
```

#### 7.2 Agent Core Implementation
Implemented `Agent` class with:
- Integration with reasoning orchestrator
- Integration with drafting strategies
- Tool registration
- Status tracking
- Task cancellation

#### 7.3 Task Planning
Implemented `TaskPlanner`:
- Decompose complex tasks into subtasks
- Identify subtask types (research, analysis, generation, etc.)
- Estimate complexity
- Select reasoning strategy based on plan

#### 7.4 Context Management
Implemented `ContextManager`:
- Token estimation
- Sliding window for long conversations
- Summarization of old context
- Memory injection support

#### 7.5 Session Management
Implemented `Session` and `SessionManager`:
- Message history tracking
- Session creation/retrieval
- Session cleanup for inactive sessions

#### 7.6 Agent API Endpoints
- `POST /api/agent/run` - Execute a task
- `POST /api/agent/chat` - Conversational interaction
- `GET /api/agent/status` - Check agent status

---

### Phase 8: Client Updates

> **Goal**: Update UI to support new agent capabilities

#### 8.1 Agent Tab
- Created AgentTab component
- Task input with natural language
- Real-time reasoning trace display
- Tool usage visualization
- Cancel/pause controls

#### 8.2 Dashboard Updates
- Connected MCP servers status
- Model provider status (API keys valid, local models loaded)
- Agent status widget
- System health overview

#### 8.3 Settings Tab
- Multi-server configuration (per-server settings storage)
- Model provider API key management
- Drafting strategy selection
- Reasoning preferences
- Memory/storage info

#### 8.4 Tools Tab Updates
- MCP tool browser (all available tools from connected servers)
- Tool search functionality
- Tool testing interface (placeholder)
- Server status display

#### 8.5 Design System Updates
- Dark cosmic theme (nebula gradients, star accents)
- Lucide-react icons throughout
- Glassmorphism effects
- Glow animations
- Updated color palette (purple/cyan/pink cosmic theme)

#### 8.6 Infrastructure
- Multi-server storage system (localStorage)
- ServerContext for app-wide server state
- Typed API client with React hooks
- Per-server metadata and preferences

---

### Phase 9: Security & Infrastructure

> **Goal**: Secure and stabilize the platform (Foundation)

#### 9.1 Tauri Security
- Configured Content Security Policy in `tauri.conf.json`
- Reviewed and restricted Tauri capabilities (already minimal)
- Improved window defaults (1200x800, min size)

#### 9.2 API Security
- Foundation for rate limiting (settings placeholder, not enforced)
- Foundation for API key authentication (settings placeholder, not enforced)
- Input validation limits defined (AGENTX_MAX_TEXT_LENGTH, AGENTX_MAX_CHAT_LENGTH)
- CORS configuration made environment-configurable
- CORS permissive in DEBUG mode for development

**Note**: Security is intentionally permissive for private server development. Foundation is in place for future hardening.

---

### Phase 10: Testing (Core)

> **Goal**: Ensure reliability through automated testing
> **Result**: 50 tests, 49 pass, 1 skipped for Docker

#### 10.1 Backend Tests
- Fixed and unskipped `test_language_detect` (POST and GET both work)
- Translation tests: multiple language pairs, error handling, long text
- MCP tool tests: servers/tools/resources endpoints, registry operations
- Reasoning framework tests: base classes, CoT, ToT, ReAct, Reflection
- Drafting strategy tests: speculative, pipeline, candidate generation
- Provider tests: registry, model config, Message/CompletionResult

#### 10.2 Integration Tests
- Full translation flow (via API endpoint tests)
- Docker service dependencies (health check with skipUnless)

---

### Phase 11: Memory System

> **Goal**: Persistent knowledge graphs with extraction, consolidation, and intelligent recall

#### 11.1–11.7: Core Memory Infrastructure
- 4-type memory architecture: episodic (PostgreSQL), semantic (Neo4j + PostgreSQL), procedural (Neo4j), working (Redis)
- `AgentMemory` unified interface with lazy-loaded database connections
- Memory models: Turn, Entity, Fact, Goal, Strategy with embeddings and salience scoring
- Channel-scoped memory with `_global` and `_default` channels
- Memory API: 12 endpoints for recall, entities, facts, strategies, stats, settings

#### 11.8: Testing & Security
- 80+ memory tests covering integration, security, and edge cases
- Tenant isolation (user_id filtering), audit logging, input validation
- Graceful degradation when databases are unavailable

#### 11.9: Extraction & Consolidation
- `ExtractionService`: LLM-based entity/fact/relationship extraction in single call
- Consolidation pipeline: scheduled jobs for extraction, entity linking, contradiction detection
- `JobRegistry` with worker threads and configurable intervals

#### 11.10: Memory Explorer UI
- Client-side memory browser with entity graph, facts list, strategy viewer

#### 11.11: Background Job Monitoring
- Job monitoring API: list, detail, history, manual trigger, enable/disable
- JobsPanel component in Memory tab with status badges and manual controls

#### 11.12: LLM-Enhanced Consolidation
- Pre-extraction relevance filter (heuristic + LLM, ~75% fewer extraction calls)
- Combined relevance + extraction in single LLM call with reasoning model support
- Confidence calibration: explicit=0.95, implied=0.85, inferred=0.70, uncertain=0.50
- Entity linking via embedding search with configurable similarity threshold
- Contradiction detection with resolution strategies (prefer_new, prefer_old, flag_review)
- User correction handling: heuristic patterns + LLM extraction + fact supersession
- `ConsolidationMetrics` dataclass for pipeline observability
- Batch entity/relationship storage with UNWIND (fixes N+1 queries)
- `claim_hash` field on Fact model for indexed duplicate detection
- Settings cache with 60s TTL refresh
- Temporal reasoning: `temporal_context` field (current/past/future) with temporal boost in retrieval
- Reinforcement signals: `last_accessed`, `access_count`, `salience` on Fact model
- Source attribution: `source_turn_id` on all extracted facts
- `RecallLayer`: 5 retrieval techniques (hybrid search, entity-centric, query expansion, HyDE, self-query)
- 50+ unit tests across extraction, metrics, confidence, temporal, and recall

---

### Phase 12: Documentation (Partial)

> **Goal**: Comprehensive backend documentation refresh

- Rewrote architecture docs (overview, API layer, memory)
- Expanded API reference: the full endpoint set documented with request/response examples
- Comprehensive API models reference (providers, agent, memory, prompts, MCP, SSE)
- Created 5 new feature docs: reasoning, drafting, MCP, providers, prompts
- Rewrote/expanded 3 existing feature docs: chat, translation, memory
- Expanded development docs: contributing guide, setup guide
- Updated configuration reference with all config layers
- Updated quickstart with examples for streaming, MCP, memory, prompts
- Mermaid diagrams throughout for architecture and data flow visualization

*Remaining: auto-generated API docs, contribution guidelines, full docstrings — see Todo.md*

---

### Phase 13: UX Overhaul — Immersive AgentX

> **Goal**: Immersive 3-page app with browser-style conversation tabs, portal-based modals, customizable agent profiles

#### 13.1 Foundation
- `ThemeProvider` context with CSS variable system, `cosmic` default theme
- `ModalContext` with stack-based modal management, `ModalPortal`, `DrawerPanel`, `ModalDialog`
- `ConversationMessage` discriminated union type system with type guards

#### 13.2 Root Layout Shell
- `RootLayout` + `TopBar` replacing sidebar TabBar with horizontal navigation
- Page routing: Start, Dashboard, AgentX

#### 13.3 Conversation Tabs
- `ConversationContext` with browser-style tab model (add, close, switch, rename, reorder)
- `ConversationTabBar` with scrollable tabs, history dropdown, localStorage persistence

#### 13.4 Agent Profiles
- `ProfileManager` backend with YAML storage, CRUD API endpoints
- Agent name injection into system prompts via `PromptConfig`
- `AgentProfileContext` frontend with profile editor modal, avatar picker, settings

#### 13.5 Modals Migration
- Settings, Memory, Tools → right-side drawer panels from toolbar icons
- Translation → centered modal dialog

#### 13.6 AgentX Page
- `AgentXPage` with `ConversationContext` integration and `ChatPanel`
- Full-width message area with immersive input bar

#### 13.7 SSE & Metadata Enhancements
- Rich SSE events: `memory_context`, extended `tool_call`/`tool_result`/`start`/`done`
- Frontend handles all new event types with typed callbacks
- Unified `ToolExecutionBlock` with animated status icons and output drawer

#### 13.8 Prompt Library
- `PromptTemplateManager` backend with YAML storage and CRUD endpoints
- Frontend modal with tag filtering, search, template preview, profile attachment

#### 13.9 Start Page & Dashboard Refresh
- `StartPage` with agent greeting and quick actions
- `DashboardPage` with health status, memory stats, DB storage metrics

#### 13.10 Cleanup
- Removed old `components/tabs/` directory, renamed to panels
- Keyboard shortcuts: Ctrl+T (new tab), Ctrl+W (close tab), Ctrl+K (command palette)
- Final CSS consistency pass

#### 13.11 Consolidation Settings UI
- Consolidation settings section in Settings drawer
- Extraction, Relevance Filter, Contradiction Detection provider settings
- TopBar integration

---

### Phase 14: Context Gating for Large Tool Outputs

> **Goal**: Hybrid context gating with task-aware compression and intent-based retrieval
> **Research basis**: ACON, Focus, A-MEM, SimpleMem

#### 14.1 Storage + Threshold
- `max_tool_result_chars` config (12K default), Redis storage with TTL
- Internal MCP tools: `read_stored_output`, `list_stored_outputs`

#### 14.2 Compression Gate
- `ToolOutputCompressor` with task-aware LLM compression prompts
- Configurable via `compression.*` settings, graceful fallback on failure

#### 14.3 Intent-Aware Retrieval
- `tool_output_chunker.py` with section detection, JSON path, semantic search
- MCP tools: `tool_output_query`, `tool_output_section`, `tool_output_path`

#### 14.4 Intra-Trajectory Compression (Focus-style)
- Round identification + LLM knowledge block generation + in-place message mutation
- Two-layer defense: trajectory compression at 75% threshold, truncation as hard fallback

#### 14.5 Read Loop Prevention
- Retrieval tool bypass in `_execute_tool_calls()`, default pagination on `read_stored_output`

#### 14.6 Agent Self-Memory
- Docker-style `agent_id` generation (adjective-adjective-noun, ~83K combinations)
- `_self_<agent_id>` memory channel with assistant self-extraction prompt
- Consolidation job Phase 2: assistant turn pass with certainty calibration
- Self-channel included in recall: `[channel, _self_{agent_id}, _global]`

#### 14.7 Bulletproof Fact Correction Pipeline
- Three-layer verification: hash gate → semantic duplicate (cosine > 0.92) → entity-scoped candidate search → temporal-aware LLM adjudication
- Temporal progression auto-resolution (new "current" supersedes old without LLM)
- Implicit correction detection at extraction time ("I switched from X to Y" → dual facts)
- Contradiction + correction detection enabled by default

#### Test Coverage
- 8 compression tests, 22 retrieval tests, 8 trajectory tests, 6 read-loop tests, 10 self-memory tests, 13 fact-correction tests

---

### Phase 15: Plan Execution (Core Complete)

> **Goal**: Execute multi-step plans with persisted state and streamed progress

- `PlanStateStore` (Redis Hash per plan, per-subtask status, best-effort) + `PlanExecutor` with sync + streaming execution, dependency-ordered subtask iteration, per-subtask trajectory compression, failure handling (skip dependents), and a final synthesis step
- `Subtask`/`TaskPlan` serialization for Redis storage
- `Agent.run` integration: MODERATE/COMPLEX plans route through the executor; SIMPLE tasks unchanged. Memory reflection + goal completion after execution
- Streaming endpoint delegates complex tasks to `PlanExecutor.execute_streaming`; SSE events `plan_start`, `subtask_start`, `subtask_complete`, `subtask_failed`, `plan_complete` (reusing `chunk`/`tool_call`/`tool_result` within subtasks)
- Plan cancellation mid-execution: `POST /api/agent/plans/cancel` sets a flag the executor checks between subtasks; `GET /api/agent/plans/{id}/status` reads Redis-tracked state
- *Deferred*: parallel subtask execution, per-subtask reasoning-strategy selection, subtask-level subgoals, plan resumption from Redis after disconnect

---

### Phase 16: Multi-Agent Conversations — Agent Alloy (In Progress, ~65%)

> **Goal**: Coordinate multiple agents via a supervisor that delegates to specialists
> **Shipped**: Agent Alloy v1 (2026-04-27) + routing/delegation/@-mention waves (through v0.21.5)

- **Alloy v1 backend**: workflow models, `WorkflowManager` (YAML CRUD), `delegate_to` tool, `AlloyExecutor`, supervisor prompts. Supervisor + specialist delegation over a shared memory channel (`_alloy_<id>`) with live `delegation_start`/`delegation_chunk`/`delegation_complete` SSE streaming, child-goal wiring, and re-delegation up to `max_delegation_depth`. CRUD endpoints (`/api/alloy/workflows`). Fully opt-in via `workflow_id`
- **Parallel / fan-out delegation** (supervisor delegates to multiple specialists at once)
- **Trace / replay UI** (`v0.20.1`): `delegation_complete` carries per-delegation tokens/duration/cost/pricing snapshot (persisted for restored runs); client `AlloyRunTraceModal` groups fan-out into runs with token/cost/wall-clock totals
- **16.1 Message attribution**: `agent_id` on `Turn` model + `conversation_logs`; per-turn attribution persisted (streaming, non-streaming, background) and restored to display names
- **16.2 Explicit routing** (`v0.21.2`): `target_agent_id` on the chat-stream request (priority `workflow_id > target_agent_id > agent_profile_id > default`); `Session.participants` hydrated from durable attribution; multi-agent awareness prompt when >1 participant
- **16.3 Per-agent tool isolation**: `allowed_tools`/`blocked_tools` on the profile enforced in `_get_tools_for_provider`
- **16.4 Ad-hoc agent-to-agent delegation** (`v0.21.3`): `delegate_to` generalized to non-workflow chats via a workflow-less `AlloyExecutor` mode, gated by `alloy.allow_adhoc_delegation`; depth-limited, no self-delegation
- **16.5 @-mention routing** (`v0.21.4`/`v0.21.5`): backend `agent/mentions.py` parses `@agent-id`/`@name` to route a turn (token stripped from the model-facing message); `AgentParticipant` Neo4j nodes + `PARTICIPATED_IN` relationships with a backfill migration; client `@`-autocomplete composer
- *Deferred*: Factory canvas frontend (visual editor), declarative route execution, loop/checkpoint/async-delegation nodes, "user as supervisor" mode, tool-output sharing across agents, per-workflow tool subsetting, and the Ambassador dual-presentation layer (16.6)

---

### Phase 17: Server Management (Complete)

> **Goal**: Make AgentX deployable beyond a single dev machine

- Optional session authentication: single root user, bcrypt password, Redis-backed sessions, gated by `AGENTX_AUTH_ENABLED`
- Dockerized production deployment via clusters (`task cluster:*`) — single or multiple isolated instances per host, with an optional Nginx + Cloudflare gateway
- Client/API version matching via `versions.yaml` (`protocol_version` + `min_client_version`), surfaced by `VersionMismatchPage`

---

### Phase 18: UX Improvements & Memory Tuning (In Progress, ~93%)

> **Goal**: Polish the client and tune the memory pipeline

- **Toolkit** (18.2): full-screen MCP server CRUD + tool browser, guided + raw-JSON editors with validation; tool metadata (tags, groups, `allowed_agent_ids`); per-agent `allowed_tools`/`blocked_tools`
- **Relay module** (18.3): consolidated chat toolbox popover — background-run inbox (Redis-Streams chat job queue + daemon worker), "No Memorization" toggle, mobile pass
- **Model metadata + picker** (18.4): full-screen `ModelPickerModal` with provider/capability filters; OpenRouter + Vercel capability/pricing/modality metadata wired through `ModelCapabilities` and `/api/providers/models`
- **Metrics** (18.5): per-tab context-window bar (persisted + backfilled) and per-turn `$` cost chip via `providers/pricing.estimate_cost` with frozen pricing snapshots (dashboard aggregation UI still pending)
- **Extraction tuning** (18.6): consolidation entity resolution (name/alias/slug reuse + alias merge), `refines_fact_id` supersedure, scope-context injection of known entities/facts, new metrics, and the `eval_consolidation` quality harness
- **Wave 2 fixes** (18.8): KaTeX math rendering, raw-HTML in tables, plan-step bubble restore, editable cached servers, MCP auto-connect on restart
- **Memory tuning** (18.9): user-selected extraction model honored, stop leaking raw past turns at chat start, `recall_user_history` tool, working-memory token-budget header, model-authored `checkpoint` tool + sidebar/badge UI
- **Plan + streaming reliability** (18.10): output token-budget clamp, "Plans in Progress" drawer + in-chat step annotation, and **detached chat runs** — runs drive a Redis-Streams fan-out on a daemon thread so they survive client disconnect, with `attach`/`cancel`/`runs` endpoints
- **Client error contract + foundation cleanup** (18.11): `ApiError` with status-derived `kind`, toast system, `useApi` hook factory, Tailwind v4 + `ui/` primitive library, and god-component / `lib/api` / `ConversationContext` splits with a Vitest harness
- **Wave 3 UX** (18.12): Start-page recents, renamable conversations, improved conversation + agent-profile selectors, splash screen, README trim

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-21 | MCP Client (not server) | AgentX consumes tools from external MCP servers |
| 2026-01-21 | Multi-strategy drafting | Support speculative, pipeline, and candidate generation |
| 2026-01-21 | Flexible reasoning | Support CoT, ToT, ReAct, Reflection patterns |
| 2026-01-21 | Keep agent_memory architecture | Well-designed, integrates with reasoning |
| 2026-01-21 | Use Todo.md for tracking | Simple, version controlled |
| 2026-01-31 | Extensible memory over backwards-compat | Memory system prioritizes easy extension; no backwards-compatibility shims |
| 2026-01-31 | Audit logging in PostgreSQL | All memory operations traceable per session with configurable verbosity |
| 2026-01-31 | Tenant isolation required | Semantic memory must filter by user_id to prevent cross-user data leakage |
| 2026-01-31 | Extraction via LLM preferred | Entity/fact extraction uses model providers for quality; spaCy as lightweight fallback |
| 2026-01-31 | Memory channels over multiple databases | Logical channel scoping (property on nodes/rows) vs multiple DBs |
| 2026-01-31 | `_global` as default channel | User-wide memory (preferences, general facts) lives in `_global` |
| 2026-01-31 | Channels are traceable scopes, not isolation | Retrieval merges active channel + `_global` |
| 2026-01-31 | Cross-channel promotion via consolidation | Prominent project facts auto-promote to `_global` |
| 2026-01-31 | Audit logs partitioned by day | Configurable retention with daily resolution |
| 2026-01-31 | LLM-only extraction, no spaCy | Entity/fact extraction uses model providers exclusively |
| 2026-02-13 | Dual interface: Chat vs Agent | Chat = minimal friction; Agent = full power for prompt engineering |
| 2026-02-13 | Agent Profiles as first-class concept | Stored configurations that personalize agent behavior |
| 2026-02-13 | Conversation branching over linear history | Power users need to explore alternatives |
| 2026-02-13 | Profile inheritance via `extends` | Child profiles inherit parent settings with overrides |
| 2026-02-13 | Global prompt library with tags | Templates shared across profiles |
| 2026-02-22 | Scheduler deferred, manual trigger priority | Manual consolidation trigger for debugging |
| 2026-02-23 | LLM providers for consolidation stages | Route through existing provider system |
| 2026-02-23 | Filter consolidation to user turns only | Extract facts from user messages only |
| 2026-02-23 | Default memory channel is _default, not _global | `_global` populated via promotion |
| 2026-03-13 | Docker-style agent IDs over UUIDs | Human-readable, memorable for multi-agent routing |
| 2026-03-13 | Self-memory channel per agent | `_self_{agent_id}` isolates agent's own knowledge |
| 2026-03-28 | Three-layer fact verification pipeline | Funnel eliminates work before LLM calls: hash → semantic → LLM |
| 2026-03-28 | Temporal progression auto-resolution | "Current supersedes current" doesn't need LLM adjudication |
| 2026-03-29 | agent_id as routing identifier, name as display | Decouples formal identity from flexible display name |
| 2026-03-29 | Plan execution prerequisite for multi-agent | Delegation is "execute another agent's plan" |

---

## Resolved Questions

| Question | Resolution |
|----------|------------|
| Which LLM providers to prioritize? | OpenAI, Anthropic, Ollama implemented; LM-Studio preferred |
| Should agent memory require authentication? | No, one server = one user. Multiple servers can exist on one system. |
| Target platforms for distribution? | Linux and Windows for now |
| Reasoning trace storage format? | Neo4j graph + PostgreSQL audit log |
| Entity extraction method? | LLM-only via model providers; spaCy too constraining |
| Audit log retention policy? | 30 days default, configurable with daily resolution |
| Memory retrieval sync or async? | Blocking for now, may scale to concurrent for complex tasks |
| Cross-channel promotion thresholds? | confidence>=0.85, access_count>=5, conversations>=2 |

---

## Deferred Items

These items were identified during development but deferred to future phases:

### From Phase 3 (MCP Client)
- UI for managing MCP server connections
- Test with standard MCP servers (filesystem, github, postgres, brave-search)
- Document tested/supported servers
- Custom MCP server templates
- Tool refresh and search/filter functionality

### From Phase 4 (Model Providers)
- TogetherProvider - Together.ai API
- LocalTransformersProvider - HuggingFace transformers
- Cost tracking and budgeting

### From Phase 6 (Reasoning)
- DebateReasoner implementation

### From Phase 9 (Security)
- Secure storage for API keys (Keyring/OS keychain)
- Encryption for sensitive config
- Audit logging for sensitive operations

### From Phase 10 (Testing)
- React component tests
- E2E tests with Playwright/Cypress

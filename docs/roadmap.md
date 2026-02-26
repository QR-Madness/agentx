# AgentX Development Roadmap

This document tracks the development history and future direction of AgentX.

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
| Phase 11: Memory System | **In Progress** | Persistent knowledge graphs |
| Phase 12: Documentation | Planned | Comprehensive docs |
| Phase 13: UI Implementation | **In Progress** | Chat and Agent interfaces |

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

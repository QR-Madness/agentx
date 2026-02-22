# AgentX Development Todo

**Project**: AgentX - AI Agent Platform with MCP Client, Drafting Models & Reasoning Framework  
**Status**: Pre-prototype  
**Last Updated**: 2026-02-22

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

## Phase 9: Security & Infrastructure ✅

> **Priority**: MEDIUM  
> **Goal**: Secure and stabilize the platform  
> **Status**: COMPLETE (Foundation)

### 9.1 Tauri Security
- [x] Configure Content Security Policy in `tauri.conf.json`
- [x] Review and restrict Tauri capabilities (already minimal)
- [x] Improved window defaults (1200x800, min size)

### 9.2 API Security
- [x] Foundation for rate limiting (settings placeholder, not enforced)
- [x] Foundation for API key authentication (settings placeholder, not enforced)
- [x] Input validation limits defined (AGENTX_MAX_TEXT_LENGTH, AGENTX_MAX_CHAT_LENGTH)
- [x] CORS configuration made environment-configurable
- [x] CORS permissive in DEBUG mode for development

### 9.3 Secrets Management
- [ ] Secure storage for API keys (Keyring/OS keychain) - deferred
- [ ] Encryption for sensitive config - deferred
- [ ] Audit logging for sensitive operations - deferred

**Note**: Security is intentionally permissive for private server development. Foundation is in place for future hardening.

---

## Phase 10: Testing (Core) ✅

> **Priority**: MEDIUM
> **Goal**: Ensure reliability through automated testing
> **Status**: COMPLETE (50 tests, 49 pass, 1 skipped for Docker)
> **Note**: Memory system tests deferred to Phase 11 (requires functional memory)

### 10.1 Backend Tests
- [x] Fix and unskip `test_language_detect` (POST and GET both work)
- [x] Add translation tests:
  - [x] Test multiple language pairs (Spanish, German, Japanese)
  - [x] Test error handling (invalid language codes - now returns 400)
  - [x] Test long text handling (paragraph translation)
- [x] Add MCP tool tests (existing tests pass):
  - [x] Test servers endpoint structure
  - [x] Test tools endpoint structure
  - [x] Test resources endpoint structure
  - [x] Test server config creation and registry operations
- [x] Add reasoning framework tests:
  - [x] Test base classes (ReasoningStatus, ThoughtType, ThoughtStep, ReasoningResult)
  - [x] Test CoT config and step extraction regex
  - [x] Test ToT TreeNode and config
  - [x] Test ReAct Tool and config
  - [x] Test Reflection Revision and config
- [x] Add drafting strategy tests:
  - [x] Test base classes (DraftStatus, DraftResult, DraftingConfig)
  - [x] Test speculative decoding config
  - [x] Test pipeline stage roles and config
  - [x] Test candidate generation scoring and config
- [x] Add provider tests:
  - [x] Test registry singleton
  - [x] Test model config retrieval
  - [x] Test Message and MessageRole
  - [x] Test CompletionResult

### 10.2 Integration Tests
- [x] Test full translation flow (via API endpoint tests)
- [x] Test Docker service dependencies (health check with skipUnless)

### 10.3 Client Tests (Optional for Prototype)
- [ ] Add React component tests (deferred)
- [ ] Add E2E tests with Playwright/Cypress (deferred)

---

## Phase 11: Memory System Activation

> **Priority**: HIGH
> **Goal**: Make the memory system functional, auditable, and extensible
> **Depends on**: Docker services (Neo4j, PostgreSQL, Redis) running
> **Design Principles**:
> - **Extensibility**: Easy to add new memory types, stores, or extraction methods without backwards-compatibility constraints
> - **Transparency**: All memory operations traceable per session/conversation based on logging settings
> - **Auditability**: Full query trace and operation audit trail in PostgreSQL
> - **Channel Scoping**: Memory organized into channels — `_global` (default, user-wide) and project channels (e.g. `my-rust-project`). Channels are traceable scopes, not isolation boundaries. Retrieval queries both the active channel and `_global`, merging results. Prominent project facts consolidate upward into global memory.

### Current State Assessment

The memory system is **architecturally complete but entirely disconnected**:
- `Agent._memory = None` is never initialized — zero code paths call `store_turn()`, `remember()`, etc.
- `SessionManager` uses in-process Python dicts (lost on restart) as the de facto memory
- Neo4j vector indexes referenced in code **do not exist** (no initialization script)
- PostgreSQL tables defined in `queries/postgres_builder.sql` but **not created** (Docker init-only)
- Entity/fact/relationship extraction functions are **stubs returning `[]`**
- Semantic memory has **no tenant isolation** (users can retrieve each other's data)
- `ContextManager.inject_memory()` exists but is **never called**
- No concept of memory channels or project scoping — all memories are flat/global
- Zero tests for any memory component

### 11.1 Database Schema Initialization ✅

> Bootstrap all three databases so memory operations don't crash at runtime

- [x] Create Neo4j initialization script (Django management command `init_memory_schema`):
  - [x] Create vector indexes: `turn_embeddings`, `fact_embeddings`, `entity_embeddings`, `strategy_embeddings`
  - [x] Create uniqueness constraints on node IDs (`User.id`, `Conversation.id`, `Turn.id`, `Entity.id`, `Fact.id`, `Strategy.id`)
  - [x] Create composite indexes for common query patterns (user_id + timestamp, user_id + channel)
  - [x] Add `channel` property indexes for all node types (Turn, Entity, Fact, Strategy, Conversation, Goal)
  - [x] Verify APOC plugin is available (required by `semantic.py` for `apoc.create.addLabels()`)
- [x] Create PostgreSQL initialization (via `queries/postgres_builder.sql` + management command):
  - [x] Enable `pgvector` extension (`CREATE EXTENSION IF NOT EXISTS vector`)
  - [x] Create `conversation_logs` table with BRIN index on timestamp, ivfflat on embedding, channel column
  - [x] Create `tool_invocations` table with indexes on conversation_id, tool_name, channel
  - [x] Create `memory_timeline` table with channel support
  - [x] Create `user_profiles` table
  - [x] Create `memory_audit_log` partitioned table for query tracing (38 daily partitions)
  - [x] Add `channel VARCHAR DEFAULT '_global'` column to all memory tables
  - [x] Add btree index on `channel` for all tables that have it
  - [x] Create `schema_version` table for tracking
  - [x] Add helper functions: `create_audit_partition()`, `drop_old_audit_partitions()`
- [x] Verify Redis connectivity and key structure:
  - [x] Document working memory key patterns (`working:{user_id}:{channel}:{conversation_id}:*`)
  - [x] Document consolidation job tracking keys
  - [x] Confirm connectivity in management command
- [x] Create `task db:init:schemas` Taskfile command to run all initialization
- [x] Create `task db:verify:schemas` Taskfile command for read-only verification
- [x] Add `agentx_ai` to INSTALLED_APPS in settings.py

### 11.2 Agent Core Integration ✅

> Wire memory into the agent lifecycle so it's actually used

- [x] Initialize `AgentMemory` in `Agent` class (`core.py`):
  - [x] Implement `_memory` property accessor (lazy-load via `get_agent_memory()`)
  - [x] Accept `channel` parameter (default `"_global"`) — passed through to AgentMemory constructor
  - [x] Respect `AgentConfig.enable_memory` flag
  - [x] Handle graceful degradation if databases are unavailable (agent works without memory)
- [x] Wire `store_turn()` into chat/run flows:
  - [x] Create `Turn` objects from `Message` objects in `Agent.chat()`
  - [x] Store turns after each user message and assistant response
  - [x] Include metadata: model used, token count, latency
- [x] Wire `remember()` into context injection:
  - [x] Call `memory.remember(query)` before building LLM context in `Agent.chat()`
  - [x] Feed retrieved `MemoryBundle` into `ContextManager.inject_memory()`
  - [x] Make retrieval configurable: top_k, time_window via `AgentConfig`
- [x] Wire `record_tool_usage()` into tool execution:
  - [x] Added callback hook to `ToolExecutor.set_usage_recorder()`
  - [x] Agent wires recorder via `mcp_client` property when memory available
  - [x] Capture: tool_name, input, output, success, latency_ms, error_message
- [x] Wire `learn_fact()` / `upsert_entity()` into extraction pipeline (see 11.3)
- [x] Call `memory.reflect()` after task completion in `Agent.run()` to trigger consolidation
- [x] Wire `add_goal()` / `complete_goal()` into task planner lifecycle

### 11.3 Extraction Pipeline ✅

> Replace stubs with real entity/fact extraction so consolidation produces actual data

- [x] Implement entity extraction (`extraction/entities.py`):
  - [x] LLM-based extraction via configured model provider (structured JSON output)
  - [x] Extract: person names, organizations, locations, concepts, technical terms
  - [x] Return `Entity` objects with type, name, description, confidence
  - [x] Make extraction model configurable (which provider/model to use for extraction)
- [x] Implement fact extraction (`extraction/facts.py`):
  - [x] LLM-based structured extraction (JSON schema output)
  - [x] Extract: claims, preferences, relationships, stated goals
  - [x] Return `Fact` objects with claim, source, confidence, subject/object entities
  - [x] Include source attribution (which turn the fact came from)
- [x] Implement relationship extraction (`extraction/relationships.py`):
  - [x] Extract entity-to-entity relationships from conversation context
  - [x] Support: works_at, knows, uses, prefers, related_to, etc.
  - [x] Return relationship triples (source_entity, relation_type, target_entity)
- [x] Wire extraction into consolidation jobs:
  - [x] `consolidate_episodic_to_semantic()` already calls extractors — verify it works with real output
  - [x] Add error handling for extraction failures (don't block consolidation)
  - [x] Add extraction metrics logging (entities/facts per consolidation run)

### 11.4 Auditability & Query Tracing ✅

> Make all memory operations transparent and traceable per session

- [x] Create `MemoryAuditLogger` class (`kit/agent_memory/audit.py`):
  - [x] Log all memory write operations (store, update, delete) with:
    - Operation type, timestamp, user_id, conversation_id, session_id, **channel**
    - Affected node/record IDs
    - Payload summary (truncated content hash, not full content)
  - [x] Log all memory read operations (retrieve, search, recall) with:
    - Query text, parameters, result count, **channels searched** (e.g. `["my-project", "_global"]`)
    - Retrieval strategy used (episodic/semantic/procedural)
    - Latency per sub-query
  - [x] Log cross-channel operations (promote_to_global) with:
    - Source channel, target channel (`_global`)
    - Promoted fact/entity IDs and promotion reason (threshold met, frequency, etc.)
  - [x] Configurable log levels:
    - `off`: No audit logging
    - `writes`: Only log mutations
    - `reads`: Log reads and writes
    - `verbose`: Full query traces with payloads
- [x] Create `memory_audit_log` PostgreSQL table (partitioned by day):
  - [x] Columns: id, timestamp, operation, memory_type, user_id, conversation_id, session_id, channel, query_hash, result_count, latency_ms, metadata (JSONB) — *created in 11.1*
  - [x] Use PostgreSQL declarative partitioning: `PARTITION BY RANGE (timestamp)` with daily partitions — *created in 11.1*
  - [x] Auto-create daily partitions (via consolidation worker `manage_audit_partitions` job)
  - [x] BRIN index on timestamp for time-range queries (per partition) — *created in 11.1*
  - [x] Btree indexes on user_id, conversation_id, operation, channel — *created in 11.1*
  - [x] Retention cleanup via `DROP PARTITION` (fast, no row-by-row delete)
- [x] Add audit logging config to `MemoryConfig` (pydantic-settings):
  - [x] `audit_log_level: str = "writes"` (off | writes | reads | verbose)
  - [x] `audit_retention_days: int = 30`
  - [x] `audit_partition_ahead_days: int = 7` (pre-create partitions this many days ahead)
  - [x] `audit_sample_rate: float = 1.0` (for high-volume read sampling)
- [x] Instrument existing memory operations:
  - [x] `interface.py`: Wrap AgentMemory methods with audit calls
  - [x] `episodic.py`: Add channel support to store/retrieve operations
  - [x] `semantic.py`: Add channel support to entity/fact operations
  - [x] `procedural.py`: Add channel support to tool recording and strategy retrieval
  - [x] `working.py`: Accept audit_logger parameter (verbose logging only)
  - [x] `retrieval.py`: Add channel support to composite retrieval
- [x] Add audit log partition management to consolidation worker:
  - [x] Drop partitions older than `audit_retention_days` (daily cleanup job)
  - [x] Pre-create future partitions (`audit_partition_ahead_days` ahead)
  - [x] Job execution tracking with audit logger
- [ ] Add `/api/memory/audit` endpoint for querying audit logs (admin only, future)

### 11.5 Channel Scoping & Data Safety

> Scope memory into traceable channels; prevent cross-user data leakage

#### Channel Architecture
- `_global` — default channel, user-wide memory (preferences, general facts, communication style)
- Project channels (e.g. `my-rust-project`) — full-featured memory containers scoped to a project
- Channels are **traceable scopes, not isolation boundaries** — retrieval queries both active channel + `_global`
- Cross-channel intersections are logged in audit trail for traceability

#### Channel Implementation
- [x] Add `channel` parameter to `AgentMemory.__init__()` (default `"_global"`):
  - [x] Store as instance attribute, pass through to all memory stores
  - [x] Add `channel` to all Neo4j MERGE/CREATE operations (episodic, semantic, procedural)
  - [x] Add `channel` to all PostgreSQL INSERT statements (conversation_logs, tool_invocations)
  - [x] Add `channel` segment to Redis key prefix: `working:{user_id}:{channel}:{conversation_id}:*`
- [x] Add `channel` filtering to all read queries:
  - [x] `episodic.py`: Filter turns by channel in Cypher WHERE and SQL WHERE
  - [x] `semantic.py`: Filter facts/entities by channel
  - [x] `procedural.py`: Filter strategies/tool stats by channel
  - [x] `working.py`: Channel already scoped via Redis key prefix
- [x] Add `channel` to data models (`models.py`):
  - [x] Add `channel: str = "_global"` field to Turn, Entity, Fact, Strategy, Goal
- [x] Create channel management API:
  - [x] `GET /api/memory/channels` — list all channels with item counts
  - [x] `POST /api/memory/channels` — create a named channel
  - [x] `DELETE /api/memory/channels/{name}` — delete a channel and all its data

#### User Scoping
- [x] Add `user_id` filtering to all semantic memory queries (`semantic.py`):
  - [x] `search_facts()`: Filter by user_id in Cypher WHERE clause (already implemented)
  - [x] `search_entities()`: Filter by user_id or scope entities to user subgraph (already implemented)
  - [x] `get_entity_graph()`: Only traverse within user's subgraph
- [x] Add `user_id` property to all Neo4j nodes (Entity, Fact, Strategy):
  - [x] Include in MERGE/CREATE operations (already implemented)
  - [x] Add to vector search post-filtering (already implemented)
- [x] Add user_id to procedural memory queries:
  - [x] `find_strategies()`: Filter by user_id (already implemented)
  - [x] `get_tool_stats()`: Scope to user
- [x] Add user_id validation at `AgentMemory` interface level:
  - [x] Validate user_id is set before any write operation
  - [x] Reject empty/null user_id with clear error

### 11.6 Extensibility Infrastructure ✅

> Make the memory system easy to extend without backwards-compatibility burden

- [x] Define clear extension points with abstract base classes (`abc.py`):
  - [x] `MemoryStore` ABC: `store()`, `retrieve()`, `delete()`, `health_check()`
  - [x] `Extractor` ABC: `extract(text, context) -> list[T]`, `extract_async()`
  - [x] `Embedder` ABC: `embed(texts)`, `embed_single(text)`, `embed_batch(texts)`, `dimensions` property
  - [x] `Reranker` ABC: `rerank(query, results)`, `rerank_async()`
  - [x] Supporting dataclasses: `ScoredResult`, `HealthStatus`
- [x] Make retrieval strategy weights configurable per-request (`retrieval.py`):
  - [x] Add `RetrievalWeights.from_dict()` and `merge()` methods
  - [x] Allow `remember(query, strategy_weights={...})` override
  - [x] Support disabling specific memory types per query (existing params)
  - [x] Add default weights to config.py
- [x] Add event hooks for memory lifecycle (`events.py`):
  - [x] `on_turn_stored(turn)` — via `MemoryEventEmitter.TURN_STORED`
  - [x] `on_fact_learned(fact)` — via `MemoryEventEmitter.FACT_LEARNED`
  - [x] `on_entity_created(entity)` — via `MemoryEventEmitter.ENTITY_CREATED`
  - [x] `on_retrieval_complete(query, results)` — via `MemoryEventEmitter.RETRIEVAL_COMPLETE`
  - [x] Implement as simple callback registry (no framework dependency)
  - [x] Support both sync and async callbacks
  - [x] Wire events into `AgentMemory` interface (`interface.py`)
- [x] Export new classes in `__init__.py`:
  - [x] ABCs: MemoryStore, Embedder, Extractor, Reranker, ScoredResult, HealthStatus
  - [x] Events: MemoryEventEmitter, EventPayload, TurnStoredPayload, FactLearnedPayload, EntityCreatedPayload, RetrievalCompletePayload
  - [x] RetrievalWeights
- [ ] Document extension patterns (deferred to Phase 12):
  - [ ] How to add a new memory store type
  - [ ] How to add a new extraction method
  - [ ] How to add a custom reranker
  - [ ] How to hook into memory events

### 11.7 Retrieval Quality ✅

> Improve retrieval beyond the current basic implementation

- [x] Implement multi-channel retrieval in `retrieval.py`:
  - [x] `retrieve()` queries both active channel and `_global` channel
  - [x] Merge results from both channels before reranking
  - [x] Tag each result with its source channel in `MemoryBundle`
  - [x] Allow caller to override which channels to search: `remember(query, channels=["_global", "my-project"])`
  - [x] Weight active channel results slightly higher than `_global` (configurable boost factor)
- [x] Fix reranking (`retrieval.py`):
  - [x] Replace conversation-diversity-only filter with proper scoring
  - [x] Add cross-encoder reranking option (configurable, off by default)
  - [x] Add relevance score normalization across memory types
- [x] Add retrieval caching:
  - [x] Cache recent retrieval results in Redis with short TTL (~60s)
  - [x] Invalidate on new writes to same user/conversation/channel scope
- [x] Add retrieval metrics:
  - [x] Track hit rates per memory type and per channel
  - [x] Track average latency per retrieval strategy
  - [x] Log to audit table (when audit level >= reads)

### 11.8 Memory System Tests

> Tests deferred from Phase 10 — requires functional memory (11.1-11.5)

- [x] Unit tests (mock databases):
  - [x] Test `MemoryEventEmitter`: callbacks, unsubscribe, emit, error handling, disable
  - [x] Test `RetrievalWeights`: defaults, from_dict, from_config, merge
  - [x] Test `MemoryAuditLogger`: log levels (off/writes/reads/verbose), timed operations
  - [x] Test `WorkingMemory`: add_turn, get/set, TTL, channel isolation
  - [x] Test `MemoryRetriever` with mocked memory stores: cache keys, normalize scores, recency decay
  - [x] Test extraction functions (entity, fact, relationship) — in `ExtractionPipelineTest`
  - [x] Test audit logger writes correct records
  - [x] Test tenant isolation (user A cannot read user B's data) — via channel scoping tests
  - [x] Test channel scoping (project channel data not returned when querying a different channel)
  - [x] Test multi-channel retrieval (active channel + `_global` both searched, results merged)
  - [x] Test `_global` channel is always included in retrieval regardless of active channel

#### Phase 11.8+ Testing Notes (from code review) ✅
> Issues discovered during comprehensive memory module review — now covered by tests in `tests_memory.py`

- [x] Security tests:
  - [x] Test `complete_goal()` access control (user can only complete their own goals)
  - [x] Test entity type whitelist validation (invalid types default to "Entity")
- [x] Correctness tests:
  - [x] Test average latency running mean calculation over many invocations
  - [x] Test extraction timeout actually fires after configured seconds
  - [x] Test async extraction works from both sync and async contexts
  - [x] Test consolidation jobs return proper metrics dictionaries
  - [x] Test entity name case normalization in relationship linking
- [x] Performance tests:
  - [x] Test Redis SCAN pagination works correctly (vs KEYS blocking)
  - [x] Test working memory TTL refresh on read access
  - [x] Test query length validation rejects oversized queries
  - [x] Test graph traversal depth limits (max 3) and result limits
- [x] Edge case tests:
  - [x] Test time_window_hours bounds validation (negative values handled)
  - [x] Test division-by-zero protection in success rate calculations
  - [x] Test consolidated timestamp set even on partial extraction failure
  - [x] Test SQL partition name validation (alphanumeric only)
- [x] Data integrity tests:
  - [x] Test embedding stored as JSON (not Python str representation)
  - [x] Test turn_index passed correctly to tool invocation recording

- [x] Integration tests (require Docker services):
  > Tests skip gracefully when Docker not running or embedding dimensions mismatch
  - [x] Test full cycle: store turn → extract → consolidate → retrieve
  - [x] Test memory persistence across API server restarts
  - [x] Test working memory TTL expiration
  - [x] Test Neo4j vector search returns relevant results
  - [x] Test PostgreSQL audit log captures operations
  - [x] Test consolidation worker runs jobs on schedule
  - [x] Test cross-channel promotion: fact in project channel meets threshold → appears in `_global`
  - [x] Test channel CRUD: create, list, delete channel and verify data cleanup
- [x] Agent integration tests:
  - [x] Test `/api/agent/chat` stores turns in memory with correct channel
  - [x] Test `/api/agent/chat` retrieves relevant context from memory
  - [x] Test `/api/agent/run` records tool usage in procedural memory
  - [x] Test graceful degradation when databases are down

### 11.9 Consolidation & Background Worker ✅

> Verify and harden the background processing pipeline

- [ ] Test consolidation worker end-to-end:
  - [ ] Verify `consolidate_episodic_to_semantic()` with real extraction output
  - [ ] Verify `detect_patterns()` correctly identifies successful strategies
  - [ ] Verify `apply_memory_decay()` reduces salience scores over time
  - [ ] Verify `cleanup_old_memories()` archives/removes appropriately
- [x] Implement `promote_to_global()` consolidation job:
  - [x] Scan project channels for facts/entities that meet **all three** promotion criteria:
    - Confidence >= `promotion_min_confidence` (default: 0.85)
    - Access count >= `promotion_min_access_count` (default: 5)
    - Referenced in >= `promotion_min_conversations` distinct conversations (default: 2)
  - [x] Add promotion threshold settings to `MemoryConfig`:
    - [x] `promotion_min_confidence: float = 0.85`
    - [x] `promotion_min_access_count: int = 5`
    - [x] `promotion_min_conversations: int = 2`
  - [x] Copy promoted facts/entities to `_global` channel (not move — preserve originals)
  - [x] Mark promoted items with `promoted_from: channel_name` in metadata
  - [x] Log each promotion as `cross_channel_promote` in audit trail with:
    - Source channel, target channel (`_global`)
    - Promoted fact/entity IDs
    - **Active threshold values at time of promotion** (snapshot the config in the log entry)
    - Which criteria the item exceeded (confidence=0.91, access_count=7, conversations=3)
  - [x] Run on same schedule as pattern detection (hourly default)
- [x] Add Taskfile command: `task memory:worker` to start consolidation worker
- [x] Add health monitoring for worker (Redis heartbeat)
- [x] Add configurable job intervals via `MemoryConfig`
- [x] Handle worker crash recovery (detect stale job locks, re-run)

### 11.10 Memory Explorer (Client) ✅

> Primitive v1 UI for inspecting memory contents — entities, facts, channels

**Note**: Embeddings are opaque vectors, but every stored memory has human-readable content alongside it (entity names, fact claims, turn content). The explorer renders the readable data; embeddings are just the search index underneath.

- [x] Create `MemoryTab` component (or subsection within existing tab):
  - [x] Channel selector dropdown (list channels via `GET /api/memory/channels`)
  - [x] Entity list view:
    - [x] Columns: name, type, channel, salience score, last accessed
    - [x] Click entity → expand to show connected facts and relationships
    - [x] Uses `get_entity_graph()` for expansion (already implemented in semantic.py)
  - [x] Fact list view:
    - [x] Columns: claim text, confidence, source channel, derived-from conversation
    - [x] Filter by channel, confidence threshold
    - [x] Show `promoted_from` badge for cross-channel promoted facts
  - [x] Strategy list view (procedural memory):
    - [x] Columns: description, tool sequence, success rate, channel
- [x] Create memory API endpoints to support explorer:
  - [x] `GET /api/memory/entities?channel=X` — list entities with pagination
  - [x] `GET /api/memory/entities/{id}/graph` — entity subgraph (facts, relationships)
  - [x] `GET /api/memory/facts?channel=X` — list facts with pagination
  - [x] `GET /api/memory/strategies?channel=X` — list strategies
  - [x] `GET /api/memory/stats` — counts per memory type, per channel
- [x] Basic search within explorer:
  - [x] Text search across entity names and fact claims (Neo4j fulltext or CONTAINS)
  - [x] No embedding-based search needed for v1 — just text matching

### 11.11 Background Job Scheduler & Monitoring

> Job scheduler for running consolidation and other background tasks, with UI for monitoring and manual triggering
> **Note**: Scheduler implementation is deferred, but monitoring UI should work with manual triggers for debugging

#### Scheduler (Deferred)
- [ ] Implement background job scheduler:
  - [ ] Configurable job intervals via `MemoryConfig` (already in config)
  - [ ] Job registration system (cron-like or interval-based)
  - [ ] Job persistence (survive API restarts)
  - [ ] Graceful shutdown handling
  - [ ] Consider: Django Q, Celery, or lightweight custom scheduler
- [ ] Register consolidation jobs:
  - [ ] `consolidate_episodic_to_semantic` — extract entities/facts from turns
  - [ ] `detect_patterns` — identify successful tool strategies
  - [ ] `apply_memory_decay` — reduce salience over time
  - [ ] `cleanup_old_memories` — archive/remove old data
  - [ ] `promote_to_global` — cross-channel promotion
  - [ ] `manage_audit_partitions` — partition maintenance

#### Job Monitoring API
- [x] Add job status endpoints:
  - [x] `GET /api/jobs` — list all registered jobs with status (running, idle, failed, disabled)
  - [x] `GET /api/jobs/{job_id}` — job details (last run, next run, run count, last error)
  - [x] `GET /api/jobs/{job_id}/history` — recent execution history with metrics
  - [ ] `GET /api/jobs/{job_id}/logs` — logs from recent runs (paginated) — deferred
  - [x] `POST /api/jobs/{job_id}/run` — manually trigger a job (for debugging)
  - [x] `POST /api/jobs/{job_id}/toggle` — enable/disable a scheduled job
- [x] Job metrics tracking:
  - [x] Execution count, success count, failure count
  - [x] Average duration
  - [x] Last run timestamp, last success timestamp
  - [x] Items processed (entities extracted, facts stored, etc.)
  - [x] Store in Redis (JobRegistry)

#### Job Monitoring UI (Client)
- [x] Create `JobsPanel` component (subsection in Memory tab):
  - [x] Job list view:
    - [x] Columns: job name, status (badge), last run (relative time), success rate
    - [x] Status badges: Running (cyan pulse), Idle (gray), Disabled (muted)
    - [x] Click row to expand details
  - [x] Job detail view:
    - [x] Metrics grid: run count, success count, failure count, average duration
    - [x] Recent execution history (last 5 runs with status, duration, items processed)
    - [ ] Log viewer (scrollable, auto-refresh when job running) — deferred
    - [x] Error display (last error)
  - [x] Manual controls:
    - [x] "Run Now" button — triggers job immediately via `POST /api/jobs/{id}/run`
    - [x] "Enable/Disable" toggle — for scheduled jobs
  - [ ] Real-time updates — deferred:
    - [ ] Poll job status while job is running (5s interval)
    - [ ] Show progress indicator for long-running jobs
    - [ ] Toast notification on job completion/failure
- [x] Styling:
  - [x] Follow existing cosmic theme
  - [x] Use status colors consistently (cyan=running, green=success, red=failed)
  - [x] Responsive layout

#### Manual Consolidation Trigger (Priority for Debugging) ✅
- [x] Add immediate consolidation endpoint:
  - [x] `POST /api/memory/consolidate` — runs full consolidation pipeline synchronously
  - [x] Request body: `{ "jobs": ["consolidate", "patterns", "promote"] }` or empty for default
  - [x] Response: job results with metrics (entities extracted, facts stored, etc.)
- [x] Add "Consolidate Now" button to Memory Explorer:
  - [x] Prominent button in Memory tab header
  - [x] Shows loading state during consolidation
  - [x] Displays results summary (duration, items processed)
  - [x] Refreshes entity/fact lists after completion
- [ ] Consolidation preview (optional) — deferred:
  - [ ] `POST /api/memory/consolidate/preview` — dry run showing what would be extracted
  - [ ] Useful for debugging extraction quality without persisting

---

## Phase 12: Documentation

> **Priority**: LOW
> **Goal**: Comprehensive documentation for users and developers

### 12.1 User Documentation
- [ ] Update README.md with:
  - Quick start guide
  - Feature overview
  - Screenshots
- [ ] Add installation guide for each platform
- [ ] Document MCP setup for AI assistants

### 12.2 Developer Documentation
- [ ] Update CLAUDE.md with MCP details
- [ ] Add API documentation (auto-generate from OpenAPI)
- [ ] Add architecture diagrams
- [ ] Document contribution guidelines
- [ ] Document memory system extension patterns (from 11.6)

### 12.3 Inline Documentation
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout Python code
- [ ] Document complex algorithms/flows

---

## Phase 13: UI Implementation (Chat & Agent)

> **Priority**: HIGH
> **Goal**: Deliver a dual-interface experience — lightweight Chat for quick interactions, powerful Agent for prompt engineering workflows
> **Design Principles**:
> - **Chat**: Minimal friction, fast, fun — get answers without configuration overhead
> - **Agent**: Full control over prompts, models, memory, tools — optimized for iterative prompt development
> - **Agent Profiles**: Stored configurations that personalize the agent's behavior, model selection, and tool access
> - **Conversation Management**: Branching, search, organization — treat conversations as first-class artifacts

### Current State Assessment

The existing UI has functional Chat and Agent tabs:
- ✅ Chat tab — fully functional with streaming, model selection, memory toggle, temperature, recent chats
- Agent tab has task execution but no conversation management
- No concept of Agent Profiles (stored configurations)
- No conversation branching, search, or organization
- Settings tab doesn't support profile management
- No prompt library or template system

### 13.1 Chat Tab — Lightweight Interface ✅

> Simple, fast, enjoyable chat experience with minimal configuration
> **Status**: COMPLETE

- [x] Chat message input:
  - [x] Auto-growing textarea with submit on Enter (Shift+Enter for newline)
  - [x] Character/token counter (subtle, non-intrusive)
  - [x] Paste image support (placeholder — detects paste, shows coming soon)
  - [x] Voice input button (placeholder — disabled, coming soon)
- [x] Message display:
  - [x] Streaming response with typing indicator
  - [x] Markdown rendering (code blocks with syntax highlighting, tables, lists)
  - [x] Copy button per message
  - [x] Regenerate button on assistant messages
  - [x] Timestamp on hover
- [x] Quick settings bar (minimal, above input):
  - [x] Model selector dropdown (default from settings)
  - [x] Temperature slider (collapsed by default in "Advanced Settings")
  - [x] "Use memory" toggle (on/off, uses `_global` channel)
- [x] Session management:
  - [x] New chat button (clears current, optionally saves)
  - [x] Recent chats list (last 10, ephemeral unless explicitly saved)
  - [ ] "Save to Agent" button (deferred to 13.2 — requires Agent conversation support)
- [x] Styling:
  - [x] Compact, centered layout (max-width ~800px)
  - [x] Subtle cosmic theme accents without overwhelming
  - [x] Fast animations, no blocking transitions

### 13.2 Agent Tab — Power Interface

> Full-featured conversation interface for prompt engineering workflows

#### 13.2.1 Conversation Sidebar
- [ ] Conversation list with:
  - [ ] Title (auto-generated or user-set)
  - [ ] Preview snippet (first user message)
  - [ ] Timestamp (relative: "2h ago", "Yesterday")
  - [ ] Active profile badge/icon
  - [ ] Unread/updated indicator
- [ ] Organization:
  - [ ] Folders/workspaces (drag-drop conversations)
  - [ ] Tags (colored labels, filterable)
  - [ ] Star/favorite toggle
  - [ ] Archive (hidden but searchable)
- [ ] Search:
  - [ ] Full-text search across all conversations
  - [ ] Filter by: date range, profile, tags, folder
  - [ ] Search within current conversation
- [ ] Bulk actions:
  - [ ] Multi-select conversations
  - [ ] Bulk delete, archive, move to folder, apply tag

#### 13.2.2 Conversation View
- [ ] Message display:
  - [ ] User messages (editable on click)
  - [ ] Assistant messages with metadata (model, tokens, latency)
  - [ ] System messages (profile's system prompt, shown collapsed)
  - [ ] Tool usage blocks (expandable: input/output/status)
  - [ ] Reasoning trace toggle (show/hide thinking steps)
  - [ ] Memory retrieval indicator (what context was injected)
- [ ] Message operations:
  - [ ] Edit message → regenerate from that point (creates branch)
  - [ ] Regenerate with different settings (model, temperature)
  - [ ] Delete message (and all descendants)
  - [ ] Pin message (always include in context)
  - [ ] Add annotation/note (user-only, not sent to model)
  - [ ] Copy message (plain text or markdown)
- [ ] Branching:
  - [ ] Visual branch indicator when conversation forks
  - [ ] Branch switcher (dropdown or tree view)
  - [ ] Compare branches side-by-side (split view)
  - [ ] Merge branch (copy messages from one branch to another)
  - [ ] Delete branch

#### 13.2.3 Input Area
- [ ] Rich input:
  - [ ] Multi-line textarea with auto-grow
  - [ ] Slash commands (`/profile`, `/clear`, `/export`, `/branch`)
  - [ ] @-mentions for injecting context (`@memory`, `@file:path`, `@tool:name`)
  - [ ] Template insertion (from prompt library)
  - [ ] Drag-drop file attachment (sent as context or tool input)
- [ ] Context controls:
  - [ ] Context window visualization (% used, token count)
  - [ ] "Include in context" checkboxes per message
  - [ ] Truncation preview (what will be cut if over limit)
  - [ ] "Continue from here" button (start fresh context from selected point)
- [ ] Action buttons:
  - [ ] Send (primary)
  - [ ] Send with options (dropdown: different model, different profile)
  - [ ] Stop generation (cancel in-flight request)
  - [ ] Retry last (regenerate last assistant message)

#### 13.2.4 Conversation Header Bar
- [ ] Profile selector:
  - [ ] Current profile name and icon
  - [ ] Quick-switch dropdown
  - [ ] "Edit profile" shortcut
- [ ] Conversation actions:
  - [ ] Rename conversation
  - [ ] Export (Markdown, JSON, HTML)
  - [ ] ~~Share (generate read-only snapshot)~~ — deferred to backlog
  - [ ] Delete conversation
- [ ] View toggles:
  - [ ] Show/hide reasoning traces
  - [ ] Show/hide tool usage
  - [ ] Show/hide memory context
  - [ ] Compact/expanded message view

### 13.3 Agent Profiles

> Stored configurations that define agent behavior, model selection, and tool access

#### 13.3.1 Profile Data Model
- [ ] Define `AgentProfile` structure:
  ```typescript
  interface AgentProfile {
    id: string;
    name: string;
    icon?: string;              // Emoji or icon identifier
    description?: string;

    // Inheritance
    extends?: string;           // Parent profile ID (inherit and override)

    // Core settings
    systemPrompt?: string;      // Optional if inheriting
    model?: string;             // Primary model ID
    fallbackModel?: string;     // Fallback if primary unavailable
    temperature?: number;
    topP?: number;
    maxTokens?: number;

    // Reasoning & Drafting
    reasoningStrategy?: 'auto' | 'cot' | 'tot' | 'react' | 'reflection';
    draftingStrategy?: 'none' | 'speculative' | 'pipeline' | 'candidate';

    // Memory
    memoryEnabled?: boolean;
    memoryChannel?: string;           // Default channel for this profile
    memoryTopK?: number;              // How many memories to retrieve
    memoryAllowedChannels?: string[]; // (Optional) Restrict to these channels — if unset, query all accessible channels

    // Tools & MCP
    enabledMcpServers?: string[];  // Server IDs
    toolAllowlist?: string[];      // Specific tools to enable (if set, only these)
    toolBlocklist?: string[];      // Tools to disable (applied after allowlist)

    // Metadata
    createdAt: string;
    updatedAt: string;
    version: number;            // For versioning/history
  }
  ```
- [ ] Profile inheritance:
  - [ ] `extends` field references parent profile ID
  - [ ] Child profile values override parent (shallow merge)
  - [ ] Resolve inheritance chain at runtime (max depth: 3 to prevent cycles)
  - [ ] UI shows "Inherits from: X" badge with link to parent
  - [ ] Prevent circular inheritance (validate on save)
- [ ] Storage:
  - [ ] localStorage: `agentx:profiles`, `agentx:profile:{id}`
  - [ ] Default profile: `agentx:defaultProfile`
  - [ ] Profile history: `agentx:profile:{id}:history` (last N versions)

#### 13.3.2 Profile Management UI (Settings Tab)
- [ ] Profile list view:
  - [ ] Grid or list of profiles with icon, name, description
  - [ ] "Default" badge on default profile
  - [ ] Quick actions: edit, clone, delete, set as default
  - [ ] Create new profile button
- [ ] Profile editor:
  - [ ] Tabbed interface: General, Model, Reasoning, Memory, Tools
  - [ ] **General tab**: name, icon picker, description, system prompt (large textarea)
  - [ ] **Model tab**: primary model selector, fallback model, temperature slider, top_p, max_tokens
  - [ ] **Reasoning tab**: strategy selector with descriptions, strategy-specific options
  - [ ] **Memory tab**: enable toggle, channel selector/creator, top_k slider
    - [ ] (Optional) Allowed channels multi-select — if set, restricts retrieval to selected channels only; default: query all
  - [ ] **Tools tab**: MCP server toggles, tool allowlist/blocklist editor
  - [ ] Save / Save as new / Discard changes
  - [ ] "Test profile" button (opens quick chat with profile applied)
- [ ] Profile operations:
  - [ ] Clone profile (creates copy with "Copy of X" name)
  - [ ] Export profile (JSON download)
  - [ ] Import profile (JSON upload, conflict resolution)
  - [ ] View history (list of saved versions)
  - [ ] Restore from history
  - [ ] Delete profile (with confirmation, cannot delete if in use)

#### 13.3.3 Built-in Profile Templates
- [ ] Ship default profiles users can clone:
  - [ ] **General Assistant**: Balanced settings, memory enabled, all tools
  - [ ] **Code Helper**: Technical system prompt, reasoning=react, code-related MCP tools
  - [ ] **Creative Writer**: Higher temperature, drafting=candidate, memory off
  - [ ] **Researcher**: reasoning=tot, memory enabled, search tools prioritized
  - [ ] **Minimal**: No memory, no tools, just model + system prompt

### 13.4 Prompt Library

> Global reusable prompt templates and snippets — shared across all profiles, organized by tags

- [ ] Library structure:
  - [ ] **System prompts**: Full system prompt templates
  - [ ] **User templates**: Reusable user message templates with placeholders
  - [ ] **Snippets**: Short text fragments to insert (boilerplate, instructions)
  - [ ] All templates are **global** (not per-profile)
- [ ] Tagging system:
  - [ ] Tags: user-defined labels (e.g., "coding", "creative", "research", "work")
  - [ ] Filter library by tags (multi-select)
  - [ ] Suggested tags based on content (optional, future)
  - [ ] Tag management UI (create, rename, delete, merge)
- [ ] Library UI:
  - [ ] Sidebar panel or modal (accessible from Agent input area)
  - [ ] Tag filter bar at top
  - [ ] Search within filtered results
  - [ ] Sort by: name, recently used, created date
  - [ ] Preview before insert
  - [ ] Edit in place
- [ ] Template features:
  - [ ] Placeholders: `{{variable}}` syntax with fill-in-the-blank on insert
  - [ ] Import/export (JSON, includes tags)
  - [ ] Share templates (copy as JSON)
  - [ ] Duplicate template
- [ ] Integration:
  - [ ] `/template` slash command in Agent input (fuzzy search by name/tag)
  - [ ] Quick insert button next to input area
  - [ ] Profile can reference a system prompt template by ID (stays in sync if template updated)

### 13.5 Conversation Persistence & Sync

> Backend support for conversation and profile storage

- [ ] API endpoints for conversations:
  - [ ] `POST /api/conversations` — create conversation
  - [ ] `GET /api/conversations` — list conversations (with pagination, filters)
  - [ ] `GET /api/conversations/{id}` — get conversation with messages
  - [ ] `PUT /api/conversations/{id}` — update conversation metadata
  - [ ] `DELETE /api/conversations/{id}` — delete conversation
  - [ ] `POST /api/conversations/{id}/messages` — add message
  - [ ] `PUT /api/conversations/{id}/messages/{msgId}` — edit message
  - [ ] `POST /api/conversations/{id}/branch` — create branch from message
  - [ ] `GET /api/conversations/{id}/branches` — list branches
  - [ ] `POST /api/conversations/{id}/export` — export conversation
- [ ] API endpoints for profiles:
  - [ ] `GET /api/profiles` — list profiles
  - [ ] `POST /api/profiles` — create profile
  - [ ] `GET /api/profiles/{id}` — get profile
  - [ ] `PUT /api/profiles/{id}` — update profile
  - [ ] `DELETE /api/profiles/{id}` — delete profile
  - [ ] `POST /api/profiles/{id}/clone` — clone profile
  - [ ] `GET /api/profiles/{id}/history` — get version history
  - [ ] `POST /api/profiles/{id}/restore` — restore from history
- [ ] Database models:
  - [ ] `Conversation` table: id, user_id, profile_id, title, folder, tags, starred, archived, created_at, updated_at
  - [ ] `Message` table: id, conversation_id, parent_id (for branching), role, content, metadata (model, tokens, latency), created_at
  - [ ] `Profile` table: id, user_id, name, config (JSONB), version, created_at, updated_at
  - [ ] `ProfileHistory` table: id, profile_id, version, config (JSONB), created_at
  - [ ] `PromptTemplate` table: id, user_id, type (system|user|snippet), name, content, placeholders (JSONB), tags (TEXT[]), created_at, updated_at
  - [ ] `Tag` table: id, user_id, name, color, created_at (for tag management)

### 13.6 Settings Tab Rework

> Reorganize settings to accommodate profiles and new features

- [ ] Settings sections:
  - [ ] **General**: Theme, language, keyboard shortcuts
  - [ ] **Agent Profiles**: Profile list, create/edit (link to 13.3.2)
  - [ ] **Model Providers**: API keys, provider health, default model
  - [ ] **MCP Servers**: Server configuration (existing)
  - [ ] **Memory**: Global memory settings, channel management
  - [ ] **Prompt Library**: Manage templates and snippets (link to 13.4)
  - [ ] **Data**: Export all data, import, clear local storage
- [ ] Settings persistence:
  - [ ] Migrate existing settings to new structure
  - [ ] Settings schema versioning for future migrations
  - [ ] Sync settings across tabs (storage events)

### 13.7 Keyboard Shortcuts & Accessibility

> Power user efficiency features

- [ ] Global shortcuts:
  - [ ] `Cmd/Ctrl + K`: Command palette (quick actions)
  - [ ] `Cmd/Ctrl + N`: New conversation
  - [ ] `Cmd/Ctrl + P`: Switch profile
  - [ ] `Cmd/Ctrl + /`: Focus search
  - [ ] `Cmd/Ctrl + Enter`: Send message
  - [ ] `Escape`: Cancel generation / close modal
- [ ] Conversation shortcuts:
  - [ ] `↑/↓`: Navigate messages
  - [ ] `E`: Edit selected message
  - [ ] `R`: Regenerate selected message
  - [ ] `B`: Create branch at selected message
  - [ ] `D`: Delete selected message
- [ ] Command palette:
  - [ ] Quick access to all actions
  - [ ] Fuzzy search
  - [ ] Recent commands
  - [ ] Profile switching
  - [ ] Conversation switching

### 13.8 Mobile-Responsive Considerations

> Ensure usability on smaller screens (Tauri window resizing)

- [ ] Responsive breakpoints:
  - [ ] Collapse sidebar at <1024px width
  - [ ] Stack header controls at <768px
  - [ ] Full-width input at <640px
- [ ] Touch-friendly:
  - [ ] Larger tap targets on mobile
  - [ ] Swipe gestures for sidebar
  - [ ] Long-press for context menu

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
- [ ] Cross-encoder reranking model for retrieval quality
- [ ] Memory export/import (JSON/SQLite backup)
- [ ] Advanced memory visualization (interactive graph rendering, embedding similarity clusters)
- [ ] Streaming memory retrieval during chat (progressive context injection)
- [ ] Conversation sharing (generate read-only shareable links/snapshots)

### Known Future Issues (from Phase 11.7+ review)

> Architectural concerns that may need addressing at scale

**Distributed Transaction Support**
- Dual-write to Neo4j + PostgreSQL has no transaction coordination
- If Neo4j write succeeds but PostgreSQL fails, data becomes inconsistent
- Mitigation: Consider CDC (Change Data Capture) or event log for consistency
- Impact: LOW for single-user; HIGH for multi-user deployment

**Connection Timeout Configuration**
- Neo4j and PostgreSQL queries have no explicit statement timeouts
- Slow queries could hang indefinitely
- Fix: Add `statement_timeout` to connection config

**Retry Logic for Transient Failures**
- No exponential backoff on transient database failures
- Operations fail immediately if database momentarily unavailable
- Fix: Add retry decorator with backoff for critical operations

**Rate Limiting on Memory Operations**
- No protection against rapid-fire memory operations
- Could exhaust database connections or memory
- Fix: Add per-user rate limits in AgentMemory

**Encryption at Rest**
- Conversation history and facts stored unencrypted
- If database compromised, all user data readable
- Fix: Enable database-level encryption or app-level encryption for sensitive fields

**Settings Cached at Import Time**
- `get_settings()` uses `@lru_cache`, loaded at module import
- Runtime `.env` changes not picked up without restart
- Mitigation: Document that settings require restart; consider cache invalidation

**Query Embedding Caching**
- Every `remember()` call generates embedding even for identical queries
- Wastes computation for repeated queries in rapid succession
- Fix: Add MRU cache for frequent query embeddings with TTL

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
| Phase 9: Security | ✅ Complete | 100% (Foundation) |
| Phase 10: Testing (Core) | ✅ Complete | 100% |
| Phase 11: Memory System | In Progress | 95% |
| Phase 12: Documentation | Not Started | 0% |
| Phase 13: UI Implementation | In Progress | 15% (13.1 complete) |

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
| 2026-01-31 | Extensible memory over backwards-compat | Memory system prioritizes easy extension; no backwards-compatibility shims |
| 2026-01-31 | Audit logging in PostgreSQL | All memory operations traceable per session with configurable verbosity |
| 2026-01-31 | Tenant isolation required | Semantic memory must filter by user_id to prevent cross-user data leakage |
| 2026-01-31 | Extraction via LLM preferred | Entity/fact extraction uses model providers for quality; spaCy as lightweight fallback |
| 2026-01-31 | Memory tests deferred to Phase 11 | Cannot test memory until schemas exist and integration is wired |
| 2026-01-31 | Memory channels over multiple databases | Logical channel scoping (property on nodes/rows) is trivial; multiple DBs would multiply infra complexity for a namespace problem |
| 2026-01-31 | `_global` as default channel | User-wide memory (preferences, general facts) lives in `_global`; project channels are full-featured containers |
| 2026-01-31 | Channels are traceable scopes, not isolation | Retrieval merges active channel + `_global`; cross-channel intersections logged in audit trail |
| 2026-01-31 | Cross-channel promotion via consolidation | Prominent project facts auto-promote to `_global` based on confidence/frequency thresholds |
| 2026-01-31 | Audit logs partitioned by day | Configurable retention with daily resolution; per-day log chunks for efficient cleanup and querying |
| 2026-01-31 | Memory retrieval is blocking | Synchronous retrieval in chat flow for simplicity; may scale to concurrent sub-queries for complex tasks later |
| 2026-01-31 | LLM-only extraction, no spaCy | Entity/fact extraction uses model providers exclusively; spaCy too constraining for open-ended semantic extraction |
| 2026-01-31 | Promotion thresholds configurable, logged | Defaults: confidence>=0.85, access_count>=5, conversations>=2; active thresholds snapshotted in each audit log entry |
| 2026-02-13 | Dual interface: Chat vs Agent | Chat = minimal friction for quick Q&A; Agent = full power for prompt engineering workflows |
| 2026-02-13 | Agent Profiles as first-class concept | Stored configurations (system prompt, model, reasoning, tools) that personalize agent behavior; switchable per conversation |
| 2026-02-13 | Conversation branching over linear history | Power users need to explore alternatives; branching at any message creates forks without losing context |
| 2026-02-13 | Prompt Library for reusable templates | System prompts, user templates with placeholders, snippets — all importable/exportable |
| 2026-02-13 | localStorage-first, API-backed persistence | Profiles and conversations in localStorage for speed; API endpoints for sync and backup |
| 2026-02-13 | Profile inheritance via `extends` | Child profiles inherit parent settings with overrides; max depth 3; prevents code duplication across similar profiles |
| 2026-02-13 | Non-restrictive memory by default | Profiles query all accessible channels unless `memoryAllowedChannels` is set; simplifies default behavior |
| 2026-02-13 | Global prompt library with tags | Templates shared across profiles (not per-profile); tags for organization and filtering |
| 2026-02-22 | Scheduler deferred, manual trigger priority | Background job scheduler implementation deferred; manual consolidation trigger and job monitoring UI added for debugging workflows |

### Blockers
- ~~**Memory activation blocked on**: Database schema initialization (11.1) must complete before any other 11.x work~~ ✅ Resolved

### Questions to Resolve
- [x] Which LLM providers to prioritize? (OpenAI / Anthropic / Ollama / Together) → OpenAI, Anthropic, Ollama implemented; LM-Studio preferred
- [x] Should agent memory require authentication? → No, one server = one user. Multiple servers can exist on one system. Simple architecture, requires rich export capability for effective long-term usage.
- [X] Target platforms for distribution? (Windows / macOS / Linux): Target for Linux and Windows for now. 
- [x] Reasoning trace storage format? (JSON / SQLite / Neo4j) → Neo4j graph + PostgreSQL audit log
- [x] Entity extraction method: LLM-based (higher quality, API cost) vs spaCy (offline, faster, less accurate)? → LLM-only via model providers; spaCy too constraining for open-ended extraction
- [X] Audit log retention policy: 30 days default — should it be configurable per deployment? Leave it configurable with daily resolution with per-day log chunks. 
- [X] Should memory retrieval be async (non-blocking) or sync (blocking) in the chat flow? Memory retrieval should be blocking for now, but potentially we may need to scale that up to concurrent queries for complex tasks...
- [x] Cross-channel promotion thresholds: what confidence/frequency values are sensible defaults? → confidence>=0.85, access_count>=5, conversations>=2; all configurable in MemoryConfig, active values snapshotted in audit log entries

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

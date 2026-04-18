# AgentX Development To-do

**Project**: AgentX - AI Agent Platform
**Status**: Pre-prototype
**Last Updated**: 2026-03-29

> For completed phases (1-14) and project history, see [docs/roadmap.md](docs/roadmap.md)

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phases 1-10 | Complete | See [roadmap.md](docs/roadmap.md) |
| Phase 11: Memory System | **Complete** | See [roadmap.md](docs/roadmap.md) |
| Phase 12: Documentation | Partial | ~60% |
| Phase 13: UX Overhaul | **Complete** | See [roadmap.md](docs/roadmap.md) |
| Phase 14: Context Gating | **Complete** | See [roadmap.md](docs/roadmap.md) |
| Phase 15: Plan Execution | **In Progress** | ~80% |
| Phase 16: Multi-Agent Conversations | Not Started | 0% |
| Phase 17: Server Management | **In Progress** | ~70% |

---

## Phase 11: Deferred Items

> Remaining items from the memory system that didn't make the cut

- [ ] Optional LLM disambiguation for ambiguous entity matches (11.12.3)
- [ ] LLM timeout enforcement (requires async/sync architecture fix)
- [ ] Calibration factors: source, recency, corroboration, contradiction
- [ ] Negative reinforcement for corrected facts
- [ ] UI: "Where did I learn this?" — show original conversation from `source_turn_id`

---

## Phase 12: Documentation

> **Priority**: LOW

- [ ] Auto-generate API docs from OpenAPI
- [ ] Document contribution guidelines
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout Python code
- [ ] Document complex algorithms/flows

---

## Phase 13: UX Overhaul — Immersive AgentX (Complete)

> Moved to [roadmap.md](docs/roadmap.md). All items complete:
> - 13.6 AgentXPage with ConversationContext + ChatPanel
> - 13.10 Removed old tabs dir, renamed to panels, keyboard shortcuts (Ctrl+T/W/K), CSS pass
> - 13.11 Consolidation settings UI + TopBar integration

---

## Phase 14: Context Gating (Complete)

> All 14.1-14.7 complete. See [roadmap.md](docs/roadmap.md) for details.

---

## Phase 15: Plan Execution

> **Priority**: HIGH
> **Goal**: Execute decomposed task plans instead of discarding them — subtask iteration, Redis state tracking, streaming progress events
> **Depends on**: Memory system (Phase 11), Context gating (Phase 14)

### 15.1 Redis Plan State Store (`plan_state.py`)

- [x] `PlanStateStore` class with Redis Hash per active plan
- [x] Key pattern: `plan:{session_id}:{plan_id}`, TTL 1 hour
- [x] Per-subtask status tracking (pending/running/complete/failed/skipped)
- [x] Best-effort Redis ops — execution continues if Redis is down

### 15.2 Plan Executor (`plan_executor.py`)

- [x] `PlanExecutor` class with sync `execute()` and async `execute_streaming()`
- [x] Subtask iteration via `TaskPlan.get_next_subtask()` respecting dependencies
- [x] Per-subtask message building (system prompt + dependency results + description)
- [x] Streaming tool-use loop per subtask (reuses provider.stream + agent._execute_tool_calls)
- [x] Trajectory compression within subtask execution
- [x] Failure handling: mark failed, skip dependents whose deps all failed
- [x] Synthesis step: final LLM call composing subtask results into coherent answer

### 15.3 Planner Serialization

- [x] `Subtask.to_dict()` and `TaskPlan.to_dict()` for Redis storage

### 15.4 Agent.run() Integration (`core.py`)

- [x] Plan execution branch for MODERATE/COMPLEX plans with >1 subtask
- [x] SIMPLE tasks use existing reasoning + direct completion path unchanged
- [x] Memory reflection and goal completion after plan execution

### 15.5 Streaming Endpoint Integration (`views.py`)

- [x] Plan assessment via `TaskPlanner.plan()` before tool loop
- [x] Async delegation to `PlanExecutor.execute_streaming()` for complex tasks
- [x] Full content tracking from chunk events for memory storage
- [x] Shared done/close/memory-storage code for both paths

### 15.6 SSE Events

- [x] `plan_start` — plan_id, subtask_count, complexity
- [x] `subtask_start` — subtask_id, description, type, progress
- [x] `subtask_complete` — subtask_id, result_preview, progress
- [x] `subtask_failed` — subtask_id, error, progress
- [x] `plan_complete` — completed_count, total_time_ms
- [x] Existing `chunk`, `tool_call`, `tool_result` events reused within subtasks

### 15.7 Deferred Items

- [ ] Parallel subtask execution (independent subtasks could run concurrently)
- [ ] Per-subtask reasoning strategy selection (use `_select_strategy` per subtask type)
- [ ] Subtask-level goal tracking (create subgoals via `parent_goal_id`)
- [ ] Plan cancellation mid-execution (check `_cancel_requested` between subtasks)
- [ ] Plan resumption from Redis state after disconnect

---

## Phase 16: Multi-Agent Conversations

> **Priority**: MEDIUM
> **Goal**: Enable multiple agents to collaborate in shared conversation threads
> **Depends on**: Plan execution (Phase 15)

### 16.1 Message Attribution

- [ ] Add `agent_id` field to `Turn` model (agent_memory/models.py)
- [ ] Add `agent_id` column to `conversation_logs` PostgreSQL table
- [ ] Set `Message.name = agent_profile.agent_id` on assistant messages in views.py
- [ ] Include `agent_id` in SSE `start` and `done` events
- [ ] Store `agent_id` on turns in background turn storage

### 16.2 Explicit Agent Routing

- [ ] Add `target_agent_id` to chat stream request parsing
- [ ] Resolve `AgentProfile` by `agent_id` (scan profiles for matching ID)
- [ ] Add `participants: dict[agent_id, AgentProfile]` to `Session`
- [ ] Build per-agent system prompt with multi-agent awareness when `len(participants) > 1`

### 16.3 Tool Isolation per Agent

- [ ] Add tool group / MCP server config to `AgentProfile`
- [ ] Use existing `allowed_tools`/`blocked_tools` on `AgentConfig` for per-agent filtering
- [ ] Each agent only sees its configured tools in `_get_tools_for_provider()`

### 16.4 Agent-to-Agent Delegation

- [ ] Define delegation protocol (structured JSON in assistant output)
- [ ] Detect delegation in streaming handler, emit `delegation` SSE event
- [ ] Chain into target agent's response flow
- [ ] Safeguards: max delegation depth, no self-delegation

### 16.5 @-Mention Routing + Graph Updates

- [ ] Parse `@agent-id` from message text for implicit routing
- [ ] Add `AgentParticipant` Neo4j node and `PARTICIPATED_IN` relationship
- [ ] Migration job: backfill from existing `Conversation.agent_id` properties

### Design Notes

- `agent_id` (Docker-style, e.g., "bold-cosmic-falcon") = formal routing identifier
- `name` (e.g., "Claude", "NodeManager") = flexible display name
- `Message.name` field carries `agent_id` on assistant messages — no provider schema changes
- Extend existing `agent/chat/stream` with optional `target_agent_id` — no new endpoints
- Memory already supports this: each agent recalls from `[channel, _self_{agent_id}, _global]`

---

## Phase 17: Seamless Server Management

> **Priority**: HIGH
> **Goal**: Establish solid server self-hosted deployment and client connection tools.
> **Depends on**: None

### 17.1 Configuration

- [x] Server holds authoritative config via `ConfigManager` + `/api/config` endpoints
- [x] Client fetches config on connect, API keys remain client-editable

### 17.2 Authentication

- [x] PostgreSQL `agentx_auth` table for root user with bcrypt password hash
- [x] Redis session storage with 24h TTL (`agentx:session:{token}`)
- [x] `AuthService` class (`api/agentx_ai/auth/service.py`)
- [x] `AgentXAuthMiddleware` gates all `/api/*` routes (`api/agentx_ai/auth/middleware.py`)
- [x] Auth endpoints: `/api/auth/status`, `/api/auth/login`, `/api/auth/logout`, `/api/auth/session`, `/api/auth/change-password`, `/api/auth/setup`
- [x] `setup_auth` management command for CLI password setup (`task auth:setup`)
- [x] Settings: `AGENTX_AUTH_ENABLED`, `AGENTX_SESSION_TTL`
- [x] Client `AuthContext` + `AuthPage` for login/setup UI
- [x] Client API client injects `X-Auth-Token` header, handles 401 responses
- [x] Auth token stored in localStorage per server
- [x] Client IP tracking (`request.agentx_client_ip`) for rate-limiting and auditing

### 17.3 Docker Deployment

- [x] `Dockerfile` for API (Python 3.12, uv, uvicorn ASGI)
- [x] `docker-compose.yml` updated with `api` service (production profile)
- [x] `.env.production.example` template for Cloudflare Tunnel deployment
- [x] Taskfile commands: `auth:setup`, `prod:build`, `prod:up`, `prod:down`, `prod:logs`, etc.

### 17.4 Multi-Cluster Deployment (Experimental)

> Goal: Run multiple isolated AgentX clusters on the same host with shared source code

- [ ] Parameterized ports in `docker-compose.yml` via env vars:
  - `API_PORT` (default: 12319)
  - `NEO4J_HTTP_PORT`, `NEO4J_BOLT_PORT` (default: 7474, 7687)
  - `POSTGRES_PORT` (default: 5432)
  - `REDIS_PORT` (default: 6379)
- [ ] Parameterized data directory (`DATA_DIR`) for isolated database volumes
- [ ] Cluster identity: `AGENTX_CLUSTER_NAME` env var, included in logs and health response
- [ ] Cross-platform launcher script (`agentx-cluster`) for managing multiple clusters
- [ ] Deployment documentation for linked config pattern

### 17.5 API Version Matching

> Goal: Ensure client and API are compatible — no compatibility layers, strict version matching

- [ ] `GET /api/version` endpoint returning:
  - `version`: semver (e.g., "0.17.0")
  - `protocol_version`: integer (e.g., 1) — bumped on breaking API changes
  - `min_client_version`: minimum compatible client semver
- [ ] Include version info in `/api/health` response
- [ ] Client checks `protocol_version` on connect, shows error if mismatched
- [ ] Client shows upgrade prompt when `version < min_client_version`
- [ ] Version constants in `api/agentx_ai/__init__.py` and `client/package.json`

### 17.6 Deferred Items

- [ ] iOS/Android builds with Tauri v2 mobile support
- [ ] Cloudflare Tunnel deployment documentation
- [ ] Rate limiting on auth endpoints (login, setup)
- [ ] Request auditing log (store auth attempts, API calls with client IP)

---

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

- [ ] iOS/Android mobile builds with Tauri v2 mobile support
- [ ] Nightly consolidation scheduler — persistent job scheduler (Django Q, Celery, or custom) with cron-like registration, restart survival, graceful shutdown
- [ ] Consolidation job logs endpoint (`GET /api/jobs/{id}/logs`)
- [ ] Real-time job progress (polling while running)
- [ ] Consolidation preview (`POST /api/memory/consolidate/preview`)
- [ ] Fact Transience (a confidence bias) — ranked on extraction for the predicted rate that the fact will be incorrect or irrelevant (e.g., "The user's home PC is slow" = high transience)
- [ ] Extraction example sets for LLM to see a large list of examples to compare from
- [ ] GPU acceleration for translation models
- [ ] Lazy model loading with progress indicator
- [ ] Multiple server support (user can log out of server, and into another one seamlessly)
- [ ] Cloud sync for memories
- [ ] Plugin system for additional tools
- [ ] Voice input/output
- [ ] Offline mode with cached models
- [ ] Cross-encoder reranking model for retrieval quality
- [ ] Memory export/import (JSON/SQLite backup)
- [ ] Advanced memory visualization (interactive graph, embedding clusters)
- [ ] Streaming memory retrieval during chat
- [ ] Conversation sharing (read-only shareable links)
- [ ] Blocking tool call approval (pause stream, user approves/rejects before execution)
- [ ] Server authentication (single access key per server, session resume on reconnect)
- [ ] Mobile-responsive breakpoints and touch-friendly gestures
- [ ] Additional themes beyond cosmic (light theme, high contrast, etc.)

---

## Known Future Issues

> Architectural concerns that may need addressing at scale

**Distributed Transaction Support**
- Dual-write to Neo4j + PostgreSQL has no transaction coordination
- Impact: LOW for single-user; HIGH for multi-user deployment

**Connection Timeout Configuration**
- Neo4j and PostgreSQL queries have no explicit statement timeouts
- Fix: Add `statement_timeout` to connection config

**Retry Logic for Transient Failures**
- No exponential backoff on transient database failures
- Fix: Add retry decorator with backoff for critical operations

**Rate Limiting on Memory Operations**
- No protection against rapid-fire memory operations
- Fix: Add per-user rate limits in AgentMemory

**Encryption at Rest**
- Conversation history and facts stored unencrypted
- Fix: Enable database-level or app-level encryption

**Query Embedding Caching**
- Every `remember()` call generates embedding even for identical queries
- Fix: Add MRU cache for frequent query embeddings with TTL

---

## Blockers

None currently.

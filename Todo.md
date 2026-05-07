# AgentX Development To-do

**Project**: AgentX - AI Agent Platform
**Status**: Prototype
**Last Updated**: 2026-05-06

**NOTE**: UI must be highly responsive between PC and mobile devices.

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
| Phase 16: Multi-Agent Conversations | **In Progress** | ~30% (Agent Alloy v1 backend shipped) |
| Phase 17: Server Management | **Complete** | 100% |

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

### 16.0 Agent Alloy v1 — Backend (shipped 2026-04-27)

> **Agent Alloy** = the multi-agent system. **Factory** = the visual editor (frontend, not yet built).
> Control flow: supervisor agent owns the conversation; specialists are invoked via a `delegate_to` tool. Opt-in per chat request via `workflow_id`.

- [x] `api/agentx_ai/alloy/` package: `models.py`, `manager.py`, `delegation_tool.py`, `executor.py`, `prompts.py`
- [x] Workflow data model: supervisor + members (each with `agent_id`, role, delegation_hint), declarative `routes` (schema only), shared memory channel `_alloy_<id>`, opaque `canvas` blob for the Factory editor
- [x] `WorkflowManager` singleton with YAML CRUD at `data/workflows.yaml` (mirrors `ProfileManager`)
- [x] Validation: one supervisor, unique members, all `agent_id`s resolve to a real profile
- [x] `delegate_to` tool descriptor built per-workflow with `enum` of allowed specialists + delegation hints
- [x] `AlloyExecutor`: spawns specialist `Agent` with shared channel, streams tokens as `delegation_chunk` SSE events, creates child `Goal` (uses parent/child wiring from 15.x), stores result as Turn in shared channel
- [x] Specialist scope: just the delegated task + access to shared workflow memory channel (no full conversation)
- [x] Live streaming: new SSE events `delegation_start`, `delegation_chunk`, `delegation_complete`
- [x] Re-delegation supported up to `alloy.max_delegation_depth` (default 3)
- [x] `agent_chat_stream` accepts optional `workflow_id`; without it, behavior unchanged (fully opt-in)
- [x] Built-in supervisor framing prompt layered on top of profile system prompt when Alloy is on
- [x] CRUD endpoints: `GET/POST /api/alloy/workflows`, `GET/PATCH/DELETE /api/alloy/workflows/<id>`
- [x] Config: `alloy.max_delegation_depth`, `alloy.specialist_inherits_supervisor_tools`, `alloy.delegation_timeout_seconds`

### 16.0.1 Agent Alloy v1 — Deferred / Next

- [ ] Factory canvas frontend (Tauri client) — backend exposes everything needed
- [ ] Declarative route execution (`on_complete`, `on_match:<predicate>`, `on_failure`) — schema accepted but ignored in v1
- [ ] Parallel / fan-out delegation (supervisor delegates to multiple specialists at once)
- [ ] Loop nodes (specialist iterates over a list)
- [ ] Human-in-the-loop checkpoint nodes
- [ ] Async delegation (specialist runs in background; supervisor continues other work)
- [ ] "User as supervisor" mode (no LLM supervisor — user manually invokes specialists from the chat UI)
- [ ] Tracing / replay UI for an Alloy run
- [ ] Tool-output sharing across agents (specialist's raw tool outputs visible to supervisor, not just final text)
- [ ] Per-workflow tool isolation (specialist inherits a *subset* of supervisor tools, not all)

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

### 17.4 Multi-Cluster Deployment

> Goal: Run multiple isolated AgentX clusters on the same host with shared source code

- [x] Parameterized ports in `docker-compose.yml` via env vars (`API_PORT`, `NEO4J_*_PORT`, `POSTGRES_PORT`, `REDIS_PORT`)
- [x] Parameterized data directory (`AGENTX_DATA_DIR`) for isolated database volumes
- [x] Cluster identity: `AGENTX_CLUSTER_NAME` env var, included in health response
- [x] Cross-platform launcher scripts (`scripts/agentx-cluster`, `scripts/agentx-cluster.ps1`)
- [x] Taskfile commands: `cluster:new`, `cluster:up`, `cluster:down`, `cluster:logs`, `cluster:status`, `cluster:auth:setup`

### 17.5 API Version Matching

> Goal: Ensure client and API are compatible — no compatibility layers, strict version matching

- [x] Version constants in `api/agentx_ai/__init__.py` (VERSION, PROTOCOL_VERSION, MIN_CLIENT_VERSION)
- [x] `GET /api/version` endpoint returning version info
- [x] Version info included in `/api/health` response (version, protocol_version, cluster)
- [x] Client version injection via Vite (`__APP_VERSION__` from package.json)
- [x] Client checks `protocol_version` on connect, shows `VersionMismatchPage` if incompatible
- [x] Client package.json version bumped to 0.17.0

### 17.6 ~~Deferred~~ Items

- [X] ~~iOS/~~Android builds with Tauri v2 mobile support
- [-] Cloudflare Tunnel deployment documentation - Pending
- [-] ~~Rate limiting on auth endpoints (login, setup)~~ Put on hold with edge-security layer provided by the Gateway.
- [-] ~~Request auditing log (store auth attempts, API calls with client IP)~~ Replaced by Gateway.

---

## Phase 18: UX Improvements & Optimization and Memory Tuning

### 18.1: Wave 1 Fixes

- [x] LLM Providers Settings do not have an OpenRouter section. — Added OpenRouter + Vercel AI Gateway cards with brand SVG icons; LM Studio/Anthropic/OpenAI re-iconed and re-badged (Offline / High-Reasoning / Cloud).
- [x] Mobile UI Fix: topbar moved to bottom on mobile (`@media (max-width: 600px)` flips RootLayout to `column-reverse`), transparent `safe-area-inset-top` margin on `.page-content`, viewport-fit=cover added so insets resolve on Android.

### 18.2: Tools Menu -> Toolkit

- [ ] Migrate our settings menu for tools into our new immersive, full-screen approach for our Tools menu, transforming it into the "Toolkit".
- [ ] The Toolkit should have ability for CRD on tools in the mcp-servers.json; guided editor is the goal, but a simple text editor + pre-save checker will suffice.
- [ ] Add tool metadata; access whitelists (allow access for specific agents), tags, groups, etc. (anything else that's a quick payoff).

### 18.3: Relay Module (RM) Foundation

- [ ] Conduct these fixes concurrently:
  - [ ] Instead of a streaming toggle, offer a way to launch a background conversation; runs without opening it.
  - [ ] Instead of a DB icon to disable/enable memory consolidation on a conversation; use a toggle that says "Remember our conversation".
  - [ ] Migrate the chat toolbar into a nice menu with a button trigger that way we can clear the chat for immersion. 
- Remember that RM will be used for file communication and controlling the different modes of the model when feature support it.
- NOTE: Before we can work with images; we'll need to improve model metadata and also our file management to target image and video output models.

### 18.4: Model Metadata, Selector Refinement

Currently our model selection and selector are a very solid foundation but are just that. We cannot have advanced capabilities until we can profile our models. We're going specialice in OpenRouter and Vercel Gateway, with anthropic for high-quality reasoning.

- Improve the model selector to use filter-based lists with a comfortable UX, and show the model selector in a modal or full-screen menu; it's cluttering the agent profile and the large lists can lag the UI.
- [ ] Vercel Gateway model selection is fairly weak, only showing context limits, we want to see capabilities, pricing, max-tokens, etc. (see more https://vercel.com/docs/ai-gateway/models-and-providers#dynamic-model-discovery)
- [ ] OpenRouter has a ton of metadata like Vercel gateway, and we need match capability to what we'll add in Vercel Gateway. (see more https://openrouter.ai/docs/guides/overview/models#model-object-schema) 

## 18.5: Metrics Overhaul

- [ ] Currently the context window display gets stuck across conversations; it should dynicamically switch per conversation tab.
- [ ] Conversations should also have cost trackers with the new model metadata from subphase 18.4
- [ ] Show token + turn metrics on the dashboard (we can defer this too)

### 18.9: Memory Tuning

- [x] Fix: consolidation extractor honors user-selected model
  - Added `combined_extraction_model` / `combined_extraction_temperature` / `combined_extraction_max_tokens` to `get_consolidation_settings()` and the POST `/api/memory/settings` allowlist (config.py, views.py).
  - Removed `ExtractionService.settings` instance cache so updates flow through the existing TTL cache instead of being pinned at first call (extraction/service.py).
  - Added "Combined Extraction" section to `MemorySettingsPanels.tsx` mirroring extraction/relevance pickers; matched fields in `ConsolidationSettings` (api.ts).
- [x] Fix: stop leaking raw past turns at conversation start
  - `_retrieve_episodic` (retrieval.py) now drops cross-conversation results whose `role != "user"`. Always-include turns from the active conversation keep all roles.
  - `MemoryBundle.to_context_string` (models.py) gained `roles` and `current_conversation_id` filters; both default to permissive so other call sites (alloy/executor, agent/context) are unchanged.
  - Streaming chat (views.py) passes `current_conversation_id=conv_id`, so cross-conversation history is opt-in via the new `recall_user_history` tool instead of being auto-dumped.
- [x] New internal tool: `recall_user_history(topic?, limit=10)`
  - Registered in `mcp/internal_tools.py`; resolves the active user/channel via the new `mcp/internal_context.py` `ContextVar` set by the streaming chat endpoint.
  - Returns deduped past-conversation user turns plus top facts; capped to 30 turns; bypasses tool-output storage via `RETRIEVAL_TOOL_NAMES`.
  - Gateable per profile via the existing Phase 18.2 allowed/blocked tool lists (no new mechanism needed).
- [x] Working memory: token budget header
  - Streaming chat appends a SYSTEM line each turn: `Context budget: ~X / Y tokens (Z% used). When usage approaches 70% consider calling the checkpoint tool…` (uses existing `streaming.helpers.estimate_tokens`).
- [x] Working memory: `checkpoint(summary, decisions, next_step)` internal tool
  - Persists per-conversation entries to Redis (`agent/checkpoint_storage.py`, 7-day TTL, capped at 8 per conversation).
  - Streaming chat re-injects a `## Checkpoints (model-authored, survive compression)` SYSTEM block every turn so anchors survive trajectory compression.
- [ ] Deferred to follow-up phase
  - Pin/anchor on arbitrary turns, `scratchpad_note` tool, `inspect_working_memory`, active-goals header, `forget` tool, plus the cached `user_recap_summary` rolling summary (skipped — `recall_user_history` covers the immediate need without adding a consolidation-job dependency).

#### 18.9.x: UI follow-ups for new memory tools

> Backend ships in 18.9; these are the client-side surfaces needed to make the new functionality discoverable.

- [ ] Checkpoints sidebar/badge in the conversation pane
  - Show a small badge with the live checkpoint count for the active conversation (read from `agent/checkpoint_storage.py` via a new `GET /api/memory/checkpoints?conversation_id=` endpoint).
  - Click opens a panel listing each checkpoint's summary, decisions, next step, and `created_at`. Provide a "Clear" button (calls `DELETE /api/memory/checkpoints?conversation_id=`).
  - Toolkit gating: surface `checkpoint` in the per-profile tool allowlist UI so users can opt agents in/out.
- [ ] Recall view for `recall_user_history`
  - When the tool is invoked mid-stream, render its result as a collapsible "User history recall" card in the chat (similar treatment to existing tool result cards) instead of raw JSON.
  - Memory drawer: add a "User History" tab that calls the same tool with no topic to give a manual browse experience.
  - Toolkit gating entry alongside `checkpoint`.
- [ ] Token-budget HUD
  - The header line is currently sent only to the model. Mirror it in the existing context-window display (referenced in 18.5 first bullet) so the user sees the same pressure indicator.
  - When the model autonomously calls `checkpoint`, flash the new entry in the sidebar so the user notices the anchor was saved.
- [ ] Combined Extraction settings polish
  - Add inline help in `MemorySettingsPanels.tsx` explaining what the "Combined Extraction" stage does and when it overrides the separate Relevance + Extraction calls.
  - Add a "Reset to default" affordance for the model field (matching the existing pattern on `extraction_system_prompt`).

---

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

- [ ] New Chat Feature: Relay Module - Message and conversation tools - files, block memory toggle (no consolidation), and more.
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
- [ ] Message injection into delegated tasks (agent interdiction tools)

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

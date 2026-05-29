# AgentX Development To-do

**Project**: AgentX - AI Agent Platform
**Status**: Prototype
**Last Updated**: 2026-05-06

**NOTE**: UI must be highly responsive between PC and mobile devices.

**Versioning**: `versions.yaml` is the single source of truth (run `task versions:sync` after
editing it). Completed work is tagged inline with the version it shipped in, e.g. `[v0.20.1]`.
Bump the version when a unit of work completes — patch for additive/back-compat features, and
bump `protocol_version` only on breaking API changes. Current: **0.21.5** (protocol 1).

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
- [x] Plan cancellation mid-execution — shipped. `PlanExecutor` checks `state.is_cancel_requested(plan_id)` between subtasks (`plan_executor.py:84,163`) and marks the plan `cancelled`; `POST /api/agent/plans/cancel` sets the flag.
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

- [-] Factory canvas frontend (Tauri client) — backend exposes everything needed
- [-] Declarative route execution (`on_complete`, `on_match:<predicate>`, `on_failure`) — schema accepted but ignored in v1
- [X] Parallel / fan-out delegation (supervisor delegates to multiple specialists at once)
- [ ] Loop nodes (specialist iterates over a list)
- [ ] Human-in-the-loop checkpoint nodes
- [ ] Async delegation (specialist runs in background; supervisor continues other work)
- [ ] "User as supervisor" mode (no LLM supervisor — user manually invokes specialists from the chat UI)
- [x] Tracing / replay UI for an Alloy run — shipped 2026-05-28 `[v0.20.1]` (static inspection, not re-run).
  - Backend: `delegation_complete` now emits per-delegation `tokens_input/output`, `duration_ms`, `cost_estimate`, `cost_currency`, `pricing_snapshot` (`alloy/executor.py`, reuses `loop_result` tokens + `providers/pricing.estimate_cost`). Metrics persist into the `delegation` blob via `_run_delegations` → `delegation_raw` (`streaming/tool_loop.py`) so restored runs keep them; no `views.py` change (rides the existing carrier).
  - Client: `DelegationMessage`/`DelegationCompleteEvent` widened with the metrics + `completedAt`; `useChatStream` stamps them; `mapServerMessages` restores them. New pure selector `lib/alloyTrace.ts` (`groupRunsFromMessages`/`latestRun`) groups the depth-1 fan-out into runs with token/cost/wall-clock totals. New full-screen `components/alloy/AlloyRunTraceModal.tsx` (registered `alloyRunTrace`), opened from a "Trace" badge in the `ChatPanel` header when the tab has delegations.
  - Tests: `AlloyDelegationMetricsTest` (executor emit + `None`-cost path + `_run_delegations` persistence), `alloyTrace.test.ts` (grouping/totals/wall-clock), `mapServerMessages` metric round-trip.
  - Deferred: per-tool timing isn't persisted (executor stores one rollup turn per delegation) → restored runs show delegation-level metrics only; supervisor done-event cost intentionally excludes specialist tokens (separate rollup task).
- [ ] Tool-output sharing across agents (specialist's raw tool outputs visible to supervisor, not just final text)
- [ ] Per-workflow tool isolation (specialist inherits a *subset* of supervisor tools, not all)

### 16.1 Message Attribution — shipped 2026-05-28

- [x] `agent_id` field on `Turn` model (`agent_memory/models.py:59`)
- [x] `agent_id` column on `conversation_logs` (migration `0003_turn_agent_id.sql` + `postgres_builder.sql`; index `idx_logs_agent`)
- [x] Persist per-turn attribution: `EpisodicMemory.store_turn_log` writes `turn.agent_id` (PG, source of truth); `store_turn` stamps the Neo4j **Turn node** `agent_id` (was conversation-level only → mis-attributed specialists). Assistant turns set `agent_id` in the streaming store (`views.py:_store_turns`) **and** the non-streaming/background path (`Agent.chat`, `core.py`). User/tool turns stay NULL.
- [x] Include `agent_id` in SSE `start` + `done` events (alongside existing `agent_name`)
- [x] Restore path: `conversations_messages` selects `agent_id`, resolves it to the current profile name via `ProfileManager`, and returns both `agent_id`+`agent_name` in metadata; client `mapServerMessages` maps `agent_name` → `AssistantMessage.agentName` (rendered by `AssistantBubble`). Historical rows (NULL) fall back to the generic name.
- Tests: `TurnAgentAttributionTest` (backend round-trip), `mapServerMessages` attribution cases (client).
- Deferred: backfill of historical rows; `Message.name` provider-schema wiring (not needed — attribution rides metadata, not the provider message).

### 16.2 Explicit Agent Routing — shipped `[v0.21.2]`

- [x] Add `target_agent_id` to chat stream request parsing — optional field on
      `/api/agent/chat/stream`; resolution priority `workflow_id > target_agent_id >
      agent_profile_id > default`. Unknown id yields an SSE `error` event (views.py).
- [x] Resolve `AgentProfile` by `agent_id` — new `ProfileManager.get_profile_by_agent_id()`
      (profiles.py); the workflow-supervisor lookup now reuses it.
- [x] Add `participants: dict[agent_id, AgentProfile]` to `Session` (session.py). Hydrated
      best-effort from the durable 16.1 attribution: `EpisodicMemory.get_conversation_agent_ids()`
      (+ `AgentMemory.get_conversation_participants()` pass-through) over distinct
      `conversation_logs.agent_id`, plus the active agent — so it survives restarts/detached runs.
- [x] Build per-agent system prompt with multi-agent awareness when `len(participants) > 1` —
      new `prompts/multi_agent.py:build_participants_block()` (roster of the *other* agents,
      modeled on `alloy/prompts.build_supervisor_prompt`), injected as a SYSTEM message in the
      streaming chat path. **Suppressed inside an Alloy workflow** (the supervisor prompt already
      frames the team). Backend-only — `participants` grows organically as the existing selector
      switches agents mid-conversation; explicit client routing UI deferred to 16.5.
      Tests: `ExplicitAgentRoutingTest` (tests.py), `ConversationAgentIdsTest` (tests_memory.py).

### 16.3 Tool Isolation per Agent

- [X] Add tool group / MCP server config to `AgentProfile`
- [X] Use existing `allowed_tools`/`blocked_tools` on `AgentConfig` for per-agent filtering
- [X] Each agent only sees its configured tools in `_get_tools_for_provider()`

### 16.4 Agent-to-Agent Delegation — shipped `[v0.21.3]`

> Chosen mechanism: **reuse Alloy's `delegate_to` tool**, generalized to non-workflow
> conversations — not the originally-sketched "structured JSON in assistant output" (the Alloy
> code flags JSON-in-output as a small-model failure mode). Gated by `alloy.allow_adhoc_delegation`
> (default off). Backend-only — the client renders the existing `delegation_*` events.

- [x] Define delegation protocol — reuses the existing `delegate_to` tool (`{agent_id, task}`). New
      `build_adhoc_delegation_tool(self_agent_id)` (`alloy/delegation_tool.py`) shares a
      `_build_descriptor` core with `build_delegation_tool`; enum = every profile except self.
- [x] Detect delegation in streaming handler, emit `delegation` SSE events — no new code: the tool
      loop's `_partition_tool_calls` already routes `delegate_to` to `agent._active_alloy_executor`,
      which streams `delegation_start/chunk/complete`.
- [x] Chain into target agent's response flow — `AlloyExecutor` generalized to a workflow-less mode
      (`AlloyExecutor(agent, session, channel=, delegator_agent_id=)`); spins up the target agent,
      streams its run back, returns the result as the delegating agent's tool result. `views.py`
      attaches the ad-hoc executor + injects the tool via two `elif` branches mirroring the workflow path.
- [x] Safeguards: max delegation depth (reuses `alloy.max_delegation_depth`) + **no self-delegation**
      (new `_validate_target`, enforced in both workflow and ad-hoc modes).
      Tests: `AdhocDelegationTest` (tests.py); `AlloyDelegationMetricsTest` still green (signature change).

### 16.5 @-Mention Routing + Graph Updates — shipped `[v0.21.4]`

> Backend-only. **Client @mention autocomplete shipped `[v0.21.5]`**: composer popover
> (`components/chat/MentionAutocomplete.tsx`) over `useAgentProfile()` profiles with ↑/↓/Enter/Tab/Esc
> nav; picking inserts the routable `@<agent_id>` slug (no API change — rides the backend text
> parser). Pure helpers in `lib/mentions.ts` (`getActiveMention`/`applyMention`/
> `extractMentionedAgentIds`, tested in `mentions.test.ts`); `handleSend` populates
> `UserMessage.targetAgentIds`; `UserBubble` emphasizes resolved mentions as `@DisplayName`
> (via non-throwing `useAgentProfileOptional`). Live in-composer highlighting intentionally deferred.

- [x] Parse `@agent-id` from message text for implicit routing — new pure `agent/mentions.py`
      (`extract_mentions` + `resolve_first_mention`; `@[\w-]+` with email/path lookbehind). Resolves
      `@agent-id` or single-word `@name` (new `ProfileManager.get_profile_by_name`). `views.py`
      sets `target_agent_id` from the first resolved mention (overriding selection; precedence
      `workflow_id > @mention > target_agent_id > agent_profile_id > default`, suppressed in
      workflows) and strips the token from the model-facing `message` only. Unmatched @tokens stay
      as plain text. The mentioned agent joins `Session.participants` via the 16.2 hydration.
- [x] Add `AgentParticipant` Neo4j node and `PARTICIPATED_IN` relationship — `EpisodicMemory.store_turn`
      MERGEs `(:AgentParticipant {id:"<conv>:<agent>"})-[:PARTICIPATED_IN]->(:Conversation)` live for
      each agent-attributed turn; constraint + index in `queries/neo4j_schemas.cypher`.
- [x] Migration job: backfill — `queries/neo4j_migrations/0003_agent_participant_nodes.cypher`
      (run via `task db:migrate`). Backfills from distinct **`Turn.agent_id`** (richer than the
      original `Conversation.agent_id` note — captures every agent that spoke, not just the first).
      Tests: `MentionRoutingTest` (tests.py), `AgentParticipantGraphTest` (tests_memory.py).

### 16.6 Ambassador Agent (dual-presentation layer) — deferred sub-phase

> **Concept**: A customizable "ambassador" agent that mediates the human↔agent
> exchange as a *second presentation layer* alongside the chat UI — enriching
> communication with zero flow-disruption. Not a thin voice feature; a relay.

- [ ] Activation toggle for the ambassador (per-conversation or global).
- [ ] **Outbound (you → agent)**: capture continuous dictation while recording;
      on manual stop, convert the captured speech into a drafted message you
      **review/edit before send** (never auto-sends).
- [ ] Relay arbitrary additional inputs you attach alongside the dictated
      message — file inputs remain available (reuse the existing input path).
- [ ] **Inbound (agent → you)**: when an agent's final message lands, the
      ambassador produces a spoken/condensed **briefing** of the message plus any
      key elements sent with it (attachments, tool artifacts, citations).
- [ ] Customizable ambassador behavior (verbosity, persona, what to summarize vs.
      read verbatim, which key elements to surface).
- [ ] Zero UI flow-disruption: the ambassador augments, never blocks, the chat
      UI — design it as a parallel channel, not a modal step.
- Design later as its own sub-phase; sits naturally on the Alloy/multi-agent
  track (an ambassador is a specialist role mediating the conversation).

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

- [x] Migrate our settings menu for tools into our new immersive, full-screen approach for our Tools menu, transforming it into the "Toolkit".
  - New `client/src/components/toolkit/` package (`ToolkitPage`, `ServerForm`) modeled on the `UnifiedSettings` shell with sub-views for Servers, Tools Browser, Groups & Tags, Access, and Raw JSON. Topbar wrench opens it as a full-screen modal; legacy `ToolsPanel`/`ToolsSection` removed. Edit modal is portaled to `<body>` to escape framer-motion transform clipping.
- [x] The Toolkit should have ability for CRD on tools in the mcp-servers.json; guided editor is the goal, but a simple text editor + pre-save checker will suffice.
  - Added `POST/PUT/DELETE /api/mcp/servers` plus `/servers/validate`; writes persist through `ServerRegistry.save_to_file` with best-effort disconnect on edit/delete. Toolkit ships both a guided `ServerForm` and a Raw JSON editor that goes through the same validate endpoint before save.
- [x] Add tool metadata; access whitelists (allow access for specific agents), tags, groups, etc. (anything else that's a quick payoff).
  - `ServerConfig` gained `tags`, `groups`, and `allowed_agent_ids`. `AgentProfile` gained `allowed_tools`/`blocked_tools`; the chat-stream view forwards both lists plus `agent_id` into `AgentConfig`, and `_get_tools_for_provider` enforces server-side `allowed_agent_ids` (default-DENY when a whitelist is set but `agent_id` is unknown).
  - Fix: `ProfileManager.update_profile` / `set_default_profile` / `delete_profile` now use `model_dump()` so `agent_id` (and any new field) is preserved on edits — previously default-toggling rotated `agent_id`s and orphaned whitelists.

### 18.3: Relay Module (RM) Foundation

The Relay module is a new UI component that consolidates various backend features. Its primary goal is to simplify the chat interface with a special toolbox, as we add more communication features to enhance user experience such as user prompt enhancment, file uploads, temporary chats (no memory), voice chat, etc.

- [x] Conduct these fixes concurrently:
  - [x] Instead of a streaming toggle, offer a way to launch a background conversation; runs without opening it.
    - Foreground chat is now always streaming; streaming toggle removed from the header. Background runs go through a new Redis-Streams chat job queue (`api/agentx_ai/background/chat_jobs.py`) with a daemon worker started from `AppConfig.ready` (`apps.py`). Exposed via `POST/GET /api/chat/background` and `GET/DELETE /api/chat/background/<id>` (`urls.py`, `views.py`); client uses `lib/api.ts` + `lib/hooks.ts` to poll an inbox surfaced inside the Relay popover.
  - [x] Migrate the DB icon that disables/enables memory consolidation on a conversation to the RM, and make it more UX friendly (ie. a "No Memorization" toggle).
    - Header memory icon removed; the toggle now lives in the Relay popover as "No Memorization" and locks in place once the conversation starts (`RelayMenu.tsx`, `ChatPanel.tsx`).
  - [x] Migrate the chat toolbar into a nice menu with a button trigger that way we can clear the chat for immersion.
    - New `client/src/components/chat/relay/` package (`RelayMenu.tsx` + `RelayMenu.css`): a single popover sitting left of the textarea consolidating the per-conversation toggles, background-run inbox, and stubs for upcoming voice/file affordances.
- Mobile pass shipped alongside the Relay foundation:
  - Single-tab mode: `ConversationContext.addTab` replaces the active tab; viewport drop collapses tabs to the active one (`ConversationContext.tsx`, `storage.ts`).
  - `.tabs-container` made flex-shrinkable so the History button stops getting pushed out (`ConversationTabBar.css`).
  - Brain (`ActiveAgentsDropdown`) + history (`ConversationHistoryDropdown`) popups now flip above their anchor when there's no room below and pin to viewport edges on small screens.
  - "+" tab-bar button hidden on mobile in favor of a "New conversation" action inside the Active Agents popup (`TopBar.tsx`, `ActiveAgentsDropdown.tsx`).
- Remember that RM will be used for file communication and controlling the different modes of the model when feature support it.
- NOTE: Before we can work with images; we'll need to improve model metadata and also our file management to target image and video output models.

### 18.4: Model Metadata, Selector Refinement

Currently our model selection and selector are a very solid foundation but are just that. We cannot have advanced capabilities until we can profile our models. We're going specialize in OpenRouter and Vercel Gateway, with anthropic for high-quality reasoning. We'll also need a solid base for workflows deciding which agents can analyze various forms of media (eg. images, videos, etc.) and which can generate various forms of media.

- [x] Improve the model selector to use filter-based lists with a comfortable UX, and show the model selector in a modal or full-screen menu; it's cluttering the agent profile and the large lists can lag the UI.
  - New `client/src/components/common/ModelPickerModal.tsx` + `.css`: fullscreen modal (portaled to `document.body` so it escapes the profile editor's framer-motion transform context), modeled on `ToolkitPage`. Left rail has provider chips + capability chips (Tools / Vision / JSON / Streaming) as AND-filters; main pane has search + rich rows showing name, provider badge, context, max-out, price chip, and capability icons.
  - `ModelSelector.tsx` now exports `fetchModelsOnce` so the modal shares the same cache; no behavior change for existing callers (Memory settings still uses the old compact selector).
  - `ProfileContent.tsx` swaps the embedded `<ModelSelector>` for a compact `.profile-model-trigger` row (`Model: <name> [ctx · tools · vision] [chevron]`) that opens the picker; new styles in `UnifiedProfileEditor.css`.
- [x] Vercel Gateway model selection is fairly weak, only showing context limits, we want to see capabilities, pricing, max-tokens, etc.
  - `vercel_provider.py` `get_capabilities` now reads `modalities.input/output`, `description`, and `pricing.currency` from the cached `/v1/models` payload (pricing + context + max-tokens were already parsed; tightened None-safety).
- [x] OpenRouter has a ton of metadata like Vercel gateway, and we need match capability to what we'll add in Vercel Gateway.
  - `openrouter_provider.py` `get_capabilities` now reads `architecture.input_modalities/output_modalities` (drives `supports_vision`), broader `supported_parameters` keys (`tools`/`function_calling`/`tool_choice` for tools; `response_format`/`json_mode`/`json_object` for JSON), and `description`.
- [x] Wire metadata through the wire format
  - `ModelCapabilities` (`providers/base.py`) gained `input_modalities`, `output_modalities`, `description`, `pricing_currency`.
  - `/api/providers/models` (`views.py`) serializes the new fields; `ModelInfo` (`lib/api.ts`) widened to match. Image/video routing remains a follow-up — this phase only stores the data.

### 18.5: Metrics Overhaul

- [x] Per-tab context window display
  - `contextInfo` moved from `ChatPanel` local state to per-tab state on `ConversationContext` so it tracks the active tab instead of getting stuck on whatever streamed last (`ConversationContext.tsx`, `storage.ts`, `ChatPanel.tsx`). The ephemeral field is stripped before persisting to localStorage so stale numbers can't rehydrate.
  - Persist `context_window` / `context_used` on the assistant `Turn` metadata so `restoreConversation` rehydrates the bar from server history. For inactive tabs and turns predating the field, `ChatPanel` backfills via the cached `/api/providers/models` payload — window from the latest assistant message's model, `used` from the last turn's tokens or a `chars / 4` estimate.
- [x] Per-turn cost estimate using Phase 18.4 model metadata
  - New `api/agentx_ai/providers/pricing.py` `estimate_cost(caps, in, out)` computes absolute-dollar cost from `ModelCapabilities` pricing and freezes a `pricing_snapshot` so historical costs stay stable if rates change later.
  - Streaming `done` event ships `model`, `provider`, `cost_estimate`, `cost_currency`, `pricing_snapshot`. Assistant `Turn` metadata persists those alongside `tokens_input`/`tokens_output`/`latency_ms`; `Turn.model` field added (`models.py`) and the episodic `INSERT` extended to write `model` + `metadata` JSONB into `conversation_logs` (columns already existed; no DDL).
  - `MetadataBar` renders a `$` cost chip. When the backend can't compute one (provider doesn't expose pricing in caps — e.g. the built-in Anthropic provider — or older turns with no metadata), it backfills from `cost_per_1k_input/output` in the same model cache so the chip still appears.
- [ ] Dashboard token + turn metrics — UI deferred, data layer ready
  - Assistant turns are now written to `conversation_logs` with `model` and a `metadata` JSONB containing tokens/cost/latency, so a future dashboard card can aggregate via a single `SELECT model, token_count, metadata FROM conversation_logs` query with no further backend work. `idx_logs_timestamp` already covers "today" windows.

### 18.6: Extraction Tuning

> Goal: fix fact overlap and entity duplication by making the extraction LLM context-aware and adding a server-side entity-resolution + fact-merge layer in consolidation. Plan file: `~/.claude/plans/look-over-the-system-eventual-hejlsberg.md`.

- [x] Semantic store helpers: `find_entity_by_name_or_alias` (name → alias → slug) and `merge_entity_aliases` (idempotent, never clobbers populated description/properties) — `kit/agent_memory/memory/semantic.py`
- [x] Consolidation entity resolution: new `_resolve_and_prepare_entities` honors LLM `existing_entity_id` → falls back to name/alias/slug lookup → fresh uuid only if truly new; merges aliases on reuse; populates `entity_map` so downstream fact and relationship linking sees reused ids. Applied to both the user path (~jobs.py:805) and the self-channel path (~jobs.py:1098). Preserves aliases/properties the prior code dropped.
- [x] Fact supersedure via `refines_fact_id` runs before the hash/semantic gates and dispatches through `_handle_contradiction` with a synthetic `prefer_new` result; falls through to the standard pipeline when the target is out of scope. Tracked via `metrics.facts_superseded_by_refine`.
- [x] `ExtractionService.check_relevance_and_extract` and `..._assistant` accept optional `known_entities` / `known_facts`; `_render_scope_context` renders them into compact `id=… name="…"` blocks (or `(none)`).
- [x] `_build_scope_context` in consolidation runs a stopword-aware capitalized-name regex against each turn, looks up matches via `find_entity_by_name_or_alias`, and attaches the top facts per matched entity (capped 8 entities × 3 facts).
- [x] Prompt updates (`prompts/system_prompts.yaml`): `{known_entities}` / `{known_facts}` placeholders, `existing_entity_id` / `refines_fact_id` in JSON schema, literal-`null` guidance for `temporal_context`, `"User <verb> ..."` claim convention, explicit negation/canonicalization/dedup rules, and worked few-shot examples for both `combined_with_relevance` and `assistant_self`.
- [x] `ConsolidationMetrics`: new `entities_reused` and `facts_superseded_by_refine` counters.
- [x] Tests in `tests_memory.py` (20 new): `NameCandidateExtractionTest`, `RenderScopeContextTest`, `EntityResolutionTest`, `RefinesFactIdSupersedureTest`, `ExtractionServiceScopeContextWiringTest`. Full memory suite still green (180 tests, 29 docker-gated skips).
- [x] E2E consolidation quality eval harness: `manage.py eval_consolidation` (`task eval:consolidation`). Seeds graded cases (L1 single-fact → L5 multi-turn / multi-entity / temporal / negation / refinement) into isolated `_eval_*` channels, runs the real consolidation pipeline against live Neo4j + a configured LLM (`--model`, e.g. `openrouter:minimax/minimax-m2.7`), and scores extracted facts/entities vs expectations. For prompt tuning + extraction regression-checking. Requires a STERILE dev instance (`--wipe`) since consolidation is global.
- [ ] Eval harness follow-ups (deferred):
  - [ ] Procedural-memory eval cases — exercise tool-usage/strategy learning (`ProceduralMemory.record_invocation`, strategy extraction). Procedural extraction has been deferred repeatedly; the harness is structured to add these cases once that path is exercised by consolidation.
  - [ ] Snapshot/restore instead of `--wipe`, once memory export/import lands (see Backlog) — so the eval can run without destroying dev data.
  - [ ] Persist eval runs (model, per-case scores, tokens) for cross-model / cross-prompt comparison over time.
- [ ] Follow-ups (deferred): one-shot Neo4j cleanup script to dedupe entities created before this fix; cross-channel entity unification; replace the regex correction patterns at `extraction/service.py:84` with an LLM-only path.

### 18.8: Wave 2 Fixes

- [x] Chats cannot render equations.
  - Wired `remark-math` + `rehype-katex` (KaTeX) into `MessageContent.tsx` and import `katex/dist/katex.min.css`; added theme overrides (inherit color, scrollable `.katex-display`) in `MessageContent.css`. Inline `$…$` and block `$$…$$` now render.
- [x] ~~Streaming in the UI seems to stop after a table; then emits the remaining chunk of text.~~ — fixed (confirmed).
- [x] Models render `<br/>` in table cells (e.g. bullet-point lists inside a cell) but it doesn't break the line.
  - Added `rehype-raw` to `MessageContent.tsx` with order `[rehypeRaw, rehypeKatex, rehypeHighlight]` so raw HTML renders while the math nodes survive. Added `MessageContent.test.tsx` locking the contract (`<br>` in a cell + KaTeX + code highlighting). `rehype-sanitize` left as a follow-up (XSS surface; local single-user app, and a sanitize schema would have to whitelist the KaTeX/`hljs-*` classes).
- [x] When re-opening a conversation with an agent that executed a plan, the steps' messages show as an error: "Unknown message type" in the UI.
  - Root cause: orphan `tool_result` rows (no paired `tool_call`) map to `type: 'tool_result'`, which had no entry in `messageRegistry` → fell through to `UnknownBubble`. Added `bubbles/ToolResultBubble.tsx` (reuses `ToolExecutionBlock`) and registered it.
- [x] ~~Fix consolidation bug~~ — already fixed. The `Neo.ClientError.Statement.TypeError` ("Can't coerce `List{Double…}`") came from passing a nested `List[List[float]]` to `db.index.vector.queryNodes`. Consolidation now uses `embedder.embed_single()` which returns a flat `List[float]` (`embeddings.py:120`), matching the working `semantic.py` query; warning text reworded. Verify on a live instance via `task eval:consolidation`.
- [x] Cached servers for the 'Connect' page cannot be edited or deleted.
  - `ServerSelector.tsx` now renders per-row edit (inline form) + delete (inline confirm) actions wired to the existing `updateServerConfig`/`deleteServer` context methods; styles added to `AuthPage.css`.
- [x] MCP tools should auto-connect on server restart if they were connected at shutdown.
  - Added a persisted `auto_connect` flag to `ServerConfig` (distinct from `auto_reconnect`), set true on connect / false on disconnect in `mcp_connect`/`mcp_disconnect` (incl. the `all` bulk paths) and saved to `mcp_servers.json`. New `MCPClientManager.connect_persisted()` restores flagged servers from `apps.py` `ready()` on a best-effort daemon thread. Flag surfaced in `_serialize_server` + preserved through the Toolkit guided/raw/`toConfigInput` editors so edits don't clobber it. Tests: round-trip + `connect_persisted` targeting (`MCPServerRegistryTest`).
  - **Follow-up fix:** the `apps.py` `ready()` startup guard only matched a bare `"uvicorn"` string, but `uv run uvicorn …` gives `argv[0]` as the full binary path → guard returned early, so neither the reconnect *nor the background chat worker* ran under the real launch. Hardened detection (argv[0] basename + asgi/wsgi target). Verified end-to-end: `brave-search` auto-connects on boot.

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

- [x] Checkpoints sidebar/badge in the conversation pane
  - New `GET/DELETE /api/memory/checkpoints?conversation_id=` (`views.memory_checkpoints` + `clear_checkpoints()` in `agent/checkpoint_storage.py`); `CheckpointEndpointTest`. Client: `Checkpoint`/`CheckpointsResponse` DTOs, `api.getCheckpoints`/`clearCheckpoints`, `useCheckpoints(conversationId)` hook.
  - `components/chat/CheckpointsBadge.tsx` (+ `.css`): count badge in the chat header keyed on `activeTab.sessionId`, opening a `DropdownPortal` popover listing each checkpoint's summary/decisions/next-step/`created_at` (reuses `formatTimestamp`) with a Clear button (toast + refetch). Hidden when count is 0.
  - Flash on autonomous save: `useChatStream` gained an `onCheckpointSaved` callback (fired when a `checkpoint` `tool_result` lands); `ChatPanel` bumps a `checkpointSignal` → badge refetches + pulses (animation gated by `prefers-reduced-motion`).
- [x] Recall view for `recall_user_history`
  - In-chat: `ToolExecutionBlock` special-cases `recall_user_history`, parsing the JSON result into a collapsible "User history recall" card (falls back to the generic block on parse failure / `success:false`).
  - Memory drawer: new "User History" section (`components/memory/UserHistoryTab.tsx`) backed by a new `POST /api/memory/user-history` endpoint (the tool itself needs an active chat `ContextVar`, so the manual browse uses an HTTP twin). Shared presentation in `components/memory/UserHistoryView.tsx` (+ `.css`) used by both surfaces.
- [-] Toolkit gating for `checkpoint`/`recall_user_history` — **deferred**: no per-profile internal-tool allowlist UI exists (Toolkit "Access" gates whole MCP servers; profile editor has only a single `enableTools` toggle). Backend `allowed_tools`/`blocked_tools` on `AgentProfile` remain unsurfaced — building that surface is its own task.
- [x] Combined Extraction settings polish
  - `ConsolidationSettingsPanel.tsx`: expanded the section description to explain the single-pass merge (~75% fewer calls) and that setting a model overrides the separate Relevance + Extraction stages; added a "Reset to default" button on the model field (clears `combined_extraction_model`).

### 18.10: Plan Executor + Streaming Reliability

> Three major bugs surfaced after 18.6 lands. Group them here because the fixes overlap (streaming lifecycle + plan step rendering share the same conversation-tab state).

- [x] ~~Plan executor hangs on final step~~ — already resolved. `_handle_failure` marks the failed subtask `completed=True` and skips dependents (`plan_executor.py:418,437`); the loop has a deadlock guard (`:168-170`) and synthesis is a single streaming call (`:466-484`). No infinite-loop path remains. Verify on a live plan run.
- [x] ~~New-conversation chat freeze~~ — resolved by prior streaming-background-job work. Verify a new conversation starts streaming without freezing on a live instance.
- [x] Output token-budget overshoot → provider 400 ("requested ~262183 tokens, max 262144"), hit during Alloy plan execution.
  - Root cause: `agent_chat_stream` adaptive budget (`views.py`) set `adaptive_max_tokens = min(max_output_tokens, context_window − input − buffer)`. OpenRouter reports `max_output_tokens == context_window` for some models, so this requested ~the whole window as output (259K) with zero slack, and `estimate_tokens` never counted the tool-schema tokens (the "1473 of tool input") → request landed 39 over.
  - Fix: clamp a capability-reported output cap to `MAX_OUTPUT_TOKENS_CEILING` (32768; explicit per-model overrides still win) and reserve estimated tool-schema tokens in the input budget. New constant in `streaming/constants.py`. Verified: reported scenario now caps output at 32768 (total ~35K ≪ 262144).
- [x] Plans UI redesign — "Plans in Progress" drawer + in-chat step annotation
  - **Global registry.** New `contexts/PlansContext.tsx` holds live plan records (fed by `useChatStream`) keyed by `planId`; the drawer merges these with a pure selector (`derivePersistedPlans`/`mergePlans`) over all tabs' persisted `plan_execution` messages, so finished plans survive reloads (live wins on collision). `PlansProvider` mounts inside `ConversationProvider`.
  - **1. Drawer.** `components/plans/PlansPanel.tsx` (+ `.css`) registered via `stubs.tsx`/`ModalPortal.tsx` and opened from a new `TopBar` entry (also in the mobile overflow). Lists each plan as its own collapsible block (compact status when collapsed; step list + progress bar + elapsed time + result previews + "open/jump to step" + cancel when expanded). Toolbar icon shows an animated construction-stripe + count badge while `activePlans>0`; frozen under `prefers-reduced-motion` (`TopBar.css`). Shared subtask/progress rendering extracted to `components/plans/PlanSubtaskList.tsx` (reused by `PlanExecutionBlock`).
  - **2. In-chat step annotation.** Messages produced inside a subtask carry a `planStep` ref (stamped in `useChatStream`; persists via localStorage). `groupMessagesBySteps` (pure, unit-tested) folds consecutive same-step messages into collapsible `StepGroup`s with a "Step k/n · title" header (finished-plan steps default-collapsed). The Plans drawer's "jump to step" scrolls + flashes the matching group/plan card via an `agentx:jump-to-step` event.
  - **Reload reconcile.** New `GET /api/agent/plans/<id>/status` (`views.agent_plan_status`, over `PlanStateStore`; `{found:false}` on TTL expiry) + `api.getPlanStatus`/`api.cancelPlan`. On drawer mount, persisted `running` plans are reconciled (completed/cancelled/stale) so no eternal spinner. Backend test: `PlanStatusEndpointTest`.
  - **Restore.** Plan synthesis turns now persist a `plan` summary in `conversation_logs` metadata; `mapServerMessages` reconstructs the `PlanExecutionMessage` card on restore (subtask-level turns aren't stored server-side, so per-step annotation on reopened convs relies on localStorage — cross-device step annotation deferred).
  - Pre-work: fixed two contract bugs — client `progress` typed as `number` (backend sends `{completed,total}`) and the unhandled `plan_cancelled` SSE event (added `onPlanCancelled` + `'cancelled'` plan status).

- [x] Seamless tab close / reopen — detached chat runs (survive disconnect)
  - **Problem:** closing a browser tab (or even switching conversation tabs) killed an in-flight chat. The SSE response was driven by the ASGI server pulling the generator; on disconnect generation halted, and since turns persist only at the *end* of `generate_sse`, nothing was saved — fatal for new chats (conversation never created), lossy for existing ones. The client compounded it by `stream.stop()`-ing on every tab switch.
  - **Detached runner.** `streaming/chat_run.py`: `start_chat_run` spawns a daemon thread that drives the existing `generate_sse` generator (handed over as-is — it references no `request` after parse, so **no body extraction needed**), fanning every SSE event into a Redis Stream (`chat_run:{id}:events`, MAXLEN+TTL) and settling run state. Generation + turn persistence are now connection-independent and play to completion. Covers plan/Alloy runs for free.
  - **Tail + re-attach.** `agent_chat_stream` is now start-run + `run_started{run_id}` + tail (`tail_chat_run` via `XREAD BLOCK` in an executor). New `GET /api/agent/chat/stream/attach?run_id=` replays from `0` + follows live; `POST /api/agent/chat/runs/{id}/cancel` cooperatively cancels (checked at event boundaries). `run_missing` emitted on TTL expiry.
  - **Client.** `streaming.ts` shares one SSE pump between `streamChat` + new `attachChatRun`; `useChatStream` gained `attach()` and split **detach** (tab switch — abort fetch only, run survives) vs **stop** (Stop button — abort + `cancelChatRun`). `activeRun` persisted on the tab (`storage.ts`); `ChatPanel` re-attaches on mount/switch (truncating to the triggering user turn so replay-from-0 doesn't duplicate), falling back to `restoreConversation` on `run_missing`.
  - Tests: `ChatRunStoreTest` (store round-trip, cancel flag, `run_missing`/stale tail paths). **Known edge:** a *new* chat abandoned past the 2h event-TTL can't be restored (client never learned its conv_id) — acceptable for v1.

### 18.11: Client Error Contract + Foundation Cleanup

> Follow-on to the backend typed-error pass (cleanup item 9: `AgentXError` hierarchy → HTTP mapper at the views boundary). The client previously discarded the backend's `{"error": "..."}` message (threw `"API request failed: <statusText>"`) and had no consistent way to show failures.

- [x] Error contract end-to-end — `ApiError` gained a status-derived `kind` (`bad_request`/`auth`/`not_found`/`upstream`/`unavailable`/`network`/`server`); `request()` now surfaces the backend message and wraps `fetch` rejections as `network`. Added `toApiError()` / `apiErrorMessage()` / `classifyStatus()` / `isApiError()` helpers in `lib/api.ts`.
- [x] Streaming errors — `streamChat` parses non-2xx stream POSTs (auth/typed errors before any SSE event) through the same path; `useChatStream.onError` raises a toast in addition to the inline assistant message.
- [x] Toast system — `NotificationContext` (`notify`/`notifyError`/`notifySuccess`) + `ui/Toaster` portal, mounted in `App.tsx`. Routed previously-silent catches (history delete/restore, background-chat enqueue) and modal save/delete errors (`apiErrorMessage`, since `ApiError` is a plain object, not an `Error`).
- [x] `useApi<T>` hook factory in `lib/hooks.ts` — collapsed the duplicated `data/loading/error/refresh` block across the read hooks (health, providers, MCP tools, agent status, memory channels/entities/facts/strategies/stats, jobs).
- [x] `ErrorBoundary` around the routed page tree; UI primitives `Badge`/`Card`/`Input`/`Textarea`/`SectionHeader`; `--space-*` spacing scale added to both themes in `lib/theme.ts`.

#### "full grade-match" client refactor
- [x] Stand up a client test harness (Vitest + Testing Library + jsdom) — `task test:client`; config in `vite.config.ts`, setup in `src/test/setup.ts`. (Prereq for the splits below.)
- [x] Tailwind v4 enabled (`@tailwindcss/vite`, Preflight off, runtime tokens bridged via `@theme inline`) + expanded `ui/` primitives (Select/Popover/Switch/Checkbox/Label).
- [x] God components, first pass: `SettingsPanel.tsx` (~996) was already superseded by `unified-settings/` and its modal key never opened — **deleted as dead code** (+ `SettingsPanel.css`, `SettingsModalContent`, the `settings` registry key). `MemoryPanel.tsx` (~933) split in place into `components/memory/` (orchestrator + list views + detail panels + graph + pagination + `formatTimestamp`), with render tests; standalone Memory Explorer modal unchanged.
- [x] God components, remaining: `MemorySettingsPanels.tsx` (894) decomposed into `components/memory-settings/` (`ConsolidationSettingsPanel` + `RecallSettingsPanel` + `index`) over reusable `fields/` components (`SettingsSection`/`SliderField`/`NumberField`/`ToggleField`/`PromptField`) built on the `ui/` primitives — the **first real adoption** of the primitive library. Added a `Slider` primitive (`@radix-ui/react-slider`); migrated the duplicated `saveMessage` banners to `useNotify()` toasts. Field + Slider render tests added.
- [x] Split the monolithic `lib/api.ts` (2,497 lines, 89 methods) → `lib/api/` folder: `types.ts` (DTOs), `errors.ts`, `version.ts`, `core.ts` (request layer + stream registry), 17 domain modules (`health`/`auth`/`providers`/`mcp`/`agent`/`relay`/`toolOutput`/`translation`/`prompts`/`promptTemplates`/`profiles`/`alloy`/`config`/`memory`/`jobs`/`history`/`streaming`), and `index.ts` facade that spreads them into `api` + re-exports the full public surface. Import specifier `'.../lib/api'` unchanged (folder barrel) → zero consumer changes. `facade.test.ts` guards 89-method parity.
- [x] Split `ConversationContext` (599 lines) into concern hooks under `contexts/conversation/`: `useConversationTabs` (tab state + persistence + CRUD — source of truth), `useTabMessages`, `useTabSettings`, `useConversationHistory`, and a pure `mapServerMessages` (extracted + unit-tested). The provider (`ConversationContext.tsx`, now 133 lines) composes them; public API (`useConversation`/`ConversationProvider`/`ConversationTab`) unchanged → zero consumer edits.
- [x] Adopt the new primitives repo-wide + fix the settings styling regression. `button-primary`/`secondary`/`ghost` → `<Button variant>` across all consumers (memory detail/list panels, modals, pages, JobsPanel, TranslationPanel, settings sections); the only remaining `button-*` references are `ui/Button.tsx` (the variant map) + its test. The `unified-settings/` sections (`Providers`/`Models`/`Appearance`/`MemoryOverview`/`Planner`/`Prompts`) now use `<SectionHeader>`/`<Card>`/`<Badge>`/`<Input>`/`<Button>` and route save/load feedback through `useNotify()` toasts (dropped the inline `configSaveMessage`/`save-message` banners). **Regression fixed:** the pass-1 deletion of `SettingsPanel.css` had orphaned the live provider/config/empty-state/section-header classes (`defs=0`) — provider icon tiles lost their gradient and the dark SVGs went near-invisible. Re-added + elevated (Aurora gradient tiles, glass hover) the provider/config/context-limits CSS in `UnifiedSettings.css`, and the generic `.section-header`/`.section-description`/`.empty-state` fallbacks in `base.css`. Smoke tests for `ProvidersSection`/`ModelsSection` added. **Deliberately left:** `.stat-badge` (distinct larger stats pill — `ax-badge` would shrink it), other `.card` divs (the `Card` primitive is class-identical, no visual gain), and `btn`/`icon-btn`-family bespoke buttons; repo-wide px→`--space-*` inside established `.css` files stays deferred (high churn, low value).

### 18.11.x: Bugfix pass `[v0.20.2]`

- [x] **Cancel never works (CSRF 403)** — `agent_chat_run_cancel` was missing `@csrf_exempt`
      (every other chat endpoint has it), so `POST /api/agent/chat/runs/<id>/cancel` returned
      `403 Forbidden (CSRF cookie not set)`; the cancel flag was never set and the detached run
      streamed to completion. This affected both mid-run cancels and cancelling a just-started
      conversation. Added the decorator (`views.py`); `agent_plan_cancel` already had it.
- [x] **Custom chrome missing on the gate pages** — the frameless drag region + window controls
      live in `TopBar`, which `RootLayout` only renders in the authenticated branch. The
      `showConnect`/`AuthPage` and `versionMismatch` early returns had no titlebar, so those
      screens were chromeless and unmovable on Windows/Linux. New `layouts/GateChrome.tsx` (a slim
      fixed drag strip + `WindowControls`, gated on `isTauri`/`showWindowControls`) rendered in
      both early-return branches.

### 18.12: Wave 3 — Start / Conversation UX + README `[v0.21.0]`

> UX polish pass on the entry surfaces (Start page, conversation + profile selectors)
> plus a README trim. Client-side.

- [x] **Recent conversations on the Start page** — Start now shows a collapsible "Recent
      Conversations" list (open tabs + server history, deduped/sorted, capped 8) with persisted
      collapse state (`agentx:startRecentsCollapsed`); refreshes history on mount. Sourced from
      `useConversation()` — the old `storage.ts` `recentChats` helpers were **dead code** (never
      called), so they weren't used. `pages/StartPage.tsx`.
- [x] **Renamable conversations** — client-side rename (no server endpoint). New
      `lib/conversationTitles.ts` (per-server `conversation_id → title` override map +
      `getDisplayTitle`). Inline rename: double-click a tab (`ConversationTabBar`) or a row /
      pencil action in `ConversationHistoryDropdown` (open tabs via `renameTab`, past convs via
      `setTitleOverride`). Restored tabs seed their title from the override. Tests:
      `conversationTitles.test.ts`.
- [x] **Improve the conversation selector** — `ConversationHistoryDropdown`: recency grouping
      (Today / Yesterday / This week / Older), `getDisplayTitle` applied, rename actions, larger
      hit-regions, Enter-opens-first-result.
- [x] **Improve the agent profile selector** — `AgentSelectorDropdown`: search/filter (shown when
      >6 profiles), default badge + clearer active state. Selection/workflow-lock behavior unchanged.
  - [x] **Redesign `[v0.21.1]`** — the stacked two-list layout cut the Workflows section to ~one card
        (two `flex:1` lists fighting under a fixed `max-height`). Rebuilt as **segmented tabs**
        ([ Agents | Workflows ]) over a **single** full-height scroll list (per-segment search,
        context footer, ↑/↓/←/→/Enter keyboard nav, opens on the active segment). Compact two-line
        rows (26px avatars, tighter padding); panel `max-height: min(460px, 70vh)` so it never
        overflows a short window. Workflow-lock semantics preserved (Agents tab shows a hint +
        highlighted supervisor when a workflow is active).
- [x] **Splash screen** — instant static splash in `index.html` (cosmic bg + logo + spinner,
      painted before the JS bundle parses), faded out in `main.tsx` after React's first frame with a
      ~400ms minimum. Note: a native window `backgroundColor` is **not** in this Tauri version's
      window schema, so a brief pre-webview-paint flash on cold open remains (HTML splash covers the
      rest); revisit if Tauri exposes the key.
- [x] **Trim the README** — `Readme.md` reduced to banner + badges + a "Built With" table + docs
      link; Features/Quick-Start/Architecture/Dev prose removed (now docs-site only).

---

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

- [ ] Generative Agent Avatar + Extended Icon Base (ie. cool robot face, or funny cat face, etc) -  blocked by image capabilities for models
- [ ] Fibonacci complexity planning scales (augment planning behaviour based on complexity)
- [ ] Disabled memory conversation prompt message banner - informs the model that memory is off for this conversation and the details are not persistent, and also that the conversation may contain confidential material.
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
- [ ] Memory export/import (JSON/SQLite backup) — also unblocks snapshot/restore for the consolidation eval harness (18.6) so it needn't `--wipe` a dev cluster
- [ ] Advanced memory visualization (interactive graph, embedding clusters)
- [ ] Streaming memory retrieval during chat
- [ ] Conversation sharing (read-only shareable links)
- [ ] Blocking tool call approval (pause stream, user approves/rejects before execution)
- [ ] Server authentication (single access key per server, session resume on reconnect)
- [ ] Mobile-responsive breakpoints and touch-friendly gestures
- [ ] Additional themes beyond cosmic (light theme, high contrast, etc.)
- [ ] Message injection into delegated tasks (agent interdiction tools)
- [ ] Custom window chrome — frameless Tauri window with our own title bar + window controls (minimize / maximize / close) and a drag region, styled to the cosmic theme. **Windows + Linux first; macOS later** (traffic-light insets + native fullscreen need separate handling). Touches `client/src-tauri/tauri.conf.json` (`decorations: false`) + a top-of-app titlebar component using the Tauri window API.

### Retrieval Quality Enhancements (migrated from docs/future-feature-pool)
- [ ] Working Memory Scratchpad — always prepend a structured scratchpad (current topic/task, active entities, recent corrections, open questions) to context for coherence/orientation
- [ ] Conversation Summarization — maintain rolling per-session and per-topic summaries; retrieval becomes `recent_turns + relevant_summaries + relevant_facts`
- [ ] Query Intent Classification — classify query before retrieval (follow-up → recency, callback → older history, new topic → broad semantic, factual recall → entities/facts); rule-based or lightweight LLM
- [ ] Negative/Correction Tracking — when `correction_detection_enabled`, mark superseded facts `temporal_context: "past"`, link corrections to originals, prioritize corrections in retrieval
- [ ] Fact Staleness Detection — add `expected_stability: transient|stable|permanent` and surface staleness warnings (relates to Fact Transience above)
- [ ] Multi-hop Entity Traversal — add a lightweight path-finding retrieval mode over the entity graph (e.g. User → works_at → Company → has_project → Project → uses_tool → Tool)

### MCP Tools (migrated from docs/future-feature-pool)
- [ ] Conversation MCP Tool — expose memory as MCP tools for external agents: `memory_recall(query, filters?)`, `memory_store(fact)`, `conversation_summary(conversation_id?)`

### Extraction Improvements (migrated from docs/future-feature-pool)
- [ ] Claude Sonnet for Extraction — switch extraction from local models to Claude Sonnet for better structured-output adherence, nuance detection, and entity resolution (cost/latency offset by async/batched consolidation)
- [ ] Improved Extraction Prompts — few-shot examples, better schema definitions, domain-specific tuning

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

**Embedding Request Queue / Serialization**
- Embeddings (`EmbeddingProvider.embed`/`embed_single`, `kit/agent_memory/embeddings.py`) are generated ad-hoc on whatever thread calls them — chat recall, consolidation jobs, strategy/entity indexing — with no coordination. The local sentence-transformers model isn't safe to call concurrently, and the remote (OpenAI) path can be rate-limited; bursts can collide or get dropped.
- Fix: route all embedding requests through a proper queue/serializer so requests are reliably served — single worker (or bounded pool) draining a queue, with batching of pending texts where possible, backpressure, and a shared path for both local and remote providers. Pairs naturally with the Query Embedding Caching item above.

---

## Blockers

None currently.

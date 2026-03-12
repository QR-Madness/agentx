# AgentX Development Todo

**Project**: AgentX - AI Agent Platform
**Status**: Pre-prototype
**Last Updated**: 2026-03-02

> For completed phases (1-10), decision log, and project history, see [docs/roadmap.md](docs/roadmap.md)

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phases 1-10 | Complete | See [roadmap.md](docs/roadmap.md) |
| Phase 11: Memory System | **In Progress** | 94% |
| Phase 11.12: LLM-Enhanced Consolidation | **In Progress** | 95% |
| Phase 12: Documentation | Not Started | 0% |
| Phase 13: UX Overhaul — Immersive AgentX | **In Progress** | 5% |

---

## Phase 11: Memory System Activation

> **Priority**: HIGH
> **Goal**: Make the memory system functional, auditable, and extensible
> **Depends on**: Docker services (Neo4j, PostgreSQL, Redis) running

### Completed Sections (11.1-11.10)
- [x] 11.1 Database Schema Initialization
- [x] 11.2 Agent Core Integration
- [x] 11.3 Extraction Pipeline
- [x] 11.4 Auditability & Query Tracing
- [x] 11.5 Channel Scoping & Data Safety
- [x] 11.6 Extensibility Infrastructure
- [x] 11.7 Retrieval Quality
- [x] 11.8 Memory System Tests (80 tests)
- [x] 11.9 Consolidation & Background Worker
- [x] 11.10 Memory Explorer (Client)

### 11.11 Background Job Scheduler & Monitoring

> Job scheduler for running consolidation and other background tasks

#### Scheduler (Deferred)
- [ ] Implement background job scheduler (Django Q, Celery, or custom)
- [ ] Job registration system (cron-like or interval-based)
- [ ] Job persistence (survive API restarts)
- [ ] Graceful shutdown handling

#### Job Monitoring API (Complete)
- [x] `GET /api/jobs` — list all registered jobs with status
- [x] `GET /api/jobs/{job_id}` — job details
- [x] `GET /api/jobs/{job_id}/history` — recent execution history
- [x] `POST /api/jobs/{job_id}/run` — manually trigger a job
- [x] `POST /api/jobs/{job_id}/toggle` — enable/disable a scheduled job

#### Job Monitoring UI (Complete)
- [x] JobsPanel component in Memory tab
- [x] Job list with status badges
- [x] Job detail view with metrics
- [x] Manual controls (Run Now, Enable/Disable)

#### Deferred
- [ ] `GET /api/jobs/{job_id}/logs` — logs from recent runs
- [ ] Real-time updates (polling while job running)
- [ ] Consolidation preview (`POST /api/memory/consolidate/preview`)

---

## Phase 11.12: LLM-Enhanced Consolidation

> **Priority**: HIGH
> **Goal**: Use LLM providers for intelligent consolidation stages

### 11.12.1 Pre-Extraction Relevance Filter (Complete)
- [x] Add `check_relevance()` to consolidation pipeline
- [x] LLM prompt: "Does this text contain memorable information? YES/NO"
- [x] Heuristic pre-filter (skip turns < 10 chars, "ok", "thanks", etc.)
- [x] Metrics: track skip rate, extraction savings (via ConsolidationMetrics)

### 11.12.2 Enhanced Fact Extraction + Condensation (Complete)
- [x] Combined relevance + extraction in single LLM call (~75% fewer calls)
- [x] Reasoning model support (nvidia/nemotron-3-nano by default)
- [x] Confidence calibration mapping (explicit/implied/inferred/uncertain)
- [x] Per-turn extraction with accumulated results
- [x] 22 unit tests for combined extraction and confidence

### 11.12.3 Entity Matching via Embedding Search
- [x] Add `link_facts_to_entities()` consolidation job
- [x] Embedding-based entity resolution (handle aliases)
- [ ] Optional LLM disambiguation for ambiguous matches

### 11.12.4 Contradiction Detection (Complete)
- [x] Add `check_contradictions()` to extraction pipeline
- [x] LLM prompt comparing new facts against recent existing facts
- [x] Contradiction resolution strategies (prefer_new, prefer_old, flag_review)
- [x] `_get_recent_facts()` helper queries Neo4j for existing facts
- [x] `_handle_contradiction()` applies resolution (supersede, skip, or flag)
- [x] Integrated into consolidation pipeline (disabled by default)

### 11.12.5 User Correction Handling (Complete)
- [x] Heuristic patterns: "actually...", "no, I meant...", "that's wrong...", etc.
- [x] `check_correction()` method with LLM extraction of original/corrected claims
- [x] `supersede_fact()` marks old facts deprecated (confidence → 0.1)
- [x] `[:SUPERSEDES]` relationship for audit trail
- [x] Integrated into consolidation pipeline (disabled by default)

### 11.12.X Infrastructure Improvements (Complete)
- [x] `ConsolidationMetrics` dataclass for pipeline observability
- [x] Batch entity/relationship storage with UNWIND (fixes N+1 queries)
- [x] `claim_hash` field on Fact model for indexed duplicate detection
- [x] Settings cache with 60s TTL refresh (UI changes take effect without restart)
- [x] 14 unit tests for metrics, claim hash, and settings cache
- [ ] LLM timeout enforcement (deferred - requires async/sync architecture fix)

### 11.12.6 Confidence Calibration (Complete)
- [x] Confidence scale: explicit=0.95, implied=0.85, inferred=0.70, uncertain=0.50
- [x] Configurable thresholds via settings
- [x] Certainty-to-confidence mapping in extraction pipeline
- [ ] Calibration factors: source, recency, corroboration, contradiction (deferred)

### 11.12.7 Reinforcement Signal (Complete)
- [x] Add `last_accessed`, `access_count`, `salience` fields to Fact model (parity with Entity)
- [x] Track access on retrieval (`vector_search_facts` increments access_count)
- [x] Use salience in retrieval scoring (`_rerank()` includes salience factor)
- [x] Fixes broken promotion system that referenced non-existent Fact.access_count
- [ ] Negative reinforcement for corrected facts (deferred)

### 11.12.8 Source Attribution (Partial)
- [x] Store `source_turn_id` on all extracted facts
- [ ] UI: "Where did I learn this?" → show original conversation

### 11.12.9 Temporal Reasoning (Complete)
- [x] Add `temporal_context` field to Fact model (current/past/future)
- [x] Extract temporal context in combined extraction prompt
- [x] Normalize temporal fields (`_normalize_temporal_fields()`)
- [x] Temporal boost in retrieval: current=1.2x, past=0.7x, future/null=1.0x
- [x] Pass temporal_context through consolidation pipeline
- [x] 14 unit tests for access tracking and temporal context

### 11.12.10 Consolidation Settings UI (Client)
- [ ] Create Consolidation Settings section in Settings tab
- [ ] Extraction, Relevance Filter, Contradiction Detection settings
- [ ] Provider presets (Local, Quality, Full Cloud)
- [ ] API endpoints: `GET/PUT /api/memory/config`

---

## Phase 12: Documentation

> **Priority**: LOW
> **Goal**: Comprehensive documentation for users and developers

### 12.1 User Documentation
- [X] Update README.md with quick start guide, feature overview, screenshots
- [X] Add installation guide for each platform
- [X] Document MCP setup for AI assistants

### 12.2 Developer Documentation
- [ ] Add API documentation (auto-generate from OpenAPI)
- [X] Add architecture diagrams
- [ ] Document contribution guidelines
- [X] Document memory system extension patterns (from 11.6)

### 12.3 Inline Documentation
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout Python code
- [ ] Document complex algorithms/flows

---

## Phase 13: UX Overhaul — Immersive AgentX

> **Priority**: HIGH
> **Goal**: Immersive 3-page app with browser-style conversation tabs, portal-based modals, theme system, customizable agent profiles, and rich inline feedback
> **Replaces**: Old 7-tab sidebar layout. Chat + Agent merged into single workspace.

**Design Decisions:**
- Tool call approval: visual-only for now (message types support future blocking approval)
- Former tabs (Settings, Memory, Tools) → right-side slide-out drawers from toolbar icons
- Translation → centered modal from toolbar
- Conversation history → dropdown near tab bar "+" button (not a persistent sidebar)
- Data is server-first, client caches only (prep for future multi-server auth)

### 13.1 Foundation (No Visible UI Changes)

> Infrastructure that all subsequent sub-phases depend on

#### Theme System
- [x] Create `ThemeProvider` context (`contexts/ThemeContext.tsx`)
- [x] Extract CSS variables from `App.css` `:root` into `ThemeDefinition` object (`lib/theme.ts`)
- [x] `ThemeProvider` applies variables to `document.documentElement.style` on mount/change
- [x] Ship `cosmic` theme as default; architecture supports adding themes later
- [x] Move non-variable CSS (resets, animations, component base styles) into `styles/base.css`
- [x] `App.css` becomes thin wrapper importing `base.css`

#### Modal/Portal System
- [x] Add `<div id="modal-root" />` to `index.html`
- [x] Create `ModalContext` (`contexts/ModalContext.tsx`) — `openModal(config)`, `closeModal(id)`, stack-based
- [x] Create `ModalPortal.tsx` — renders children into `#modal-root` via `ReactDOM.createPortal`
- [x] Create `DrawerPanel.tsx` — slide-in panel (left/right), backdrop overlay, sizes (sm/md/lg)
- [x] Create `ModalDialog.tsx` — centered overlay modal
- [x] Lazy-loaded component registry to avoid circular imports

#### Message Type System
- [x] Create `lib/messages.ts` — discriminated union: `UserMessage | AssistantMessage | ToolCallMessage | ToolResultMessage | MemoryInjectionMessage | SystemMessage | ErrorMessage`
- [x] Each type carries specific metadata (tool names, confidence scores, model info, etc.)
- [x] Type guards for each message type
- [x] `ToolCallMessage.status` field: `pending | approved | rejected | completed` (prep for future blocking)

### 13.2 Root Layout Shell

> Replace sidebar TabBar with horizontal top bar + page routing

#### RootLayout & TopBar
- [x] Create `RootLayout.tsx` (`layouts/`) — top bar + full-height page content area
- [x] Create `TopBar.tsx` (`layouts/`) with sections:
  - Left: Logo + active agent name (dynamic from profile)
  - Center-left: Page nav pills (Start, Dashboard, AgentX)
  - Center: ConversationTabBar placeholder (wired in 13.3)
  - Right: Toolbar icons (Settings, Memory, Tools, Translation) → open modals
- [x] Page switching via `useState<'start' | 'dashboard' | 'agentx'>`
- [x] Migrate `App.tsx` from `TabBar` + 7 display-toggled divs to `RootLayout` + page routing
- [x] Initially, "AgentX" page renders existing `ChatTab` as-is (incremental migration)

### 13.3 Conversation Tabs

> Browser-style tabs in top bar, each representing a conversation

#### ConversationContext
- [ ] Create `ConversationContext` (`contexts/ConversationContext.tsx`)
- [ ] `ConversationTab` model: `id`, `title`, `sessionId`, `profileId`, `messages: ConversationMessage[]`, `isStreaming`, timestamps
- [ ] Actions: `addTab()`, `closeTab(id)`, `switchTab(id)`, `renameTab()`, `reorderTabs()`
- [ ] Persistence: `agentx:server:{id}:convTabs` (tab list), `agentx:server:{id}:conv:{tabId}:msgs` (messages)
- [ ] Max ~20 tabs, messages capped at ~200 per tab in localStorage (backend has full history)

#### ConversationTabBar
- [ ] Create `ConversationTabBar.tsx` (`layouts/`) — horizontal scrollable tabs with close buttons
- [ ] "+" button to add new tab (inherits active profile)
- [ ] History button (clock icon) opens dropdown of past conversations
- [ ] Active tab highlighted; only visible on AgentX page
- [ ] Wire into `TopBar` center area

### 13.4 Agent Profiles (Frontend + Backend)

> Customizable agent profiles with names that carry across UI and prompts

#### Backend
- [ ] Create `ProfileManager` class (`agent/profiles.py`) — pattern follows `prompts/manager.py`
- [ ] Storage: `data/agent_profiles.yaml` with default profiles
- [ ] API endpoints: `GET/POST /api/agent/profiles`, `GET/PUT/DELETE /api/agent/profiles/<id>`
- [ ] Views in `views.py` for profile CRUD
- [ ] Agent name injection: prepend `"Your name is {profile.name}."` to system prompt in `agent/core.py`

#### Frontend
- [ ] Create `AgentProfileContext` (`contexts/AgentProfileContext.tsx`)
- [ ] `AgentProfile` model: `id`, `name` (the "X"), `avatar`, `defaultModel`, `temperature`, `promptProfileId`, `reasoningStrategy`, `enableMemory`, `memoryChannel`, `enableTools`, `isDefault`, timestamps
- [ ] Fetch from server, cache in localStorage per-server
- [ ] `activeProfile`, `setActiveProfile(id)`, `createProfile()`, `updateProfile()`
- [ ] `getAgentName()` used by TopBar logo and StartPage greeting
- [ ] Create `ProfileEditorModal.tsx` — form: name, avatar, model, temperature, prompt profile, reasoning, memory/tools toggles
- [ ] Add profile API methods to `lib/api.ts` and `useAgentProfiles()` hook to `lib/hooks.ts`

### 13.5 Modals Migration

> Convert former tabs into modal/drawer panels opened from toolbar

- [ ] `SettingsModal.tsx` — wrap `SettingsTab` content in `DrawerPanel` (right, md). Trigger: gear icon
- [ ] `MemoryExplorerModal.tsx` — wrap `MemoryTab` content in `DrawerPanel` (right, lg). Trigger: database icon
- [ ] `ToolsModal.tsx` — wrap `ToolsTab` content in `DrawerPanel` (right, md). Trigger: wrench icon
- [ ] `TranslationModal.tsx` — wrap `TranslationTab` content in `ModalDialog` (center). Trigger: globe icon
- [ ] Wire toolbar icons in `TopBar` to open each modal via `useModal().openModal()`
- [ ] Remove old tab entries from `App.tsx` page routing

### 13.6 Merged AgentX Page

> Core of the overhaul — combines Chat + Agent into one rich workspace

#### AgentXPage
- [ ] Create `AgentXPage.tsx` (`pages/`) — reads active conversation tab from `ConversationContext`
- [ ] Full-width message area + input bar (maximally immersive)
- [ ] Profile badge in header shows agent name + model

#### Message Components
- [ ] `MessageBubble.tsx` — renders any `ConversationMessage` via switch on type:
  - `user`: avatar + content + edit action
  - `assistant`: agent avatar/name + markdown + expandable thinking + metadata bar + actions (copy, regenerate, pin)
  - `tool_call`: compact inline block with tool name, arguments preview, status badge
  - `tool_result`: compact block with tool name, result preview, success/fail, duration
  - `memory_injection`: collapsible facts with confidence bars, entities with type badges
  - `system`: subtle centered text
  - `error`: red-tinted block
- [ ] `MessageInput.tsx` — auto-grow textarea, Enter/Shift+Enter, model badge, temperature, profile selector, send/stop
- [ ] `ToolCallBlock.tsx` — tool name + collapsible arguments JSON + status badge (visual-only, no blocking yet)
- [ ] `MemoryInjectionBlock.tsx` — facts list with confidence, entities with type badges, collapsible (default collapsed)
- [ ] `ThinkingBlock.tsx` — collapsible reasoning content, step count, streaming indicator
- [ ] `MetadataBar.tsx` — per-message footer: model name, token count (in/out), latency
- [ ] `ConversationHistoryDropdown.tsx` — dropdown near tab bar "+" for browsing past conversations
- [ ] Reuse existing `chat/MessageContent.tsx` (markdown renderer) inside `MessageBubble`

### 13.7 SSE & Metadata Enhancements

> Extend streaming backend to emit richer events for new message types

#### Backend (views.py — `agent_chat_stream` / `generate_sse`)
- [ ] New SSE event `memory_context`: emit after memory retrieval, before first chunk. Data: `{ facts, entities, query }`
- [ ] Extend `tool_call` event: add `tool_call_id` field for linking to results
- [ ] Extend `tool_result` event: add `tool_call_id` and `duration_ms`
- [ ] Extend `start` event: add `profile_name`, `agent_name`, `model_display_name`
- [ ] Extend `done` event: add `tokens_input`, `tokens_output`, `profile_name`, `agent_name`

#### Frontend (lib/api.ts)
- [ ] Update `streamChat()` to handle `memory_context`, `tool_call`, `tool_result` events
- [ ] Add callbacks: `onToolCall`, `onToolResult`, `onMemoryContext`
- [ ] AgentXPage converts SSE events into typed `ConversationMessage` objects in the message list

### 13.8 Prompt Library

> Profile-agnostic prompt templates with tag-based organization

#### Backend
- [ ] Create `PromptTemplateManager` (`prompts/templates.py`) — follows `PromptManager` pattern
- [ ] Storage: `data/prompt_templates.yaml`
- [ ] Endpoints: `GET/POST/PUT/DELETE /api/prompts/templates`
- [ ] Fields: `id`, `name`, `content`, `tags[]`, `placeholders[]`, `type` (system/user/snippet)

#### Frontend
- [ ] Create `PromptLibraryModal.tsx` — opened from toolbar or slash command in input
- [ ] Tag filter sidebar, search bar, template list
- [ ] Template preview with placeholder highlighting
- [ ] "Use in conversation" action (inserts into input)
- [ ] "Attach to profile" action (sets as profile's prompt)
- [ ] Add template API methods to `lib/api.ts`

### 13.9 Start Page & Dashboard Refresh

#### Start Page
- [ ] Create `StartPage.tsx` (`pages/`) — centered agent avatar + "Hello, I'm {agentName}" greeting
- [ ] Quick actions: "New Conversation", "Open Dashboard"
- [ ] Minimal placeholder (future: onboarding, tips, recent activity)

#### Dashboard Refresh
- [ ] Create `DashboardPage.tsx` (`pages/`) — refactored from `DashboardTab`
- [ ] Keep: health status grid, server banner
- [ ] Add: token usage metrics, memory stats (from `/api/memory/stats`), active conversation count
- [ ] Add: running agents indicator
- [ ] Remove: quick action buttons (replaced by toolbar/page nav)

### 13.10 Polish & Cleanup

- [ ] Remove `components/tabs/` directory (all old tab components)
- [ ] Remove `components/TabBar.tsx`
- [ ] Remove old `styles/*Tab.css` files
- [ ] Clean up dead `TabId` type and related code
- [ ] Final CSS consistency pass
- [ ] Keyboard shortcuts: Cmd+T (new tab), Cmd+W (close tab), Cmd+K (command palette placeholder)

### Dependency Graph

```
13.1 Foundation ──────┬──→ 13.2 Root Layout ──→ 13.3 Conv Tabs ──→ 13.6 AgentX Page ──→ 13.7 SSE
                      ├──→ 13.5 Modals ────────┘                                            │
                      └──→ 13.4 Profiles (backend can start early) ──────────────────────────┘
                                     └──→ 13.8 Prompt Library
13.2 ──→ 13.9 Start/Dashboard (can parallel with 13.6)
ALL ──→ 13.10 Cleanup
```

---

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

- [ ] GPU acceleration for translation models
- [ ] Lazy model loading with progress indicator
- [ ] Multiple server support (user can log out of server, and into another one seamlessly)
- [ ] Cloud sync for memories
- [ ] Plugin system for additional tools
- [ ] Voice input/output
- [ ] Mobile companion app
- [ ] Offline mode with cached models
- [ ] Cross-encoder reranking model for retrieval quality
- [ ] Memory export/import (JSON/SQLite backup)
- [ ] Advanced memory visualization (interactive graph, embedding clusters)
- [ ] Streaming memory retrieval during chat
- [ ] Conversation sharing (read-only shareable links)
- [ ] Blocking tool call approval (pause stream, user approves/rejects before execution)
- [ ] Server authentication (single access key per server, session resume on reconnect)
- [ ] Multi-agent collaboration modes (multiple agents/models working together on a task)
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

**Settings Cached at Import Time**
- `get_settings()` uses `@lru_cache`, loaded at module import
- Mitigation: Settings require restart; consider cache invalidation

**Query Embedding Caching**
- Every `remember()` call generates embedding even for identical queries
- Fix: Add MRU cache for frequent query embeddings with TTL

---

## Blockers

None currently.

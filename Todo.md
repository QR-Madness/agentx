# AgentX Development Todo

**Project**: AgentX - AI Agent Platform
**Status**: Pre-prototype
**Last Updated**: 2026-02-26

> For completed phases (1-10), decision log, and project history, see [docs/roadmap.md](docs/roadmap.md)

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phases 1-10 | Complete | See [roadmap.md](docs/roadmap.md) |
| Phase 11: Memory System | **In Progress** | 90% |
| Phase 11.12: LLM-Enhanced Consolidation | Not Started | 0% |
| Phase 12: Documentation | Not Started | 0% |
| Phase 13: UI Implementation | **In Progress** | 15% |

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

### 11.12.1 Pre-Extraction Relevance Filter
- [ ] Add `filter_relevant_turns()` to consolidation pipeline
- [ ] LLM prompt: "Does this text contain memorable information? YES/NO"
- [ ] Heuristic pre-filter (skip turns < 10 chars, "ok", "thanks", etc.)
- [ ] Metrics: track skip rate, extraction savings

### 11.12.2 Enhanced Fact Extraction + Condensation
- [ ] Improve extraction prompt (user-stated facts only, atomic facts)
- [ ] Condense verbose statements to atomic facts
- [ ] Batch extraction (combine multiple turns)

### 11.12.3 Entity Matching via Embedding Search
- [ ] Add `link_facts_to_entities()` consolidation job
- [ ] Embedding-based entity resolution (handle aliases)
- [ ] Optional LLM disambiguation for ambiguous matches

### 11.12.4 Contradiction Detection
- [ ] Add `check_contradictions()` to extraction pipeline
- [ ] LLM prompt: "Do these facts contradict each other?"
- [ ] Contradiction resolution strategies (keep_both, prefer_recent, flag_review)

### 11.12.5 User Correction Handling
- [ ] Pattern matching: "actually...", "no, I meant...", "that's wrong..."
- [ ] Find and supersede corrected facts
- [ ] Log corrections in audit trail

### 11.12.6 Confidence Calibration
- [ ] Define confidence scale (0.9+ explicit, 0.7-0.9 implied, 0.5-0.7 inferred)
- [ ] Calibration factors: source, recency, corroboration, contradiction

### 11.12.7 Reinforcement Signal
- [ ] Track memory usage in chat (retrieved and used in response)
- [ ] Boost salience for frequently-used memories
- [ ] Negative reinforcement for corrected facts

### 11.12.8 Source Attribution
- [ ] Store `source_turn_id` on all extracted facts
- [ ] UI: "Where did I learn this?" → show original conversation

### 11.12.9 Temporal Reasoning
- [ ] Add temporal fields to Fact model (valid_from, valid_until)
- [ ] Extract temporal context from text
- [ ] Retrieval: prefer current facts over outdated ones

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
- [ ] Update README.md with quick start guide, feature overview, screenshots
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
> **Goal**: Dual-interface experience — lightweight Chat for quick interactions, powerful Agent for prompt engineering

### 13.1 Chat Tab — Lightweight Interface (Complete)
- [x] Chat message input (auto-grow, Enter to send, character counter)
- [x] Message display (streaming, markdown, code highlighting, copy/regenerate)
- [x] Quick settings bar (model selector, temperature, memory toggle)
- [x] Session management (new chat, recent chats list)

### 13.2 Agent Tab — Power Interface

#### 13.2.1 Conversation Sidebar
- [ ] Conversation list with title, preview, timestamp, profile badge
- [ ] Organization: folders, tags, star/favorite, archive
- [ ] Search: full-text, filter by date/profile/tags
- [ ] Bulk actions: multi-select, delete, archive, move

#### 13.2.2 Conversation View
- [ ] Message display with metadata (model, tokens, latency)
- [ ] Message operations: edit, regenerate, delete, pin, annotate, copy
- [ ] Branching: visual indicator, branch switcher, compare side-by-side

#### 13.2.3 Input Area
- [ ] Rich input: slash commands, @-mentions, template insertion
- [ ] Context controls: window visualization, include checkboxes
- [ ] Action buttons: send, send with options, stop, retry

#### 13.2.4 Conversation Header Bar
- [ ] Profile selector with quick-switch
- [ ] Conversation actions: rename, export, delete
- [ ] View toggles: reasoning traces, tool usage, memory context

### 13.3 Agent Profiles
- [ ] Define `AgentProfile` data model (system prompt, model, reasoning, memory, tools)
- [ ] Profile inheritance via `extends`
- [ ] Profile Management UI in Settings tab
- [ ] Built-in profile templates (General Assistant, Code Helper, Creative Writer, etc.)

### 13.4 Prompt Library
- [ ] Library structure: system prompts, user templates, snippets
- [ ] Tagging system with filter/search
- [ ] Template features: placeholders, import/export
- [ ] Integration: `/template` slash command, quick insert

### 13.5 Conversation Persistence & Sync
- [ ] API endpoints for conversations (CRUD, messages, branches, export)
- [ ] API endpoints for profiles (CRUD, clone, history, restore)
- [ ] Database models: Conversation, Message, Profile, ProfileHistory, PromptTemplate, Tag

### 13.6 Settings Tab Rework
- [ ] Reorganize: General, Agent Profiles, Model Providers, MCP Servers, Memory, Prompt Library, Data

### 13.7 Keyboard Shortcuts & Accessibility
- [ ] Global shortcuts: Cmd+K (command palette), Cmd+N (new), Cmd+P (switch profile)
- [ ] Conversation shortcuts: navigate, edit, regenerate, branch
- [ ] Command palette with fuzzy search

### 13.8 Mobile-Responsive Considerations
- [ ] Responsive breakpoints (collapse sidebar <1024px)
- [ ] Touch-friendly tap targets and gestures

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
- [ ] Advanced memory visualization (interactive graph, embedding clusters)
- [ ] Streaming memory retrieval during chat
- [ ] Conversation sharing (read-only shareable links)

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

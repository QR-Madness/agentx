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

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

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
- [ ] Mobile companion app
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

### ERRORS TO FIX (DELETE ME AFTER)

1.) Plans fail to init: [API] INFO 2026-03-31 17:24:29,283 views Using context limits for lmstudio/nvidia/nemotron-3-nano: window=128,000, max_output=8,192
[API] DEBUG 2026-03-31 17:24:29,283 views No memory context to emit (bundle: False)
[API] DEBUG 2026-03-31 17:24:29,283 views Adaptive max_tokens: 8192 (context_window=128000, estimated_input=531, max_output=8192, available=125469)
[API] ERROR 2026-03-31 17:24:29,283 planner Planning failed: 'coroutine' object has no attribute 'content'

2.) Adaptive Tokens will cause an error on Anthropic saying it's Exceeding Max output tokens for haiku (8192>4096), also some more issues in this log: (context window and max tokens should be scoped to the model): [API] DEBUG 2026-03-31 17:47:20,867 views Adaptive max_tokens: 8192 (context_window=1000000, estimated_input=422, max_output=8192, available=997578)
Batches: 100%|██████████| 1/1 [00:00<00:00,  7.27it/s]
[API] DEBUG 2026-03-31 17:47:21,115 planner Created goal 9e1cb937-c2f5-4b88-a36f-59a9408e5a77 for task plan
[API] INFO 2026-03-31 17:47:21,115 tool_loop Tool round 1: 2 messages, ~422 tokens, limit=32000
[API] DEBUG 2026-03-31 17:47:21,115 anthropic_provider Anthropic stream: model=claude-3-haiku-20240307, messages=2
[API] INFO 2026-03-31 17:47:21,115 base [DEBUG LLM REQUEST] Anthropic (stream):

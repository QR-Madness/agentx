# Phase 15 — Plan Execution (Core Complete)

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

## Phase 15: Plan Execution (Core Complete)

> **Goal**: Execute decomposed task plans instead of discarding them — subtask iteration, Redis state tracking, streaming progress events
> Core shipped (15.1–15.6 + cancellation). Moved to [roadmap.md](docs-site/src/content/docs/roadmap.md):
> `PlanStateStore` + `PlanExecutor` (dependency-ordered subtasks, per-subtask trajectory
> compression, failure skip, synthesis), `Agent.run`/streaming integration, SSE plan events,
> and mid-execution cancellation. Only the deferred follow-ups below remain.

### 15.10 Execution safety (part 1) — shipped

- [x] **"Stuck in web_search"** — `_tavily_search` called the SDK with no timeout (SDK default
      ~60s), blocking the synchronous tool loop and stalling Stop. Capped via `search.timeout`
      (default 15s) on both backends.
- [x] **Clean termination + faithful resume.** `PlanExecutor` wraps its loop in `try/finally`:
      a hard Stop (GeneratorExit) mid-subtask resets the in-flight subtask to `pending` and marks
      the plan `interrupted` (resumable) instead of stuck `running`; cooperative cancel
      `clear_cancel`s the flag. `views.py` Stop handler persists the interrupted plan card (was
      persisting nothing on the plan path). Client `mapServerMessages` trusts the persisted plan
      status (was rewriting incomplete plans to `failed`, hiding the resume offer); `interrupted`
      added across the status union + resume detection. Covered by `PlanTerminationTest`.
- [x] **Plan conversations 404 on restore (the "first turn not persisted" bug).** `store_turn`
      embeds then writes Neo4j→PG and **re-raises on any failure**, so one failing shared resource
      (the embedder, or Neo4j) dropped the PostgreSQL `conversation_logs` row that restore reads —
      leaving the conversation un-openable ("Failed to restore"). The plan-turn persist loop now
      isolates failures **per turn** and, on `store_turn` failure, falls back to a direct
      `conversation_logs` write (no embedding/graph needed) so restore always has the user +
      assistant rows (`views.py::_persist_turns._run`).
- [x] **Resume offer never appeared.** An interrupted plan that stops *before synthesis* has blank
      assistant content, so `build_assistant_turn` dropped it (skip-empty rule) → the plan card +
      its resume nudge were lost on restore. Now a turn carrying `metadata["plan"]` is kept even
      when blank (`streaming/persistence.py`), and `mapServerMessages` no longer renders an empty
      assistant bubble under the card. Covered by `build_assistant_turn` + `mapServerMessages` tests.
- [x] **Cooperative mid-loop cancel.** `streaming_tool_loop` now takes a `cancel_check` (default:
      the ambient run-cancel flag via `current_run_id`) checked at each round boundary and before
      tool execution, so Stop/Cancel lands *between rounds* within a subtask, not only at subtask
      boundaries. `PlanExecutor` passes a check that also ORs the plan-cancel flag. Purely
      cooperative — **no `run_in_executor`/`shield`/ThreadPoolExecutor** (see below).
- [x] **Resume context re-attachment.** Plan subtask tool calls/results aren't persisted (only the
      synthesis is), so a plan cancelled *before synthesis* left the resuming model with just the
      question + plan card — no view of what earlier steps produced. The resume nudge
      (`buildResumeNudge`, `client/src/lib/planResume.ts`) now carries each completed step's
      **result** (from the `/status` Redis snapshot, ~1500-char capped) so prior context rides into
      the resume turn. (Durable per-step tool-turn persistence beyond Redis's 1h TTL is a separate,
      larger follow-up — see deferred below.)
- [ ] **Deferred — persist plan subtask tool turns.** Capture per-subtask tool calls/results,
      aggregate them in `PlanResult`, and write them to `conversation_logs` so a restored plan shows
      its tool cards faithfully and survives beyond the Redis TTL. (Today only the user turn + final
      synthesis/plan-card are stored; `hydrate_session_from_history` also skips tool rows.)
- [x] **Dev-tooling:** `task plan:inspect <session_id> [plan_id]` — dumps the Redis plan state
      (status/subtasks/cancel/ttl/resumable) **and** the session's `conversation_logs` rows
      (turn_index/role/content-len/has-plan), the observability that root-caused the restore bug.
- [ ] **Deferred — prompt mid-tool cancel.** Truly-instant Stop *during* a single tool call needs
      off-thread tool execution; the first attempt (`run_in_executor` + `asyncio.shield` in
      `streaming_tool_loop`) **deadlocked `gen.aclose()`** (Stop hung, turns never persisted) and was
      reverted. The dead `feat/plan-executor-hardening` branch was **retired** (its only unique
      content was that deadlocking approach + it lacked the `search.timeout` fix); rebuild from
      scratch with a design that doesn't block generator close, and reproduce the hang in a test
      *first*. The cooperative between-rounds check above is the salvaged-safe half.
- [ ] **Deferred — shared plan state-machine module** (statuses + transitions) imported by
      executor/state-store/views/client, to kill the `complete`/`completed` wire-vocab drift.

### 15.9 Main-agent plan composition — shipped

- [x] **Plans wouldn't compose** (the separate planner was context-blind + brittle). Root cause:
      the chat path called `TaskPlanner.plan(message)` with **no conversation context**, using a
      rigid `SUBTASK N:`/`TYPE:` prompt that `_parse_plan` regex-scraped — any model that answered
      in markdown/JSON/prose matched zero blocks → single-step fallback → `len(steps) > 1` false →
      never decomposed (lowering `complexity_threshold` couldn't help; the failure is downstream of
      the rank gate). Fix: the chat path now composes the plan with the **main agent model** via
      `TaskPlanner.compose_with_model(provider, model_id, messages, …)` — it reuses the
      already-assembled turn context (system prompt + memory + history) and takes a **structured
      JSON plan** back (`_extract_json_object`, tolerant of fenced/embedded JSON; coercion helpers
      for type/deps/tools; `_normalize_steps` for ids/deps/cap). The model decides whether to
      decompose (returns `{"plan": null}` → normal single-pass turn). A cheap `_assess_complexity`
      heuristic gates whether the extra call runs at all (SIMPLE/trivial turns skip it; the config
      `planner.complexity_threshold` still tunes moderate-vs-complex). The legacy `plan()` +
      `SUBTASK` path stays for non-chat callers. Covered by `PlannerComposeTest`.

### 15.8 Fixes — shipped `[v0.21.8]`

- [x] **Executor looped on one subtask** ("step 3 of 9"). `Subtask.id` was used as a list index
      everywhere but `_parse_plan` set it from the LLM's `SUBTASK N` numbering, so non-contiguous/
      duplicate numbering made `mark_complete` flip the wrong slot → the running subtask never
      completed → re-selected forever. Fix: `_normalize_steps` reindexes to `steps[i].id == i` and
      remaps/sanitizes dependencies; plus a no-progress safety guard in the executor loop.
- [x] **Over-decomposition** ("giant plans for simple things"). `_assess_complexity` rewritten to
      require genuine multi-step structure (sequence markers / multiple action clauses / length),
      not a lone keyword; `planner.decompose` prompt now mandates the fewest subtasks (single-step
      allowed) with a hard cap; `planner.max_subtasks` (default 6) enforced in `_parse_plan`; default
      `complexity_threshold` raised to **complex**. Settings now seed the prompt editor with the
      live default (`/api/config` `planner.decompose_default`) + a Reset-to-default action.

### 15.7 Deferred Items

- [ ] Parallel subtask execution (independent subtasks could run concurrently) — **prerequisite now
      met**: the embedding request queue/serializer + cache shipped (`[v0.21.6]`,
      `kit/agent_memory/embedding_queue.py`), so concurrent subtasks' recall/embedding bursts are
      serialized safely. Remaining work is the parallel scheduler in `PlanExecutor` itself.
- [ ] Per-subtask reasoning strategy selection (use `_select_strategy` per subtask type)
- [x] Subtask-level goal tracking (create subgoals via `parent_goal_id`) — shipped. The
      planner's `_create_goal_for_plan` creates a child `Goal(parent_goal_id=plan.goal_id)`
      per subtask (skipped for single-step plans) and stamps `step.goal_id`; `PlanExecutor`
      closes each out through the agent hook seam (`_complete_subtask_goal` →
      `on_goal_complete` → `MemoryRecorder.complete_goal`) on complete/fail/abandon.
      Certified by `SubtaskGoalTrackingTest`.
- [x] Plan cancellation mid-execution — shipped. `PlanExecutor` checks `state.is_cancel_requested(plan_id)` between subtasks (`plan_executor.py:84,163`) and marks the plan `cancelled`; `POST /api/agent/plans/cancel` sets the flag.
- [x] Plan resumption from Redis state after disconnect — shipped (**single-agent**). B1 durable
      full-plan serialization (`Subtask`/`TaskPlan` `to_dict`/`from_dict`; `PlanStateStore.create`
      writes a `plan_json` snapshot; `load_plan` rebuilds + overlays live status, `is_resumable`).
      B2 `PlanExecutor.execute_streaming(resume_plan_id=…)` emits a `plan_resumed` snapshot then
      continues the loop (dedup is automatic — terminal subtasks are pre-marked). B3
      `POST /api/agent/plans/{plan_id}/resume` (rebuilds a single-agent Agent + hydrated context,
      detached run, persists synthesis; `GET .../status` now reports `resumable`) + an
      in-conversation "Resume plan" affordance on the plan card (`PlanExecutionBlock`, via
      `useChatStream.resume` → `streamingApi.resumePlan`). B4 docs.
      Covered by `PlanSerializationTest` + the resume streaming test. **Alloy plan resumption
      remains a separate follow-up** (needs `_active_alloy_executor` re-attached for the
      per-subtask `delegate_to` injection).
- [ ] Alloy plan resumption — resume a workflow-scoped plan. The resume endpoint must rebuild the
      `AlloyExecutor` (`workflow_id` from the plan/Redis) and attach it as `_active_alloy_executor`
      so resumed subtasks keep the `delegate_to` tool; otherwise a supervisor subtask loses
      delegation on resume. Deferred from the single-agent resume slice above.



# AgentX Development To-do

**Project**: AgentX - AI Agent Platform
**Status**: Prototype
**Last Updated**: 2026-06-03

**NOTE**: UI must be highly responsive between PC and mobile devices.

**Versioning**: `versions.yaml` is the single source of truth (run `task versions:sync` after
editing it). Completed work is tagged inline with the version it shipped in, e.g. `[v0.20.1]`.
Bump the version when a unit of work completes — patch for additive/back-compat features, and
bump `protocol_version` only on breaking API changes. Current: **0.21.29** (protocol 1).

> For completed phases and project history, see [roadmap.md](docs-site/src/content/docs/roadmap.md)

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phases 1-11, 13, 14, 17 | **Complete** | See [roadmap.md](docs-site/src/content/docs/roadmap.md) |
| Phase 12: Documentation | Partial | ~60% |
| Phase 15: Plan Execution | **Core Complete** | Core shipped; parallelism/resumption deferred |
| Phase 16: Multi-Agent Conversations | **In Progress** | ~72% (16.0–16.5 + 16.6 ambassador foundation & TTS voice shipped; 16.7 Ambassador v2 rework planned; Factory UI deferred) |
| Phase 18: UX + Memory Tuning | **In Progress** | ~98% (18.9 done; eval procedural cases + run persistence done; memory import/export shipped `[v0.21.22]` → eval snapshot/restore now unblocked) |

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

> Moved to [roadmap.md](docs-site/src/content/docs/roadmap.md). All items complete:
> - 13.6 AgentXPage with ConversationContext + ChatPanel
> - 13.10 Removed old tabs dir, renamed to panels, keyboard shortcuts (Ctrl+T/W/K), CSS pass
> - 13.11 Consolidation settings UI + TopBar integration

---

## Phase 14: Context Gating (Complete)

> All 14.1-14.7 complete. See [roadmap.md](docs-site/src/content/docs/roadmap.md) for details.

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

---

## Phase 16: Multi-Agent Conversations

> **Priority**: MEDIUM
> **Goal**: Enable multiple agents to collaborate in shared conversation threads
> **Depends on**: Plan execution (Phase 15)

> **Agent Alloy** = the multi-agent system. **Factory** = the visual editor (frontend, not yet built).
> Control flow: supervisor agent owns the conversation; specialists are invoked via a `delegate_to` tool. Opt-in per chat request via `workflow_id`.

### Shipped (16.0–16.5) — moved to [roadmap.md](docs-site/src/content/docs/roadmap.md)

- **16.0 Agent Alloy v1 backend** (2026-04-27): `alloy/` package, workflow model + `WorkflowManager` YAML CRUD, `delegate_to` tool, `AlloyExecutor` (shared `_alloy_<id>` channel, child goals, `delegation_*` SSE streaming, depth-limited re-delegation), supervisor framing prompt, `/api/alloy/workflows` CRUD, `alloy.*` config.
- **Parallel / fan-out delegation** + **trace/replay UI** (`[v0.20.1]`): per-delegation tokens/duration/cost/pricing-snapshot persisted; client `AlloyRunTraceModal` groups fan-out into runs.
- **16.1 Message attribution**: `agent_id` on `Turn` + `conversation_logs`; persisted across streaming/non-streaming/background and restored to display names.
- **16.2 Explicit routing** (`[v0.21.2]`): `target_agent_id`, `Session.participants` hydration, multi-agent awareness prompt.
- **16.3 Per-agent tool isolation**: `allowed_tools`/`blocked_tools` enforced in `_get_tools_for_provider`.
- **16.4 Ad-hoc agent-to-agent delegation** (`[v0.21.3]`): workflow-less `AlloyExecutor` mode gated by `alloy.allow_adhoc_delegation`, depth-limited, no self-delegation.
- **16.5 @-mention routing** (`[v0.21.4]`/`[v0.21.5]`): `agent/mentions.py` parsing, `AgentParticipant` Neo4j nodes + backfill migration, client `@`-autocomplete composer.
- **Multi-agent attribution**: attribution is now per-agent, not a singleton "agent". Agents are first-class `Entity(type="Agent")` (canonical `properties.agent_id`, name as prose, prior names as aliases); facts attributed to a specific agent (`subject_agent` name → resolved `subject_agent_id`) route to that agent's `_self_` channel — so a directive aimed at Mobius lands in Mobius's memory, not Atlas's. Roster-aware extraction prompts + per-turn responder resolution for "you"; assistant self-extraction routes each turn by its own producing `agent_id`. Display names stamped onto `Turn`/`AgentParticipant` at write-time (`get_conversation_roster`); rename-safety via Agent-entity aliases; `dedupe_entities` skips Agent nodes; deterministic legacy backfill (`task memory:backfill-agent-attribution`).

### 16.x Deferred / Next

- [-] Factory canvas frontend (Tauri client) — backend exposes everything needed
- [-] Declarative route execution (`on_complete`, `on_match:<predicate>`, `on_failure`) — schema accepted but ignored in v1
- [ ] Loop nodes (specialist iterates over a list)
- [ ] Human-in-the-loop checkpoint nodes
- [ ] Async delegation (specialist runs in background; supervisor continues other work)
- [ ] "User as supervisor" mode (no LLM supervisor — user manually invokes specialists from the chat UI)
- [ ] Tool-output sharing across agents (specialist's raw tool outputs visible to supervisor, not just final text)
- [ ] Per-workflow tool isolation (specialist inherits a *subset* of supervisor tools, not all)
- [ ] Trace UI follow-up: persist per-tool timing (executor currently stores one rollup turn per delegation → restored runs show delegation-level metrics only); fold specialist tokens into the supervisor done-event cost rollup
- [ ] Attribution follow-up: backfill historical NULL `agent_id` rows
- [ ] **Attribution quality in compound messages** — the `debug_attribution` harness shows
      that on a *mixed* user turn ("I prefer metric… also Mobius, cite sources… Jeff, be
      concise"), a small extraction model (gpt-4o-mini) left the per-agent directives in the
      active channel instead of homing them to each agent's `_self_`, and dropped some facts.
      Clean single directives route correctly. Follow-up: tune the `combined_with_relevance`
      prompt (or default to a stronger extraction model) so multi-directive turns split +
      attribute reliably; add a golden-output regression once stabilized.
- [ ] **Full-roster DI provider** — resolve user-named agents that aren't conversation
      participants (today they demote to `third_party`); inject the full profile roster into
      consolidation without coupling the kit to `ProfileManager`.
- [ ] **Agent social/delegation graph** — mine cross-agent facts ("Atlas is faster at SQL
      than Mobius") into a graph that informs Agent Alloy routing ("who's good at what").
- [ ] **Delegation handbook (the "Dossier")** — a **global, cross-agent registry of who to
      delegate to**: one entry per agent profile with its role/specialties/strengths (seeded from
      the profile's system prompt + capability blurb, hand-editable, optionally enriched by the
      social/delegation graph above). Global to **all** agents — any agent (and the top-level
      ambassador) reads it to pick the right delegate, so routing isn't re-derived per turn. This is
      the **curated/explicit** counterpart to the *mined* social graph (graph informs it; handbook is
      the authoritative, editable source). Backs the ambassador's `list_agents`/roster awareness
      (§16.7) and ad-hoc/Alloy delegation routing with one shared lookup. Stored once (not per
      conversation); surfaced read-only to agents as a compact "who's who," editable in the UI.
- [ ] **Per-agent identity seeding** — on profile create, seed the agent's `_self_` channel
      with an identity fact/entity ("I am Mobius, id …") for stronger self-recall.
- [ ] **Debug-harness extensions** — record/replay real conversations into scenarios;
      assertion-based regression suite (golden attribution outcomes) runnable in CI when a
      provider is configured; extract the shared cluster snapshot/wipe/restore util used by
      `eval_consolidation` into a module both commands import.
- [ ] **Memory capability registry** — a code-side `@capability(...)`/registry that
      `architecture/memory-capabilities.md` is generated from or validated against, so the
      manifest can't silently drift from code (the deferred half of the drift decision).

### 16.6 Ambassador Agent (dual-presentation layer) — foundation shipped `[v0.21.32]`

> **Concept**: A customizable "ambassador" agent that runs *parallel* to a
> conversation as the **middleman of information** between the conversation and the
> user — a dedicated interpreter for large-context / complex situations that reads
> the conversation on demand **without polluting** the main transcript or the
> agent's context. Not a thin voice feature; a relay. An ambassador is a normal
> (reliable) agent profile **plus an `ambassador` section** — customized like any
> other profile.

**Shipped (foundation):**
- [x] **Per-turn briefing core**: a CC button on each assistant reply CCs the
      ambassador to brief *that turn*; a right-side **Ambassador** panel (subscribed
      to the active conversation) streams + persists the briefing.
- [x] **Parallel + non-polluting**: dedicated endpoints (`/api/agent/ambassador/*`)
      on the detached-run infra (reconnect/replay/cancel) writing to a Redis
      **sidecar** under the `ambassador:` prefix — never `conversation_logs` /
      `conv_summary:`; `start_chat_run(indexed=False)` keeps the run out of the
      conversation-recovery list. Reads conversation context `SELECT`-only.
- [x] **Profile section**: `AgentProfile.ambassador` (`AmbassadorConfig`:
      enabled/briefing_prompt/verbosity + null speech seam); global default picked
      in **Settings → Ambassador** (`ambassador.profile_id`).
- [x] **Bulletproofing**: graceful empty-provider/error degradation
      (`resolve_with_fallback`, never raises); idempotent re-CC; reload/tab-switch
      replay from the sidecar. Tests: storage round-trip, pollution regression,
      recovery-isolation, graceful degradation.
- [x] **Token-streaming + verbosity budget + cancel-settle** `[v0.21.33]`: the
      briefing now streams via `provider.stream` (per-delta `ambassador_chunk` +
      sidecar `append_chunk`); output budget scales with verbosity
      (`_VERBOSITY_TOKENS`, capped by `ambassador.max_tokens`). **Fix:** a cancel
      (GeneratorExit from `gen.aclose()`) now settles the sidecar to `cancelled`
      (preserving partial text) instead of leaving it stuck on `streaming` — no
      more perpetual "briefing…" spinner after reload. Panel refreshed with
      per-turn status chips + streaming cursor. Tests: streaming round-trip,
      cancel-settle.
- [x] **Human voice + turn-substance grounding** `[v0.21.34]`: rewrote the persona
      so the briefing speaks **to you** (second person, names the agent) instead of
      narrating "the user asked… the assistant replied." And the briefing now sees
      **what the agent actually did** — the client gathers the turn's tool calls,
      cited sources, and table/diagram exhibits (`lib/ambassadorTurn.ts::gatherTurnContext`,
      compact + capped) and posts them as `artifacts`; `_render_artifacts` weaves them
      into the prompt so it interprets the turn, not just the prose. `agent_name` +
      `artifacts` added to `/ambassador/brief-turn`. Tests: artifact-grounded prompt.
      **Name-resolution fix:** the briefed agent's name is now resolved
      (`resolveTurnAgentName`: stamped `agentName` → producing profile by `profileId`
      → conversation profile/`getAgentName()`) at both CC entry points, so restored
      turns (which lack a stamped `agentName`) still get named instead of degrading to
      "your agent". Tests: `client/src/lib/ambassadorTurn.test.ts`.
      **Thinking-truncation fix:** ambassadors think freely, so the token cap now
      budgets `_THINKING_HEADROOM` (reasoning) + a per-verbosity answer allowance,
      instead of a tight cap a thinking model (e.g. Gemini) would spend reasoning —
      which truncated the visible briefing mid-sentence. Visible length is governed
      by a firm prompt LENGTH LIMIT, not the cap; `ambassador.max_tokens` is now an
      optional hard ceiling (unset by default). `finish_reason=length` is logged.
      Tests: budget headroom + ceiling.

- [x] **Outbound relay (you → agent)** `[v0.21.36]`: relay a message into the
      conversation from the ambassador panel — a real **user turn** (or a **steer**
      into the running turn), so the ambassador stays a non-participant (the invariant
      holds: it never speaks into the transcript as itself; the user is the author).
      Client seam: `ConversationContext.{registerRelay,relayToConversation}` — ChatPanel
      registers its tab's send/steer handler; the panel got an Ask/Relay mode toggle.
      The ambassador's value-add is **drafting**: `POST /ambassador/draft` →
      `AmbassadorService.draft_relay_message` shapes a rough intent into a ready-to-send
      first-person message (ghostwriter, not speaker; degrades to the raw intent with no
      provider) which you review/edit before sending ("Refine"). Tests: draft degrade +
      provider completion. Deferred: dictation (speech → intent) feeds this same draft seam.

**Deferred (seams in place):**
> **Superseded by §16.7 (Ambassador v2):** the briefing/Q&A flow below is being
> reframed into a conversational, tool-using ambassador with its own thread — the
> "no auto-brief on open", empty-conversation, and cross-conversation gaps are tracked
> there. The items here remain accurate for the *current* one-shot implementation.
- [ ] **Activation toggle per-conversation** (today: global default + the active
      tab's context).
- [ ] **Dictation (speech → relay)**: capture continuous dictation; on stop, feed the
      captured speech as the *intent* into the existing `/ambassador/draft` → review/edit
      → relay seam (never auto-sends). File inputs remain available (reuse the input path).
- [x] **Spoken briefing (inbound) — TTS** `[v0.21.63]`: the ambassador speaks its
      briefings + Q&A aloud. `ModelProvider.synthesize_speech` (implemented by
      `OpenRouterProvider` via OpenAI-compatible `/audio/speech` → MP3; base raises);
      `AmbassadorService.synthesize` resolves the profile's `ambassador.{speech_model,
      voice,speech_speed}` block **strictly** (no chat fallback; precedence override →
      profile → `config.ambassador.*` → shipped default `microsoft/mai-voice-2`),
      degrading to a typed `SpeechUnavailable` → `422`. New `POST /ambassador/speak`
      returns raw `audio/mpeg`. Client: `lib/audio.ts::SpeechPlayer` (blob cache +
      queue + autoplay-unlock) behind `hooks/useSpeech.ts`; `AmbassadorPanel` per-item
      speaker buttons + an opt-in immersive **voice mode** (auto-speaks new briefings,
      `prefers-reduced-motion`-aware orb, Esc-to-exit). `voice_mode`/`speech_model`/`voice`
      surfaced in the ambassador profile editor's **Voice** card (speech-capable model
      picker via `ModelPickerModal requireCapability="speech"`). Tests: OpenRouter
      synth (mock httpx) + `supports_speech` cap + base-raise; `AmbassadorService.synthesize`
      precedence/degradation; `SpeechPlayer` cache/state vitest.
- [x] **Two-way voice (STT, the user-speaks half)** `[v0.21.64]`: hold-to-talk captures mic
      (Web Audio → WAV; webkit2gtk's MediaRecorder is broken), transcribes via OpenRouter
      `/audio/transcriptions`, and routes the transcript through the ambassador's intent inference
      (`/ambassador/voice-command`) → spoken answer or a reviewable relay (never auto-sent). Voice
      surface rework + intent routing landed in `[v0.21.65]` (see §16.6 vision below).
- [x] **Free-form Q&A** `[v0.21.35]`: ask the ambassador anything about the
      conversation from the panel (`POST /ambassador/ask` → `AmbassadorService.answer_question`,
      a Q&A persona/prompt over the shared `_stream_and_settle` streaming core). Persists
      under the disjoint `qa:` sidecar family (replays via `/ambassador/{conversation_id}`
      → `{briefings, qa}`); client-stable `qa_id`; grounded on a wider transcript window
      + latest-turn artifacts. Panel gained a pinned ask input + a Q&A thread. Tests:
      qa storage round-trip/isolation, answer streaming, qa prompt grounding.

#### 16.6 Voice Mode — UX vision (north star for the post-STT UI rework)

> **The bar:** Microsoft Personal Copilot's voice mode is the closest thing to a
> *perfect* immersive voice experience (OpenAI's was close too) — but both have the
> **same fatal gap: no stable push-to-talk**, and they lack **retake** and **pre-send
> confirmation**. The Ambassador's voice mode should feel like a **Discord voice call**:
> minimal, immersive, calm — and fix exactly those gaps. The current panel will need a
> **major UI rework** to reach this; the STT pass below ships the plumbing behind a
> *placeholder record button*, not this surface.

**In-panel rework shipped** `[v0.21.65]` — `components/ambassador/VoiceSurface.tsx` + the
`[Voice | Text]` tabs in `AmbassadorPanel`; backend `route_voice_command`. The items below marked
`[x]` landed in that slice; the floating CC player + recording lifecycle stay deferred.

- [x] **Immersive panel = three things only** `[v0.21.65]`: `VoiceSurface` is a **push-to-talk
      icon** (hero), **captions**, and a **settings** popover — nothing else. Voice **leads** for a
      voice-enabled ambassador (opens on the Voice tab; Text is the 2nd tab).
- [x] **Stable push-to-talk** `[v0.21.65]`: press/hold/release via mic button + Space (ignored
      while typing); pointerup/leave end a hold; **hold default + tap-toggle** setting
      (localStorage `agentx:voice:pttMode`); toggle-mode **max-duration auto-stop** + stop on
      unmount; distinct idle/listening/transcribing/thinking/speaking states. **Barge-in**: talking
      cuts off the ambassador.
- [x] **Retake confirmation** `[v0.21.65]`: a `relay` draft offers **retake** (discard + re-record)
      before sending.
- [x] **Pre-send confirmation** `[v0.21.65]`: an instruction routes to a **relay draft** shown for
      review/edit; you send (or discard). Never auto-sent. (Questions are answered spoken; a
      first-class **"relay that instead"** override recovers a misroute.)
- [x] **Captioning** `[v0.21.65]`: captions for **both** sides — your transcript + the ambassador's
      spoken line; toggleable; never audio-only.
- [x] **Voice settings popover** `[v0.21.65]`: PTT hold/toggle + captions on/off (localStorage).
      Voice/STT model + per-model **voice dropdown** live in the profile Voice card
      (`lib/voiceCatalog.ts` + `components/common/VoicePicker.tsx`).
- [x] **Voice command intent routing** `[v0.21.65]`: the ambassador **infers intent** —
      `route_voice_command` returns `{action: answer|relay, text}`; answers persist `qa:`, relays go
      through the confirm strip → `relayToConversation`. Forward-compatible `target` for future
      cross-agent delegation.
- [ ] **Headless floating CC sticky player**: CC'ing an ambassador *from a message* spawns a
      small **floating, draggable sticky button** (not the full panel) — **pause/play**, a
      small **close** button, and **keyboard PTT capture when focus isn't in an input**. A
      mini now-playing pill that rides above the conversation.
- [~] **Text mode = a second tab** `[v0.21.65]`: the typed panel is now the **Text** tab (turns,
      Q&A replay with speaker buttons, ask/relay). Playback/history/replay are there; **archival +
      recording housekeeping** still pending (see recording lifecycle below).
- [ ] **Recording lifecycle (manual, user-owned)**: recordings are **not** auto-GC'd — users
      own their own lifecycle. Surface **how much storage** recordings use and give one-tap
      **clear** options (all / by age / by conversation); **recommend clearing at a size
      threshold** (we don't mind heavy usage, we just want it visible). Auto-GC is **deferred**
      — decide the store first (IndexedDB client-side vs server sidecar) and what's even kept
      (user audio? synthesized TTS? captions/transcripts only?).
- [x] **Tauri/runtime mic permission** `[v0.21.64]`: macOS Info.plist
      `NSMicrophoneUsageDescription` + `com.apple.security.device.audio-input` entitlement,
      CSP `media-src`, **and** a Linux webkit2gtk setup hook
      (`src-tauri/src/lib.rs::enable_webview_microphone`) that flips on `enable-media-stream`
      (otherwise `navigator.mediaDevices` is absent in the packaged webview) and auto-grants the
      user-media `permission-request`. Requires an **app rebuild** to take effect (Rust change).
      Windows WebView2 prompts by default. (Verify on each packaged target.)

### 16.7 Ambassador v2 — conversational, tool-using, parallel relay — **planned**

> **Why a rework.** 16.6 shipped the ambassador as a *stateless per-turn briefer*
> hard-bound to the active conversation tab: `conversationId = activeTab.sessionId`,
> every brief/ask/voice-command is a **one-shot completion** over a read-only
> transcript snapshot. That model has three problems the user hit:
> 1. **It briefs the immediate turn on open** instead of waiting for you to ask what
>    you want — but the ambassador runs *parallel to* the conversation, not *off* its
>    latest turn. (Surfaces: the prominent "Brief the latest turn" CTA + the voice
>    tab's auto-speak of the freshest item make "open ⇒ briefed" the default feel.)
> 2. **It breaks on an empty conversation** — with no turns, the grounding context is
>    empty and the briefing/voice path has nothing to operate on.
> 3. **It can only see one conversation, one snapshot.** There's no way to ask "what
>    have my agents discovered across my long-running sessions?" — the north-star use
>    case (come back to the cluster, ask the ambassador for an application-wide
>    summary of everything in flight).
>
> **The reframe.** The ambassador becomes a **real conversational agent with its own
> persistent thread**, running parallel to (not derived from) your work, that
> **observes the conversation world through a curated read-only tool belt** and acts
> only when you ask. You talk *to it*; it decides what to look at. The load-bearing
> **no-pollution invariant is unchanged** — it still writes nothing into
> `conversation_logs`/`conv_summary:`, and its tools are SELECT-only.
>
> **The bigger picture — the ambassador is the app's top-level agent.** It sits *above*
> the work, not beside one conversation. The agent profiles (the "settings agents" —
> the workers you configure) **serve it**: it surveys what they're doing, briefs you,
> and (future) dispatches work to them. The human has **dual entry** — talk to the
> ambassador (the front door / orchestrator) *or* drop straight into a worker
> conversation through the UI; both are first-class. This is the through-line that ties
> the slices together: 16.7 builds the ambassador *up* from a per-conversation briefer
> to the orchestration layer. Read-side (survey/brief) lands here; the write-side
> (dispatch/delegate to workers) reuses the relay/`target` seam from 16.6 and grows
> into real cross-agent delegation. The no-pollution invariant still holds at every
> step — orchestration is reads + relays, never the ambassador ghost-writing a worker's
> transcript as itself.

**Design pillars**

- **Wait-for-ask, never auto-brief.** Opening the panel (text or voice) lands on a
  calm idle state — no briefing fires until you ask. "Brief the latest turn" stays as
  an *explicit* affordance; the voice tab no longer auto-speaks on open. The
  ambassador is parallel, not a reaction to your last turn.
- **The ambassador owns its own conversation.** A durable, multi-turn ambassador
  **thread** (roles `user`/`ambassador`, with tool-call records) — persisted in the
  `ambassador:` sidecar, never `conversation_logs`. Two scopes: a thread *about* a
  main conversation (default), and a free-standing **command-deck** thread that isn't
  bound to one tab (for cross-conversation queries). Replaces the disjoint
  per-turn-briefing + flat Q&A records with one coherent thread.
- **A curated, read-only ambassador tool belt** (distinct from the main agent's
  MCP/internal tools — an ambassador-private registry; nothing it calls can write a
  transcript). The agentic loop replaces the one-shot completions.
- **Voice confirms the inferred tool/intent.** "Can you summarize this?" →
  `summarize_conversation`; "explore more on this" → `explore_turn` — two different
  tool calls the ambassador names and **confirms by voice** before running, then
  speaks the result. Generalizes `route_voice_command` from `{answer|relay}` to
  `{answer|relay|tool}`.
- **Stability is a feature.** Empty conversation, no active tab, missing provider,
  tool failure — all degrade gracefully (the never-raise invariant now wraps the tool
  loop, not just a single completion).

**Slice 0 — stop auto-briefing + empty-conversation safety (fast, isolated)** — shipped (uncommitted)
- [x] **No brief on open.** Decoupled open from brief: the per-message CC button
      (`ChatPanel.handleAmbassador`) now *only opens* the panel — it no longer fires
      `ccTurn`. Briefing is an explicit action inside the panel ("Brief the latest turn"
      / per-turn) or an ask. The command-palette "Open Ambassador" was already idle; the
      voice tab already opens idle (existing `seedSpoken()` prevents replay). Button
      tooltip updated to "Open the Ambassador (parallel…)".
- [x] **Empty-conversation safety.** `_build_qa_prompt` / `_build_voice_command_prompt`
      now tell the model plainly when the transcript is empty (instead of a context-free
      prompt it could hallucinate against), so brief/ask/voice degrade to a grounded
      "nothing here yet" instead of erroring. Tests: `test_qa_prompt_handles_empty_conversation`,
      `test_answer_question_degrades_on_empty_conversation`.
- [ ] **Decouple the panel from a required `activeTab`** where the global mode will
      live (the empty-state copy already exists; make the ask/voice paths null-safe).
      → folded into Slice 1 (the ambassador thread isn't bound to a saved `sessionId`).

**Slice U — coexisting panel: slide-in, non-modal (the "stop hogging the screen" rework)**

> **The problem (touching reality).** Today the ambassador is a **modal drawer**:
> `lib/surfaces.ts::ambassador` opens `type:'drawer', size:'xxl'` (`min(1000px,100vw)`)
> inside `DrawerPanel` → `.drawer-backdrop` (`position:fixed; inset:0;
> rgba(0,0,0,.6) + blur(4px); z-index:1000`, `aria-modal`, click-to-close). So even
> though the panel is on the right, the **whole conversation is dimmed, blurred, and
> click-locked** behind it. You can't watch the agent work *and* talk to the
> ambassador — which is the entire point of a parallel relay. It must **slide in
> beside** the conversation, both panels live.
>
> **Mechanisms considered:**
> - **(A) Docked layout region (push/shrink) — recommended.** Make the ambassador a
>   first-class column in `AgentXPage` (already `ConversationSidebar | ChatPanel`): add
>   a right-side `AmbassadorDock` that, when open, *shrinks ChatPanel via flex* rather
>   than overlaying it. No backdrop, no blur, both interactive. **Reuse the
>   `ConversationSidebar` pattern** (collapsible + resizable, `agentx:conv-sidebar-*`)
>   for `agentx:ambassador-dock-{open,width}`. Default ~420–480px (not 1000 — a
>   coexisting dock leaves the conversation room), resizable 340–680. *Concern:* the
>   open-state must move out of the modal system into layout state; the entry points
>   (CC button, palette, topbar) toggle the dock via a shared store (keep `surfaces.ts`
>   as the single descriptor so they don't drift). *Concern:* narrow screens can't
>   push — fall back to the current full-screen sheet under ~900px (`useIsMobile`).
> - **(B) Non-modal floating drawer (overlay, no backdrop).** Smallest change: add a
>   `modal:false` variant to `DrawerPanel`/`ModalContext` that drops `.drawer-backdrop`
>   (no dim/blur, click-through) and `aria-modal`. The panel still floats over the
>   right edge but the conversation stays visible/interactive. *Concern:* it *covers*
>   the right slice of the conversation instead of giving it its own space; pair with a
>   narrower default width. Good interim step toward (A).
> - **(C) Split-pane with draggable divider.** (A) plus a VS-Code-style secondary-
>   sidebar drag handle — most powerful, most work. The resize mechanism already exists
>   (ConversationSidebar), so (A) naturally grows into (C).
>
> **Recommendation:** ship **(A)** — a docked, resizable, collapsible, non-modal right
> panel in `AgentXPage`, mobile-falling-back to the sheet. It's the truest "see both
> panels" and reuses an in-repo pattern (sidebar resize/collapse). (B) is the cheap
> interim if we want it *today*; (C) is (A) + a drag handle once it's docked.

- [x] **Mechanism: (A) docked push/shrink**, built *with* a resize handle (close to
      (C) — reuses the `ConversationSidebar` resize/collapse pattern). `AmbassadorDock`
      is a right column in `AgentXPage`'s flex row (after `.agentx-chat`), so opening it
      shrinks the conversation; `AmbassadorDock.css` left-edge resizer.
- [x] **Dock state + persistence**: `AmbassadorDockContext` (`agentx:ambassador-dock-{open,width}`),
      mounted under `ModalProvider`. One entry point — `useOpenAmbassador()` — used by both
      the per-message CC button (`ChatPanel`) and the palette ("Open Ambassador" now also
      navigates to chat first), so they can't drift.
- [x] **Non-modal**: no backdrop/blur, the conversation stays fully interactive; explicit
      close button (reuses `.shell-close-btn`), not Esc (the dock is persistent).
- [x] **Responsive**: docks at `≥880px`; below that (reactive media query) callers fall
      back to the existing full-screen sheet (`DrawerPanel` is already full-screen ≤600px).
      Width clamps 340–680 (default 440 — leaves the conversation room).
- [ ] **Payoff check (needs a visual pass)**: watch the agent stream on the left while
      the ambassador briefs/voices on the right; verify resize/collapse, the ≥880px↔sheet
      crossover, and mobile sheet. *(Couldn't run the app here — built + typechecked only.)*

**Slice 1 — ambassador thread (its own conversation)**

> **1a shipped (`0.21.75`): conversational memory.** The ambassador now *has its own
> conversation* — `answer_question` feeds prior settled Q&A back as real
> user/assistant dialogue turns (`AmbassadorService._thread_history`, read-only over
> the `qa:` sidecar, oldest-first, capped, in-flight turn excluded), so a follow-up
> ("what about the second one?") has context. Zero client churn — the existing Q&A
> store *is* the thread for now. Test: `test_thread_history_gives_qa_continuity`.
> **1b shipped (`0.21.82`): the `amb_thread:` thread model.** Briefings + Q&A are now
> one ordered **thread** ("Inquiry") under `amb_thread:{thread_id}` (entry-oriented — one
> record per ambassador turn carrying its optional `question`; ordered by `created_at`;
> `thread_id` defaults to the conversation id) carrying its own `title`. Decision:
> **entry-oriented, not message-split** (preserves in-place streaming + briefing
> idempotency). The pre-1b per-family public API is preserved as **thin projections** over
> the entry store, so `ambassador.py` + the existing tests barely changed; legacy
> `ambassador:` records still replay (one-place fold in `list_thread`). Tool-call chips are
> **persisted on the entry** (`set_entry_tool_calls`, inside `_agentic_answer`) so they
> survive a reload — closing the Slice-2 follow-up. The client renders one **Inquiry**
> stream (`AmbassadorContext.threadFor`; briefings as their own turns via `BriefingItem`,
> the per-turn CC trigger staying in the Turns strip) with inline rename in the switcher
> (`titleFor`/`renameThread`, empty → chat title). New endpoint
> `GET/PATCH /api/agent/ambassador/thread/{thread_id}`; the `{conversation_id}` endpoint is
> a back-compat shim. Tests: `AmbassadorStorageTest` (unify+order, tool-call persistence,
> title, legacy fold).

- [x] **Sidecar thread model** (`0.21.82`): one `amb_thread:` entry family + thread meta
      (`title`); the briefing/Q&A public API preserved as projections; legacy records fold
      in via `list_thread`. (Entry-oriented rather than a `role`/`content` message list.)
- [x] **Thread scoping** (`0.21.82`): `thread_id` is a real param, defaulting to the main
      `conversation_id`. Standalone command-deck thread ids (not bound to a tab) are
      Slice 4's to mint — the seam is in place.
- [x] **`GET/PATCH /api/agent/ambassador/thread/{thread_id}`** (`0.21.82`) replays the
      thread (`{thread_id, title, entries}`) + renames it; the `{conversation_id}` endpoint
      is now a back-compat shim.
- [x] **Client thread state** (`0.21.82`): `AmbassadorContext.threadFor` renders one ordered
      Inquiry stream (briefings + Q&A + persisted tool chips); `refresh` replays the thread.
      *(The two internal maps are kept as the source of truth with `threadFor` as a merge
      selector + projected `briefingsFor`/`qaFor` views — lower-risk than a physical state
      collapse, same UX.)*
- [ ] **Voice continuity follow-up:** the spoken router (`route_voice_command`) persists to
      the thread but still doesn't *read* prior Q&A as history — give voice the same
      continuity the typed path has.

**Slice 2 — the read-only tool belt + agentic loop** — shipped + consolidated (`0.21.76`–`78`)

> **Ungated + unified (`0.21.78`).** The tools never fired because (a) **voice had no
> tools** and (b) the answer path **pre-stuffed the whole transcript** beside the tools,
> so the model never needed them. Fixed by a consolidation onto **one streaming agentic
> core** (`_agentic_answer`): tools always on (no gate), **tool-first grounding**
> (`_LEAN_GROUNDING_TURNS` only — the model reads via tools for depth), and **voice
> answers routed through the same core** (`route_voice_command` classifies →
> `_answer_to_text`), so spoken questions get the same tools + continuity. Also DRY'd
> provider resolution (`_resolve_answerer`), collapsed the qa/tools personas into one
> capability-stating `_answer_persona` (+ `_TOOLS_NOTE`), and reuse `_stream_and_settle`
> for the degrade. Tests rewritten for the streaming core (tool fires, voice drives a
> tool, provider-rejects-tools → grounded fallback).

- [x] **Ambassador tool registry** (`agent/ambassador_tools.py`): SELECT-only,
      separate from `mcp/internal_tools`; `execute_tool` dispatch never raises.
      - `summarize_conversation(conversation_id?)` — "summarize this".
      - `explore_conversation(topic?, conversation_id?)` — "explore more on this".
      - `read_conversation(conversation_id)` — a specific session (after a survey).
      - `list_conversations(limit?)` — the cross-conversation survey primitive
        (`conversation_history.list_recent_conversations`, new read-only enumerator).
      Backed by `load_recent_turns` (read-only). Test: `test_ambassador_tools_are_read_only_and_degrade`.
      *Deferred:* `read_conversation_results` (exhibits/sources) — needs bibliography
      extraction plumbing; `explore_turn` by message-id — needs turn-id plumbing.
- [x] **Agentic turn loop** (`AmbassadorService._answer_with_tools`): bounded
      (`ambassador.max_tool_rounds`, default 4) provider tool-calling loop; emits
      `ambassador_tool_call`/`ambassador_tool_result` SSE; **never-raise** wraps the loop
      and **degrades to a grounded one-shot** on any failure (e.g. a provider that
      rejects `tools`). Client SSE pump ignores the new events (default no-op), so it's
      forward-safe. Tests: `test_answer_with_tools_executes_then_answers`,
      `test_answer_with_tools_falls_back_when_provider_rejects_tools`.
- [x] **Tools persona** (`_build_tools_persona`): the Q&A persona + a note that it has
      read-only tools and should fetch what it needs, then answer in its own voice
      (no markdown, names the agent, never reads tool output back).
- [x] **Voice answers fold through the same loop** (`0.21.78`) — spoken questions drive
      tools + continuity via the shared core.
- [x] **Client tool-call chips** (`0.21.80`) — `streamAmbassador` now surfaces
      `ambassador_tool_call`/`_result`; `AmbassadorContext` captures them onto the
      qa/briefing record (`toolCalls`); `AmbassadorPanel` renders live `ToolChips`
      (spinner → check) in Q&A + briefings (`lib/ambassadorTools.ts::toolChipLabel`).
      *Live-only* (not persisted to the sidecar — gone on reload).
- [x] **Persist tool calls to the sidecar so chips survive reload** (`0.21.82`, with Slice 1b
      thread model — `set_entry_tool_calls` on the entry).
- [ ] **Follow-ups:** surface tool activity in the *voice* path too (it answers server-side,
      no live SSE today); tool chips in `QaItem`'s avatar still use the generic mark.

**Slice 1d — parallel operator (de-couple the panel from turns)** — shipped (`0.21.83`)

> The ambassador is **an operator in its own right**, not a turn-by-turn briefer (UI kept
> *boringly stable*). The panel is **de-coupled from turns** — the Turns strip is gone. It's
> conversation-level: **"Brief this conversation"** (`briefConversation` → an `ask`) + starter
> chips + free-form ask/relay; it scopes to conversations via its read-only tools (no per-turn
> `artifacts` passed). **Per-turn = CC**: the chat's `MessageActions` button forwards a turn
> *into* the Inquiry as an `ask` ("brief me on this turn: …") — like an email into the thread —
> and opens the panel (`ChatPanel.handleAmbassador`). Header redesigned (compact command bar /
> accent-gradient **voice hero**); body **auto-scrolls** (`hooks/useStickyScroll` + jump pill);
> a **⋯ menu** (brief / rename / clear → `DELETE /thread/{id}` via `clearThread`); answers carry
> **copy** + relative timestamps. `AmbassadorConversationSwitcher` has an `inline` variant.
> **Voice mirrored (`0.21.84`, Slice 1e):** voice + text now share **one body** — the Inquiry
> stream *is* the transcript (a spoken question persists as a `qa:` entry), so only the footer
> differs (text composer ↔ `VoiceBar` = PTT mic + `voiceCommand` + relay-confirm + settings). The
> full-screen `VoiceSurface` (orb + caption log) and `lib/voiceCaptions.*` were deleted; the panel
> renders the body unconditionally and swaps the footer on `voiceActive`. **Remaining:** refine CC
> semantics (a CC'd turn as a first-class *message* that retains prior state, vs. today's framed ask).

**Slice 1c — conversations overhaul + active-conversation context** — shipped (`0.21.81`)

> The ambassador was welded to the chat tab (`conversationId = activeTab.sessionId`).
> Fixed: **independent focus** (`focusedConversationId`, "stays put", switched via
> `AmbassadorConversationSwitcher` + "current conversation" jump); **loading is pure
> display** (`locallyStreamedRef` — voice auto-speak only fires for items streamed *this
> session*, so switching/reopening never re-synthesizes TTS or speaks history — the
> cost+speech bug); **active-conversation context** (`active_conversation` {id,title} on
> ask/voice → `_active_conversation_note`, so it knows where the person is *now* even when
> focused elsewhere); **dropped the single "active agent"** (`execute_tool` param renamed
> `focused_conversation_id`; multi-agent answer persona names each agent from its own
> conversation). Relay targets the focused conversation when open; per-turn briefing stays
> gated to the active tab (needs in-memory messages).
- [ ] **Switcher over server history too** (today: open tabs only) — the full command deck.
- [~] **Nameable ambassador conversations — "Inquiries"** — **manual rename shipped (`0.21.82`)**:
      each thread carries its own `title` (`amb_thread:{id}:meta`); the switcher has inline rename
      (`titleFor`/`renameThread`), empty → chat title. **Name: an _Inquiry_** (UI noun, plural
      "Inquiries") — leans into its read-only, investigative role ("what have my agents
      discovered?") and dodges the `survey`/`search`/`research` verbs already used in code.
      **Remaining:** auto-titling (*"Inquiry · {chat title}"* mirror / *"Inquiry — {date}"* or a
      first-question summary for a **standalone** thread) is only worth it once threads can be
      standalone (a "weekly review" inquiry, a cross-conversation survey, a command-deck session)
      with no chat title to borrow — that lands with Slice 4's standalone `thread_id` minting.

**Command deck + ad-hoc delegation — roadmap (foundation laid in 1c)**
- [ ] **Agent roster awareness** — a read-only `list_agents` tool (names + capability blurb
      from each profile's system prompt) so the ambassador knows the roster. Reads from the
      global **Delegation Handbook ("Dossier")** (§16 multi-agent roadmap) once it lands.
- [ ] **Capability/strength modelling + recognize the primary agent from history.**
- [ ] **Ad-hoc delegation** — the ambassador (top-level agent) dispatches work to the right
      agent, reusing the relay `target` seam. The active-conversation context + per-conversation
      agent names from 1c are the inputs.
- [ ] **Swarm paradigm — aides gather, the ambassador stays high-level.** Today the tool belt
      reads **full transcripts into the ambassador's own context** (each read ~`_READ_TOKEN_BUDGET`,
      bounded by `_MAX_TOOL_ROUNDS`) — fine for one conversation, but a cross-conversation survey
      ("what have my agents discovered?") bloats context fast and gets expensive. Instead, the
      ambassador should **delegate to a swarm of cheap aide models**: each aide reads/condenses ONE
      conversation (or shard) read-only and returns a **high-level digest**, so the ambassador only
      ingests condensed summaries, never raw transcripts. Keeps its context lean + parallelizes the
      survey. Shape: `summarize_conversation`/`read_conversation` (and `survey_conversations`) become
      **fan-out to aide jobs** (their own model tier, e.g. `consolidation.feature_default_model`),
      results merged for the ambassador. Reuses the read-only tool belt as the aides' capability;
      ties into ad-hoc delegation (aides are the read-side, worker agents the write-side). Bound
      fan-out width + per-aide budget; never-raise per aide (one bad read doesn't sink the survey).

**Slice 3 — voice mode confirms the tool call**
- [ ] **`route_voice_command` → `{action: answer|relay|tool, ...}`.** When the spoken
      intent maps to a tool (`summarize_conversation`/`explore_turn`/…), return the
      inferred tool + args; the `VoiceSurface` confirm strip (already there for relay)
      shows a **"Summarize this conversation?" / "Explore that turn?"** confirm before
      running, then the ambassador runs the tool-loop and **speaks the result**.
- [ ] **Barge-in + retake** extend to tool confirmations (discard/redo before run).
- [ ] Captions name the action ("Summarizing…", "Exploring that turn…") so voice is
      never a silent wait.

**Slice 4 — command deck: the ambassador as orchestrator (the north-star)**
- [ ] **Standalone, top-level ambassador surface** not bound to a tab (a command-deck
      entry in the command palette / a global Ambassador surface) — the app's front
      door: "What have my agents discovered?" The worker conversations are still
      reachable directly through the UI (dual entry); this is the layer *above* them.
- [ ] **`survey_conversations`** enumerates active/recent worker sessions (over
      `/api/conversations`), pulls each one's results read-only
      (`read_conversation_results`), and the ambassador composes an **application-wide
      summary** across long-running sessions. Read-only, never-raise, degrades per
      conversation (one offline session doesn't sink the survey).
- [ ] **Dispatch seam (write-side, later).** The orchestration write-side — the
      ambassador handing a task to a worker — reuses the relay/`target` (16.6) and grows
      into real cross-agent delegation. v1 stays read + relay (you confirm); the seam is
      target-extensible so a future ambassador can start/steer a worker run directly.
      The survey above is the read-side of that same world.

**Stability & invariants (apply across all slices)**
- [ ] No-pollution regression tests extended to the thread + tool loop (nothing reaches
      `conversation_logs`/`conv_summary:`; tools are SELECT-only).
- [ ] Never-raise tests: empty conversation, no provider, tool error, no active tab —
      each degrades to a clean spoken/text notice.
- [ ] Docs: update `CLAUDE.md` (Ambassador section), `OpenApi.yaml` +
      `docs-site/.../api/endpoints.md` (new `thread`/tool SSE events), and the endpoint
      table here. Version + Release-Notes bump travels with each shippable slice.

**Open questions (decide before building the relevant slice)**
- Thread persistence depth: keep the full ambassador thread in Redis (TTL'd like
  today) vs. promote to a durable store for the command-deck history? (Recording
  lifecycle from 16.6 is the sibling decision.)
- Tool-belt surface: do tool calls show as chips in the text thread (transparency) or
  stay invisible (just the answer)? Lean transparent — they're the proof of grounding.
- Command-deck scope: all conversations, or a user-pinned working set?

### Design Notes

- `agent_id` (Docker-style, e.g., "bold-cosmic-falcon") = formal routing identifier
- `name` (e.g., "Claude", "NodeManager") = flexible display name
- `Message.name` field carries `agent_id` on assistant messages — no provider schema changes
- Extend existing `agent/chat/stream` with optional `target_agent_id` — no new endpoints
- Memory already supports this: each agent recalls from `[channel, _self_{agent_id}, _global]`

---

## Phase 17: Seamless Server Management (Complete)

> **Goal**: Solid self-hosted server deployment + client connection tools.
> Complete (17.1–17.5). Moved to [roadmap.md](docs-site/src/content/docs/roadmap.md): authoritative
> `ConfigManager` config, optional session auth (bcrypt + Redis sessions, `AGENTX_AUTH_ENABLED`),
> Docker production stack (`task prod:*`), multi-cluster deployment (`task cluster:*`), and strict
> client/API version matching (`protocol_version` + `VersionMismatchPage`). Android Tauri mobile
> shipped; rate-limiting + request-audit logging dropped in favor of the edge Gateway.

### 17.6 Open

- [ ] Cloudflare Tunnel deployment documentation (pending)

### 17.7 Release & Packaging

- [x] **Client release CI** — `.github/workflows/client-release.yml`: manual-dispatch matrix
      (Windows nsis/msi + Linux deb/AppImage/rpm) building Tauri installers + SHA-256 checksums;
      version flows through `versions.yaml`.
- [x] **Server packaging — local vs isolated clusters**: single-source compose
      (`docker-compose.yml` pulls the image; `docker-compose.build.yml` overlay builds locally;
      `docker-compose.cluster.yml` → `docker-compose.gateway.yml`); self-initializing API
      entrypoint (`docker/entrypoint.sh`) + in-image `agentx` ops CLI (`docker/agentx`);
      `.github/workflows/api-release.yml` publishes `qrmadness/agentx-api` to Docker Hub;
      `task deploy:bundle` generates the isolated deployment bundle; docs at
      `deployment/self-hosting`.
- [ ] **Image groundskeeping** — the API image is ~12 GB uncompressed (~4.4 GB compressed on
      Docker Hub). Slim it: multi-stage build, drop build toolchain/caches from the final layer,
      reconsider the bundled CUDA torch wheel (CPU-only base + optional CUDA variant), and trim
      the nvm/Node install. Heavy pull for self-hosters today.
- [ ] arm64 / multi-arch API image (amd64-only today; QEMU build deferred).
- [ ] Offline "models-baked" image variant (first run downloads ~5 GB to the HF cache volume).
- [x] **GitHub Releases Automation** — the `release` job in `.github/workflows/client-release.yml`
      drafts a `client-v{version}` GitHub Release (draft for manual publish; `-suffix` → prerelease)
      with SHA-256 checksums, supported-server notes (protocol/min-client/api version from
      `versions.yaml`), and the installers attached. Download links on `deployment/self-hosting`.
- [ ] Shared-infra local clusters (one DB stack, namespaced) — deferred per the isolation-axis design.
- [ ] **Cloudflare gateway for isolated clusters** — the gateway overlay (Nginx + cloudflared tunnel,
      `docker-compose.gateway.yml` + `clusters/template/cloudflared/config.yml.example`) is wired only
      for **local** clusters (`cluster:up` includes it when `nginx.conf` exists; documented in
      `deployment/clusters`). The **isolated** bundle (`task deploy:bundle` → ships only
      `docker-compose.yml` + `docker-compose.gpu.yml`) has **no** gateway/cloudflared option, and
      `deploy/.env.example`/README never mention public exposure — yet isolated is the production path.
      Decide: ship a `docker-compose.gateway.yml` + a `cloudflared/config.yml.example` in the bundle
      (image-based, no `build.yml` overlay) and document `AGENTX_PUBLIC_HOST`/`AGENTX_GATEWAY_TOKEN`
      in `deployment/self-hosting`, or explicitly state isolated = bring-your-own-reverse-proxy.
- [ ] **cloudflared SSE/streaming timeout** — in `clusters/template/cloudflared/config.yml.example`
      the comment credits `noHappyEyeballs: true` with holding streaming (chat SSE) connections open
      past ~100s, but that flag is IPv4/IPv6 connection racing, not a response/idle timeout. Verify
      long SSE streams actually survive the tunnel + nginx; if they get cut, set the real knobs
      (nginx `proxy_read_timeout`/buffering off for SSE; confirm cloudflared has no hard cap) and fix
      the misleading comment.
- [ ] **Isolated-deploy doc accuracy** (from the v0.21.31 isolated-cluster smoke test):
      - `deploy/README.md` + `deploy/dockerhub-overview.md` say first boot downloads "~5 GB" of
        models; observed it's really ~2.3 GB (the `BAAI/bge-m3` embedding model) — translation
        (`NLLB`) is **lazy** (`/api/health` showed `translation: not_loaded`). Correct the figure.
      - Add a note that **Docker Desktop** only bind-mounts host-shared paths, so the bundle must be
        unpacked under a shared dir (e.g. `$HOME`, not `/tmp`) or `compose up` fails with
        "mounts denied". (Native docker engine is unaffected.)

---

## Phase 18: UX Improvements & Optimization and Memory Tuning (In Progress, ~90%)

> Polish the client and tune the memory pipeline. Shipped waves moved to
> [roadmap.md](docs-site/src/content/docs/roadmap.md):
> **18.1** Wave 1 fixes (provider settings, mobile topbar) · **18.2** Toolkit (MCP server CRUD +
> tool browser, tags/groups/`allowed_agent_ids`, per-agent `allowed_tools`/`blocked_tools`) ·
> **18.3** Relay module (background-run inbox, "No Memorization" toggle) · **18.4** model metadata +
> `ModelPickerModal` (OpenRouter/Vercel capabilities + pricing) · **18.5** per-tab context bar +
> per-turn cost chip · **18.6** extraction tuning (entity resolution, `refines_fact_id` supersedure,
> scope context, `eval_consolidation` harness) · **18.8** Wave 2 fixes (KaTeX, table HTML, plan-step
> restore, editable cached servers, MCP auto-connect) · **18.9** memory tuning (`recall_user_history`,
> token-budget header, `checkpoint` tool + badge UI) · **18.10** plan/streaming reliability (token
> clamp, Plans drawer + step annotation, detached chat runs) · **18.11** client error contract +
> foundation cleanup (`ApiError`/toasts/`useApi`, Tailwind v4 + `ui/` primitives, god-component /
> `lib/api` / `ConversationContext` splits) · **18.11.x** cancel-CSRF + gate-page chrome fixes ·
> **18.12** Wave 3 entry-surface UX (Start recents, renamable conversations, selector redesigns,
> splash, README trim).

### 18.x Shipped — moved to [roadmap.md](docs-site/src/content/docs/roadmap.md)

> Dashboard redesign + usage metrics (`[v0.21.9]`/`[v0.21.10]`), extraction eval-harness + cleanup
> (18.6: procedural cases, snapshot/restore, persisted eval runs, `dedupe_entities`), working-memory
> follow-ups (`scratchpad_note`, `forget`/`remember_this`/provenance, cached recap), and the
> per-profile internal-tool gating UI (18.9.x). All shipped — see roadmap.

### 18.13 Layered Prompt Composer ("Prompt Stack") — ✅ Complete `[v0.21.47]`

> Replace the single-base-prompt model with a durable, block-based stack of editable
> prompt **layers**. Each built-in layer ships a `default` (sidecar, owned by the app,
> updated by releases) + an optional user `override`; effective = override ?? default,
> so untouched layers keep getting release improvements while edits are pinned and
> never silently overwritten. Locked decisions: **global stack only** (per-agent prompt
> stays in the profile editor, read-only in preview), **debounced autosave**, **no named
> presets in v1** (retire legacy prompt-profiles into the stack). Fixes the original
> durability bug — the old global/sections edits were in-memory only and lost on restart.

- [x] **1a — model + store** `[v0.21.43]`: `PromptLayer` (default/override/`default_version`/
      `base_version`; `effective`/`modified`/`update_available`); `BUILTIN_LAYERS` (global
      prompt split into versioned blocks); `LayerStore` durable via `ConfigManager`
      (`prompts.layers`, write-through) — override/reset/acknowledge/enable/reorder + custom
      CRUD + `compose()`. Tests: precedence, override persists, default-change → update/ack/
      reset, custom CRUD/reorder/compose.
- [x] **1b — wire + migrate** `[v0.21.44]`: `PromptManager.compose_prompt` now sources the
      global content from `LayerStore.compose()` (lights up all 3 live sites — `core.py`,
      streaming `views.py`, `alloy/executor.py` — unchanged). Folded the default "General"
      profile's sections (structured-thinking/concise-output/safety-constraints) into
      `BUILTIN_LAYERS` so the stack is the *complete* default prompt (byte-parity), and dropped
      the default-profile auto-attach (`is_default` guard) to avoid double-injection. One-time
      legacy migration (`_ensure_layers_migrated`, guarded by `prompts.layers_migrated`) imports
      any customized legacy global into a reserved `legacy-global` custom layer. `/prompts/global*`
      kept as back-compat shims over the store (`set_singleton_override` upsert). Scope: governs
      only the conversational prompt — `SystemPromptLoader` feature prompts untouched. Tests:
      parity, no-duplication, override→live, shim, singleton idempotency.
- [x] **2 — layer API** `[v0.21.45]`: `GET/POST /api/prompts/layers` (list `{layers, composed}`
      / create custom), `PATCH/DELETE /api/prompts/layers/{id}` (content→override, title,
      enabled / delete custom), `POST /{id}/reset`, `POST /{id}/acknowledge`, `POST
      /layers/reorder`. Typed client (`promptsApi.{list,create,update,delete,reset,acknowledge,
      reorder}PromptLayer(s)` + `PromptLayer` type). Docs: CLAUDE.md table, OpenApi (`PromptLayer`
      schema + paths, spec lints clean), endpoints.md. Tests: `PromptLayerApiTest` (list/create/
      patch/delete/reset/reorder/404 via RequestFactory). Compose-preview enrichment (dynamic
      injections + active agent) deferred to the editor (Phase 3).
- [x] **3 — block-stack editor UI** `[v0.21.46]` (Settings → Intelligence → **System Prompt**):
      side-by-side two-pane composer — draggable layer cards (`@dnd-kit`, collapse via
      framer-motion, Built-in/Custom badge, enable Switch, ● edited / ▲ update dots), inline
      `Textarea` edit w/ 600ms debounced autosave + ~token count, **reset-to-default**, **diff
      modal** (`diff` lib; update: Keep[ack] / Adopt[reset] / Load-default merge-assist; edited:
      Reset), live composed preview (client `composeStack` mirrors backend, unit-tested) + custom
      layer add/delete. Narrow widths collapse to one column + Preview dialog. New: `SystemPromptSection`
      + `prompt-stack/{LayerCard,ComposedPreview,LayerDiffModal}` + `PromptStack.css` +
      `lib/promptStack.ts`(+test). Libs added: `@dnd-kit/*`, `diff`.
- [x] **4 — snippet library** `[v0.21.47]`: reframed `template_manager` as the **Prompt Library** —
      snippets **insert-as-layer** (reuse `PromptLibraryModal` `mode='insert'` → `createPromptLayer`)
      and the enhancer rewrites a layer **in place** (`Enhance` button → `/api/prompts/enhance`,
      one-click undo). Extended `onInsert(content, name?)` to seed the layer title. Dropped the
      misleading "replaces PromptProfile/PromptSection" docstring (it's a building-block library,
      not a replacement — composition is owned by `LayerStore`).

### 18.14 Prompt System Round 2 + Ambassador-as-profile-kind — ✅ Complete `[v0.21.48–54]`
- [x] **Dialog z-index fix** `[v0.21.48]`: `ui/Dialog` raised above the legacy app modals
      (z-1000/1001) → "Insert from library"/diff/preview popups no longer render behind Settings.
- [x] **Per-agent prompt editor** `[v0.21.49]`: reusable `common/PromptEditor` (tokens + in-place
      Enhance/undo + library-insert-replaces) + `common/EffectivePromptPreview` (name → agent prompt
      → global stack, true backend order).
- [x] **Ambassador is its own profile `kind`** `[v0.21.50 backend / .51 client]`: `AgentProfile.kind`
      (`agent`|`ambassador`) + `is_default_ambassador`; separate defaults (`get/set_default_ambassador`,
      agent-default never an ambassador); engine-level chat exclusion (default-agent/routing/delegation);
      `AmbassadorConfig` persona overrides (briefing/qa/draft) over code defaults + `system_prompt` =
      Communications voice; migration seeds a default ambassador, never converts the default agent.
      Endpoint `set-default-ambassador` + `ambassador/persona-defaults`. Editor adapts by kind
      (`OverridablePromptField` default/override/reset/diff); Settings → Ambassador slimmed to a
      default-ambassador picker + New/Edit.
- [x] **Diff coloring everywhere + memory-prompt diff** `[v0.21.52]`: moved `.layer-diff` CSS into
      `LayerDiffModal.css` on `--feedback-success/-error` tokens; memory extraction/relevance
      `PromptField` gains a Diff button (default vs override).
- [x] **More avatar icons** `[v0.21.53]`: curated set 11 → ~57 (static/tree-shaken).
- [x] **Autosave** `[v0.21.54]`: agent profiles + memory recall/consolidation settings autosave
      (debounced, baseline-diff = no save-loops, hydration keyed on `profile.id` = preserves cursor).

### 18.15 Prompt/Ambassador follow-ups (observed while in the code) — open
- [ ] **Finish moving ambassador settings onto the profile**: `ambassador.model` +
      `max_context_turns` still live in global `config.ambassador.*` (split-brain). Move per-ambassador
      knobs onto the ambassador profile (the user's "ambassadors will have lots of settings" intent).
- [ ] **Retire the vestigial PromptProfile/PromptSection path**: the "Select Base Template"
      (`prompt_profile_id`) control + legacy `/prompts/profiles*`/`/sections*` are now confusing
      alongside the layer stack + library-insert. Plan a deprecation/removal once nothing depends on it.
- [ ] **Client tests for the new logic**: autosave baseline-diff, `EffectivePromptPreview` compose
      order, `OverridablePromptField` reset/diff — only `promptStack` has unit coverage today.
- [x] ~~Theme leak straggler (`.profile-avatar-option.selected` → `--cosmic-violet`)~~ — **not a leak.**
      `--cosmic-violet` is a theme-adaptive *alias* (defined per theme = the accent), so it follows
      themes correctly. Optional cosmetic cleanup: rename the legacy alias to `--accent-primary`
      app-wide (15+ files) — zero visual change, low priority.
- [x] **Prompt placeholders** `[v0.21.55]`: whitelist `{agent_name}`/`{date}`/`{time}` substituted
      at compose time (`prompts/placeholders.py`, applied in `compose_system_prompt` + the ambassador
      persona builders — so an override's `{agent_name}` now resolves). Client: `PromptEditor` "Insert
      placeholder" menu (`lib/promptPlaceholders.ts`) + preview highlighting (`HighlightedPrompt`).
      Tests: `PromptPlaceholderTest`. Also raised `DropdownMenu` z-index above the legacy modals
      (same class of bug as the Dialog fix). **Follow-up:** add the insert-placeholder affordance to
      the global `LayerCard` editor too (substitution + highlight already work there).
- [ ] **`SKILL.md` standardization (parked, user-flagged)**: adopt the standard skills-file pattern;
      let the prompt/layer machinery host composable skills; guard so profile-typing/layering never
      gates a user's skill access. Its own initiative.

### 18.16 Agent-profile control-center + icon picker — ✅ Complete `[v0.21.56]`
- [x] **Redesigned editor**: hero identity header (AvatarPicker tile + `agentAccent` aura, inline
      name, `CopyChip` agent-id, kind/default badges, tags, description) over a `ControlCard` grid
      per tab (Model / Generation / System Prompt-full / Delegation|Ambassador / Tools / Memory),
      each with an at-a-glance header summary. All field logic + `useProfileEditorState` autosave
      preserved (presentation-only re-house).
- [x] **Consolidated icon picker** (`common/AvatarPicker`): searchable, categorized modal (recents
      + live preview + stagger), with a disabled **Generate** seam for AI icons next. Catalog grown
      to ~95 icons with `category`/`keywords` (`lib/avatars.ts`).
- [x] **New reusable primitives**: `ui/SegmentedControl` (reasoning + picker tabs; sliding motion,
      a11y radiogroup), `ui/CopyChip`, `common/ControlCard`, `lib/agentAccent.ts` (deterministic
      per-agent color — foundation for chat/Alloy identity later). Gradient temperature slider +
      dynamic label. Tests: `agentAccent`, `avatars`. Reduced-motion respected.
- [ ] **Follow-up**: generated icons (the Generate tab); a light `ProfileNav` polish (apply the
      agent accent to the active item); retire now-dead `.profile-section-card`/accordion CSS.

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

### ▶ FOUNDATION — real next-session priority order (barring the genome/advisor/evolution meta-layer)

> The "fancy" meta-layer (Agent Genome, Settings Advisor, evolution) is captured below but gated on
> foundation. Do these first; they're user-facing, correctness, or reliability — not strategy.

1. ~~**Chat legibility slice**~~ — **shipped.** (a) ~~compact collapsible tool-call rendering~~ — slim
   one-row card (tool-type icon + name + key arg/query + folded `· N results · 1.2s` meta; status =
   colored left border; footer/output relocated into the expanded view), `ToolExecutionBlock` so every
   inheritor (tool_call/tool_result bubbles, delegation nested tools, checkpoints) gets it. (b)
   ~~web-search query inline~~ — `web_search`/`web_research` show the quoted `query` + result count on
   the collapsed row. (c) ~~per-phase SSE `status` events~~ — coarse feed on the run-bus
   `emit_status()` seam (see the Observability cluster; deep sub-phases now a drop-in).
2. ~~**Live steering — message interruption / queue**~~ — **v1 shipped (queue mode).** A steer
   (`POST /agent/chat/runs/{id}/steer`) is pushed to a per-run Redis queue (`chat_run.push_steer`),
   drained by `streaming_tool_loop` (`streaming/steering.py`, run resolved via the `current_run_id`
   contextvar) at the tool-result boundary **and** at the would-end (continues the turn instead of
   ending), folded in as a fresh user turn. Echoed as a `steer` bus event so all clients show the
   bubble inline; client composer stays live (Stop **+** Steer). **Follow-ups:** hard interrupt (abort
   the in-flight provider stream) + persisting the steer as a real `conversation_logs` turn. See
   cluster below.
3. **Stable memory core** — kill transient memory injection (`remember(query=message)` re-ranks every
   turn); inject a stable high-salience core + recall as a supplement. Correctness; rides the Slice-6
   `assemble_turn_context` preamble budget.
4. **Finish the reliability guarantees** — extend the Slice-5 model fallback to the remaining feature
   sites (reasoning/drafting/`planner`/`alloy`, still raw `get_provider_for_model`); **hydrate the
   Alloy + background-chat paths** (Slice-6 follow-up) so multi-agent/queued chats also resume warm.
5. **Cost + gaps** — **per-turn search credit budget** (Tavily spend), **configure the global default
   model** (UI gap), and the **full persisted tool outputs** debugging surface (heavier backend).
6. **Tech-debt sweep** — consolidate the 4 token estimators (→ `tiktoken`), retire dead context knobs
   (`auto_summarize_at`/`max_messages`/stale `ContextConfig`/superseded `prepare_context`).

> ⭐ **Major missing capability — File Workspaces & Document RAG** (see section below). Slots near the
> top once the chat-legibility slice lands: today agents can search learned *memory* + the *web* but
> can't be handed a file/codebase/folder. Bigger build than 1–5 but capability-defining, and mostly
> *reuse* — schedule it as its own slice.
>
> The **Settings Manifest** (keystone) is the bridge: foundational *and* the precondition for the
> meta-layer — reasonable once the above land, since it directly cleans up settings + validation.

### ⭐ Workspaces & Document RAG (major foundation gap)

> Confirmed absent: no upload endpoint, no file/document store, no workspace, no ingestion. Agents can
> RAG over their *learned memory* (Neo4j + pgvector) and the *web* (Tavily), but a user can't hand them
> a PDF / codebase / folder of docs. **Retrieval is trivial on our stack** (reuse chunker + embeddings
> + pgvector) — the real design is the **pattern**: a persistent *workspace* with a searchable
> **manifest**, and conversation→workspace tagging that **injects the file list** so the agent is
> *aware* of its corpus (a mini data-warehouse) rather than blindly semantic-searching a blob.

- [ ] **Workspace as a first-class entity** — a named, persistent container of files + metadata (CRUD,
      like agent profiles), **not** per-conversation. A **conversation is tagged to a workspace** (a
      field on the conversation/session); attach/switch is a UI action. (A workspace can later be shared
      by an Alloy team as a common knowledge base.)
- [ ] **Workspace manifest (the catalog / "data-warehouse index")** — per file `{filename, type, size,
      auto-generated tags, short summary}`, kept queryable. Retrieval is **two-tier**: (1) **manifest
      search** by filename/tag/summary → the right *file*; (2) **semantic chunk search** → the right
      *passage*. Mirrors the shipped `tool_output_section` (list → fetch) + `tool_output_query`
      (semantic) pattern, just persisted.
- [ ] **Manifest injected into context (stably)** — the tagged workspace's **file list** rides the
      Slice-6 `assemble_turn_context` preamble as a **stable** system block (names + tags only, bounded;
      aligns with the "stable core, minimal transient" principle), so the agent always knows *what it
      has* before retrieving. This awareness is what makes it a workspace, not just a vector store.
- [ ] **Ingestion (reuse, don't rebuild)** — parse (pdf/text/md/code) → **chunk**
      (`agent/tool_output_chunker.py`) → **auto-tag + summarize** (reuse the extraction/LLM infra) →
      **embed** (`kit/agent_memory/embedding_queue.py` + provider) → **pgvector** (`document_chunks`) +
      a manifest row. Upload endpoint (multipart) + durable file store + composer drop-zone + Workspace drawer.
- [ ] **Retrieval tools + citations** — `workspace_search` (manifest: name/tag), `document_query`
      (semantic chunks), `read_document` (paginated) — registered like the existing stored-output tools;
      hits auto-capture a `citation` exhibit (`source_type: "doc"`) → the conversation **Bibliography**.
- [ ] **Storage backend + quota (Docker)** — **three-store separation** (don't put bytes in Postgres):
      **(1) bytes → a blob store**, content-addressed by **sha256** (free dedup + integrity) — local
      disk at `${AGENTX_DB_DIR:-./data}/workspaces/{workspace_id}/{sha256}` for dev (matches the
      Neo4j/PG/Redis bind-mount pattern), swappable for a **MinIO (S3-compatible) container in
      `docker-compose`** on the production / multi-cluster path (Phase 17 `prod:*`/`cluster:*`).
      **(2) manifest/metadata → Postgres** (`workspaces` + `documents`: `filename, content_type,
      size_bytes, sha256, storage_key, tags[], summary, status`). **(3) vectors → Postgres + pgvector**
      (`document_chunks`, `vector(N)` + HNSW, mirroring `init_memory_schema`'s `fact_embeddings`). Join
      key = `document.id`/`storage_key`. **Per-workspace + per-user quotas** enforced at upload
      (`SUM(size_bytes)` vs a configurable byte budget; reject + notify), plus per-file size/type
      allow-lists. Wire into `task db:init` (create the dir) + `db:status`.
- [ ] *(later)* code-aware/AST chunking, folder/repo ingestion, `web_crawl` → workspace (crawl a site
      *into* a workspace), cross-workspace search.
- [ ] *(future)* **Content-addressed store doubles as snapshots / restore points** — because blobs are
      keyed by sha256 (Git-like), a "snapshot" is just a manifest of hashes and **incremental snapshots
      are cheap** (only changed blobs written). Generalizes today's bespoke snapshot/restore
      (`eval_consolidation --snapshot`, memory export/import) into one uniform primitive across **config
      + memory graph + workspaces + genomes**. This is the **safety net for the meta-layer**: it's what
      makes the autonomy envelope / evolution *reversible* (the parents can experiment because a bad
      mutation rolls back). Ties [[autonomy-envelope]] ↔ restore.

### Chat UX & Tool-Call Rendering (density + observability)

> Tool calls — and everything that inherits the tool-call block (checkpoints, exhibit fallbacks,
> web-search cards) — dominate the transcript and hide what matters. Make the chat readable.

- [x] **Compact tool-call rendering** — shipped. `ToolExecutionBlock` is now a slim one-row card by
      default (tool-type lead icon + name + key arg/query + folded `· N results · 1.2s` meta; status =
      colored left border; Call ID / args JSON / result footer + View Output relocated into the
      expanded view). Every inheritor (tool_call/tool_result bubbles, delegation nested tools,
      checkpoints, `recall_user_history`) gets it.
- [x] **Web-search shows its query inline** — shipped. `web_search`/`web_research` surface the quoted
      `query` + parsed result count on the collapsed row (`ToolExecutionBlock` `primaryPreview` +
      `resultCount`). The auto-captured Sources/citation exhibit stays a separate row (linking them is
      a follow-up).
- [ ] **Full tool-call outputs (persisted)** — only a small slice of a tool result is shown today,
      but full outputs are valuable for debugging agent thinking. Persist complete outputs
      (PostgreSQL or similar) and let the UI expand to the whole thing (lazy-loaded), beyond the
      streamed/truncated preview.

### Backend Observability — live operation status over SSE

- [x] **Per-phase status events (coarse)** — shipped. A typed `status` event
      (`{phase, label, detail?, group?, progress?}`) gives the chat a live activity line
      (`recalling` → `composing` → `thinking` → `running_tool` → `reading`) instead of a silent
      "thinking". **Key realization:** the chat path doesn't go through `Agent.run` — `generate_sse`
      inlines the phases and the live client *tails the run's Redis event bus* (`chat_run`), not the
      generator. So all status routes through one ambient `emit_status()` (`streaming/status.py`, run
      resolved from a `ContextVar` set in `chat_run._drive_run`) that appends straight to the bus —
      replays on re-attach for free, throttled/coalesced. Emit points: `views.py::generate_sse`
      (recalling/composing) + `streaming/tool_loop.py` (thinking/running_tool/reading, shared with plan
      exec). Client: `onStatus` → `streamReducer` `activity` → the `ChatPanel` spinner line.
- [ ] **Per-phase status events (deep sub-phases)** — the deferred fine grain: `embedding` /
      `reranking` inside `remember()`/`RecallLayer`, `reasoning_step N` inside the reasoner. Now a
      **drop-in** `emit_status("embedding", …)` — phases are reserved in `STATUS_PHASES` and the
      `{detail, group, progress}` contract fields + client `activity` shape already carry them. Only
      real work: the **embedding daemon thread** can't see the `ContextVar`, so its queued job must
      carry `run_id` → `emit_status(…, run_id=job.run_id)` (the explicit arg already exists). Rides the
      **same tool-loop boundary as Live Steering** (below); build the boundary once.
- [ ] **Context Inspector ("what's in the model's head this turn")** *(my idea, from the Slice-6
      context work)* — now that `assemble_turn_context` builds one well-defined message list, expose it:
      a per-turn debug view showing exactly what was sent to the model (system preamble blocks:
      checkpoints / scratchpad / summary / memory; the verbatim transcript that fit; the new turn) with
      **per-block token counts** and the budget breakdown (verbatim vs reserved vs window). Pairs with
      the per-tab context bar + the "full tool outputs" item — the single best lens for debugging agent
      behavior. Cheap to surface (the assembler already has all of it); gate behind a dev/inspect toggle.

### Live Steering — message interruption & queue (steer a running agent)

> Today a turn is fire-and-forget: once it starts you can only let it finish or **hard-cancel** it
> (`/runs/{run_id}/cancel`). You can't say "wait, also check X" or "stop — you're off track, do Y
> instead" without throwing away the whole run. Steering mid-run is essential for long/agentic turns,
> and it's the most-forgotten gap. Foundation #2 — this is the design cluster for it.

- [x] **Inject-into-running-turn** — shipped. `POST /api/agent/chat/runs/{run_id}/steer` (`{message,
      mode}`, owner-only) pushes to a per-run **steer queue** (`chat_run.push_steer` →
      `chat_run:{run_id}:queue`) **and** echoes a `steer` bus event so all clients render the bubble.
- [x] **Drain at safe boundaries** — shipped. `streaming_tool_loop` drains (`streaming/steering.py`,
      run resolved via the `current_run_id` contextvar) at the **tool-result boundary** *and* at the
      **would-end** (folds the answer-so-far + steer, then `continue`s instead of ending), so the agent
      re-plans mid-trajectory or keeps going after a steer.
- [ ] **Two modes** — only **queue** (fold at the next safe boundary) shipped; the `mode` field is
      carried but **interrupt** (abort the in-flight provider stream / tool wait + re-prompt) is a
      follow-up.
- [x] **Client** — shipped. Composer stays **live during streaming** (`ChatPanel` shows Stop **+**
      Steer; Enter routes to `stream.steer`); the `steer` event appends a `steered` user bubble via
      `useChatStream.onSteer` (flush-then-append, dedupe by id) so live + re-attached clients match.
- [x] **Persist the steer as a real turn** — shipped. Folded steers are captured on
      `ToolLoopResult.steers` and persisted as `user` turns (`metadata.steered` + `steer_round`/
      `after_tools`/`phase` — a procedural-memory signal); restored on reload (`mapServerMessages`
      `steered`). Turn-shaping extracted to pure builders in `streaming/persistence.py`.
- [x] **Hard-stop persists the partial turn** — shipped. A Stop (`GeneratorExit` in `generate_sse`)
      saves progress up to the stop (user + completed tools + steers + partial assistant text,
      `metadata.interrupted`) via the same `_persist_turns` orchestrator; the assistant bubble restores
      with a "stopped" tag. Detach/tab-close still persists normally (run plays on). **Follow-ups:**
      procedural *consumption* (consolidation mining `metadata.steered`), plan-execution-path steer +
      partial capture, background chat-queue jobs, richer `tool_call_id` linkage.
- [ ] **Shares plumbing** — the same boundary still wants to power **Blocking tool-call approval** +
      the in-run **Exhibit `choice`** round-trip (see Future Enhancements). The drain boundary +
      `current_run_id` contextvar are now in place to build on.

### Conversation Context & Checkpoints

- [x] **Include prior conversation context every turn (near-verbatim)** — shipped `[v0.21.30]`. The
      in-memory `SessionManager` is now **rehydrated** from the durable `conversation_logs` transcript
      on a cold session (`agent/conversation_history.py`, before the new turn), so resumed/restored
      conversations keep their history. Per-turn context is assembled by
      `ContextManager.assemble_turn_context` — SYSTEM preamble + recent **verbatim** transcript up to
      `context.verbatim_budget_ratio` (0.7) of the model's real window, oldest overflow covered by the
      rolling summary. The memory recall's old current-conversation turn-dump (a band-aid) is dropped
      to avoid double-injection. Tests: `ConversationContextTest`.
- [x] **Context-window-based summary/compression triggering** — shipped `[v0.21.30]`. The rolling
      summary (what fired "early" on a fixed message count) is now **token-triggered**:
      `SessionManager.maybe_update_summary` summarizes aged-out turns only when the verbatim transcript
      crosses `verbatim_budget_ratio` of the window (keeping a `recent_floor`), and the summary is
      **persisted** in Redis so it survives a cold rebuild. (The model-authored `checkpoint` tool has
      no auto-trigger — also hardened: anchor-preserving eviction + a `replace` mode.)
- [ ] **Redis/Postgres-backed live session store** — rehydrate-from-logs (shipped) re-reads the DB on
      a cold session; a durable session store would survive restarts without the per-turn read and
      across workers.
- [ ] **Rolling summary as a first-class `conversations` column** (vs. the current Redis TTL) for
      durability beyond 30 days.
- [ ] **Hydrate the Alloy / background-chat paths** too — this slice rehydrates the main streaming
      chat; the multi-agent + queued-chat paths build their own context.
- [ ] **Stable memory core (kill transient memory injection)** — today the injected memory is
      **transient**: `views.py` calls `agent.memory.remember(query=message)` **every turn**, so the
      facts/entities re-rank against the current message and shift turn-to-turn (the agent "sees" a
      fact one turn, not the next). Inject a **stable, high-salience core** (durable facts/entities for
      this user/channel) as a persistent preamble block consistent every turn, with query-specific
      recall as a small *supplement* on top. Goal: minimal transient context. Slots into the same
      `assemble_turn_context` SYSTEM-preamble budget (and is exactly what the Context Inspector would
      surface).

### Memory Area UX Cleanup

- [~] **Redesign the Memory area** — Memory moved from a cramped right-side `xxl` drawer to a
      **full-screen modal** (`type:'modal', size:'full'`, mirroring Settings/Toolkit; own
      `.memory-modal-content` host with a definite height so the panel fills the dialog and scrolls
      internally). Also fixed the procedure "When when …" render doubling (shared `procedureHeadline`/
      `_prefix_trigger` helpers, render-only). **Still TODO:** the *drastic cleanup* +
      **document every feature in-UI** (per-control abstract descriptions) — deferred from this pass.
- [ ] **Memory-mending agent (memory janitor)** — an agent that actively explores the memory graph
      and *repairs* it: find orphaned facts (no `[:ABOUT]`), broken/dangling links, duplicate or
      contradictory entities/facts, stale context, and weakly-connected clusters; propose/apply mends
      (link, merge, supersede, prune). Build on the new manual fact↔entity link + the existing
      lifecycle ops (`dedupe_entities`, `link_facts_to_entities`, `check_contradictions`,
      `promote_to_global`) and the Fact→Entity surfacing. Likely an **Agent Alloy specialist**
      ("Memory" agent) so it reuses delegation + can run on a schedule; surface proposed mends in the
      Memory explorer for review/approve. (Requested follow-up to the Memory Explorer pass.)

### Engineering Hardening (observed while in the code, Slices 5–6)

> Grounded tech-debt / consistency items noticed during the model-fallback + context work.

- [ ] **Type the plan executor's subtask status (kill stringly-typed sentinels)** — subtask state is
      encoded as magic prefixes on the *result* string (`[FAILED: …]` / `[SKIPPED: …]` / `[ABANDONED: …]`)
      and re-parsed by `str.startswith` in **three** places (`plan_executor.py` `_build_synthesis_messages`,
      `_handle_failure`, `_completed_count`, and `views.py::_subtask_status`). Replace with a real
      `status` enum/field on `Subtask` (keep the result string for the human-readable reason). Cross-file;
      own pass. *(Filed from the PlanExecutor cleanup that did #1 typed `PlanResult` + #2 sync safety-net
      parity.)*
- [ ] **Unify the plan executor's sync/async engines (optional)** — `execute()` and `execute_streaming()`
      duplicate the subtask loop (select → run → mark/handle-failure → safety-net → synthesize). Parity is
      now restored (safety net mirrored), but the two skeletons can still drift. Sharing the loop body is
      awkward across sync/async; revisit only if `Agent.run` (the sync path) grows.
- [ ] **Extend the universal model fallback to the remaining feature sites** — Slice 5 wired
      `resolve_with_fallback` into memory/recall/recap/compression, but **reasoning** (CoT/ToT/ReAct/
      Reflection), **drafting** (speculative/pipeline/candidate), `agent/planner.py`, and
      `alloy/executor.py` still call `registry.get_provider_for_model` directly, so a missing/unreachable
      model there can still hard-fail those features. Route them through `resolve_with_fallback`
      (passing the agent model as `preferred_fallback`) for the same "never crash the turn" guarantee.
- [ ] **Consolidate token estimation (4 copies)** — `estimate_tokens` now exists in
      `streaming/helpers.py`, `agent/context.py`, `agent/session.py`, and `agent/conversation_history.py`,
      all the same rough `len/4`. Unify into one shared util — and consider using **`tiktoken`** (already
      pulled in transitively by `tavily-python`) for accurate counts, which would tighten the new
      context budget.
- [ ] **Retire dead/legacy context knobs** — now that assembly is token-based: `Session.auto_summarize_at`
      has a dead `pass` branch, `Session.max_messages` is a vestigial count cap, `ContextConfig` defaults
      are stale (`summary_model="gpt-3.5-turbo"`, unused `tokens_per_message_estimate`), and the old
      `ContextManager.prepare_context` is superseded by `assemble_turn_context`. Prune them and make the
      budget-header nudge reference the configurable `context.verbatim_budget_ratio` (it hardcodes "70%").
- [ ] **Proactive provider-health refresh for the fallback path** — `registry._provider_health` (used to
      skip a known-down provider) is only populated when something calls `/api/providers/health` (the
      dashboard poll). A small periodic background refresh would make the "unreachable" fallback tier
      proactive instead of only learning from a failed call.
- [ ] **Decouple transcript persistence from memory extraction (optional)** — "No Memorization"
      conversations persist **nothing** to `conversation_logs`, so they can't be rehydrated or browsed
      after a restart. A transcript-only durable record (independent of memory *extraction*) would let
      them survive a cold session while still honoring "don't learn from this." Weigh against the
      toggle's intent (some users may want zero persistence).

- [x] **Bulletproof fact→entity linking** — root cause of facts not showing under their entities was a
      silent name-resolution gap in consolidation: facts linked entities only via an exact batch-map
      lookup, dropping cross-batch / alias / variant names with no log. Fixed with
      `_resolve_fact_entity_ids` (batch map → `find_entity_by_name_or_alias` → auto-create stub entity)
      wired into both the user and self fact-storage paths, plus `fact_entity_links_recovered` /
      `fact_entity_stubs_created` metrics and a `link_autocreate_stub_entities` flag. (The "use an LLM to
      map relations" idea was the hacky path — the deterministic resolver already existed.)
- [x] **Subject-aware attribution** — consolidator was mixing the user up with the agent because it
      mapped turn-role → subject rigidly (assistant self-extractor absorbed relayed user facts; user
      extractor force-prefixed every claim with "User"). Now both extractors emit a per-fact
      `subject` (user|agent|third_party) and consolidation routes each fact to the matching channel
      (agent → `_self_{agent_id}`, user/third-party → active channel), so either turn role can
      contribute correctly-attributed facts.
- [x] **Subject-aware attribution → per-agent** — the singleton "agent" subject couldn't tell
      Mobius from Atlas (every directive stored as the generic "User wants agent to …"). Now the
      extractor names the specific agent (`subject_agent` → resolved `subject_agent_id`, agent_id =
      source of truth) and consolidation homes each fact to *that* agent's `_self_` channel; agents
      are first-class entities; legacy "Agent …" facts are renamed by a deterministic backfill. (See
      Phase 16 multi-agent attribution.)
- [x] **Backfill orphaned facts** — reworked `link_facts_to_entities` (the scheduled
      `entity_linking` job) into a deterministic, full-history repair: per-(user,channel) name/alias/slug
      index + claim n-gram matching → `(Fact)-[:ABOUT]->(Entity)` edges (`method='backfill_namematch'`),
      no 7-day window, channel-scoped, reports `facts_still_orphan`. Dropped the broken entity-embedding
      dependency (consolidation entities have no embeddings). Remaining (optional): a `task memory:relink`
      / admin endpoint to trigger it on demand instead of only on the 30-min schedule.
- [ ] **Type-check the test suite (django-stubs)** — `tests.py` / `tests_memory.py` currently disable
      pyright framework-noise rules at file level (Django test-client return types, Optional model
      getters, mocked sessions) because no stubs are configured. Add `django-stubs` (+ a pyright/mypy
      config, settings module wiring) and `types-redis` so the test suite gets real type coverage, then
      drop the file-level `# pyright: ...=false` directives. Watch for new stricter-typing fallout on
      Django models. Source already type-checks clean at baseline 0.
- [x] **Memory panel: Fact→Entity display** — shipped. `list_facts` now returns `entities[]`
      ({id,name,type}) alongside `entity_ids`; `FactDetail` renders a clickable "Mentioned entities"
      section that navigates to the entity (`MemoryPanel` onNavigateEntity). Folded together with the
      link tool below.
- [x] **Entity-relationship type consistency** — shipped. Doc'd the canonical edge as
      `(:Entity)-[:RELATES_TO {type, …}]->(:Entity)` in `queries/neo4j_schemas.cypher` (the named
      types `WORKS_FOR`/`RELATED_TO`/… were never written — zero live readers). Also fixed
      `get_entity_facts_and_relationships` to surface the semantic `r.type` property via
      `coalesce(r.type, type(r))` so the graph view stops labelling every edge "RELATES_TO".
- [ ] Global Default Model (ultimate fallback model) not Configurable
- [ ] Store Consolidation costs
- [ ] Chat steaming affect is very disorientating: use animation smoothing avoid ripping the page scroll around
- [ ] Generative Agent Avatar + Extended Icon Base (ie. cool robot face, or funny cat face, etc) -  blocked by image capabilities for models
- [ ] Fibonacci complexity planning scales (augment planning behaviour based on complexity)
- [ ] Disabled memory conversation prompt message banner - informs the model that memory is off for this conversation and the details are not persistent, and also that the conversation may contain confidential material.
- [ ] Nightly consolidation scheduler — persistent job scheduler (Django Q, Celery, or custom) with cron-like registration, restart survival, graceful shutdown
- [ ] Consolidation job logs endpoint (`GET /api/jobs/{id}/logs`)
- [ ] Real-time job progress (polling while running)
- [ ] Consolidation preview (`POST /api/memory/consolidate/preview`)
- [ ] Fact Transience (a confidence bias) — ranked on extraction for the predicted rate that the fact will be incorrect or irrelevant (e.g., "The user's home PC is slow" = high transience)
- [ ] Extraction example sets for LLM to see a large list of examples to compare from
- [x] GPU acceleration for translation models — shipped `[v0.21.6]`. Shared `kit/device.py`
      `resolve_device()` (`AGENTX_DEVICE`: auto/cpu/cuda/cuda:N); `translation.py` moves both NLLB-200
      + detection models `.to(device)` and moves tokenizer inputs onto it in both hot paths (they ran
      CPU-only before); `embeddings.py` passes `device=` to `SentenceTransformer`. Device surfaced at
      `GET /api/health` → `compute` + logged at load. Docs: Windows Setup + GPU Acceleration pages.
- [ ] Lazy model loading with progress indicator
- [ ] Multiple server support (user can log out of server, and into another one seamlessly)
- [ ] Cloud sync for memories
- [ ] Plugin system for additional tools
- [ ] Voice input/output
- [ ] Offline mode with cached models
- [ ] Cross-encoder reranking model for retrieval quality
- [ ] Streaming memory retrieval during chat
- [ ] Conversation sharing (read-only shareable links)
- [ ] Blocking tool call approval (pause stream, user approves/rejects before execution) — the same
      pause/hold-run/resume subsystem would also enable the **blocking in-run Exhibits `choice`**
      round-trip (the user's click becomes the `tool_result` and resumes the same turn, vs. the
      shipped next-turn model). Build once, both benefit.
- [ ] Server authentication (single access key per server, session resume on reconnect)
- [ ] Mobile-responsive breakpoints and touch-friendly gestures
- [ ] Additional themes beyond cosmic (light theme, high contrast, etc.)
- [ ] Message injection into delegated tasks (agent interdiction tools)
- [ ] Custom window chrome — frameless Tauri window with our own title bar + window controls (minimize / maximize / close) and a drag region, styled to the cosmic theme. **Windows + Linux first; macOS later** (traffic-light insets + native fullscreen need separate handling). Touches `client/src-tauri/tauri.conf.json` (`decorations: false`) + a top-of-app titlebar component using the Tauri window API.
- [ ] macOS runner for the client release matrix — add a `macos-latest` leg to `.github/workflows/client-release.yml` (currently Windows + Linux only). Builds `.dmg`/`.app` (`tauri_bundles: dmg,app`); `client/src-tauri/tauri.macos.conf.json` already exists. Needs Apple Developer signing + notarization (certs/secrets) for distributable builds — without them the app is unsigned/Gatekeeper-blocked.

### Agent Genome & Cognitive Evolution (intelligence-focused)

> External idea (Copilot, codebase-blind) evaluated against the actual code. The genome's real value
> is **unification + wiring**: consolidating our scattered cognitive knobs (reasoning strategy,
> ToT branching, Reflection, temperature, delegation config, tool gating) into one tunable per-profile
> struct read per task. The JSON schema is trivial; wiring each gene to a real lever — and giving the
> vague ones (`abstraction_level`, `evidence_strictness`, `tool_bias`) a concrete meaning — is the
> work. Dependency-ordered; the evolution loop is a *research bet*, not an engineering task.

- [ ] **(1, foundation) Reasoning-quality scoring (LLM-as-judge)** — score an agent's reasoning trace
      on coherence / groundedness / foresight / abstraction / self-correction, stored per task. The
      existing `eval_consolidation` harness is **memory-only**, so this is new; reuse the provider layer
      + Reflection's critique-prompt patterns. Independently valuable (powers the **Context Inspector**
      + dashboards) even if evolution never ships. Build this first.
- [ ] **(2) Agent genome — unify cognitive knobs on `AgentProfile`** — a tunable struct
      (`planning_depth`, `branching_factor`, `abstraction_level`, `self_critique_strength`,
      `evidence_strictness`, `delegation_aggressiveness`, `tool_bias`) read per task. **Wire genes to
      existing levers**: `planning_depth`→reasoning strategy + ToT depth / `planner.max_subtasks`;
      `branching_factor`→ToT beam width; `self_critique_strength`→**Reflection** passes (already exists);
      `delegation_aggressiveness`→`alloy.*` thresholds. **Operationalize the unwired genes**
      (`abstraction_level`, `evidence_strictness`→a verification/fact-confidence pass, `tool_bias`→
      tool-choice prompting). Half maps to machinery we have; the value is one coherent control surface.
- [ ] **(3) Context-adaptive genome expression** — modulate genes by derived signals (uncertainty,
      time/risk, tool availability): e.g. high uncertainty → deeper planning, high risk → stricter
      evidence. Downstream of (2); needs uncertainty/risk signals we'd have to derive (not free).
- [ ] **(4) Genome presets = "thinking styles"** — named bundles (careful-analyst, creative-strategist,
      fast-executor) extending the existing `DEFAULT_PROFILES`; Alloy can assign a style to a specialist.
      Falls out of (2) cheaply.
- [ ] **(5, EXPLORATORY — research bet, gate it) Offline genome evolution + intelligence control loop**
      — actor (AgentX) / critic (LLM judge from #1) / environment (a *reasoning* eval harness) →
      store task→trace→score→genome, mutate, keep top-K, discard worst; plus an SLO controller that
      nudges genes when the rolling score drifts. **Risks to respect:** LLM-judge scores are noisy +
      gameable, and auto-tuning a controller off them invites oscillation / reward-hacking. Treat as a
      time-boxed experiment with a **kill criterion** (must beat a fixed-genome baseline on held-out
      tasks), not a shippable feature. Depends on (1)+(2). *(Note: the "online self-critique" half of
      Copilot's #7 already exists as the Reflection strategy.)*

### Settings Advisor + Settings Manifest (the control-plane interface)

> Conceptual frame — the **family model**: **parents** = the Settings Advisor *and* evolution as one
> governance layer with standing authority over the **children** (agents), who act only within the
> config/genome the parents give them (children may *petition* — failures, low reasoning scores,
> uncertainty — but the parents decide). The **user is an associate of the parents** — a *peer*, not a
> boss and not a child: co-decides, gets explanations, sets the **bounds** the parents may act within,
> and keeps ultimate veto. So evolution is not a separate machine — it's **the parents doing long-term
> child-rearing autonomously *within those bounds***; the Advisor is the same governance acting in the
> moment / with the associate. Both run one primitive: *propose a config/genome diff → validate against
> the manifest → (optionally) eval its effect → apply (auto if within bounds, else escalate)*.
> The Advisor's voice follows from "associate": transparent peer — "here's what I see, here's what I'd
> do, your call" — never subservient, never commanding.

- [ ] **(keystone) Settings Manifest** — a canonical registry of every config key
      (`{path, type, default, range, description, "how it works abstractly", affected feature}`).
      Today this knowledge is scattered as inline comments in `config.py` + ad-hoc UI hints. One
      manifest collapses **four** items into itself: it feeds the **Settings Advisor**, lets the
      **settings-overhaul panel** auto-generate a clean UI, supplies the **"document every feature
      in-UI"** + **Memory Area cleanup** descriptions, and gives `/api/config/update` real validation.
      Build this first.
- [ ] **`@Settings` Advisor agent** — a built-in agent profile addressed via the shipped @-mention
      routing (16.5). Free-rein **read** access: the Settings Manifest, the docs-site (a docs-search
      tool), and a **conversation-diagnostic** tool (transcript + the **Context Inspector** + logs/
      metrics) so it can answer "**why did X happen**" and pinpoint the setting responsible. Proposes
      fixes as a **confirmed `form`/`choice` exhibit** that writes via `/api/config/update` —
      **read-broad, write-gated** (user confirms; never silent writes). Uses a **long-context model
      (Opus 1M)** to swallow a whole conversation for diagnosis; budget its own context carefully
      (reuse `assemble_turn_context`). *(Depends on: Settings Manifest; the `form` exhibit element for
      rich apply-a-fix UI — `choice` covers simple toggles until then. This agent is the consumer that
      makes the observability cluster — Context Inspector, SSE status, reasoning scoring — pay off.)*
- [ ] **Shared "control-plane change" primitive** — a single path that takes a config/genome **diff**,
      validates it against the manifest, applies it, and (optionally) evals its effect. The Advisor
      drives it human-confirmed; the evolution subsystem (above) drives it autonomously within bounds.
      Unifying these means evolution is just "the Advisor on auto, gated" — not a separate machine.
- [ ] **Autonomy envelope (the safety keystone)** — a per-system policy object the *associate* (user)
      grants the *parents*: which genes/settings may be auto-tuned and within which ranges, what is
      always escalate-and-confirm (cost, API keys, destructive resets, model swaps), and the
      log/notify behavior. This is what makes evolution **bounded child-rearing** rather than an
      unsupervised mutation loop, and gives the Advisor its collegial-but-empowered footing. Low-risk →
      act + log; high-risk → escalate to the associate. Every control-plane change is checked against it.
- [ ] **Child→parent petition channel** — agents emit governance signals (repeated failures, low
      reasoning scores, high uncertainty, tool errors) that the parents consume as inputs for tuning a
      child. The children do the work and surface what's hurting them; the parents decide the fix.

### Open Platform — De-walling the Garden

> AgentX is today an MCP *consumer* (it eats external tool servers). This track makes it
> interoperable in the other direction: scriptable data I/O, an outward-facing contract, and
> egress. All items carry a schema-versioned envelope (`schema_version` + `agent_id`/channel
> scoping) to honor the v0.20 "migratable across platforms" rule. Cross-references the content-part
> protocol below (an export payload is the same typed structure).

- [x] **Scriptable memory import/export (JSON)** — shipped `[v0.21.22]`. Round-trippable full-graph
      export (conversations/turns, facts/entities, strategies, goals, tool-invocations + PG audit
      mirror) and idempotent re-import that `MERGE`-es every node on its stable id. New
      `kit/agent_memory/portability/` (`schema`/`exporter`/`importer`); `AgentMemory.export_memory`/
      `import_memory` delegators; `POST /api/memory/{export,import}`; `task memory:export|import`
      commands; **merge** + **replace** (scoped `DETACH DELETE`) modes. Exports are **text-only** —
      embeddings are regenerated from text on import (`[v0.21.24]`), so files are small (~9× smaller),
      deterministic, git-diffable, and portable across embedding models (the *Memory-as-VCS* basis is
      now the default behavior). `MemoryPortabilityTest` covers round-trip, idempotency, replace-wipe,
      text-only-export + recompute, and schema-version rejection. Client: Export/Import buttons in the
      Memory drawer (`lib/fileTransfer.ts` browser Blob/FileReader helper — first file I/O in the
      client). Unblocks the 18.6 eval snapshot/restore. Supersedes the old "Memory export/import
      (JSON/SQLite backup)" item.
- [ ] **Import dry-run / preview-diff** — show the graph delta before committing (mirrors the existing
      `POST /api/memory/consolidate/preview` pattern) so a bad hand-edit can't silently nuke the graph.
      *Natural next follow-up to the shipped import/export above.*
- [ ] **Verify-on-import (default on, opt-out)** — route imported facts through the three-layer
      `check_contradictions` pipeline rather than trusting JSON verbatim; opt-out flag for trusted/raw
      restores. *Natural next follow-up to the shipped import/export above.*
- [ ] **Memory-as-VCS (diffable, hand-editable snapshots)** — the text-only export + idempotent
      MERGE-on-id import (`[v0.21.24]`) already gives the core loop: export → commit/hand-edit →
      import re-applies, re-embedding from text → branchable/restorable memory states. Remaining
      polish: canonical/sorted-key JSON (+ optional NDJSON) for clean `git diff`; a
      `memory:snapshot`/`memory:restore` task pair; per-node content hash to recompute only changed
      embeddings (today import re-embeds every node); a "memory log" of snapshots.
- [ ] **Import conflict policy** — skip / overwrite / rename strategies for cross-instance id
      collisions (merge currently overwrites — importing another instance's export reassigns shared-id
      nodes to the importing user).
- [ ] **Recompute PG audit-mirror embeddings on import** — import leaves
      `conversation_logs.embedding` NULL (recall uses the recomputed Neo4j `turn_embeddings`); fill it
      from content (cheap via the embedding cache) if anything starts querying the PG vector column.
- [ ] **Per-user export** — export/import is single-user (`DEFAULT_USER_ID = "default"`) until auth
      lands; scope by authenticated user when multi-user ships.
- [ ] **Expose AgentX outward** — publish its own memory/agents as an MCP *server* + promote
      `OpenApi.yaml` to a stable public REST contract. (Subsumes the "Conversation MCP Tool" item under
      MCP Tools below — `memory_recall` / `memory_store` / `conversation_summary`.)
- [ ] **Egress / webhooks** — outbound events so external systems can react to agent activity
      (run lifecycle, new facts, goal completion).

### Exhibits — Rich Agent-Authored Content (declarative content-part protocol)

> The agent presents structured content the client renders from a registry — rather than
> hand-rolling raw HTML (a security/consistency liability). Vocabulary: a **Gallery** (a
> conversation's array of exhibits) → **Exhibit** (one declaratively-arranged unit, amendable by
> stable `id`) → **Element** (typed building block). Producer is the declarative internal
> `present_exhibit` tool (not fence-scraping) — the same mechanism interactive elements need.
> Visual sibling to the 16.6 Ambassador Agent (which mediates via voice/briefing); this mediates
> visually. Same typed structure doubles as the export/integration payload above.

**Shipped (Slices 1–5, `[v0.21.25]`–`[v0.21.29]`) → [roadmap.md](docs-site/src/content/docs/roadmap.md):**
protocol + `present_exhibit` tool, `mermaid`/`choice`/`table`/`citation` elements, `web_search` +
`web_research` citation auto-capture, capability-aware Tavily web tools (search/extract/map/crawl/
research via the `tavily-python` SDK), universal model fallback + bulk/inherited memory-stage models,
and the static client-only conversation **Bibliography** ("Sources" drawer).

Open:
- [ ] **`text` element** + absorbs the former "Advanced memory visualization (interactive graph,
      embedding clusters)" item as a registered element type.
- [ ] **Active-citation context-injection** — fold `active` sources' `quote`s back into the agent's
      context (bounded) so it can reference tracked sources later (the "tracked in the chat" payoff).
      Still to do: `memory` citations deep-linking into the Memory drawer fact, and `web_extract` →
      `active` citation promotion.
- [ ] **Per-turn web search/research credit budget** — Tavily burns credits fast. Allot each turn a
      **credit budget** (config `search.credits_per_turn`, e.g. 15) that web tools spend by a
      **weighted cost** (`web_search` ~1, `web_extract` ~2, `web_crawl` ~5, `web_research` ~10 —
      tunable, mirroring real API cost). **Every web tool result returns `credits_remaining`** so the
      model self-rations; once exhausted, calls return a clear "budget exhausted" error instead of
      looping. Track the per-turn tally in the tool loop / internal context (reset each turn).
- [ ] **Truly long `web_research` → background job** — move minutes-long research off the synchronous
      tool path onto the `/api/chat/background` queue so it can't block a turn.
- [ ] **Configure the global default model (UI gap)** — two latent keys exist
      (`preferences.default_model`, `models.defaults.chat`) with **no settings editor**. Live turns
      are safe (agent profiles carry a model = the fallback floor), but background consolidation has
      no floor without one. Add a picker in settings.
- [ ] **`form` (multi-field) interactive element** — multiple inputs submitted together as one
      turn; builds on the `choice` next-turn mechanism.
- [ ] **`grid` (and richer) layouts** + a dedicated browsable **Gallery panel** (drawer) listing a
      conversation's exhibits.
- [ ] **Inline-fence fallback** — also render the model's *native* ` ```mermaid ` fences (no tool
      call) by parsing them into exhibits, for models that under-reach for the tool.
- [ ] **Exhibits in delegation streams** — extend the typed event to `delegation_chunk` so a
      specialist's diagrams surface too.

### Translation Quality Overhaul (pluggable `TranslationKit` backend)

> NLLB-200 graded 5/10 — but we just invested in it (GPU accel `[v0.21.6]`, `LanguageLexicon` ISO-code
> bridging), so the move is *pluggable backend*, not rip-and-replace. (Caveat on the eval: Mistral
> grading NLLB output while itself relying on NLLB is a soft/circular benchmark.)

- [ ] **Pluggable translation backend behind `TranslationKit`** — interface so backends swap without
      touching the `LanguageLexicon` code-bridging or call sites.
- [ ] **LLM-provider translation path** — route high-value pairs through the existing model-provider
      stack (reuses the provider abstraction, no new dependency) and keep NLLB-200 as the cheap offline
      fallback.
- [ ] **Evaluate stronger open models** — SeamlessM4T / MADLAD-400 / Tower as alternative offline
      backends; pick on a non-circular eval.

### Web Search & Delegation — shipped + deferred

> Shipped (see plan `~/.claude/plans/i-can-t-do-it-unified-pond.md`): internal `web_search` tool
> (**Tavily** primary + **Brave REST** fallback, in-tool retry + short-TTL cache; Brave MCP server
> auto-connect disabled), `search.*` config + **Settings → Web Search** UI; **parallel fan-out
> delegation** (`alloy.max_parallel_delegations`, reentrant `AlloyExecutor`, queue fan-in in
> `_run_delegations`); **delegatable agent profiles** (`available_for_delegation` flag + filter,
> tool-gating persistence-bug fix, **Settings → Multi-Agent** toggle, **Researcher** preset); profile
> editor **hybrid Tabs+Accordion** UX. SearXNG was dropped in favor of Tavily (no proxy/blacklist ops).

Deferred — **Search Router** subsystem ("browsing on autopilot"); a delegatable Researcher already
covers ~80% of this via its own tool loop:
- [ ] **`fetch_page` tool** — trafilatura (static) now, Playwright (JS-heavy) later — lets a
      Researcher read full pages, not just snippets.
- [ ] **Autonomous browse loop** — search→fetch→follow→synthesize with confidence/termination
      heuristics (a `research` tool or ReAct-derived `ResearchAgent` via `reasoning/orchestrator.py`).
- [ ] **Group-based tool gating** — consume the latent `groups` field (`mcp/server_registry.py`) to
      route a set of web tools into a managed lane (today's per-profile `allowed_tools` suffices).
- [ ] **Router lifecycle subsystem** — per-tool rate limiting, session state, shared cache, backend
      rotation beyond the in-tool Tavily→Brave fallback.
- [ ] **SearXNG self-hosted backend** — optional fully-self-hosted `web_search` backend (needs
      residential/ISP proxy in `settings.yml`); slots behind the existing pluggable tool.

### ⭐ Active Memory Recall — close the query-formulation gap

> The real gap isn't *retrieval*, it's **query formulation**. Recall is `remember(query=message)` —
> the literal user turn is the query. But "should we go after this idea?" has near-zero overlap with
> the facts that matter (our goals, prior decisions, constraints, what we're mid-build on). A human
> partner first asks *themselves* "what are we trying to accomplish? what are we working on?" — an
> **implicit question** the LLM has to synthesize. HyDE only half-helps (it hypothesizes an *answer*;
> here there's no answer yet, only the implicit questions). Build recall in **three tiers** — the model
> needs memory in-context *before* it generates, so this can't be streamed in after the fact. Ties
> foundation #3 (stable memory core) + the Retrieval Quality items below.

- [ ] **Tier 2 — smart pre-turn recall (start here; the 80/20)** — replace `remember(query=message)`
      with: (1) a **conversational query rewrite / step-back** (one fast-model call: "should we go
      after this?" → "active goals; current project scope; prior strategic decisions; known
      constraints/risks"); (2) **anchor retrieval on `get_active_goals()` + `session.summary`**, not
      the raw message (both already exist, recall ignores them); (3) fan the sub-queries out
      concurrently. Synchronous but cheap; fixes the completion gap on every turn.
- [ ] **Fast recall model knob** — add `recall.expansion_model` (rewrite/step-back/expansion) defaulting
      to a fast tier (local `nemotron-nano` like `combined_extraction_model`, or a Haiku/Flash-class
      cloud model). The expensive chat model never touches recall; sub-queries parallelize.
- [ ] **Tier 1 — passive working-set watchdog (always-on, state-driven)** — a debounced background
      updater keeps a compact "here's our head right now" digest fresh as turns accumulate (goals +
      recent decisions + salient entities + open questions); piggyback the rolling-summary update that
      already fires. Injected as the **stable core** every turn (foundation #3); recall is the
      *supplement*. This is the "thinking in their own head" — maintenance, not search. Supersedes the
      **Working Memory Scratchpad** item below.
- [ ] **Tier 3 — agentic deep recall (on-demand, observable)** — an LLM-callable `deep_recall("what do
      we know bearing on this?")` that runs a **multi-hop compounding** loop (retrieve → read → the gaps
      become the next query → retrieve again; FLARE/IRCoT/self-RAG family), synthesizing a working
      brief. It "blocks" the turn only when the model *chooses* to think harder — the human behavior —
      and **streams its steps over the `status`/`delegation_*` SSE infra** so the user sees it think.
      **Implement as an Agent Alloy specialist** (a delegated "Memory" agent) to reuse delegation
      streaming + depth limits wholesale — which dissolves the "SSE vs blocking agent" question: it's a
      delegated agent that streams. (Decision to pin: Alloy specialist vs standalone internal tool.)
- [ ] **Compounding extraction — keep it ephemeral** — during multi-hop recall, synthesize a
      retrieval-time brief but **don't write durable facts inline** (pollution risk); instead *queue*
      interesting discoveries for the existing 15-min consolidation, which owns durable writes.

### ⭐ Procedural Memory — a constant thought (encode → replay → activate)

> **Pre-flight verdict: it's wired but inert** — four independent breaks. ① internal tools never
> record (chat loop → `call_tool_sync` → `execute_internal_tool` bypasses the `tool_executor` recorder;
> only external MCP tools record). ② `detect_patterns` keys off `(:Conversation)-[:RESULTED_IN]->
> (:Outcome {success:true})`, but `Outcome` nodes are created **only** in the eval harness — never the
> live path → zero strategies learned. ③ even if learned, `MemoryBundle.to_context_string` renders
> turns/facts/entities/goals but **omits strategies** → they never reach the prompt. ④ the
> consolidation worker runs only via `task memory:worker`; `task dev` doesn't start it. Don't revive the
> coarse `detect_patterns` ("Use {tools} for {task_type}") — rebuild it **brain-modeled**. Unifies with
> the **Active Memory Recall** tiers above (Tier-1 watchdog = the reflex core; Tier-3 = deliberate
> self-query). Seed signal: the persisted steer corrections (`metadata.steered`).
>
> **Slice 1 update:** ③ and ④ **fixed** (reflex-core renders into the prompt; the autonomous worker now
> awaits coroutine jobs — it was silently dropping the async `consolidate` — and `task dev` runs it).
> The new corrections/rules → `Procedure` path runs the brain-modeled distill (encode → distill →
> reflex-core activation). ① and ② (internal-tool sequence evidence + a live Outcome/success signal,
> which feed tool-**sequence** procedures) remain for a later slice.

> **Brain model.** (a) *General → specific*: the model already has the general baseline, so store only
> the **delta** — the project/user/domain "how we do it here," corrections, learned habits. (b) *The
> asymmetry is encoding + recall, not retention*: agents store durably (no rehearsal needed), but a
> stored-yet-never-recalled procedure is **functionally forgotten** → engineer **active recall**, not
> rehearsal. (c) *It's a constant thought*: three always-on loops, not a batch afterthought.

- [ ] **Useful-pattern spec (the first step)** — a pattern is stored only if it passes: (1) a
      **high-signal event** — corrections/**steers** (`metadata.steered`, the best signal) › explicit
      rules ("always…/we prefer…") › failure→recovery › repetition › novel successful sequences; (2)
      **baseline-deviation** — a fast model discards anything a competent agent would already do by
      default (store the *delta*); (3) **reusable + evidenced** (scoped, backed by a real event).
      Scope hierarchy `_global → user → project → _self_{agent} → conversation`; recall prefers
      most-specific.
- [x] **Loop 1 — Encode (every turn, cheap)** — **Slice 0 shipped.** Stages `correction` candidates
      (from the persisted `steers_data` — `after_tools`/`steer_round`/`phase`) + `explicit_rule`
      candidates (`procedural.detect_explicit_rule`, heuristic, no LLM) into a new `procedure_candidates`
      PG table, from the `_persist_turns` daemon on the streaming chat path (both normal + hard-stop).
      Count surfaced on `/api/memory/stats`. **Remaining for the loop:** failure-marker capture +
      repetition detection. [[metadata.steered]]
- [~] **Loop 2 — Replay / distill (consolidation = "sleep")** — **Slice 1 shipped** the candidate→
      Procedure half: the async `distill_procedures` consolidation job (`consolidation/jobs.py`, in the
      registry default pipeline + autonomous worker) reads pending `procedure_candidates`, groups by
      derived scope (corrections with an `agent_id` route to `_self_{agent_id}`; explicit rules stay on
      their channel), runs `ExtractionService.distill_procedure` (baseline-deviation filter with
      signal-aware deference — explicit rules are kept unless redundant), and **strengthens** a
      cosine-similar existing Procedure (`procedural_dedupe_threshold`) instead of duplicating; candidates
      flip to `distilled`/`discarded` (+`distilled_into`). **Remaining:** tool-**sequence** replay /
      invariant-core abstraction, `ReflectiveReasoner` correction-reflection, Fix ② (emit a success/
      Outcome signal) + Fix ① (record internal-tool sequences as evidence).
- [~] **Loop 3 — Activate (every turn; the hard part) — gate, don't retrieve** — procedural recall ≠
      content similarity (a conditional trigger→sequence isn't *similar* to the prompt). **Index by
      trigger, query by situation.** Four modes: **reflex core** (top general/project procedures
      injected every turn, maintained not searched = Tier-1 watchdog) — **shipped in Slice 1**
      (`ProceduralMemory.get_reflex_procedures` top-`strength` over recall channels, attached at the
      `remember()` boundary, rendered by `MemoryBundle.to_context_string`, gated by `reflex_core_enabled`/
      `reflex_core_limit`; Fix ③ done); **activation nerve** (match a *situation descriptor* built from
      goals+summary+fast-model intent-tag+entities+next-tool against trigger conditions — fires the
      procedure); **point-of-action** (action-bound procedures inject at the tool-call boundary);
      **deliberate** (`recall_procedures` self-query = Tier-3) — all three remain.
- [x] **`Procedure` model (richer than `Strategy`)** — **Slice 1.** New Neo4j `Procedure` node +
      `models.Procedure` `{trigger (NL) + trigger_features, body (replayable), rationale, scope, strength
      (replay/reinforce count), evidence_refs, signal_kinds}` with `procedure_embeddings` vector index;
      `learn_procedure`/`reinforce_procedure`/`find_procedures`/`list_procedures` reuse the (dead)
      `learn_strategy`/`reinforce_strategy` write pattern. Inspect via `GET /api/memory/procedures` +
      `/api/memory/stats` `procedures` count. Fix ④ done — `ConsolidationWorker` now awaits coroutine
      jobs (the autonomous `consolidate` was silently no-op'ing) and `task dev` runs the worker; manual
      `task memory:distill-procedures`.

### Retrieval Quality Enhancements (migrated from docs/future-feature-pool)
- [ ] Working Memory Scratchpad — always prepend a structured scratchpad (current topic/task, active entities, recent corrections, open questions) to context for coherence/orientation
- [ ] Conversation Summarization — maintain rolling per-session and per-topic summaries; retrieval becomes `recent_turns + relevant_summaries + relevant_facts`
- [ ] Query Intent Classification — classify query before retrieval (follow-up → recency, callback → older history, new topic → broad semantic, factual recall → entities/facts); rule-based or lightweight LLM
- [ ] Negative/Correction Tracking — when `correction_detection_enabled`, mark superseded facts `temporal_context: "past"`, link corrections to originals, prioritize corrections in retrieval
- [ ] Fact Staleness Detection — add `expected_stability: transient|stable|permanent` and surface staleness warnings (relates to Fact Transience above)
- [ ] Multi-hop Entity Traversal — add a lightweight path-finding retrieval mode over the entity graph (e.g. User → works_at → Company → has_project → Project → uses_tool → Tool)
- [x] Memory graph relationship connections — shipped as a Fact↔Entity link tool. Backend
      `link_fact_to_entity`/`unlink_fact_from_entity` (user-scoped ABOUT MERGE/DELETE) +
      `POST/DELETE /api/memory/facts/{id}/entities`; `FactDetail` "Mentioned entities" section has a
      searchable, channel-scoped entity picker (link) + per-chip unlink, optimistically updating the
      fact. (Deferred: drag-to-connect directly on the ReactFlow canvas.)

### MCP Tools (migrated from docs/future-feature-pool)
- [ ] Conversation MCP Tool — expose memory as MCP tools for external agents: `memory_recall(query, filters?)`, `memory_store(fact)`, `conversation_summary(conversation_id?)`

### Extraction Improvements (migrated from docs/future-feature-pool)
- [ ] Claude Sonnet for Extraction — switch extraction from local models to Claude Sonnet for better structured-output adherence, nuance detection, and entity resolution (cost/latency offset by async/batched consolidation)
- [ ] Improved Extraction Prompts — few-shot examples, better schema definitions, domain-specific tuning

### Logging & Observability Overhaul (`logging_kit`)
- [x] Central color/category/run-tag console via QueueHandler→QueueListener; `AGENTX_LOG_*` flags (decorations on by default); plain/json modes; secret redaction; third-party noise tamed
- [x] ASCII startup banner + status table (`AGENTX_LOG_BANNER`)
- [x] Compact LLM request cards (`AGENTX_LLM_LOG_LEVEL` off|summary|full; legacy `DEBUG_LOG_LLM_REQUESTS`→full)
- [x] In-memory ring buffer + `/api/logs`, `/api/logs/stream` (SSE), `/api/logs/categories`; auth-gated via middleware + `AGENTX_LOG_API_ENABLED`
- [x] Compressed log archive: rotating gzip file handler + `/api/logs/archive*` endpoints + client browse/download
- [ ] Optional structured JSONL sidecar for the archive (machine-queryable history)
- [ ] Stamp `conversation_id`/`agent_id` ContextVars at the chat entry (run_id already wired) for richer per-turn correlation
- [ ] Dedupe/rate-limit consecutive identical lines (`… (×N)`); client run-tag → transcript cross-link

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

**~~Query Embedding Caching~~** — RESOLVED `[v0.21.6]`
- Identical queries now hit an LRU+TTL cache (`EmbeddingCache`, keyed `(provider:model, text)`) in
  front of the queue (`kit/agent_memory/embedding_queue.py`). Tunable via `EMBEDDING_CACHE_*`.

**~~Embedding Request Queue / Serialization~~** — RESOLVED `[v0.21.6]`
- All embedding calls funnel through one process-wide daemon worker (`EmbeddingDispatcher` →
  `_EmbeddingQueue`, `kit/agent_memory/embedding_queue.py`): serialized so the thread-unsafe local
  model never runs concurrently, with opportunistic batching, bounded-queue backpressure, and
  exponential-backoff retry on transient (remote) failures. The public `embed`/`embed_single` API is
  unchanged, so all ~40 call sites were untouched. Lazy-started; bypassable via
  `EMBEDDING_QUEUE_ENABLED=false`. Covered by `EmbeddingQueueTest`.

---

## Blockers

None currently.

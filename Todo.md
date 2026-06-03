
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
| Phase 16: Multi-Agent Conversations | **In Progress** | ~65% (16.0–16.5 shipped; Factory UI + ambassador deferred) |
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
- [ ] Subtask-level goal tracking (create subgoals via `parent_goal_id`)
- [x] Plan cancellation mid-execution — shipped. `PlanExecutor` checks `state.is_cancel_requested(plan_id)` between subtasks (`plan_executor.py:84,163`) and marks the plan `cancelled`; `POST /api/agent/plans/cancel` sets the flag.
- [ ] Plan resumption from Redis state after disconnect

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
- [ ] **Per-agent identity seeding** — on profile create, seed the agent's `_self_` channel
      with an identity fact/entity ("I am Mobius, id …") for stronger self-recall.
- [ ] **Debug-harness extensions** — record/replay real conversations into scenarios;
      assertion-based regression suite (golden attribution outcomes) runnable in CI when a
      provider is configured; extract the shared cluster snapshot/wipe/restore util used by
      `eval_consolidation` into a module both commands import.
- [ ] **Memory capability registry** — a code-side `@capability(...)`/registry that
      `architecture/memory-capabilities.md` is generated from or validated against, so the
      manifest can't silently drift from code (the deferred half of the drift decision).

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

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

### ▶ FOUNDATION — real next-session priority order (barring the genome/advisor/evolution meta-layer)

> The "fancy" meta-layer (Agent Genome, Settings Advisor, evolution) is captured below but gated on
> foundation. Do these first; they're user-facing, correctness, or reliability — not strategy.

1. **Chat legibility slice** *(highest visible payoff; the user's most-repeated complaint)* — bundle
   three from the **Chat UX & Tool-Call Rendering** + **Observability** clusters: (a) **compact,
   collapsible tool-call rendering** (`chat/bubbles/ToolCallBubble`, `ToolResultBubble`,
   `ToolExecutionBlock`/`ToolResultBlock` — dense one-liner by default; everything inheriting the
   block gets it), (b) **web-search query inline** on the collapsed row, (c) **per-phase SSE `status`
   events** (`streaming/tool_loop.py` emit → `lib/api/streaming.ts` → `useChatStream`) so the chat
   shows a live activity line instead of a silent "thinking". Mostly client + one focused backend emit.
2. **Stable memory core** — kill transient memory injection (`remember(query=message)` re-ranks every
   turn); inject a stable high-salience core + recall as a supplement. Correctness; rides the Slice-6
   `assemble_turn_context` preamble budget.
3. **Finish the reliability guarantees** — extend the Slice-5 model fallback to the remaining feature
   sites (reasoning/drafting/`planner`/`alloy`, still raw `get_provider_for_model`); **hydrate the
   Alloy + background-chat paths** (Slice-6 follow-up) so multi-agent/queued chats also resume warm.
4. **Cost + gaps** — **per-turn search credit budget** (Tavily spend), **configure the global default
   model** (UI gap), and the **full persisted tool outputs** debugging surface (heavier backend).
5. **Tech-debt sweep** — consolidate the 4 token estimators (→ `tiktoken`), retire dead context knobs
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

### Chat UX & Tool-Call Rendering (density + observability)

> Tool calls — and everything that inherits the tool-call block (checkpoints, exhibit fallbacks,
> web-search cards) — dominate the transcript and hide what matters. Make the chat readable.

- [ ] **Compact tool-call rendering** — the tool-call block is bulky and space-hungry; redesign it
      to a dense, collapsible one-liner by default (icon + tool + key arg), expandable on demand.
      Everything inheriting the block (checkpoints, web-search, exhibit source-fallbacks) gets the
      slimmer treatment.
- [ ] **Web-search shows its query inline** — you currently can't see what the agent searched
      without drilling into the card. Surface the query (+ result count) on the collapsed row; pair
      with the auto-captured Sources so results are one glance away.
- [ ] **Full tool-call outputs (persisted)** — only a small slice of a tool result is shown today,
      but full outputs are valuable for debugging agent thinking. Persist complete outputs
      (PostgreSQL or similar) and let the UI expand to the whole thing (lazy-loaded), beyond the
      streamed/truncated preview.

### Backend Observability — live operation status over SSE

- [ ] **Per-phase status events** — between "message sent" and the response, the UI just says
      "thinking" then dumps the answer. Emit granular status over the existing SSE stream so the
      client *always* knows the backend phase: recalling/embedding, composing context, reasoning
      step N, building a tool call, running a tool, compressing, synthesizing, etc. A typed `status`
      event (phase + human label) the chat renders as a live activity line.
- [ ] **Context Inspector ("what's in the model's head this turn")** *(my idea, from the Slice-6
      context work)* — now that `assemble_turn_context` builds one well-defined message list, expose it:
      a per-turn debug view showing exactly what was sent to the model (system preamble blocks:
      checkpoints / scratchpad / summary / memory; the verbatim transcript that fit; the new turn) with
      **per-block token counts** and the budget breakdown (verbatim vs reserved vs window). Pairs with
      the per-tab context bar + the "full tool outputs" item — the single best lens for debugging agent
      behavior. Cheap to surface (the assembler already has all of it); gate behind a dev/inspect toggle.

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

- [ ] **Redesign the Memory area** (drastic cleanup, mirroring the agent-profile editor pass) and
      **document every feature in-UI** — each control gets a clear, abstract description of what it
      does and how it works, so the panel is legible without reading the code.

### Engineering Hardening (observed while in the code, Slices 5–6)

> Grounded tech-debt / consistency items noticed during the model-fallback + context work.

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
- [ ] **Memory panel: Fact→Entity display** — `client/src/components/memory/FactDetail.tsx` ignores
      `entity_ids` entirely (Entity→Fact works in EntityDetail; the reverse is missing). Have
      `/api/memory/facts` return `{id,name,type}` for ABOUT'd entities (query already does
      `OPTIONAL MATCH (f)-[:ABOUT]->(e)`) and add a clickable "Mentioned entities" section.
- [ ] **Entity-relationship type consistency** — consolidation stores all entity↔entity relations as
      `[:RELATES_TO {type}]` (jobs.py `_batch_store_relationships`) while `queries/neo4j_schemas.cypher`
      documents specific types (`RELATED_TO`, `WORKS_FOR`, …). Pick one model and align graph
      traversals/queries.
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

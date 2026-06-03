
# AgentX Development To-do

**Project**: AgentX - AI Agent Platform
**Status**: Prototype
**Last Updated**: 2026-05-29

**NOTE**: UI must be highly responsive between PC and mobile devices.

**Versioning**: `versions.yaml` is the single source of truth (run `task versions:sync` after
editing it). Completed work is tagged inline with the version it shipped in, e.g. `[v0.20.1]`.
Bump the version when a unit of work completes — patch for additive/back-compat features, and
bump `protocol_version` only on breaking API changes. Current: **0.21.24** (protocol 1).

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

### 18.x Open / Deferred

- [x] **Dashboard redesign** (clean/mean/powerful) — shipped `[v0.21.10]`. Dropped the two
      4-card status grids + maintenance banner for a slim `SystemStatusStrip` of
      colored pills (API · Neo4j · PG · Redis · Providers · MCP · Agent) with a `Details`
      toggle that expands the full subsystem breakdown inline (same hooks; same data).
      `UsageMetricsSection` is now the hero: added a **Projected Month** KPI tile
      (client-side MTD × days-in-month / days-elapsed), swapped the daily chart for a
      dual-axis recharts `ComposedChart` (tokens bars + cost line), and added side-by-side
      **Top Models** and **Per-Agent** leaderboard tables. Backend: `usage_metrics` view
      gained a `by_agent` aggregation (`GROUP BY COALESCE(agent_id,'_default')`); type +
      OpenAPI + endpoints doc updated. Agent display names resolved via
      `useAgentProfile().profiles` (slug → name), fallback to slug / `Default agent`.
- [x] **Dashboard token + turn metrics** (18.5) — shipped `[v0.21.9]`. New read-only
      `GET /api/metrics/usage?days=` endpoint (`views.usage_metrics`) aggregates assistant turns from
      `conversation_logs` (`metadata` JSONB → tokens/cost/latency, `idx_logs_timestamp` window) into
      totals + by-model + daily series, with graceful `unavailable` degradation. Client adds
      `metricsApi.getUsageMetrics` + `useUsageMetrics` and a Dashboard **Usage & Cost** section
      (`components/dashboard/UsageMetricsSection.tsx`) with summary tiles + recharts bar/area charts
      (7/14/30-day toggle). Shared formatters extracted to `lib/format.ts` (reused by `MetadataBar`).
      Note: the earlier "no further backend work needed" assessment was inaccurate — `/api/memory/stats`
      only counts Neo4j nodes, so the aggregation endpoint had to be added.
- [-] **Extraction eval-harness follow-ups** (18.6):
  - [x] Procedural-memory eval cases (tool-usage/strategy learning) — shipped `[v0.21.21]`.
        `eval_consolidation` now carries procedural cases (any `EvalCase` with a
        `tool_sequence`): seeds a tool trajectory + success/failure `Outcome` **directly**
        (bypassing the extraction LLM — embedding-only), runs `detect_patterns()`, and scores
        the learned `Strategy` via the `SUCCEEDED_IN` edge back to the eval conversation
        (strategies aren't channel-tagged). Includes a negative case (`proc_failed_run`) that
        asserts the detector's `WHERE success:true` gate learns nothing. **Note:** the
        *production* agent loop never writes `:Outcome` nodes, so `detect_patterns()` is a
        no-op outside this eval — wiring that into the live loop is a separate follow-up
        (see Backlog/Phase 15-16 procedural tracking).
  - [x] Snapshot/restore instead of `--wipe` — shipped `[v0.21.23]`. `eval_consolidation --snapshot`
        exports the whole cluster (every user, with embeddings) to `data/eval_snapshots/<ts>.json`,
        wipes, runs, then restores it in a `finally` (survives an eval error); bypasses the sterility
        gate non-destructively. Provider usability is validated **before** the wipe (fail-fast).
        `--restore <file>` is a standalone recovery path. Reuses the `[v0.21.22]` portability module.
        Covered by `EvalSnapshotRestoreTest`.
  - [x] Persist eval runs (model, per-case scores, tokens) for cross-model / cross-prompt
        comparison — shipped `[v0.21.21]`. `--save`/`--no-save` (default on) + `--output-dir`
        write `data/eval_runs/<ts>-<run4>_<model>_<full|quick>.json` (per-case results,
        `summary.by_kind`, full `ConsolidationMetrics`) and append `index.jsonl`. Extraction
        vs procedural tallies are kept **separate** — extraction quality varies by model, but
        procedural is embedding-only and constant across models, so a single number would
        muddy comparison.
- [x] **Extraction cleanup** (18.6) — shipped. Two changes land together:
      (a) `python manage.py dedupe_entities` (Django management command, wired via
      `task db:dedupe:entities[:apply]`) collapses pre-resolution-fix duplicates in
      Neo4j. Default is dry-run inside an explicit transaction (rollback); `--apply`
      commits. Survivor pick: salience DESC → access_count DESC → first_seen ASC.
      `--cross-channel` adds a per-user pass that prefers `_global` survivors to fold
      duplicates spread across `_default`/`_self_*`/`_alloy_*` (matches recall's
      `[active_channel, _self_<id>, _global]` lookup order). Uses
      `apoc.create.relationship` to rewrite incoming/outgoing edges, then a
      parallel-rel collapse on the survivor.
      (b) `check_correction()` retires the 10-pattern `CORRECTION_PATTERNS` regex
      pre-filter in `extraction/service.py` — every gated turn now goes straight to
      the correction LLM, which already returns `CORRECTION: NO` for non-matches.
      The cheap `< 10 chars` / `constants.skip_patterns` guard stays purely as a cost
      gate. Tests in `tests_memory.py::CorrectionDetectionTest` rewritten against a
      mocked provider; new `DedupeEntitiesAliasMergeTest` covers the alias-merge
      helper.
- [x] **Working-memory follow-ups** (18.9) — redundancy-pruned after review and shipped in waves:
  - [x] `scratchpad_note` tool + read-back (folds in `inspect_working_memory`) — shipped `[v0.21.18]`.
        New `agent/scratchpad_storage.py` mirrors checkpoint storage (Redis, re-injected each turn so
        it survives compression); the tool's no-arg/`read` mode returns notes + checkpoints + active
        goals (never the transcript). Injected next to checkpoints in `views.py`. Also removed the
        dead `push_thought`/`get_thoughts`/`set_active_goal`/`get_active_goal` helpers in `working.py`.
  - [x] `forget` (soft-supersede) + salience boost ("remember this") + "where did I learn this?"
        provenance — shipped `[v0.21.19]`. `salience` threaded through `update_fact`; new
        `AgentMemory.boost_salience`/`forget_fact`/`get_fact_provenance` (+ `episodic.get_turn_by_id`);
        `_internal.remember_this`/`_internal.forget` tools; `POST /api/memory/facts/{id}/remember`,
        `.../forget`, `GET .../provenance`; Memory-drawer fact actions (Remember / Source / Forget).
  - [x] cached `user_recap_summary` rolling summary — shipped `[v0.21.20]`. New
        `kit/agent_memory/recap.py` builds a durable cross-conversation recap (recent turns + facts →
        the configured summary model) and caches it in Redis (`user_recap:{user_id}:{channel}`, 30-day
        TTL); refreshed at the end of `consolidate_episodic_to_semantic` (gated by
        `user_recap_enabled`). `recall_user_history` + `POST /api/memory/user-history` now fill their
        `summary` field from the cache, surfaced as a "Recap" block in `UserHistoryView`.
  - **Cut after review (already covered):** active-goals header (the memory bundle already injects a
        `## Active Goals` section), model-driven pin/anchor turns (duplicates `checkpoint`).
- [x] **Per-profile internal-tool gating UI** (18.9.x) — shipped. New
      `ToolAccessSection` inside the Profile Editor (under the existing
      `Enable Tools` toggle) surfaces `AgentProfile.allowed_tools` /
      `blocked_tools` with a "Allow all / Limit to selected" mode chip
      (mirrors the Toolkit AccessView UX), a per-server grouped checklist
      pulled from `useMCPTools()`, and a separate Blocked-tools chip list
      with a `<details>` add-to-block picker. `_internal.<name>` group is
      pinned first so users can opt in/out of `checkpoint`,
      `recall_user_history`, etc. Read-only "Enabled for: X, Y" badges in
      the Toolkit Tools Browser surface the gating from the other side.
      Backend now matches on fully-qualified `server.tool` keys
      (`Agent._get_tools_for_provider`) — the docstrings already promised
      this; legacy bare-name entries log a startup warning via
      `profiles._warn_unqualified_tool_names`. Also fixed a stale TS type:
      `MCPTool.server` is now populated (was always blank — backend returns
      `server_name`). New `ToolGatingTest` + `ProfileUnqualifiedToolWarningTest`
      in `tests.py` cover FQ matching, allow/block precedence,
      internal-tool gating, and the warning path.

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

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

- [x] **Content-part protocol (Exhibits) + Mermaid element** — shipped `[v0.21.25]`. Declarative
      Gallery→Exhibit→Element model with a `schema_version` envelope and an `exhibit` SSE event.
      Producer: internal `present_exhibit({ id?, title?, layout?, elements:[...] })` tool
      (`mcp/internal_tools.py`); `streaming/exhibits.py` owns the Pydantic models +
      `ALLOWED_ELEMENT_TYPES` allow-list + validation. `streaming/tool_loop.py` surfaces a
      `present_exhibit` call as a typed `exhibit` event (suppressing its `tool_call`/`tool_result`
      cards) while the tool body still returns a result so the model can re-present on error.
      Client: `lib/exhibits.ts` types, `onExhibit` streaming callback, `useChatStream` upsert-by-id
      (amend in place), `ExhibitBubble` + `elementRegistry` (unknown type → safe source-as-code
      fallback) + `messageRegistry` entry, `MermaidElement` (dynamic-imported mermaid@11,
      `securityLevel:'strict'`, error fallback), and `mapServerMessages` restore (rebuilds exhibits
      from stored `present_exhibit` tool turns, last-per-id wins). `PresentExhibitToolTest` (backend)
      + exhibit restore cases (`mapServerMessages.test.ts`). Slice 1 = `mermaid` element + `stack`
      layout only; the model/protocol carry the full tree.
- [ ] **More element types** — `table`, `text`, `citation` (wire to `memory_context`/`web_search`);
      absorbs the former "Advanced memory visualization (interactive graph, embedding clusters)" item
      as one registered element type.
- [x] **Interactive `choice` element (next-turn round-trip)** — shipped `[v0.21.26]`. A `choice`
      element (`{type:'choice', prompt?, options[]}`) renders option buttons; clicking one submits
      it as the user's **next turn** via the existing send path (`ChatPanel.submitChoice`, a
      `useCallback`-stable mirror of `handleSend` incl. `workflow_id`, busy-guarded against
      in-flight runs). Backend: `ChoiceElement` in a Pydantic **discriminated** `Element` union
      (options stripped/de-duped/non-blank, ≤10); `present_exhibit` schema/description widened.
      Client: element-renderer contract refactored to pass the full typed element + context
      (`ElementRenderProps`); `MermaidElement` memo keyed on element identity so choice
      callbacks/flags don't thrash diagram re-renders; `answeredValue` on the exhibit message
      disables/marks the choice (persists via localStorage; cleared on amend). Tests: backend
      choice cases + `ChoiceElement` component test + restore case. Single-agent next-turn model;
      blocking in-run variant → Backlog.
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

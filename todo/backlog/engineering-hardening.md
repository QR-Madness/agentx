# Engineering Hardening

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Engineering Hardening (observed while in the code, Slices 5–6)

> Grounded tech-debt / consistency items noticed during the model-fallback + context work.

- [ ] **Dependency CVE triage + bump** — `task audit` (pip-audit) flags ~30 known CVEs; the
      actionable ones cluster in auth/web-facing deps: **`pyjwt`** (×8 → 2.13.0), **`django`**
      (CVE-2026-25673 → 5.2.12), **`python-multipart`** (→0.0.27), `requests` (→2.33.0), `urllib3`
      (→2.6.3), `starlette`, `python-dotenv`, `sqlparse`, `idna`, `filelock`. A few (`torch`,
      `transformers`) have no fix without a major bump — accept/track separately. Do a focused
      bump-and-test pass (not a blind `uv lock --upgrade`); re-run `task test` after. *(Surfaced by
      the static-analysis pass; pip-audit is advisory in `task release:check`.)*
- [x] **Paid down the 48 pyright errors → baseline 0** — the file read `0` but the tree had 48
      pre-existing (proven at the pre-session commit), accumulated because the gate was wired into
      nothing that runs. Cleared all 48 + the +1 I'd introduced: hoisted the `generate_sse`
      persist-handler vars to bind on every path, narrowed `plan` via `assert`, cast the sync-redis
      `ResponseT`/`smembers` returns, fixed a **real** stale import (`_qa_persona` → `_answer_persona`,
      which would have `ImportError`'d the `/persona-defaults` endpoint live), and converted
      `PromptLayer`/`AmbassadorConfig` positional `Field(x, …)` → `Field(default=x, …)` (pyright only
      honors the keyword form). Baseline now 0 and CI-enforced. **Latent pattern worth a sweep:** any
      pydantic model built with omitted positional-`Field()`-default fields will re-trigger this —
      consider a repo-wide `Field(x,` → `Field(default=x,` pass.

#### Workspace DX gates (from Fable's Super-Refinements — see [Repo-Questions.md](../../Repo-Questions.md))

> Built already: anchor-aware link checking + CLAUDE.md context-budget gate (`check_docs.py`),
> the gate-script protocol doc (`scripts/README.md`). The rest, by leverage:

- [x] **(WS-1, highest) CI workflow (`ci.yml`)** — **shipped** (push to master + PRs): `uv sync
      --frozen` + `bun install --frozen-lockfile` (lockfile checks) → ruff · pyright-baseline · tsc ·
      docs:check (links/anchors/parity) · test:quick (WS-4 made it deterministic). Python pinned
      3.14 to match the baseline. **Remaining:** (a) the workflow is
      **unverified in Actions** (couldn't run CI locally) — first push may need an action-version
      bump (`setup-uv`/`setup-bun`/`setup-task`); (b) optionally turn on branch protection requiring
      the `gates` check.
- [x] **(WS-2) `task check:fast` + opt-in `task hooks:install`** — **shipped.** `check:fast`
      (~0.2s) = `docs:check` + `lint:python` + Release-Notes-marker check; `hooks:install` writes a
      plain 3-line `.git/hooks/pre-commit` calling it (no pre-commit framework — see Decisions
      "Rejected"; bypass via `git commit --no-verify`). Opt-in, so it's non-cumbersome. Catches drift
      at commit time, not release-time archaeology.
- [x] **(WS-6) `scripts/check_config_keys.py`** — **shipped** as `task lint:config-keys` (advisory
      scout, ast-parses `DEFAULT_CONFIG`). Read pattern is **scoped to `config.get()` handles** so the
      prompt-loader / plain-dict `.get("a.b")` calls don't masquerade as config reads (that scoping was
      essential — unscoped it flagged `loader.get("planner.decompose")` etc.). `--inventory` emits the
      75-leaf key list = the **Settings Manifest seed**. *Dead-knob axis is opt-in (`--dead-knobs`):
      noisy because many keys are read by f-string (`f"providers.{p}.{k}"`) — would need f-string-prefix
      detection to be gate-worthy.* Standalone/advisory, not wired into a gate.
- [x] **Declare the undeclared ambassador config keys** — **shipped `[v0.21.227]`.** Added
      `ambassador.{speech_model,voice,transcription_model}: None` to `DEFAULT_CONFIG`
      (behavior-preserving — the resolution chain still falls to the shipped code default);
      `lint:config-keys` no longer flags them as read-but-undefined.
- [~] **(WS-7) client static-analysis parity** — **`bun audit` shipped** (`task lint:client:audit`
      + advisory in `release:check`, mirroring `pip-audit`), and removed one dead barrel
      (`components/memory/index.ts`). **knip deferred:** its high-signal output here is mostly false
      positives needing an ignore-list — `@tauri-apps/plugin-opener` (used Rust-side), `react-hook-form`
      (staged for the planned `form` exhibit), `concurrently` (used by `task dev`, not client src),
      `@types/*` (needed by tsc) — and 152 "unused export/type" findings are noise for a typed API
      client. Worth adding *with* a tuned `knip.json` (rules: `files`+`dependencies` only, +
      `ignoreDependencies`) — not lean to force now.
- [ ] **Client dependency CVE triage** — `task lint:client:audit` flags 8 (1 critical / 4 high / 3
      moderate), mostly **dev-only** (`vite` dev-server path-traversal/file-read, `picomatch` ReDoS).
      A Tauri-bundled prod app doesn't ship the vite dev server, so impact is low; bump on the next
      client-dep pass (`bun update`) + re-run `tsc`/`vitest`. Advisory (non-blocking) in `release:check`.
- [x] **(WS-4) `test:quick` deterministic + in CI** — the "7 pre-existing 401 failures" were **not
      inherent**: `MCPClientTest` inherited ambient `AGENTX_AUTH_ENABLED` (true in this dev's local
      `.env`) and didn't authenticate. Fixed the *real* way — not a baseline/`expectedFailure`, which
      would have gone red in clean CI on "unexpected success" — via `@override_settings(
      AGENTX_AUTH_ENABLED=False)` on `MCPClientTest`, matching the 4 sibling endpoint classes.
      `task test:quick` is now green regardless of local auth and **runs in CI**. *(If the full `task
      test` is ever CI-gated, sweep the same override onto any other endpoint test that 401s.)*
- [×] **(WS-5) TODO-comment pointer-or-perish** — **won't build (doesn't earn its keep).** The whole
      codebase has **2** inline `TODO`s (`mcp/client.py` config-search, `views.py:4095` auth placeholder);
      a gate + ratchet for 2 markers is over-tooling. Revisit only if `grep -rE "TODO|FIXME"` climbs into
      the dozens.
- [~] **Type the plan executor's subtask status (kill stringly-typed sentinels)** — **derivation
      centralized `[v0.21.227]`.** Added a typed `SubtaskStatus(StrEnum)` + shared `classify_result()`/
      `subtask_status()`/`build_plan_card()` in `planner.py`; the scattered `result.startswith("[FAILED")`
      reads (`plan_executor.py` `_completed_count`/`_subtask_status`/`_resume_plan_summary`/synthesis/
      dependency-skip + the byte-identical `views.py` plan-card duplicate) now route through them, and the
      two duplicated plan-card builders collapsed into one `build_plan_card`. **Deferred (own pass):** the
      *authoritative `status` field on `Subtask`* — the result string still carries the sentinel (it's the
      in-memory encoding bridged to Redis by `plan_state._overlay_result`); making the field authoritative
      means threading it through every mutation + the resume overlay + `to_dict`/`from_dict`, a riskier
      change kept out of the housekeeping bundle.
- [ ] **Unify the plan executor's sync/async engines (optional)** — `execute()` and `execute_streaming()`
      duplicate the subtask loop (select → run → mark/handle-failure → safety-net → synthesize). Parity is
      now restored (safety net mirrored), but the two skeletons can still drift. Sharing the loop body is
      awkward across sync/async; revisit only if `Agent.run` (the sync path) grows.
- [x] **Extend the universal model fallback to the remaining feature sites** — **shipped `[v0.21.92]`
      (Foundation #4a).** Routed the chat stream, reasoning (CoT/ToT/ReAct/Reflection), drafting
      (candidate + pipeline stages), `agent/planner.py`, `plan_executor.py`, `agent/core.py`,
      conversation summarization, the prompt enhancer, and `alloy/executor.py` through
      `resolve_with_fallback`/`complete_with_fallback`. Specialized roles (speculative draft/target,
      ambassador TTS/STT) and the explicit availability probes (`validate()`, cost estimation) stay
      strict by design. **Two follow-ups split out below** (sub-model `preferred_fallback` threading;
      streaming first-call resilience).
- [ ] **Thread the agent model as `preferred_fallback` at reasoning/drafting sub-model sites** —
      Foundation #4a routed these through `resolve_with_fallback` but with **no `preferred_fallback`**,
      so a broken sub-model (`cot_config.model`, a drafting `stage.model`, `verifier_model`) falls
      straight to the *global* default rather than the agent's own working model. The reasoner/drafter
      classes hold only their own config (`CoTConfig` etc.), no handle to the agent's `default_model` —
      threading one is a constructor-signature change across the reasoning + drafting subsystems
      (deferred from #4a as scope creep). Worth it so a sub-feature degrades to the *active agent* model,
      not the global floor. Cross-file; own pass.
- [ ] **Streaming model-fallback is first-call-fragile** — the streaming sites (chat `views.py`, plan
      resume, Alloy specialists) use `resolve_with_fallback`, which only skips providers **already
      cached-unhealthy**; unlike `complete_with_fallback` it has no runtime-retry, and a stream-only path
      never *feeds* the health cache. So the **first** turn against a configured-but-down provider still
      hard-fails (it's only caught once a failed call or a `/providers/health` poll marks it unhealthy).
      Minimal fix: wrap the stream kickoff so a provider/connection error **before first token** marks the
      provider unhealthy + retries `resolve_with_fallback` once. Full **mid-stream** failover is the
      separate deferred "hard interrupt" work (abort + resume the in-flight provider stream). Pairs with
      the proactive provider-health refresh item below (a background refresh would also pre-empt this).
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

### Streaming engine — code-health report (2026-07-14, post-goldens review)

> State assessment from the golden-corpus session (v0.21.222). **Verdict: mid-refactor healthy.**
> The load-bearing seams are real, small, and now contract-tested — the event bus as the single
> transport is the architecture's best decision (the goldens exploited it directly), the driver
> now owns termination, steering/work-order folds are isolated pure-ish helpers, and
> `streamReducer.ts` (195 lines, pure) + `dispatchSseEvent` (one exported switch) are exemplary.
> The debt is concentrated in two fat generators — and the corpus's strategic value is precisely
> that it converts them from "scary monolith" into "refactorable monolith". Ranked:

- [ ] **`_run_tool_loop` is ~430 lines of one async generator** (`streaming/tool_loop.py`, 1510
      total) — round loop + steer folds + work-order barrier + auto-continue + finalize nudge +
      exhaustion synthesis as inline branches. At the edge of readability; the NEXT feature that
      touches it should first extract the would-end branch (steers → length → barrier → nudge)
      into helpers. Sibling `_execute_and_emit_tools` mixes SSE emission, persistence capture, and
      exhibit derivation — same treatment.
- [ ] **`views.py generate_sse` is the real monolith** (views.py 8.5k lines; the chat generator is
      its biggest block, with recurring "factored out to stay under the type-checker's complexity
      budget" comments admitting it). Streaming concerns keep accreting in views.py (executor
      attach, persist orchestration, plan branch). Long-term: extract a `streaming/chat_turn.py`
      family — a big, risky move to be done BEHIND the golden corpus, which now makes it safe.
- [ ] **`useChatStream.ts` (800) / `ChatPanel.tsx` (1750)** — the hook mirrors state into 9 refs
      (documented, but it's a hand-rolled sync layer) and now carries two parallel delegation id
      maps; the append/update message-store semantics are re-implemented as a test double in the
      goldens. Extract a tiny shared message-store module (one implementation, used by
      `useTabMessages` and the tests) before the next handler lands.
- [ ] **Small duplications**: `_sse()` exists in both `alloy/executor.py` and
      `streaming/tool_loop.py`; `formatCost`/`formatDuration` are modal-local in
      `AlloyRunTraceModal` + `WorkOrderCard` vs the shared `lib/format.ts` (unify on lib/format);
      the second SSE endpoint's own `close` (views.py ~4400) is driver-deduped like the chat one
      but the pattern deserves one shared terminator.

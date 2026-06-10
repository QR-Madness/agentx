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
- [ ] **(WS-2) `task check:fast` + opt-in `task hooks:install`** — the sub-5s subset (docs+parity+
      ruff-on-changed+notes-marker) behind a plain 3-line `.git/hooks/pre-commit` (no pre-commit
      framework — see Decisions "Rejected"). Catches drift at commit time, not release time.
- [ ] **(WS-6) `scripts/check_config_keys.py`** — cross-reference config keys (defined in
      `config.py`/`memory_settings`/YAML) against read sites; flag defined-but-unread + read-but-
      undefined (the `_global`-vs-`_default` bug class). Warnings-first. Doubles as the Settings
      Manifest seed inventory; folds into Foundation #6's dead-knob sweep.
- [ ] **(WS-7) client static-analysis parity** — `knip` (unused exports/files/deps in `client/src`)
      + `bun audit` as `lint:client:audit`, beside the Python `ruff`+`pip-audit`. Single-config;
      knip warnings-only until the first sweep.
- [x] **(WS-4) `test:quick` deterministic + in CI** — the "7 pre-existing 401 failures" were **not
      inherent**: `MCPClientTest` inherited ambient `AGENTX_AUTH_ENABLED` (true in this dev's local
      `.env`) and didn't authenticate. Fixed the *real* way — not a baseline/`expectedFailure`, which
      would have gone red in clean CI on "unexpected success" — via `@override_settings(
      AGENTX_AUTH_ENABLED=False)` on `MCPClientTest`, matching the 4 sibling endpoint classes.
      `task test:quick` is now green regardless of local auth and **runs in CI**. *(If the full `task
      test` is ever CI-gated, sweep the same override onto any other endpoint test that 401s.)*
- [ ] **(WS-5) TODO-comment pointer-or-perish** — gate inline `TODO`/`FIXME` to carry a doc pointer
      (`TODO(todo/backlog/foo.md): …`); enforce as a ratcheted count so `grep TODO` and the `todo/`
      tree can't disagree about what's owed.
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

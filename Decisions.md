# Decisions & Invariants

> **Purpose:** the load-bearing decisions and invariants that future-me must **not relitigate or
> accidentally reverse**. When a change seems to violate one of these, stop — either it's a mistake,
> or this entry is out of date and should be edited *in the same change* with the reasoning.
>
> Two kinds of entry:
> - **Invariants (INV-n)** — properties that must always hold. Each names its **guard** (the test
>   that fails if it breaks). `⚠ no guard` = a property currently protected only by prose — a
>   candidate for a collected `tests_invariants.py` (Memory-Roadmap T3 / [`todo/backlog/`](todo/backlog/)).
> - **Decisions (ADR-n)** — a path chosen over alternatives, with the *why* (so the rejected option
>   isn't quietly re-attempted).
>
> Linked from the [CLAUDE.md Documentation Map](CLAUDE.md#documentation-map). The mechanical half of
> anti-drift (links, version, index) is enforced by `task docs:check`; this file is the *judgement*
> half a script can't check.

---

## Invariants

### INV-1 — Memory is optional; it never blocks a turn
Every memory op degrades gracefully when Neo4j/PG/Redis are down. A failed recall/extraction/
consolidation must not fail the chat turn. (Memory-Roadmap §0.1 formalizes this into L0–L3 levels.)
**Guard:** memory integration tests skip-gracefully without Docker (`tests_memory.py`); per-stage
fallback via `resolve_with_fallback`. *Partial — the explicit L-level contract (§3.11) is not yet tested.*

### INV-2 — The Ambassador never writes to `conversation_logs` / `conv_summary:`
It is a parallel operator on a Redis **sidecar** only; its tool belt is **SELECT-only**
(`execute_tool` never mutates). This is the feature's whole reason to exist — breaking it pollutes
the main transcript. The **aide swarm** (16.7) upholds this: aides only *read* + call a model and
return a string, and their digest cache lives under `amb_aide:` (the sidecar family), never
`conv_summary:`/`conversation_logs`. The **dispatch** write-side (16.7) upholds it too: it enqueues a
real **user** turn (you authored it) into the *worker's* brand-new conversation via
`enqueue_background_chat` — the ambassador never writes a transcript as itself.
**Guard:** `AmbassadorStorageTest` (pollution regression, recovery-isolation; incl. aide-cache
isolation); `AmbassadorDispatchEndpointTest` (dispatch routes only through the background-chat worker).

### INV-3 — `agent_id` is the durable identity key; names are aliases
Rename-safe: a profile rename propagates to the Agent entity's `aliases`; `dedupe_entities` **never
merges `Agent` nodes** (keyed by `agent_id`). Never fabricate an `agent_id` — unknown names demote to
`third_party`.
**Guard:** multi-agent attribution tests (`AgentSelfMemoryTest`); `dedupe_entities` Agent-skip path.
*Verify the rename→aliases propagation has a direct assertion — ⚠ may be guard-thin.*

### INV-4 — Exports are text-only; embeddings are always derived
Memory export carries text, never vectors. Import re-embeds from text (portable across embedding
models). Embeddings are a rebuildable projection, never canonical.
**Guard:** `MemoryPortabilityTest` (text-only-export + recompute, round-trip, idempotency).

### INV-5 — CoT / thinking is process, not result — never persisted
Reasoning/thinking streams live (shown collapsed) but is **never** written to `conversation_logs`.
(Old turns persisted before this rule still restore stored `metadata.thinking`.)
**Guard:** ⚠ no dedicated test — the persistence builders in `streaming/persistence.py` enforce it by
construction. Candidate for `tests_invariants.py`.

### INV-6 — Durable writes belong to consolidation
The hot path (recall/synthesis) may **queue** discoveries but never mints durable facts inline
(pollution + attribution risk). Consolidation owns all durable writes.
**Guard:** ⚠ no test — design invariant. Active-Recall Tier 3 ("keep it ephemeral",
[`todo/backlog/memory-recall.md`](todo/backlog/memory-recall.md)) depends on it.

### INV-7 — Channels are traceable scopes, not isolation boundaries
…except `_self_{agent_id}`, which **is** an attribution boundary. Scope hierarchy
`_global → user → project → _self_{agent} → conversation`; recall prefers most-specific. Recall
searches `[active_channel, _self_{agent_id}, _global]`.
**Guard:** recall/attribution tests (`tests_memory.py`). *Default-channel drift (`_global` vs `_default`)
is a known hazard — Memory-Roadmap §1.4 `ChannelRef` is the fix.*

### INV-CTX-1 — No silent context loss
A turn may leave the model's verbatim view **only when covered** by the persisted compaction target;
the JIT pre-assembly refresh (`views.py::_ensure_summary_coverage`) sizes that compaction to the turns
`fit_history` is about to drop, **before** assembly. When the summarizer is unavailable, the dropped
turns must instead appear in the deterministic `history_digest` block that same turn (no model call,
never silence). Rehydration surfaces coverage gaps (`session.metadata["history_overflow"]` when the
row cap is hit with no restored summary) rather than truncating silently. The verbatim ceiling
(`context.verbatim_budget_ratio`, 0.9) and the post-turn summary trigger
(`context.summary_trigger_ratio`, 0.85) are **separate knobs** — the old 0.7 double duty is gone.

**Single compaction target (Slice 1c, v0.21.179).** The compaction target is now the structured
**conversation-state object**, not the prose rolling summary: `_ensure_summary_coverage` calls
`SessionManager.maybe_compact_to_state`, which rolls aged-out turns into `ConversationState.digest`
(rendered by the `conversation_state` ledger block, priority 68 — registered right **after** the
coverage call, and above the legacy summary's 65, so coverage is retained longer). Rules that must
not drift: (a) **exactly one** compaction pass runs per over-budget turn — state-compaction OR the
legacy prose summary (`context.conversation_state_compaction_enabled`, default ON; the prose path is
the flag-off/legacy fallback), **never both** (no double-compression); (b) the digest is a **single
rolling field re-summarized in place**, NOT the append-capped `narrative` slot — so it stays bounded
without the newest-N cap ever dropping old coverage; (c) the prose `session.summary` block stays only
as **read-back for pre-1c conversations**; (d) the `history_digest` deterministic fallback still fires
whenever the structured pass fails. **Guard:** `ConversationContextTest` JIT-coverage tests +
`ConversationStateTest` digest/compaction tests (`tests.py`).

---

## Decisions

### ADR-1 — Tool execution stays synchronous (no off-thread / asyncio)
**Decision:** the streaming tool loop runs tools synchronously; cancellation is **cooperative**
(`GeneratorExit` lands at a `yield`), made prompt by capping tool wall-clock (e.g. `search.timeout`).
**Why:** the off-thread attempt (`run_in_executor` + `asyncio.shield`) **deadlocked `gen.aclose()`** —
Stop hung and turns never persisted. Rebuild a truly-instant mid-tool cancel only with a design that
doesn't block generator close, and **reproduce the hang in a test first**.
**Source:** Todo Phase 15.10 (`todo/phases/phase-15-plan-execution.md`).

### ADR-2 — The Ambassador thread is entry-oriented, not message-split
**Decision:** one record per ambassador turn (carrying its optional `question`), ordered by
`created_at`, under `amb_thread:{thread_id}`. The pre-1b per-family API is preserved as **thin
projections** over the entry store.
**Why:** preserves in-place streaming + briefing idempotency; a role/content message list would have
broken both. **Source:** Todo §16.7 Slice 1b.

### ADR-3 — Ambassador tools are always on (no gate)
**Decision:** the read-only tool belt is unconditionally available; grounding is **tool-first**
(pre-load only `_LEAN_GROUNDING_TURNS`, the model *reads* for depth).
**Why:** experimental app; the earlier gated/pre-stuffed design meant the tools never fired (the model
already had the whole transcript). Matches the workspace-wide "ship experimental features ON" stance.
**Source:** Todo §16.7 Slice 2.

### ADR-4 — Global system prompt is a layered stack; `effective = override ?? default`
**Decision:** built-in layers ship a versioned `default` overlaid by a user `override`; **global stack
only** (per-agent prompt stays in the profile), debounced autosave, **no named presets in v1**.
**Why:** untouched layers keep getting release improvements while edits are pinned and never silently
overwritten — fixes the original lost-on-restart durability bug. **Source:** Todo §18.13.

### ADR-5 — Model resolution falls back everywhere; a chat swap is surfaced, not silent
**Decision:** **(updated, Foundation #4a)** *every* model-using site resolves through
`resolve_with_fallback` / `complete_with_fallback` so a missing/unhealthy provider never hard-fails the
turn — including the main chat path, which now falls back **but emits a `model_fallback` status notice**
so the swap is visible rather than silent. Strict-by-design exceptions (a generic chat fallback would be
*wrong*): speculative-decoding draft/target, ambassador TTS/STT, and explicit availability probes
(`validate()`, cost estimation). Kill-switch: `models.fallback_enabled`. *(Supersedes the original
"chat stays strict" decision — surfacing the swap resolved the "silent swap is worse" concern.)*
**Why:** crashing a turn because one provider blipped is the worse failure; visibility (the notice)
addresses the original objection. **Source:** Model Resolution (`Development-Notes.md`); ModelFallbackTest.

### ADR-6 — `conversation_logs` is the transcript record; memory extraction is decoupled (direction)
**Decision (target):** a turn's durable truth is its `conversation_logs` row; graph/vector/working-
memory are **rebuildable projections**. "No Memorization" = log without projection.
**Why:** the dual-write gap already bit us once (the 15.10 restore-404). This is the spine of
Memory-Roadmap §1.1; recorded here as the *intended* direction so new write paths don't deepen the
dual-write coupling. **Status:** aspirational — not yet the implemented write path.

### ADR-7 — Ruff is configured lean; rules that fight our idioms are off by design
**Decision:** `[tool.ruff.lint]` selects a *curated* set (`E4/E7/E9,F,B,C4,PIE,UP,ASYNC,S`), not
"everything." Deliberately **off**: `E501` (line length — the codebase uses a compact-block style,
[[project_ruff_check_not_format]]); `S110`/`S112` (try/except/pass *is* the never-raise idiom, INV-1);
`S311` (random is agent-id/jitter, never crypto); `S101` (asserts are deliberate). `S608` is
**per-file-ignored** on the three audited raw-SQL files (`views.py`, `memory/semantic.py`,
`eval_consolidation.py`) — their SQL binds all values (`%s`/`:name`/Cypher
`$params`) and interpolates only static fragments/identifiers; per-line `noqa` was rejected because
the violations sit on multi-line `f"""` openings (can't append a comment without churning hot query
code). `S608` stays active everywhere else to catch *new* raw SQL.
**Why:** in a utilitarian repo, a linter that emits hundreds of findings against intentional idioms
trains you to ignore it. Lean rules that *earn their keep* stay green and get read. The real fix for
the raw-SQL surface is the §1.5 repository layer (Memory-Roadmap), not `noqa`-decoration.
**Guard:** `task lint:python` runs clean at HEAD; `task docs:check` + `task audit` (pip-audit CVEs)
round out the gate. **Source:** this session's static-analysis pass; pyproject comments point here.

### ADR-8 — OpenAPI is spec-first and hand-maintained; no DRF/generator
**Decision:** `OpenApi.yaml` stays the authoritative, hand-written contract; **code is checked
against it**, never generated from it. Path drift is caught by `check_api_parity.py` (in
`task docs:check`); response-shape drift will be caught by `schemathesis` running
`response_schema_conformance` against the dev server (planned `task api:spec:test`, non-gating
first). Promote the parity check to `--strict` (undocumented routes → error) once stable;
`ALLOW_UNDOCUMENTED` is the escape hatch.
**Why:** this codebase has **no serializers** (responses are hand-picked dicts —
[[project_profile_serialization_handpicked]]), so drf-spectacular would mean writing a serializer
layer for ~111 routes *just to document* — a large invasive rewrite that still can't express the
SSE/multipart/stream surfaces where drift actually hurts. A generator would also make code
authoritative and demote the spec to an artifact, contradicting the spec-first posture every other
doc here takes. Endpoint count doesn't flip this; only a public/multi-tenant API would (the authz
rewrite would carry DRF then). **Source:** [Repo-Questions Q2](Repo-Questions.md#q2--openapi-without-drf-hand-maintained--parity-check-or-adopt-a-generator).

### ADR-9 — PostgreSQL schema is Alembic; Neo4j stays on the home-grown runner
**Decision:** the memory **Postgres** schema is managed by **Alembic** (`alembic upgrade head`; wired
into the Docker entrypoint + `task db:migrate:pg`). The cutover **adopted the existing schema in place**:
revision `0001_baseline` applies the frozen `alembic/baseline.sql` (the former `postgres_builder.sql`,
all `CREATE … IF NOT EXISTS`), so `upgrade head` no-ops a populated DB and only stamps `alembic_version`.
Every future PG change is a **new revision**; `alembic/baseline.sql` is **frozen** (sha-gated by
`scripts/check_alembic.py`, which also enforces a **single head** — both run in `task docs:check`).
**Neo4j** keeps the home-grown Cypher runner (`migrate_schema`, now Neo4j-only): the Python Neo4j
migration libs are 0.1.x/single-maintainer and `neo4j-migrations`/Liquibase are JVM — wrong for a
`uv`/`bun` stack with ~3 graph migrations.
**Why:** the bootstrap-vs-migrations **dual source** let `usage_events` ship into the baseline file yet
never reach a running DB until a manual re-init. Alembic gives one forward-only, auto-applied PG system;
the frozen baseline + single-head gate make that class of bug impossible.
**Caveat:** the baseline assumes a DB is fresh **or** already at the pre-Alembic head — it doesn't
re-run intermediate `ALTER`s; continuous auto-init means deployed instances already are.
**Source:** this session; `alembic/README`.

### ADR-10 — Docker Compose is the deployment engine; the manager owns overlay assembly
**Decision:** deployments stay on **Docker Compose**; the **deployment manager** (`manager/` —
`agentx-manager` CLI + web GUI container, one testable core) is the single owner of overlay assembly.
A cluster's shape is an **explicit persisted spec** (`.manager-state.json`: kind `source|image`,
gateway / tunnel `none|token|named` / expose / gpu / shell), not file-presence sniffing; every cluster
runs under its own compose project (`-p agentx-<name>`, fixing the shared-default-project collision);
tracked single-file bind mounts are **hashed**, so `restart` force-recreates services whose config
changed (the nginx.conf inode gotcha, fixed by construction). The isolation axis collapses to
`kind=source|image` in one codepath — `task cluster:*` are thin wrappers.
**Why:** compose already carried the product end-to-end (self-init entrypoint, in-image CLI,
profiles); the gaps were assembly UX, lifecycle safety, and observability — a manager problem, not an
orchestrator problem.
**Security invariant:** the manager drives the Docker socket (root-equivalent) → loopback bind +
bearer token by default; non-loopback requires explicit `AGENTX_MANAGER_BIND`; it is **never** routed
through the gateway/tunnel.
**Source:** deployment overhaul plan (this session); `manager/README` posture in
`docs-site/.../deployment/manager.md`.

### ADR-11 — Model roles are an implicit overlay tier; explicit values always win
**Decision:** the scattered utility-model settings cluster into three roles
(`models.roles.{fast_utility,deep_reasoning,summarizer}`, single source `agentx_ai/model_roles.py`).
The role tier is **implicit**: a member follows its role only while its own value is empty/"inherit";
_setting_ a role never rewrites member keys (they would destroy custom values); an
unset role makes the tier a byte-identical no-op. `role:<name>` is a valid explicit value anywhere;
the registry expands it defensively so the sentinel never reaches a provider lookup. Roles produce ONE
concrete requested model — ADR-5 fallback + swap-surfacing run after, unchanged. Membership excludes
planner/ambassador-answerer/images (their empty means "follow the agent model", a different semantic —
bucket b below); the **ambassador aide is a member** (`fast_utility`).
**Coverage invariant (v0.21.176):** every general-purpose LLM feature-model setting must resolve
through a family — a `ROLE_MEMBERS` source, OR one of two documented lists in `model_roles.py`:
`INHERITS_AGENT_MODEL` (bucket b — falls through to the calling agent's own model) or
`EXEMPT_SPECIALIZED` (bucket c — embeddings/reranker/image-gen/TTS-STT + the speculative
draft↔target matched pair). A `*_model` with no bucket fails the guard. Consolidation stage defaults
flipped `lmstudio:…` → `"inherit"` (they were shadowing a set role); an **explicit, user-initiated**
`POST /api/models/roles/adopt` clears an existing install's concrete stage overrides so they follow
the role — distinct from role-setting (user-triggered, clears-to-inherit, never writes a role's value
into a key).
**Why:** ~15 "which model runs this internal job?" knobs across two stores were unintelligible; an
overlay keeps every existing chain intact while giving users three meaningful levers — and a member
that ships pinned to a concrete model silently defeats the overlay.
**Guard:** `ModelRolesTest` (behavior-preservation matrix, sentinel containment, role-to-role refs
ignored); `ModelFamilyCoverageTest` (three-bucket coverage + consolidation defaults inherit);
`ModelRolesEndpointTest` (incl. the adopt action). **Source:** settings overhaul D1 (user-locked:
implicit tier) + Slice 0 model-family audit.

---

## Rejected — do not relitigate

Options weighed and declined (with the reason, so they don't return as "good ideas"):

- **DRF / drf-spectacular migration** — no serializers exist; it's a rewrite that documents nothing
  the spec-first posture doesn't already cover. See ADR-8 / Repo-Questions Q2.
- **A code-derived OpenAPI generator** — would make code authoritative and demote the hand-written
  spec to an artifact, against the repo's spec-first posture.
- **Coverage mandates / line-coverage thresholds** — measure lines, not safety. The eval harnesses
  (`eval_consolidation`, planned `eval_recall`) are this repo's real quality instruments.
- **pre-commit framework / husky / commitlint** — each imports a config ecosystem to do what a
  3-line git hook + the existing gates already do (see the planned `task hooks:install`).
- **k3s / Kubernetes / Helm as the deployment story** — a real orchestrator buys self-healing and
  declarative state at the cost of forcing k3s onto every self-hoster and rewriting the entire deploy
  path (ingress, PVCs for bind-mounted data, the dind shell backend). Compose + the manager covers the
  actual gaps (assembly, lifecycle, GUI). Revisit only if multi-node scheduling becomes a real
  requirement; a community Helm chart could then wrap the same images. See ADR-10.
- **Monorepo orchestrators (nx / turbo / moon)** — Taskfile is already the orchestration layer;
  a second task runner is drift by construction.
- **CODEOWNERS / PR-template machinery** — human-team review-routing ceremony; in an agent-first
  workspace the Documentation Map is the ownership model and gates outrank reviewers.
- **Storing the memory degradation L-level on a capability/store** — it's derivable from `stores`;
  a stored copy can contradict it (Repo-Questions Q1).
- **Deriving bitemporal `valid_to` from `expected_stability`** — invents world-time data and
  silently re-ranks recall (Repo-Questions Q4).

> Source: Fable's "Considered and rejected" list (Repo-Questions) + the Q1–Q4 resolutions.

---

## Maintenance

- Add an entry when you make a decision you'd be annoyed to see re-litigated, or discover a property
  that must hold. Keep each to a few lines; link the test and the `todo/`/roadmap item.
- When you add an INV with `⚠ no guard`, also drop a line in the relevant `todo/backlog/` file so the
  guard gets written (the `tests_invariants.py` collection is Memory-Roadmap T3).
- If a change *intentionally* reverses an entry, edit the entry in the same change with the new why —
  don't leave it contradicting the code.

# AgentX Memory System — Improvement & Experimental Roadmap

> **Audience:** Claude Code working in this repo. Canonical plan for hardening and advancing
> the memory system (`api/agentx_ai/kit/agent_memory/` + consumers in `views.py`, `agent/`,
> `streaming/`). **Read alongside the TODO** — this doc is reconciled against it as of
> **v0.21.88** and must stay that way. Keep both updated per the docs-maintenance rule in
> `CLAUDE.md` and `architecture/memory-capabilities.md`.
>
> **The TODO is now split** (v0.21.88): [`Todo.md`](Todo.md) is the index, detail lives under
> [`todo/`](todo/). This roadmap's memory items pair with the backlog files
> [`todo/backlog/memory-recall.md`](todo/backlog/memory-recall.md),
> [`procedural.md`](todo/backlog/procedural.md),
> [`retrieval-extraction.md`](todo/backlog/retrieval-extraction.md),
> [`conversation-context.md`](todo/backlog/conversation-context.md), and
> [`open-platform.md`](todo/backlog/open-platform.md). §7 maps every area and lists the open
> reconciliation items.
>
> **Status tags:** `[NEW]` not tracked anywhere else · `[REFINES: …]` extends a TODO item
> (link to it, don't duplicate it) · `[TRACKED: …]` already in the TODO — listed only for
> sequencing context, do the work under the TODO item · `[PARTIAL]` some half shipped.

---

## 0. Core Invariants (do not break)

1. **Memory is optional.** Every memory op degrades gracefully when Neo4j/PG/Redis are down;
   nothing memory-related may block a chat turn. (§3.11 formalizes this into levels.)
2. **Channels are traceable scopes, not isolation boundaries** — except `_self_{agent_id}`,
   which IS an attribution boundary. The procedural-memory scope hierarchy
   `_global → user → project → _self_{agent} → conversation` (Todo §Procedural) is the
   canonical ordering; recall prefers most-specific.
3. **`agent_id` is the durable identity key; names are aliases** (rename-safe via Agent-entity
   aliases; `dedupe_entities` never merges Agent nodes).
4. **Exports are text-only; embeddings are always derived, never canonical.**
5. **The Ambassador never writes to `conversation_logs`/`conv_summary:`** (sidecar only;
   its 16.7 tool belt is SELECT-only). The no-pollution regression tests guard this.
6. **CoT/thinking is process, not result** — never persisted.
7. **Durable writes belong to consolidation.** Hot-path recall/synthesis may *queue*
   discoveries but never mints facts inline (locked in Todo §Active Recall "keep it ephemeral").

---

## 1. Structural Fixes (Tier 1 — stability)

### 1.1 Event-sourced memory: `conversation_logs` as the write-ahead log `[NEW — elevates Todo "Known Future Issues: Distributed Transaction Support"]`

Todo.md rates the dual-write gap "LOW for single-user; HIGH for multi-user" — but the
15.10 restore-404 bug (one failing shared resource dropped the PG row; patched with a
per-turn fallback in `_persist_turns`) proves it's already a single-user problem. The same
class recurs wherever two stores must agree (orphaned `ABOUT` edges needed the
`link_facts_to_entities` repair; rename propagation is best-effort).

**Change.** Invert the write path:
- A turn write **succeeds when the `conversation_logs` row commits.** Nothing else.
- Neo4j graph + vector indexes + working-memory mirrors become **derived, rebuildable
  projections** built by consolidation (which already sweeps unconsolidated turns).
- `manage.py rebuild_memory_projections [--channel] [--since]` reconstructs the graph from
  the log — subsuming dual-write repair, embedder migration, and graph-schema evolution.
- This also resolves Todo §Engineering "Decouple transcript persistence from memory
  extraction": a transcript-only durable record *is* the log; "No Memorization" = log
  without projection.

**Sequencing — RESOLVED ([Repo-Questions Q3](Repo-Questions.md#q3--event-sourced-write-path-11-the-no-flag-day-cutover)):**
the 3-step shape holds, with two amendments. **Step 0 (do first):** the log isn't
rebuild-complete today — `conversation_logs` has no `turn_id`/`user_id`, so a rebuild would
mint fresh Turn UUIDs and orphan every `Fact.source_turn_id`. Add + backfill both columns *now*,
while the graph is still authoritative to cross-fill from. Then: (1) ship the 1.2 checker;
(2) reorder `store_turn` to PG-first + best-effort Neo4j with a `projected_at` marker — this *is*
the no-flag-day flip (same method, PG authoritative in one commit) and the `_persist_turns`
fallback dies in the **same PR** as proof; (3) extract `project_turn(row)` so the live path and
`rebuild_memory_projections` are **one function, two callers** (fidelity true by construction);
(4) land §2.1's per-turn `(consolidated_at, pipeline_version)` columns + point consolidation's
reads/watermark at PG *before* trusting rebuild as recovery; (5) shadow-rebuild one channel, diff,
then the invariant goes into `Decisions.md`. Riskiest seam = identity/bookkeeping that lives only
in the projection (Turn UUIDs, `c.consolidated` watermark), not content.

### 1.2 Cross-store invariant checker `[REFINES: Todo "Memory-mending agent (memory janitor)"]`

The janitor agent (Todo §Memory Area UX) needs a deterministic substrate before an LLM
proposes mends. Ship a `verify_invariants` consolidation job (daily) asserting:

- Per-conversation turn counts match Neo4j `(:Turn)` ↔ `conversation_logs`.
- No `(:Fact)` with zero `[:ABOUT]` edges (the recovered class — assert the
  `_resolve_fact_entity_ids` fix holds).
- No `[:SUPERSEDES]` cycles; no `superseded_at`/retired fact reachable by recall.
- Every `_self_{agent_id}` channel maps to a live/aliased Agent entity.
- Embedding dims match the active embedder (1.3); `procedure_candidates` aren't stuck
  `pending` beyond N days; **the consolidation worker is alive** (the Slice-1 fix ④ showed
  it can silently not run).

Results → `memory_audit_log` metrics + a Dashboard `SystemStatusStrip` pill. The janitor
agent then consumes checker findings as its work queue (deterministic detection, LLM-proposed
mends, user-approved apply — matching the janitor's review/approve design). Also the natural
home for **verify-on-import** (Todo §Open Platform): run the checker scoped to an import.

### 1.3 Embedder provenance on every vector `[NEW]`

`vector(1024)` is hardcoded (4 Neo4j indexes + 3 PG tables); provider switch = destructive
reset; the 0.92 duplicate gate and `procedural_dedupe_threshold` are silently bge-m3-calibrated.

- Store `(embedder_name, embedder_version, dims)` on every embedded node/row (the export
  envelope already records `embedder` — push it into live stores).
- Similarity thresholds become a per-embedder config map.
- With 1.1, migration = background re-embed projection. Synergy: Todo §Memory-as-VCS's
  planned **per-node content hash** ("recompute only changed embeddings") is the same
  mechanism — build the hash once, both import and migration use it.

### 1.4 Typed `ChannelRef` `[NEW]`

One string property carries ≥5 meanings (`_global`, user, project, `_self_*`, `_alloy_*`),
and defaults have drifted across the codebase (`_global` vs `_default` vs `_all` depending
on the endpoint) — a fact stored under one default and queried under another vanishes.

```python
@dataclass(frozen=True)
class ChannelRef:
    kind: Literal["global", "user", "project", "self", "alloy", "conversation"]
    owner: str | None
    def wire(self) -> str: ...
    @classmethod
    def parse(cls, s: str) -> "ChannelRef": ...
    def recall_set(self, agent_id: str | None) -> list[str]: ...
    def specificity(self) -> int: ...   # the procedural scope-hierarchy rank
```

- Mirrors the procedural scope hierarchy exactly, so procedure scoping (Todo §Procedural)
  and fact scoping share one model with one specificity ordering.
- Per-kind policy: self-channels never promotion targets; alloy channels get
  workflow-lifetime retention; `recall_set` derives `[active, _self_, _global]` instead of
  convention at call sites.
- Workspaces (Todo §⭐ Workspaces) will need the same scoping for `document_chunks` — design
  `ChannelRef` so a workspace id slots in as a kind later.

### 1.5 Repository layer — ban raw Cypher/SQL outside `kit/agent_memory/` `[NEW]`

`views.py` still holds raw psycopg cursors, hand-written Cypher, and direct SQLAlchemy;
tenancy depends on each query remembering `user_id` (`DEFAULT_USER_ID = "default"` + TODO,
and Todo §Open Platform "Per-user export" confirms multi-user is coming). Scoped repository
methods where a query **cannot exist without `(user_id, ChannelRef)`**, + a CI grep banning
`Neo4jConnection.session()` / `raw_connection()` outside the kit. Do this *before* the
Settings Manifest / multi-user waves so the audit surface is one module.

### 1.6 Redis durability tiers `[REFINES: Todo "Redis/Postgres-backed live session store" + "Rolling summary as a first-class conversations column"]`

`allkeys-lru` @ 512MB now backs semi-durable state: checkpoints (7-day, designed to survive
compression), scratchpad, plan snapshots gating resumability, detached-run buffers, the
rolling summary, voice/Q&A sidecars, `amb_thread:` (16.7 Slice 1). Under pressure LRU evicts
any of it — a resumable plan or a checkpoint silently vanishing.

- Promote **checkpoints, plan snapshots, rolling summary** to PostgreSQL (the summary item
  is already tracked; fold checkpoints/plans into the same pass — they're
  `conversation_logs`-adjacent, not cache).
- Remaining Redis splits into documented tiers: `volatile-lru` cache (translation,
  embedding cache, recap, prefetch) vs. `noeviction` state (run buffers, steer queues,
  ambassador sidecars). Resolve the 16.7 open question ("thread persistence depth") with
  this tier decision, not ad hoc.

### 1.7 Bitemporal facts `[REFINES: Todo "Fact Transience", "Fact Staleness Detection", "Negative/Correction Tracking"]`

Three backlog items are the same missing dimension: `temporal_context ∈ {current,past,future}`
conflates *when true* with *when learned*. Replace with:

- `valid_from`/`valid_to` (world time) + `asserted_at`/`superseded_at` (transaction time,
  partially present). Correction = close interval + assert new fact — mechanical, auditable,
  and "what did we believe on March 1st?" becomes a query.
- **Fact Transience / `expected_stability`** becomes a *predicted valid-interval length*
  set at extraction ("user's PC is slow" → short horizon) feeding decay (2.4) and staleness
  warnings — one field family serves all three backlog items.
- Recall's 1.2×/0.7× multipliers become interval-overlap functions; keep the labels as a
  derived view for back-compat.

**Migration — RESOLVED ([Repo-Questions Q4](Repo-Questions.md#q4--bitemporal-facts-17-non-destructive-migration-from-temporal_context)):**
additive only — keep `temporal_context`, never derive `valid_to` from `expected_stability` (that
invents world-time and silently re-ranks recall). Backfill only the honest bound: `current` →
`valid_from=asserted_at`, `valid_to=NULL`; `past` → `valid_to=asserted_at`; `future`/NULL → no
timestamps (label only). Stamp `temporal_provenance='backfill_v1'` so a later pass (generative
replay §3.9) refines only those rows. **Scoring is two-path keyed by provenance** — `backfill_v1`
uses today's label table (bit-identical ranking, by construction), `extracted` uses the overlap
function; default-flip gated on `eval_recall` (§2.7). Corrections set `valid_to=now()` (the one
moment world-time *is* known). Ship as the first numbered migration under §2.6; the importer runs
old exports through the **same** backfill function (one mapping, two callers).

### 1.8 Context Ledger `[SHIPPED v0.21.90 — was: shipped `assemble_turn_context` + Todo Foundation #3 "Stable memory core" + "Context Inspector"]`

v0.21.30 shipped the token-budgeted assembler — but it fit **only the transcript** and kept
every SYSTEM block by fiat, while the preamble kept growing (checkpoints, scratchpad, summary,
memory bundle, participants, reflex core, budget header, soon the workspace manifest). When
sidecars grew, the transcript silently shrank. **v0.21.90 lands the ledger** (`agent/context_ledger.py`).

- ✅ The assembler is now a **ledger**: each contributor registers a `LedgerBlock(priority,
  min_tokens, max_tokens, shrink_fn, mandatory)`; `assemble_ledger` allocates by priority,
  shrinking then dropping under pressure. Ships `shrink_tail` / `shrink_lines_newest_n` (checkpoints
  → newest N) / `shrink_memory_to_facts` (memory bundle → facts-only). `assemble_turn_context` is now
  a thin **wrapper** (all-mandatory blocks) so the other callers stay byte-identical.
- ✅ **Foundation #3 landed here**: the stable high-salience core (`AgentMemory.get_salient_core` →
  cheap non-vector `ORDER BY salience`, maintained-not-searched) is a high-priority persistent block
  (prio 70, with the reflex procedures); query-specific `remember` recall is a lower-priority (30),
  droppable supplement that now yields to the transcript. Recall is deduped against the core by id.
- ✅ **Context Inspector** seam: `LedgerResult.allocations` carries per-block requested/granted/status
  + budget/used totals (logged today; a UI view is the remaining follow-up).
- ✅ Superseded (`superseded_at`) / retired (`temporal_context=past`, salience 0.05) facts are excluded
  from the core at the query; the allocator reads the configurable `context.verbatim_budget_ratio`
  (no hardcoded "70%" in the path).
- ⏭ **Follow-ups:** migrate `agent/core.py` + `alloy/executor.py` to native ledger blocks (they keep
  the wrapper for now); build the **workspace manifest** (Todo §⭐ Workspaces) *as a ledger block from
  day one*; surface the allocation report as a real Context Inspector view.

---

## 2. Quality & Lifecycle (Tier 2)

### 2.1 Per-turn consolidation watermark + pipeline-version stamp `[NEW]`
One per-conversation timestamp forces global reset + full reprocess for any extraction
change. Stamp `(consolidated_at, pipeline_version)` per turn; version bumps make
re-consolidation incremental and let the persisted eval runs (shipped, 18.6) compare
pipeline versions on the same corpus. Also what generative replay (3.9) targets.

### 2.2 Break the salience feedback loop `[NEW]`
Access → salience → rank → access entrenches early facts. Pick ≥2:
- **Score-decomposition logging** per recalled item (vector vs salience vs recency vs
  temporal) into `memory_audit_log` — also feeds the Context Inspector and is the
  prerequisite for 3.5.
- **Usage-based reinforcement:** increment `access_count` only when the fact demonstrably
  mattered — survived ledger allocation, or surfaced in an `active` citation (the shipped
  citation exhibits give this signal for free).
- **Exploration floor:** reserve 1 top-k slot for a low-salience/high-similarity fact.
- **Batch/sample reinforcement writes** (one Neo4j write per item per turn is waste).

### 2.3 Empirical confidence calibration `[NEW]`
`explicit=0.95/implied=0.85/inferred=0.70/uncertain=0.50` is static. The eval harness +
persisted runs exist — fit per-extraction-model calibration (isotonic over adjudicated
outcomes), stored in `memory_settings.json` keyed by `extraction_model`. Subsumes the
Phase-11 deferred "Calibration factors: source, recency, corroboration, contradiction"
(those become features of the fitted model, not more hand constants).

### 2.4 Spaced-repetition decay `[REFINES: shipped Procedure `strength`]`
Procedures already do reinforcement counts; facts still get flat `×0.95^days`. Port the
pattern: `stability` + `last_reinforced` on Fact; successful *use* (2.2's signal) extends
half-life multiplicatively. `expected_stability` (1.7) sets the initial half-life. The
Phase-11 deferred "Negative reinforcement for corrected facts" is the symmetric case:
a supersession event *shortens* the superseded family's stability.

### 2.5 Unified tombstones `[NEW]`
`forget_fact` soft-retires; the `cleanup` job hard-deletes low-salience entities — hard
deletion from a background job conflicts with auditability, the event log (1.1), and
Memory-as-VCS. Unify on tombstones (`retired_at`, recall-excluded, retention-swept); hard
delete only for explicit user `{hard:true}`.

### 2.6 Versioned graph migrations `[REFINES: Todo "Memory capability registry"]`
Backfills are one-off commands ("did this cluster get it?" is unanswerable). Mirror Django:
`graph_migrations/` registry of ordered idempotent Cypher + a `(:SchemaVersion)` node,
applied by `init_memory_schema`/entrypoint. Convert existing backfills into numbered
migrations. The capability registry (Todo, deferred) and this share a shape — a code-side
registry the manifest/docs validate against; build them as siblings.

**Registry schema — RESOLVED ([Repo-Questions Q1](Repo-Questions.md#q1--memory-capability-registry-the-minimal-schema-memory-roadmap-t2)):**
a passive, import-time `@capability` registry in `kit/agent_memory/capabilities.py` whose only job
is to be the *join key* — it stores nothing a consumer already owns. Fields:
`name` (`<layer>.<slug>` — **the one-way door**, doc anchor + checker key), `layer`, `summary`,
`stores: frozenset[Store]`, `flags` (config dot-paths *by reference*, not inlined metadata),
`status`, `degrades_to` (pointer to a thinner capability), `channel_scope`, `entrypoint`
(decorator-filled). **L-level is derived, never stored** (`min_level = lowest L whose available
stores ⊇ cap.stores`; degradation is the `degrades_to` pointer, not a second number). The doc
*matrix* is generated/validated row↔entry (prose stays hand-written); the §1.2 checker reads
`stores`+`flags` for applicability and registers its own assertions; the Settings Manifest joins on
`flags` by path. **When built, the checker must also assert every `degrades_to` resolves to a
registered capability** (cheap integrity check).

### 2.7 `eval_recall` golden-set suite `[NEW — prerequisite for 3.5; sibling of Todo "Debug-harness extensions"]`
`RecallMetrics` exists but there's no retrieval analog of `eval_consolidation` — with 5
techniques (+ cross-encoder reranking and the Active-Recall tiers coming), regressions are
invisible. Seeded corpus + (query → expected-fact-ids) golden set; recall@k/MRR per technique
and fused; runs on the shared snapshot/restore util Todo already wants extracted into a
module. Add the Active-Recall query-rewrite step as a scored stage once Tier 2 ships.

### 2.8 Token estimation `[TRACKED: Todo Foundation #6 — "Consolidate the 4 token estimators (→ tiktoken)"]`
Already tracked; sequencing note only: do it **before** the Context Ledger (1.8), since the
ledger's allocation quality is bounded by estimator accuracy. Feed provider-reported actual
usage back into a per-model ratio as the cheap fallback where tiktoken lacks the tokenizer.

### 2.9 Connection hardening `[TRACKED: Todo "Known Future Issues"]`
Statement timeouts, retry-with-backoff, per-user rate limits on memory ops — all listed in
Known Future Issues. Bundle them into the repository layer pass (1.5): one module boundary
means one place to wrap timeout/retry/limit decorators instead of forty call sites.

---

## 3. Experimental: Reactivity to Complex Situations (Tier 3)

Flag-gated, independently shippable. Reconciled against the two ⭐ Todo tracks — **Active
Memory Recall** (Tiers 1–3) and **Procedural Memory** (Loops 1–3, Slices 0/1 shipped) — which
are the spine; items below either slot into them or extend them.

### 3.0 Alignment map: this doc ↔ the ⭐ Todo tracks

| Todo track | This doc adds |
|---|---|
| Active Recall **Tier 1** (watchdog/stable core) | Lands as a Ledger block (1.8); community summaries (3.7) give it a mid-altitude source |
| Active Recall **Tier 2** (query rewrite + goal/summary anchoring) | Bandit (3.5) learns *when* the rewrite/HyDE spend pays; prefetch (3.4) hides its latency |
| Active Recall **Tier 3** (agentic `deep_recall` as Alloy specialist) | PPR (3.6) is its strongest hop primitive; ephemeral-brief rule = Invariant 7 |
| Procedural **Loop 1** (encode; Slice 0 shipped) | Flash facts (3.2) generalize the same heuristic fast path to *facts* |
| Procedural **Loop 2** (distill = "sleep"; Slice 1 partial) | Generative replay (3.9) is the same job pointed at contested *facts* |
| Procedural **Loop 3** (activate; reflex core shipped) | Reflexion scoring (3.8) closes the loop with a measured violation signal |

### 3.1 Event-driven consolidation (idle trigger) `[REFINES: Todo "Nightly consolidation scheduler"]`
The 15-min sweep means the system is blind to its own last 14 minutes. Debounced
per-conversation idle trigger (no activity for N min + unconsolidated turns → targeted
consolidation; the per-conversation lock machinery already exists). The sweep stays as the
safety net. Fold into the persistent-scheduler work when it lands — idle triggers are just
another registration in that scheduler.

### 3.2 Flash facts (two-speed memory) `[REFINES: shipped Procedural Slice 0]`
`detect_explicit_rule` already proves the pattern: heuristic, no-LLM, hot-path capture.
Generalize to facts: a cheap extractor (preferences, names, corrections) writes
**provisional facts** (`source="flash"`, low confidence, short TTL, working-memory-visible)
recallable *next turn*; consolidation later confirms (claim-hash match → upgrade) or
tombstones. Fast hippocampal store / slow neocortical consolidate — the two-speed split the
Procedural track's brain model already endorses. *Depends:* 2.5, 1.4.

### 3.3 Surprise-gated encoding `[REFINES: shipped relevance pre-filter]`
The heuristic pre-filter already cut ~75% of extraction calls; add the other direction.
Surprise score per turn = embedding distance vs working-memory centroid + recall-hit-rate
("did memory predict this?"). High surprise → eager consolidation (3.1) + checkpoint
suggestion; low → batch with the sweep, skip the relevance LLM. Concentrates spend exactly
where the situation is *changing*. Also a free input to the Tier-1 watchdog ("open
questions" ≈ recent high-surprise turns).

### 3.4 Anticipatory recall (prefetch) `[NEW]`
After `done`, background-recall the predicted next topics (embed last answer + active
goals → hybrid search → warm the embedding cache + a `prefetch:` key). The Ledger checks
prefetch first; topic match = cache-read latency, miss costs nothing visible. This is what
makes Tier 2's extra rewrite call latency-free in the common case. *Depends:* 1.8, 1.6
cache tier.

### 3.5 Contextual bandit over recall techniques `[NEW]`
Five techniques run on static toggles; HyDE/self-query (and Tier 2's rewrite, and the
backlog cross-encoder reranker) are LLM/compute-priced. Thompson-sampling selector:
features = query shape + channel kind + intent class (the backlog "Query Intent
Classification" item becomes the bandit's feature extractor rather than a separate
rule-table); reward = 2.2's usage signal; `eval_recall` (2.7) is the offline guardrail.
Learns to spend expensive techniques only on query shapes where they pay.

### 3.6 Personalized PageRank over the entity graph `[REFINES: Todo "Multi-hop Entity Traversal"]`
The backlog wants lightweight path-finding; PPR (HippoRAG-style) is the stronger form:
seed on matched entities, rank facts by stationary probability over
`ABOUT`/`RELATES_TO`/`SUPERSEDES` (Neo4j GDS, fallback to bounded traversal). Surfaces
*bridge facts* between the query's entities — what Tier-3 `deep_recall` hops on. Gate
behind `recall.ppr_enabled`.

### 3.7 Graph community summaries `[NEW — fills the altitude gap]`
Recap (shipped) is conversation-level; facts are shard-level; nothing sits between.
Consolidation job: community detection (Leiden) per channel → LLM-summarized
`(:CommunitySummary)` nodes (re-summarized only on membership-hash change). Three recall
altitudes: fact → community → recap. The Ledger's `shrink_fn` gets a principled degradation
(facts → community summary under budget pressure), and the Tier-1 watchdog digest gets a
cheap, pre-computed source. Also the natural unit for the **Ambassador's
`survey_conversations`** (16.7 Slice 4) — an application-wide summary is a read over
community summaries, not N full-transcript reads.

### 3.8 Reflexion loop: score the reflex core `[REFINES: shipped Procedural Loop 3 reflex core]`
The reflex core injects top-strength procedures; nothing measures whether they *work*.
Close it: a reflex procedure in-prompt + a steer matching its trigger in the same turn ⇒
log `procedure_violation`; repeated violations auto-flag for re-distillation or demotion.
Steer-free turns with a matched trigger ⇒ reinforce. This is also the missing piece of the
**activation nerve** (Loop 3 remaining): the situation-descriptor matcher needs a
true/false-positive signal to tune against, and violations are exactly that. Converts
procedures from write-only artifacts into a measured control system — and produces the
per-agent quality signal the Genome track's "reasoning-quality scoring" wants, for free,
on the procedural axis.

### 3.9 Generative replay ("sleep-time compute") `[REFINES: Procedural Loop 2's "consolidation = sleep" framing]`
Loop 2 distills procedures nightly; point the same machinery at **facts**: sample
contradiction-flagged / low-confidence / high-salience clusters, re-read original turns
(provenance links exist), re-adjudicate with full cross-conversation context. Budget-capped
(N clusters/night, `feature_default_model` fallback chain). With 2.1, replay is just
targeted re-consolidation at a higher pipeline effort level. Also the consumer for the
backlog "Store Consolidation costs" — replay needs a budget, budgets need cost tracking.

### 3.10 Multi-agent shared-channel write discipline `[REFINES: Todo 16.x "Attribution quality in compound messages" + "Agent social/delegation graph"]`
The compound-message attribution gap (mixed directives mis-homed by small extraction
models) is the symptom; the structural risk is contested supersession once Factory/routes
scale fact traffic. Before that:
- **Provenance-scoped supersession:** auto-supersede only with same resolved `subject` +
  compatible provenance (same agent, or user > agent). Cross-agent contradictions →
  `flag_review` with both attributions, never auto-resolve.
- Last-writer-wins banned on shared channels; bitemporal (1.7) lets both assertions coexist
  until adjudicated.
- Per-agent fact-rate metrics in `by_agent` usage (a flooding specialist must be visible) —
  and these same rates are raw material for the planned agent social/delegation graph
  ("who's reliable at what").

### 3.11 Degradation ladder `[REFINES: shipped `resolve_with_fallback` + Todo Foundation #4]`
Model fallback is shipping site-by-site; memory stores need the analog. An explicit ladder
the Ledger consults, published by health:

| Level | Available | Behavior |
|---|---|---|
| L0 | all | full recall (techniques + graph + procedures) |
| L1 | PG+Redis | vector-only over `conversation_logs` embeddings; no graph/entities |
| L2 | Redis | working memory + checkpoints + flash facts (3.2) |
| L3 | none | transcript only; turns queue for later projection (safe under 1.1) |

Each memory call declares its minimum level. Turns "memory is optional" from scattered
try/excepts into a designed, testable contract — and 1.1 makes L3 lossless.

### 3.12 Conformal duplicate/contradiction thresholds `[NEW]`
0.92 (facts) and `procedural_dedupe_threshold` are point estimates. Fit conformal bands per
embedder from the eval corpora: above upper band auto-merge, below lower auto-pass, the
uncertain middle → LLM adjudication. The verification funnel keeps its shape with
statistically guaranteed boundaries that re-fit automatically on embedder change (1.3).

---

## 4. Sequencing (reconciled with Todo.md's FOUNDATION order)

Todo's Foundation list (chat legibility ✅, steering ✅, stable memory core, fallback
extension, cost/gaps, tech-debt sweep) takes precedence. This roadmap interleaves:

| Wave | Items | Notes |
|---|---|---|
| **W1 observe** | 1.2 invariant checker · 2.2 score logging · Context Inspector (Todo) | Cheap; everything after becomes measurable. Inspector + ledger bookkeeping co-design. |
| **W2 foundations** | 1.4 ChannelRef · 1.6 Redis tiers · 2.5 tombstones · 2.6 graph migrations · 2.8/2.9 (tracked) | Highest bug-prevention per line. 2.8 before W4's ledger. |
| **W3 invert** | 1.1 event-log write path + rebuild · 1.3 embedder provenance | The structural payoff; W1 verifies it. Unblocks "No Memorization" durability + Memory-as-VCS hashing. |
| **W4 quality + Foundation #3** | 1.8 Context Ledger **with the stable-core block (Tier 1 watchdog)** · 1.7 bitemporal · 2.1 watermark · 2.7 eval_recall · 2.3/2.4 | The ledger and the Todo stable-core item are one deliverable — don't build them separately. |
| **W5 reactive** | 3.1 idle trigger · 3.2 flash facts · 3.3 surprise gating · 3.11 ladder · **Active Recall Tier 2** (Todo) | Tier 2 is the 80/20 per Todo; 3.4 prefetch immediately after to hide its latency. |
| **W6 frontier** | 3.4 prefetch · 3.5 bandit · 3.6 PPR · 3.7 communities · 3.8 reflexion scoring · 3.9 replay · 3.12 conformal · **Active Recall Tier 3** (Todo) | Each flag-gated + eval-guarded (2.7). Tier 3 as Alloy specialist per Todo's lean. |
| **Gate before Phase 16 scale-up (Factory / 16.7 Slice 4)** | 3.10 write discipline | Multi-agent fact traffic multiplies supersession contention; land discipline first. |

**Explicit non-duplication:** do *not* re-implement under this doc anything tagged
`[TRACKED]` — work those under their Todo.md items and check them off in both places.

---

## 5. File Map

| Change | Primary files |
|---|---|
| 1.1 / 1.2 | `consolidation/jobs.py` · new `management/commands/rebuild_memory_projections.py` · `streaming/persistence.py` · `views.py::_persist_turns` |
| 1.3 / 3.12 | embedding modules · `extraction/service.py` (dup gate) · `embedding_queue.py` · `config.py` |
| 1.4 | new `kit/agent_memory/channels.py`; sweep `views.py`, `memory/interface.py`, `recall/layer.py`, `memory/procedural.py` (scope ranks) |
| 1.5 / 2.9 | new `kit/agent_memory/repositories/`; strip raw queries from `views.py`; timeout/retry decorators |
| 1.6 | `connections.py` · `agent/checkpoint_storage.py` · `agent/plan_state.py` · `agent/session.py` (summary) · compose Redis config · `ambassador_storage` tiering |
| 1.7 / 3.10 | `models.py` (Fact) · `extraction/service.py` · contradiction pipeline (`check_contradictions`, `check_correction`) |
| 1.8 / 3.4 / 3.11 | `agent/context.py` (assembler → Ledger) · `views.py` stream assembly · the Tier-1 watchdog block |
| 2.x lifecycle | `consolidation/jobs.py` · `memory/semantic.py` · new `graph_migrations/` |
| 2.7 / 3.5 | `recall/layer.py` · new `management/commands/eval_recall.py` · shared snapshot/restore module (Todo wants it extracted anyway) |
| 3.1–3.3 | `consolidation/worker.py` · new `memory/flash.py` · relevance pre-filter |
| 3.6–3.7 | `memory/semantic.py` + Neo4j GDS · new `CommunitySummary` job · 16.7 `survey_conversations` consumer |
| 3.8–3.9 | `memory/procedural.py` (violation logging, activation-nerve tuning) · `consolidation/jobs.py` (replay) · `streaming/steering.py` (steer↔trigger match) |

---

## 6. Definition of Done (per item)

(a) config flag defaulting to current behavior where behavioral; (b) graceful degradation
when its store is down (declare an L-level per 3.11); (c) a row in
`architecture/memory-capabilities.md` (and the capability registry once 2.6's sibling
lands); (d) eval coverage — `eval_consolidation` for write-path, `eval_recall` for
read-path, no-pollution tests for anything Ambassador-adjacent; (e) extend the invariant
checker (1.2) for any new cross-store agreement; (f) **update both this doc and the matching
[`todo/`](todo/) file** (see §7.1) — a shipped item gets its tag flipped here and its checkbox
there in the same PR.

---

## 7. TODO cross-reference & open reconciliation (as of v0.21.88)

> The TODO was split from a monolithic `Todo.md` into [`todo/`](todo/) (per-phase + per-theme).
> §7.1 maps each area of this roadmap to the backlog file that owns the `[TRACKED]`/`[REFINES]`
> items; §7.2 is the **enrichment queue** — findings from the v0.21.88 reconciliation pass that
> aren't yet folded into the §-sections above. **Fable:** work §7.2 by editing the noted section,
> then strike the item here.

### 7.1 Section ↔ backlog-file map

| This roadmap | Lives in (TODO) |
|---|---|
| Active-Recall tiers (3.0, 1.8, 3.4–3.6) | [`todo/backlog/memory-recall.md`](todo/backlog/memory-recall.md) |
| Procedural loops + reflexion (3.0, 2.4, 3.8, 3.9) | [`todo/backlog/procedural.md`](todo/backlog/procedural.md) |
| Retrieval techniques, extraction, calibration (2.3, 2.7, 3.5, 3.12) | [`todo/backlog/retrieval-extraction.md`](todo/backlog/retrieval-extraction.md) |
| Context ledger, summary/checkpoint durability (1.6, 1.8) | [`todo/backlog/conversation-context.md`](todo/backlog/conversation-context.md) |
| Janitor, capability registry, attribution (1.2, 2.6, 3.10) | [`todo/backlog/conversation-context.md`](todo/backlog/conversation-context.md) (Memory Area UX) + [`todo/phases/phase-16-multi-agent.md`](todo/phases/phase-16-multi-agent.md) (16.x) |
| Event-log, repository layer, connection hardening (1.1, 1.5, 2.9) | [`todo/backlog/engineering-hardening.md`](todo/backlog/engineering-hardening.md) + [`todo/known-future-issues.md`](todo/known-future-issues.md) |
| Import/export, Memory-as-VCS, embedder provenance (1.1, 1.3) | [`todo/backlog/open-platform.md`](todo/backlog/open-platform.md) |
| Token estimators, dead-knob sweep (2.8, 1.8) | [`todo/backlog/foundation.md`](todo/backlog/foundation.md) (#6) + [`todo/backlog/engineering-hardening.md`](todo/backlog/engineering-hardening.md) |
| Survey/community summaries → Ambassador (3.7) | [`todo/phases/phase-16-multi-agent.md`](todo/phases/phase-16-multi-agent.md) (16.7 Slice 4) |
| Workspace channel scoping (1.4) | [`todo/backlog/workspaces.md`](todo/backlog/workspaces.md) |

### 7.2 Open reconciliation items (enrichment queue)

Findings from the v0.21.88 cross-check that sharpen the plan but aren't yet woven into the
sections above. Each names the section to edit and the cross-referenced backlog file.

- [ ] **Encryption at Rest is missing from the §2.9 sweep.** It's the one
      [`known-future-issues.md`](todo/known-future-issues.md) item with no home in this doc, and we
      already ship the pattern to reuse — `logging_kit/archive_crypto.py` (Scrypt KEK wrapping an
      AES-256-GCM DEK). **Fold into §1.5 / §2.9** as a repository-layer-transparent field encryption
      sub-point.
- [ ] **One fitted-artifact key, not three.** §2.3 keys calibration by `extraction_model`; §3.12
      keys conformal bands by `embedder`; §2.1 stamps `pipeline_version`. But an extraction-*prompt*
      change ([`retrieval-extraction.md`](todo/backlog/retrieval-extraction.md): few-shot sets,
      improved prompts, Sonnet-for-extraction) invalidates §2.3 without touching model or embedder.
      **Make §2.1's `pipeline_version` the shared invalidation key** for §2.3 + §3.12.
- [ ] **§1.5 is the universal access seam — say so.** Reframe it to carry not just tenancy + the
      raw-query ban but also at-rest encryption (above), the §2.9 timeout/retry/limit decorators, and
      §3.11's per-call minimum L-level. *Build the boundary once, decorate it five times* — cleans up
      the W2→W6 dependency chain.
- [ ] **§1.2 invariant list needs the rename-safety assertion.** Invariant 3 (names are aliases) is
      best-effort today; add *"every live profile name appears in its Agent entity's `aliases`"* to
      the checker, not just "channel maps to a live entity."
- [ ] **§1.1 also enables multi-cluster + fixes the disabled-memory banner.** `rebuild_memory_projections`
      + §2.6 migrations are the primitives Phase-17 `prod:*`/`cluster:*` and "Multiple server support"
      ([`misc.md`](todo/backlog/misc.md)) need — draw that edge. And once L3/projection split exists,
      the *"disabled-memory banner"* ([`misc.md`](todo/backlog/misc.md)) reads real projection state,
      not a flag.
- [ ] **W1 "observe" should pull in the consolidation-observability trio** — `/jobs/{id}/logs`,
      real-time progress, `consolidate/preview` ([`misc.md`](todo/backlog/misc.md)). Cheap; the janitor
      (§1.2) and generative replay (§3.9) both want `consolidate/preview` as their dry-run surface.
- [ ] **Map or retire "Streaming memory retrieval during chat"** ([`misc.md`](todo/backlog/misc.md))
      — it's arguably subsumed by §3.4 prefetch + Active-Recall Tier 2; say which, so it isn't built
      twice.
- [ ] **Flag the §1.8 ↔ Foundation #6 overlap.** §1.8 already claims it kills the hardcoded 70%
      (Foundation #6's dead-knob half) and §2.8 is `[TRACKED]` for the estimator half — note that
      Foundation #6 ([`foundation.md`](todo/backlog/foundation.md)) is split across both so the §6 DoD
      accounting doesn't double-count.

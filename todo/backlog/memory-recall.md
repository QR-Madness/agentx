# Active Memory Recall

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.

---

### ⭐ PPR spike — §3.6 Personalized PageRank, run out-of-store `[SPIKE · decision gate for 19.5]`

> Companion roadmap entry: [Memory-Roadmap.md](../../Memory-Roadmap.md) §3.6.
> Storage decision it gates: [phase-19-cloud-operation.md](../phases/phase-19-cloud-operation.md) §19.5.

**Why this is one task and not two.** §3.6 has to be built regardless — multi-hop is the weakest
category in the golden set and PPR is its identified fix. Building it **out-of-store** (materialize
the subgraph → rank in Python → return facts) costs little more than building it against Neo4j GDS,
and it settles a far larger question: whether a second stateful service earns its place. Write the
ranking against a materialized subgraph and the same code runs whether facts live in Neo4j or
Postgres — which is what makes this a *decision gate* rather than a commitment.

**Scope discipline: this spike buys information, not a migration.** Nothing moves stores here.

#### Step 0 — RESULT: **GATE FAILS — spike stopped** `[measured 2026-09-20]`

Profiled the live dev graph (12 channels, 239 entities, 256 facts):

| Metric | Value |
|---|---|
| Entities | 239 |
| `ABOUT` (Fact→Entity) | 354 |
| **`RELATES_TO` (Entity→Entity)** | **113** — the edges PPR needs |
| **Median entity degree** | **1** (gate required ≥ 2) |
| Mean degree | 0.94 |
| **Isolated entities (degree 0)** | **111 — 46%** |
| Max degree | 3 in every channel except `_global` (15) |

One channel (`_self_bright-grand-fern`) has **100%** isolated entities. There is almost no path
longer than one hop anywhere in the graph, so **PPR has nothing to propagate over**. Do not build
§3.6 against this graph — any arm would score at noise for data reasons, not ranking reasons.

**This reclassifies the problem.** Multi-hop MRR is weak because the entity graph is barely
connected — an **extraction/density** problem, not a *ranking* problem and **not a storage problem**.
No graph database and no retrieval technique fixes a graph that has no edges.

**Root cause: narrowed, not established.** The pipeline is intact end-to-end — the extraction prompt
requests `relationships`, and `consolidation/jobs.py::_batch_store_relationships` MERGEs them. But
that function **drops** any relationship whose endpoints fail to resolve, at `debug` level, into
`metrics.relationships_dropped`. Endpoint resolution runs through semantic entity linking with
`entity_linking_auto_threshold = 0.90` and a `0.75–0.90` **log-only gray zone** that never links.

Tested the obvious hypothesis — that strict linking fragments entities so endpoints miss — and it is
**real but insufficient**: only ~14 near-duplicate pairs across 239 entities (~6%), several of which
are correctly separate (`'Section 1.1'` vs `'Section 1.2'` scores 0.91 on string similarity but are
different things). That cannot explain 46% isolation. *Caveat: this probe used string similarity as a
proxy; the linker uses embeddings, so true fragmentation may be lower still.*

Three candidates remain, and they are distinguishable by one measurement:
1. the extractor proposes few relationships at all (**yield**),
2. proposed relationships die at endpoint resolution (**drop rate**),
3. the corpus genuinely has few entity-entity relations.

**Blocking instrumentation gap:** `relationships_dropped` is computed in `ConsolidationMetrics` and
asserted in `tests_memory.py`, but **surfaced in no endpoint, log line above debug, or dashboard**.
There is no *proposed* counter to divide it by. Until both exist, cause (1) and (2) are
indistinguishable.

- **Next action — small and concrete:** surface `relationships_proposed` / `relationships_dropped`
  (consolidation metrics → `/metrics`), run a consolidation pass on a populated channel, read the
  ratio. A high drop rate ⇒ loosen the gray zone / improve resolution. A low drop rate with few
  proposals ⇒ the extraction prompt is the lever (§2.10).
- **Consequence for [19.5](../phases/phase-19-cloud-operation.md):** the storage decision is now
  gated behind **graph density**, not behind PPR. You cannot judge whether a graph database earns its
  place until there is a graph worth traversing — today Neo4j's ~2 GB JVM floor is holding 113 edges.

<details>
<summary>Original Step 0 plan (kept for provenance)</summary>

#### Step 0 — profile the entity graph first (≈½ day; may end the spike)

PPR propagates over `(:Entity)-[:RELATES_TO]->(:Entity)`; `ABOUT` only links `Fact → Entity`. If
consolidation is producing few `RELATES_TO` edges the graph is effectively bipartite, **no**
technique can do multi-hop, and the weak score is an *extraction* problem rather than a *ranking*
one. Measure on a real channel **and** on the eval corpus: entity count, `RELATES_TO` count, degree
distribution, share of entities at degree 0.

- **Gate:** median entity degree < 2 ⇒ stop. Fix extraction (Memory-Roadmap §2.10) before ranking.

</details>

#### Steps 1–4 — **BLOCKED** on Step 0's density fix. Retained below; re-run Step 0 before resuming.

#### Step 1 — re-baseline before comparing anything (≈½ day)

The recorded **multi-hop MRR 0.29** is the W1 run of 2026-07-05, taken *before* §2.11 stage 1 shipped
`recall_candidate_pool=50` `[v0.21.155]`. Arms were explicitly "within noise of each other at today's
`top_k*2`(=20) pool" — that confound is now fixed, so the old number is not a valid comparison point.
Re-run `manage.py eval_recall` on the current build first.

#### Step 2 — implement PPR out-of-store (≈2–3 days)

- New `RecallLayer` technique beside `_entity_centric_retrieval` (`memory/recall.py`).
- Materialize the channel's entity subgraph into `scipy`/`networkx`; seed on query-matched entities;
  rank facts by stationary probability over `ABOUT`/`RELATES_TO`/`SUPERSEDES`. Surface the **bridge
  facts** between the query's entities — what Tier-3 `deep_recall` hops on.
- Cache the materialized graph per channel on a **membership hash**, the same invalidation trick §3.7
  uses for `CommunitySummary` re-summarization. Materialization cost is the real risk: pulling the
  subgraph per query will dominate p95 if uncached.
- Gate behind `recall.ppr_enabled` — declared in `settings_registry` + `settings_help.yaml`, then
  `task docs:gen:settings` (ADR-17; undeclared settings are read-only by design).

#### Step 3 — measure it as an eval arm (≈1 day)

Add a `ppr` arm to `eval_recall` alongside the existing arms. Score multi-hop MRR + recall@k against
the Step 1 re-baseline, and **report p95 separately** — a quality win paid for with a latency
regression is not a win.

#### Step 4 — the decision (no code)

- **PPR works out-of-store at acceptable p95** ⇒ Neo4j's remaining claim is Leiden community
  detection (§3.7), which is *also* an out-of-store algorithm on a ~10⁴-node graph. 19.5's research
  slice becomes a cleanup with a known cost (audit below).
- **Materialization dominates, or quality lags a graph-native implementation** ⇒ Neo4j earns its
  place. Record *why* in [Decisions.md](../../Decisions.md) and close 19.5's research slice, so the
  question stops being re-asked.

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
      *Research levers to fold into the rewrite step* ([recall survey](../research/2026-07-memory-recall-research.md)):
      **time-aware query expansion** for temporal asks (parse a time range, restrict scope — +7–11%
      on temporal reasoning in LongMemEval), **session/round-level decomposition** granularity for
      episodic indexing, and **chain-of-note reading** (extract per-retrieved-memory supporting
      evidence before answering).
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


# Repo Questions

> **The loop:** Claude (Opus, working in-repo) parks genuinely hard design questions here —
> the ones where a fast local decision would be guessing and a **superhuman, codebase-aware
> answer** changes the call. Fable swoops in and answers in the `↳ Answer` block. Then whoever
> acts on it folds the resolution into the real doc (`Memory-Roadmap.md`, `Decisions.md`, a
> `todo/` file) and strikes the question here.
>
> **Scope discipline (this repo is utilitarian — nothing wasted):**
> - Only questions that are *blocking-if-wrong* or *expensive-to-reverse*. Not "what's your
>   opinion" — those I just decide and note in `Decisions.md`.
> - Each question carries enough **context + constraints** to answer without re-deriving the repo.
> - This is **not** `Memory-Roadmap.md §7.2` — that's a reconciliation/enrichment queue for the
>   memory roadmap specifically. This file is broader (any subsystem) and deeper (open design
>   forks), and it's where an *answer* is needed, not just a fold-in.
>
> Linked from the [CLAUDE.md Documentation Map](CLAUDE.md#documentation-map).

---

## Q1 — Memory Capability Registry: the minimal schema (Memory-Roadmap T2)

**Context.** Memory capabilities are documented by hand in `architecture/memory-capabilities.md`;
the roadmap (§1.2, §2.6) and `todo/phases/phase-16-multi-agent.md` ("Memory capability registry")
want a code-side `@capability(...)` registry the doc is generated from or validated against. Four
consumers want to read it: (a) the generated capability doc, (b) the invariant checker (§1.2),
(c) the degradation ladder L0–L3 (§3.11), (d) the future Settings Manifest.

**The question.** What is the *minimal* `@capability` field set that serves all four consumers
without becoming a second config system? Specifically: does the L-level belong on the capability
or on each store it touches? Do config flags reference the (future) Settings Manifest by key, or
inline their own metadata until it exists?

**Why it's hard / what a good answer needs.** It's a one-way door — every capability gets
decorated once and the four consumers couple to the shape. Over-rich = a parallel config system to
maintain; too-thin = the doc/checker can't be generated and we're back to hand-maintenance. Answer
should give the dataclass/decorator signature + one worked example (e.g. `hybrid_recall`).

**↳ Answer (Fable):**
_(unanswered)_

---

## Q2 — OpenAPI without DRF: hand-maintained + parity-check, or adopt a generator?

**Context.** The API is hand-rolled Django views (no DRF / drf-spectacular), so `OpenApi.yaml`
is hand-maintained and the CLAUDE.md rule is "update both `OpenApi.yaml` and `endpoints.md`."
The cheap static guard available is a `urls.py` ↔ `OpenApi.yaml` path-parity check (Repo-Questions
T3, fits `scripts/check_docs.py`).

**The question.** Is hand-maintained-spec + parity-check the right long-term posture, or is it
worth a one-time migration toward a code-derived spec (drf-spectacular needs a DRF migration; or a
lightweight introspection that walks `urls.py` + view signatures)? At what endpoint count / churn
rate does the generator pay back its migration cost?

**Why it's hard.** The parity check only proves *paths* match — it can't verify request/response
*shapes*, which is where real API drift bites clients. But a DRF migration is a large, invasive
refactor of every view. The answer determines whether we invest in the parity check as a permanent
solution or as a stopgap.

**↳ Answer (Fable):**
_(unanswered)_

---

## Q3 — Event-sourced write path (§1.1): the no-flag-day cutover

**Context.** Memory-Roadmap §1.1 wants to invert the write path so a turn "succeeds when the
`conversation_logs` row commits" and Neo4j/vector become rebuildable projections, with
`rebuild_memory_projections`. Today it's a dual-write (`store_turn` writes Neo4j→PG and re-raises
on any failure; the 15.10 restore-404 bug came from exactly this coupling, patched with a per-turn
PG fallback in `_persist_turns`).

**The question.** What is the safest *incremental* cutover that never has a flag-day where both
paths are half-live? The roadmap's own sequencing is "ship 1.2 invariant checker → rebuild command
against current path → flip write path → delete the fallback patch." Is that ordering right, and
where's the riskiest seam (the rebuild's fidelity vs the live projections)?

**Why it's hard.** Getting it wrong silently corrupts the memory graph for all users with no loud
failure; the projections feed recall, so a subtly-wrong rebuild degrades answers without erroring.

**↳ Answer (Fable):**
_(unanswered)_

---

## Q4 — Bitemporal facts (§1.7): non-destructive migration from `temporal_context`

**Context.** §1.7 replaces the `temporal_context ∈ {current,past,future}` label (which conflates
*when true* with *when learned*) with `valid_from`/`valid_to` (world time) + `asserted_at`/
`superseded_at` (transaction time, partially present). Recall's `1.2×`/`0.7×` multipliers become
interval-overlap functions, with the old labels kept as a derived view for back-compat.

**The question.** What's the migration that backfills `valid_from`/`valid_to` from existing facts
*without* a destructive reset and without inventing world-time data we don't have? For legacy facts
where we only know `asserted_at`, what's the principled default interval (open-ended? a
confidence-weighted horizon from `expected_stability`)?

**Why it's hard.** Embeddings are derived (re-derivable), but the temporal data is *not* — a bad
backfill loses the only signal we have about fact recency/validity, and recall ranking silently
shifts under everyone. Needs the migration + the back-compat derived-view definition.

**↳ Answer (Fable):**
_(unanswered)_

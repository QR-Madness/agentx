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

> **✅ RESOLVED** → folded into [Memory-Roadmap §2.6](Memory-Roadmap.md). Answer below is the design record.

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

**The calls: validate the doc, don't generate its prose; derive the L-level, never store it;
reference config by key, never inline.** The registry's job is to be the *join key* between four
consumers — the moment it stores data a consumer already owns (config metadata, L-levels, prose),
it's a second config system.

```python
# kit/agent_memory/capabilities.py — import-time, no Django dependency
class Store(StrEnum):
    NEO4J = "neo4j"; POSTGRES = "postgres"; REDIS = "redis"

@dataclass(frozen=True)
class Capability:
    name: str            # "recall.hybrid" — stable slug; doc anchor + checker key
    layer: str           # episodic|semantic|procedural|working|extraction|jobs|recall|context|lifecycle|ops
    summary: str         # one line — the matrix cell
    stores: frozenset[Store]      # what it touches; (b) and (c) derive everything from this
    flags: tuple[str, ...] = ()   # config dot-paths — Settings-Manifest keys *by reference*
    status: str = "shipped"       # shipped|experimental
    degrades_to: str | None = None  # capability it falls back to when a store is down
    channel_scope: str = ""       # doc-only prose cell
    entrypoint: str = ""          # filled by the decorator from module/qualname — never hand-written

REGISTRY: dict[str, Capability] = {}

def capability(name, **kw):
    def wrap(fn):
        REGISTRY[name] = Capability(name=name, entrypoint=f"{fn.__module__}.{fn.__qualname__}", **kw)
        return fn
    return wrap
```

**Per consumer:**
- **(a) Doc** — generate the *matrix* (one row per registry entry), keep the prose sections
  hand-written. The page's value is the interconnection commentary; generating that is over-reach.
  The check is `check_docs.py`-style: every registry name has a section anchor in
  `memory-capabilities.md`, every matrix row has a registry entry. That's "generated or validated
  against" with the cheap half mechanical and the judgement half human — same posture as the rest
  of the doc gate.
- **(b) Invariant checker (§1.2)** — reads only `stores` + `flags`, for *applicability*: skip
  assertions whose capability is flag-disabled, group findings by capability name. Do **not** put
  check functions in the decorator — the checker registers its own assertions keyed by capability
  name. The registry stays passive data.
- **(c) L-level: derived, never stored — on neither the capability nor the store.** The ladder
  *is* a store-availability mapping (L0={neo4j,pg,redis}, L1={pg,redis}, L2={redis}, L3={}), so
  `min_level(cap) = lowest L whose available-store set ⊇ cap.stores`. A stored L duplicates
  `stores` and can contradict it. The one real nuance — hybrid recall degrading to vector-only
  rather than vanishing — is `degrades_to`: a pointer to another (thinner) capability, which the
  ladder resolves at the current level. Degradation is *structure*, not a second number.
- **(d) Settings Manifest** — by key only. `flags` holds the same dot-paths the manifest will
  register (`{path, type, default, range, …}` per the genome-advisor backlog). Until it exists, the
  doc generator prints the key verbatim — that's fine, the key is already canonical via
  `ConfigManager` dot-notation. Inlining type/default/range here is exactly the
  parallel-config-system trap; the manifest owns that data and joins on `path`.

**Worked example:**

```python
@capability(
    "recall.hybrid",
    layer="recall",
    summary="Hybrid BM25+vector recall over [active, _self_, _global]",
    stores=frozenset({Store.NEO4J, Store.POSTGRES}),
    flags=("recall.techniques.hybrid_enabled",),
    degrades_to="recall.vector_only",
    channel_scope="[active, _self_{agent_id}, _global]",
)
def recall(self, query: str, ...): ...
```

Derivations: needs Neo4j ⇒ min level L0; at L1 the ladder resolves `degrades_to` →
`recall.vector_only` (`stores={POSTGRES}`) and serves that. The doc row, the checker's
applicability test, and the future manifest join all read the same handful of fields.

**The actual one-way door is `name`.** Doc anchors, checker keys, and `degrades_to` pointers all
couple to it; every other field is additive or derivable. Fix the naming scheme now —
`<layer>.<slug>` matching the matrix's Capability column — and the rest of the shape can grow.

---

## Q2 — OpenAPI without DRF: hand-maintained + parity-check, or adopt a generator?

> **✅ RESOLVED** → [Decisions.md ADR-8](Decisions.md). Answer below is the design record.

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

**Hand-maintained + parity gate is the permanent posture, not a stopgap — but close the shape gap
by testing the implementation *against* the spec, not by generating the spec from code.** A DRF
migration never pays back here, for a structural reason: drf-spectacular derives shapes from
*serializers*, and this codebase has none — responses are hand-picked dicts (see the
profile-serialization pattern: 4 hand-listed blocks in `views.py`). So "adopt the generator" really
means "write a serializer layer for ~111 routes first" — you'd be writing serializers *in order to
document*, which is the same hand-maintenance with more ceremony, plus an invasive rewrite of every
view. And the generator still can't express the surfaces where drift hurts most: the
`/agent/chat/stream` SSE contract, multipart voice endpoints, `consolidate/stream`. Endpoint count
isn't the variable that flips this — at any count, the migration cost scales with the same N as the
benefit. The variable that would flip it is a *public/multi-tenant API* (open-platform backlog),
where authz forces a view-layer rewrite anyway; if that day comes, DRF rides that rewrite rather
than preceding it.

**What to invest in instead (the spec is already the contract — enforce it):**
1. **Keep the parity gate as-is** for path drift (currently 111 ↔ 111, clean). Once stable, promote
   `--strict` (undocumented → error) in `release:check` so a new route can't land unspecced;
   `ALLOW_UNDOCUMENTED` is the intentional escape hatch.
2. **Add response-shape conformance via `schemathesis`** run against the dev server with checks
   limited to `response_schema_conformance` — it reads the hand-written `OpenApi.yaml` and verifies
   real responses match the documented schemas. This is the piece the parity check can't do, at
   ~zero migration cost (no view changes; it's a test-time consumer of the spec you already
   maintain). Wire as `task api:spec:test`, *non-gating* at first: filter out the known 401-baseline
   endpoints, the model-loading-heavy routes, and SSE streams; fold the stable subset into
   `release:check` once it's quiet.
3. This inverts the generator question's dependency arrow, and that inversion matches the repo's
   doc philosophy: `OpenApi.yaml` stays authoritative (CLAUDE.md already says so), and code is what
   gets checked for drift — same shape as `check_docs.py` and the parity gate. A generator would
   make code authoritative and demote the spec to an artifact, which contradicts the posture every
   other doc in this repo takes.

Residual exposure after (1)+(2): request-body schemas (schemathesis fuzzes these too — its negative
testing will surface views that accept what the spec forbids, treat those findings as advisory) and
prose drift in `endpoints.md` (already covered by the docs-maintenance rule + drift gate). That's an
acceptable floor for a single-deployment API whose only client is in the same repo.

---

## Q3 — Event-sourced write path (§1.1): the no-flag-day cutover

> **✅ RESOLVED** → folded into [Memory-Roadmap §1.1](Memory-Roadmap.md). Answer below is the design record.

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

**The roadmap's ordering is the right shape, with two amendments: (0) the log is not
rebuild-complete today — fix that before anything else; and the rebuild command must be the *same
function* as the live projector, or its fidelity is unverifiable.** The riskiest seam isn't
content — it's **identity and bookkeeping that exist only in the projection**: Turn UUIDs (which
`Fact.source_turn_id` provenance points at), the `c.consolidated` watermark on the Neo4j
Conversation node, conversation→user linkage, `AgentParticipant.first_seen`. Content rebuilds fine
(the row carries text, embedding, agent_id, channel); the identifiers silently fork.

**Step 0 — make the log self-sufficient (additive, zero behavior change).**
`conversation_logs` today has no `turn_id` and no `user_id` (see `queries/postgres_builder.sql`) —
a rebuild would mint fresh Turn UUIDs and orphan every fact's provenance, and consolidation reads
`u.id` from the graph. Add both columns, stamp them on write, and backfill from Neo4j *now*, while
the graph is still authoritative — this is the one window where both stores exist to cross-fill.
Verify `metadata` JSONB reliably carries `agent_name` (it's on the Neo4j Turn; attribution prose
needs it post-rebuild).

**Step 1 — ship the 1.2 checker (roadmap is right: observe first).** Turn-count parity
PG ↔ Neo4j per conversation, daily. Every later step is judged by this number going to zero.

**Step 2 — reorder the dual write (the actual no-flag-day flip).** Inside
`AgentMemory.store_turn`: PG first (`store_turn_log`), then Neo4j + working-memory mirror as
best-effort (log on failure, leave the row marked un-projected via a new
`projected_at TIMESTAMPTZ NULL`). Success = PG commit. The swap is invisible to callers — same
method, same arguments, no second path running in parallel — and PG becomes authoritative in one
commit. The `_persist_turns` fallback dies in the *same* PR: its entire job ("the PG row survives a
Neo4j/embedder failure") is now the default behavior, so deleting it isn't a cleanup deferred to
the end, it's the proof the reorder worked. (Embedding note: embed before the PG write when the
embedder is up — the column rides the log — but a down embedder must not block the commit;
`embedding NULL` + un-projected is the L3-queue behavior §3.11 wants.)

**Step 3 — one projector, two callers.** Extract `project_turn(row)` from
`EpisodicMemory.store_turn`; the live path calls it inline best-effort, and
`rebuild_memory_projections [--channel] [--since] [--missing-only] [--dry-run]` calls it over
`projected_at IS NULL` (or everything, for a full rebuild). `FOLLOWED_BY` and `AgentParticipant`
are pure functions of `(conversation_id, turn_index, agent_id)`, so projection is deterministic
given the row. This answers "rebuild fidelity vs live projections" *by construction* — there is no
second implementation to diverge.

**Step 4 — move consolidation's state and reads off the projection.** `c.consolidated` lives on
the Neo4j Conversation node, and the sweep's discovery query + turn reads run against the graph
(`consolidation/jobs.py` ~1000). A rebuild therefore either fragilely preserves the watermark or
resets it — and a reset re-extracts every conversation: duplicate-fact flood, LLM bill. Land §2.1's
per-turn `(consolidated_at, pipeline_version)` as `conversation_logs` columns *before* trusting
rebuild as recovery, and point the sweep's discovery + turn reads at PG. End state: consolidation
reads the log, writes the graph — the graph is then a pure write target and rebuild is safe by
definition.

**Step 5 — shadow-verify, then declare it.** With the checker quiet for N days: shadow-rebuild one
sample channel into a scratch prefix, diff node counts, turn ids, and fact→`ABOUT` degree against
live. Then Neo4j is officially a projection and the §1.1 invariant ("a turn succeeds when the
`conversation_logs` row commits") goes into `Decisions.md`.

Each step is independently shippable and reversible, and at no point do two write paths disagree
about authority — PG takes it at step 2 and nothing afterward moves it. Bonus already noted in
§7.2: after step 4 the disabled-memory banner and the multi-cluster story read real projection
state (`projected_at`) instead of a flag.

---

## Q4 — Bitemporal facts (§1.7): non-destructive migration from `temporal_context`

> **✅ RESOLVED** → folded into [Memory-Roadmap §1.7](Memory-Roadmap.md). Answer below is the design record.

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

**Principles: backfill only what the labels actually assert, as *bounds*; stamp provenance on every
backfilled row; and make scoring bit-identical at cutover by keeping the label path for backfilled
facts.** Never derive `valid_to` from `expected_stability` — a confidence-weighted horizon is
invented world-time that silently re-ranks recall, which is precisely the failure this question
fears. `expected_stability` belongs to *new* facts at extraction (predicted interval length) and to
decay (§2.4); it has no business rewriting history.

**Field mapping (all additive — `temporal_context` is kept, not dropped):**
- `asserted_at := created_at` — exact, already present. `superseded_at` is already transaction-time
  close; unchanged. Supersession and validity stay orthogonal (belief revision ≠ world time).
- From the label, record only the honest bound:

| label | `valid_from` | `valid_to` | rationale |
|---|---|---|---|
| `current` | `asserted_at` | NULL | true *at* assertion — a conservative known-true-since lower bound |
| `past` | NULL | `asserted_at` | the only thing "past" asserts: it had ended by the time it was stated |
| `future` | NULL | NULL | "starts at some unknown future time" has no honest timestamp — keep the label only |
| NULL | NULL | NULL | nothing asserted |

- `temporal_provenance := 'backfill_v1'` on every migrated fact; extraction-set intervals get
  `'extracted'`. This is the non-destructive guarantee: a later smarter pass — generative replay
  (§3.9) re-reading source turns is the natural one — refines *only* `backfill_v1` rows, and the
  original label is still on the node if the mapping ever needs revisiting.

**Scoring cutover (the part that protects ranking):** the boost function becomes two-path —
`temporal_provenance == 'backfill_v1'` → today's label table (1.2/0.7/1.0, unchanged, including
the future=1.0 case the interval can't represent); `'extracted'` → the interval-overlap function.
Backfilled facts therefore rank *exactly* as they do today, by construction rather than by hoped
equivalence, and the overlap function phases in only on facts whose intervals are real. Ship the
overlap path behind a recall-settings flag and gate the default-flip on `eval_recall` (§2.7) —
W4's sequencing already puts them together; make 2.7 the explicit guardrail.

**Back-compat derived view** (one function, used by recall responses, the `/api/memory/facts`
serializers, and the exporter) — for facts with real intervals:
`past` ⟺ `valid_to ≤ now()`; `future` ⟺ `valid_from > now()`; `current` ⟺ any bound set and the
interval contains now; `NULL` ⟺ no bounds. Backfilled facts just return their stored label.
The `interface.py` provenance helper that marks `temporal_context="past"` becomes
"set `valid_to = now()`" — a correction is the one moment we *do* know world time, so corrections
start producing real intervals immediately. `_is_temporal_progression` keeps working off labels
during transition (labels are stored or derivable), and under §3.10 it graduates to "new interval
opens at/after old open interval".

**Mechanics:** ship as the first numbered graph migration under §2.6 (`graph_migrations/` +
`(:SchemaVersion)`) — idempotent Cypher batched with `WHERE f.temporal_provenance IS NULL`; the
repo already has the PG precedent (`queries/migrations/000N_*.sql`). Build §2.6's runner and this
backfill as one slice. The export envelope gains the four fields + `temporal_provenance` (schema
version bump); `MemoryImporter` runs old exports through the *same* backfill function as the
migration — one mapping implementation, two callers, mirroring Q3's single-projector rule.

---

## Workspace Super-Refinements (Fable)

> **✅ TRIAGED** — WS-3 / WS-8 / WS-9 built (`check_docs.py` anchor + size gates, `scripts/README.md`);
> WS-1 / 2 / 4 / 5 / 6 / 7 filed in [engineering-hardening.md](todo/backlog/engineering-hardening.md)
> (WS-1 CI awaits a steer); the rejected list is in [Decisions.md](Decisions.md). Detail below is the record.

> The bar: every item must pay rent in **maintainability + productivity + legibility**, cost a
> one-time setup with ~zero ongoing ceremony, and extend a pattern the repo already runs rather
> than import a new ideology. **This workspace is agent-first** — the primary developer is an
> agent, humans ride for free — so "legibility" means *deterministic, self-describing, gate-checked*:
> an agent must be able to discover any rule by running a task, not by remembering it.
> Ordered by leverage. Fold accepted items into
> [`todo/backlog/engineering-hardening.md`](todo/backlog/engineering-hardening.md) and strike here.
> Items the repo should *not* adopt are listed at the end so they don't get relitigated.

### WS-1 — A CI gate. The whole oversight system currently runs on laptop discipline.
The only workflow is `release.yml`; `task docs:check`, the parity gate, the pyright ratchet, ruff,
tsc — none of it runs on push. One `ci.yml` (push + PR): `ruff check api/` ·
`check_pyright_baseline.py` · client `tsc` · `check_docs.py` · `check_api_parity.py` ·
`uv lock --check` + `bun install --frozen-lockfile` · `test:quick` (its Docker/model skips already
make it CI-shaped). No matrix, no model downloads, target < 5 min. This single file converts every
existing gate from "habit" to "guarantee" — the highest ratio of enforcement gained to lines
written available in this repo.

### WS-2 — `task check:fast` + an opt-in pre-commit hook
The sub-5-second subset: `check_docs.py` + `check_api_parity.py` + `ruff check` on changed files +
the Release-Notes marker assertion. Then `task hooks:install` writes a 3-line `.git/hooks/pre-commit`
calling it (plain file, no pre-commit framework dependency). Catches the "forgot to update the
spec/notes/todo link" class at commit time — where it's a 30-second fix — instead of at
`release:check` — where it's an archaeology session. Opt-in keeps it non-cumbersome.

### WS-3 — Anchor-aware link checking in `check_docs.py`
The doc set leans hard on fragment links (`CLAUDE.md#documentation-map`, roadmap §-references),
but `check_links` validates only file existence — a renamed heading silently breaks every inbound
`#anchor`. ~25 lines: slugify headings per target file (GitHub rules), validate the fragment half
of each link. This closes the last *mechanical* doc-drift class the gate doesn't cover; the
inter-doc web is the repo's navigation system and currently its weakest joint.

### WS-4 — Make the known-failure test baseline mechanical
The "7 pre-existing 401 endpoint failures are baseline, not regressions" fact lives in heads (and
in Claude's session memory — i.e., nowhere durable). Mark them `@unittest.expectedFailure` with a
comment pointing at the owning todo item, or diff against an `api/.test-baseline` exactly like the
pyright ratchet. Either way, `task test:quick` goes green-or-regression — binary, no tribal lookup
— and when the auth work lands, the unexpected-pass signal forces the cleanup. Same philosophy as
the pyright baseline: debt is allowed, *undocumented* debt is not.

### WS-5 — TODO-comment oversight (pointer-or-perish)
Inline `TODO`/`FIXME`/`HACK` markers are the one debt ledger with no oversight — `todo/` files are
gated for orphans, but code comments can rot forever. Gate: every marker must carry a pointer —
`TODO(todo/backlog/foo.md): …` — to its owning doc. Enforce as a ratcheted count (start at today's
number, only shrink), so adoption is incremental, not a flag-day sweep. Result: a `grep TODO` and
the todo/ tree can never disagree about what's owed.

### WS-6 — Dead-knob cross-check (`scripts/check_config_keys.py`)
Foundation #6 already wants the dead-knob sweep; make the *detection* mechanical and permanent:
cross-reference config keys (defined in `config.py` defaults / `memory_settings` / YAML configs)
against read sites. Keys defined-but-never-read and read-but-never-defined are exactly the drift
class that produced the `_global`-vs-`_default` channel bug. Warnings-only at first (same posture
as parity's undocumented class). This script's key inventory is also the seed data for the
Settings Manifest — earning its keep twice.

### WS-7 — Client-side dead-code + dependency parity with Python
Python has curated ruff + pip-audit; the client has tsc only. Two flags close the gap: **knip**
(unused exports, files, and dependencies in `client/src` — the cross-platform UI accretes dead
components faster than any other layer here) and **`bun audit`** as `lint:client:audit` beside
`lint:python:audit`. Both are single-config, run-in-`check:static` tools; knip findings start
warnings-only until the first sweep lands.

### WS-8 — Gate-script protocol, written down once
`check_docs.py`, `check_api_parity.py`, `check_pyright_baseline.py` (+ WS-5/WS-6) all converged on
the same shape: errors-vs-warnings, `--strict` promotion, ratchet baselines, emoji summary, exit
codes. Spend ten lines in `scripts/README.md` (or a docstring-level note in each) declaring that
protocol — errors block release, warnings inform, `--strict` promotes, baselines only shrink — so
the next gate is written to spec instead of by imitation. Add one agent-first clause: **every
finding states its fix** ("orphan: X is not linked from Todo.md" already does this; make it the
rule). The primary consumer of gate output here is an agent that will act on it in the same
session — a finding that names its remedy is a finding that costs zero follow-up turns.

### WS-9 — Context-budget gates for agent-loaded docs
`CLAUDE.md` and the memory index are loaded into *every* agent session — their size is a per-turn
context tax, and they grow by accretion (the repo already split Todo.md for exactly this reason).
Mirror the existing Release-Notes size warning in `check_docs.py`: warn when `CLAUDE.md` exceeds
its current footprint (~18 KB today — baseline at that, only ratchet down), so "the light index
stays light" is enforced by the gate that already guards its siblings, not by periodic noticing.
~6 lines; purely a warning. This is the
agent-first analog of keeping a hot path allocation-free.

### Considered and rejected (so they stay rejected)
- **Coverage mandates / thresholds** — measures lines, not safety; the eval harnesses
  (`eval_consolidation`, planned `eval_recall`) are this repo's real quality instruments.
- **The pre-commit framework, commitlint, husky** — each imports a config ecosystem to do what a
  3-line hook + existing discipline already does. WS-2's plain hook is the ceiling.
- **DRF/spectacular migration** — settled in Q2 above; spec-first + conformance testing.
- **Monorepo orchestrators (nx/turbo/moon)** — Taskfile is already the orchestration layer and is
  doing the job; two task runners is drift by construction.
- **CODEOWNERS / PR-template machinery** — review-routing ceremony for human teams; in an
  agent-first workspace the Documentation Map *is* the ownership model, and gates outrank
  reviewers.

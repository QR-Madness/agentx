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
the main transcript.
**Guard:** `AmbassadorStorageTest` (pollution regression, recovery-isolation).

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

### ADR-5 — The main chat path stays strict on model resolution
**Decision:** chat uses `get_provider_for_model` (the agent's chosen model is the floor); only
*non-chat features* (memory/recall/recap/reasoning/…) use `resolve_with_fallback` to never hard-fail.
**Why:** silently swapping the user's chosen chat model is worse than surfacing the error; background
features failing the turn is the thing we actually want to prevent. **Source:** CLAUDE.md / Model
Resolution (`Development-Notes.md`).

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
**per-file-ignored** on the four audited raw-SQL files (`views.py`, `memory/semantic.py`,
`eval_consolidation.py`, `init_memory_schema.py`) — their SQL binds all values (`%s`/`:name`/Cypher
`$params`) and interpolates only static fragments/identifiers; per-line `noqa` was rejected because
the violations sit on multi-line `f"""` openings (can't append a comment without churning hot query
code). `S608` stays active everywhere else to catch *new* raw SQL.
**Why:** in a utilitarian repo, a linter that emits hundreds of findings against intentional idioms
trains you to ignore it. Lean rules that *earn their keep* stay green and get read. The real fix for
the raw-SQL surface is the §1.5 repository layer (Memory-Roadmap), not `noqa`-decoration.
**Guard:** `task lint:python` runs clean at HEAD; `task docs:check` + `task audit` (pip-audit CVEs)
round out the gate. **Source:** this session's static-analysis pass; pyproject comments point here.

---

## Maintenance

- Add an entry when you make a decision you'd be annoyed to see re-litigated, or discover a property
  that must hold. Keep each to a few lines; link the test and the `todo/`/roadmap item.
- When you add an INV with `⚠ no guard`, also drop a line in the relevant `todo/backlog/` file so the
  guard gets written (the `tests_invariants.py` collection is Memory-Roadmap T3).
- If a change *intentionally* reverses an entry, edit the entry in the same change with the new why —
  don't leave it contradicting the code.

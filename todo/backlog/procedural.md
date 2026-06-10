# Procedural Memory

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.

---

### ⭐ Procedural Memory — a constant thought (encode → replay → activate)

> **Pre-flight verdict: it's wired but inert** — four independent breaks. ① internal tools never
> record (chat loop → `call_tool_sync` → `execute_internal_tool` bypasses the `tool_executor` recorder;
> only external MCP tools record). ② `detect_patterns` keys off `(:Conversation)-[:RESULTED_IN]->
> (:Outcome {success:true})`, but `Outcome` nodes are created **only** in the eval harness — never the
> live path → zero strategies learned. ③ even if learned, `MemoryBundle.to_context_string` renders
> turns/facts/entities/goals but **omits strategies** → they never reach the prompt. ④ the
> consolidation worker runs only via `task memory:worker`; `task dev` doesn't start it. Don't revive the
> coarse `detect_patterns` ("Use {tools} for {task_type}") — rebuild it **brain-modeled**. Unifies with
> the **Active Memory Recall** tiers above (Tier-1 watchdog = the reflex core; Tier-3 = deliberate
> self-query). Seed signal: the persisted steer corrections (`metadata.steered`).
>
> **Slice 1 update:** ③ and ④ **fixed** (reflex-core renders into the prompt; the autonomous worker now
> awaits coroutine jobs — it was silently dropping the async `consolidate` — and `task dev` runs it).
> The new corrections/rules → `Procedure` path runs the brain-modeled distill (encode → distill →
> reflex-core activation). ① and ② (internal-tool sequence evidence + a live Outcome/success signal,
> which feed tool-**sequence** procedures) remain for a later slice.

> **Brain model.** (a) *General → specific*: the model already has the general baseline, so store only
> the **delta** — the project/user/domain "how we do it here," corrections, learned habits. (b) *The
> asymmetry is encoding + recall, not retention*: agents store durably (no rehearsal needed), but a
> stored-yet-never-recalled procedure is **functionally forgotten** → engineer **active recall**, not
> rehearsal. (c) *It's a constant thought*: three always-on loops, not a batch afterthought.

- [ ] **Useful-pattern spec (the first step)** — a pattern is stored only if it passes: (1) a
      **high-signal event** — corrections/**steers** (`metadata.steered`, the best signal) › explicit
      rules ("always…/we prefer…") › failure→recovery › repetition › novel successful sequences; (2)
      **baseline-deviation** — a fast model discards anything a competent agent would already do by
      default (store the *delta*); (3) **reusable + evidenced** (scoped, backed by a real event).
      Scope hierarchy `_global → user → project → _self_{agent} → conversation`; recall prefers
      most-specific.
- [x] **Loop 1 — Encode (every turn, cheap)** — **Slice 0 shipped.** Stages `correction` candidates
      (from the persisted `steers_data` — `after_tools`/`steer_round`/`phase`) + `explicit_rule`
      candidates (`procedural.detect_explicit_rule`, heuristic, no LLM) into a new `procedure_candidates`
      PG table, from the `_persist_turns` daemon on the streaming chat path (both normal + hard-stop).
      Count surfaced on `/api/memory/stats`. **Remaining for the loop:** failure-marker capture +
      repetition detection. [[metadata.steered]]
- [~] **Loop 2 — Replay / distill (consolidation = "sleep")** — **Slice 1 shipped** the candidate→
      Procedure half: the async `distill_procedures` consolidation job (`consolidation/jobs.py`, in the
      registry default pipeline + autonomous worker) reads pending `procedure_candidates`, groups by
      derived scope (corrections with an `agent_id` route to `_self_{agent_id}`; explicit rules stay on
      their channel), runs `ExtractionService.distill_procedure` (baseline-deviation filter with
      signal-aware deference — explicit rules are kept unless redundant), and **strengthens** a
      cosine-similar existing Procedure (`procedural_dedupe_threshold`) instead of duplicating; candidates
      flip to `distilled`/`discarded` (+`distilled_into`). **Remaining:** tool-**sequence** replay /
      invariant-core abstraction, `ReflectiveReasoner` correction-reflection, Fix ② (emit a success/
      Outcome signal) + Fix ① (record internal-tool sequences as evidence).
- [~] **Loop 3 — Activate (every turn; the hard part) — gate, don't retrieve** — procedural recall ≠
      content similarity (a conditional trigger→sequence isn't *similar* to the prompt). **Index by
      trigger, query by situation.** Four modes: **reflex core** (top general/project procedures
      injected every turn, maintained not searched = Tier-1 watchdog) — **shipped in Slice 1**
      (`ProceduralMemory.get_reflex_procedures` top-`strength` over recall channels, attached at the
      `remember()` boundary, rendered by `MemoryBundle.to_context_string`, gated by `reflex_core_enabled`/
      `reflex_core_limit`; Fix ③ done); **activation nerve** (match a *situation descriptor* built from
      goals+summary+fast-model intent-tag+entities+next-tool against trigger conditions — fires the
      procedure); **point-of-action** (action-bound procedures inject at the tool-call boundary);
      **deliberate** (`recall_procedures` self-query = Tier-3) — all three remain.
- [x] **`Procedure` model (richer than `Strategy`)** — **Slice 1.** New Neo4j `Procedure` node +
      `models.Procedure` `{trigger (NL) + trigger_features, body (replayable), rationale, scope, strength
      (replay/reinforce count), evidence_refs, signal_kinds}` with `procedure_embeddings` vector index;
      `learn_procedure`/`reinforce_procedure`/`find_procedures`/`list_procedures` reuse the (dead)
      `learn_strategy`/`reinforce_strategy` write pattern. Inspect via `GET /api/memory/procedures` +
      `/api/memory/stats` `procedures` count. Fix ④ done — `ConsolidationWorker` now awaits coroutine
      jobs (the autonomous `consolidate` was silently no-op'ing) and `task dev` runs the worker; manual
      `task memory:distill-procedures`.


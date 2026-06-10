# Active Memory Recall

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.

---

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


# Research Mode — Follow-ups

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Research Mode (deep, cited, self-reviewing research) — follow-ups

> **Shipped (v1, `[v0.21.201]`):** per-conversation `research_mode` (🔭 composer chip, gated by
> `research.enabled`) → elevated cost-aware search budget (`search.research_per_turn_limit`,
> `web_research.budget_weight`), budget/cost awareness stamped on every search result
> (`search_budget.snapshot()` → `est_cost_usd`), accurate metering (Brave costed, extract/map/crawl
> billed), bounded + `web_research`-caching search chain, a rigorous evidence-grounded self-reviewing
> **research prompt** (`prompts/system_prompts.yaml::research.system`), forced flat single-agent path,
> durable Project-doc deliverable, and a **Settings → Research Mode** section. Internals:
> [Development-Notes.md → Research Mode](../../Development-Notes.md). Single-agent topology by design.

Open:
- [ ] **Structured `form` Exhibit for scope/depth intake** — a modular multi-field element (depth /
      breadth / audience / output format in one card) surfaced as a **user-invoked deep-research
      aide**, replacing v1's prose clarifying questions. Build points: `lib/exhibits.ts` element union
      + `elementFromWire`/`KNOWN_ELEMENT_TYPES`, a renderer under `components/chat/exhibits/`,
      `elementRegistry.ts`, and a backend `present_exhibit` emit; reuse the `choice` submit-back +
      amend-on-same-id lifecycle. Keep it modular so the same element powers delegated deep-research.
      (Pairs with [exhibits.md](exhibits.md).)
- [ ] **Delegated deep-research (peer-review + combined reports)** — a lead + specialist team via
      `alloy/` delegation: fan out research to specialists, then a reviewer (gather → write → review).
      Needs context-calculation work: delegated specialists compose their own persona in
      `alloy/executor.py::_build_specialist_messages` and don't inherit per-turn ledger blocks, so the
      research framing must be passed to them; combined-report assembly needs budget/context
      accounting. (Pairs with [translation-web-search.md](translation-web-search.md).)
- [x] **Anti-premature-stop / delivery guard** `[v0.21.202]` — the prompt alone under-delivered on
      the first live run (zero deliverable), so v1.1 shipped the coded guard: a one-shot
      `finalize_nudge` (proactive near round exhaustion + reactive at a natural stop, gated on
      `ToolLoopResult.docs_written == 0`), a generic **round-exhaustion synthesis floor**
      (`for/else` in `_run_tool_loop` — a turn can never again end silently with
      `finish_reason=tool_calls`), and the **`web_research` async fix** (Tavily research() is
      initiate-only; the report arrives via `get_research` polling — v1 always got an empty report
      in ~210 ms and called it success).
- [ ] **Brave capability expansion** — a curated **Goggle** for source-quality re-ranking (boost
      primary/authoritative sources, discard content farms); the **`llm-context`** endpoint as a
      grounding backend (pre-extracted, token-budgeted content — could collapse search→extract into
      one call and cut extract spend); and **`answers`** as an alternate deep-research backend
      (complement/fallback to Tavily `web_research`).
- [ ] **Dollar-denominated spend ceiling + composer cost-meter** — gate research on cumulative $
      (not just call count), with a live per-conversation cost chip in the composer.
- [ ] **Inline source URLs in saved reports** — v1.1's acceptance report names its sources (graded
      table) but doesn't hyperlink them; the URLs exist in the conversation's auto-captured
      `citation` exhibits (Bibliography). Bridge them into the document the agent saves (prompt
      guidance or a post-write enrichment pass).
- [ ] **Catalog-miss guardrail UX** — Research Mode logs a warning when the model resolves the
      8192-default context (`views._research_finalize_nudge` sibling `_research_min_output`), but
      the user only finds out server-side. Surface an in-chat notice + one-click "set Model Limits"
      (the v3 acceptance run needed a manual `context_limits.models.{id}` override to un-starve a
      1M-context model).
- [ ] **Report editorial pass** — small polish: ground the report's header date via the persona
      `{date}` (v3 wrote "July 2025" against later-dated sources), section-numbering hygiene.

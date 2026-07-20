# Exhibits — Rich Agent-Authored Content

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Exhibits — Rich Agent-Authored Content (declarative content-part protocol)

> The agent presents structured content the client renders from a registry — rather than
> hand-rolling raw HTML (a security/consistency liability). Vocabulary: a **Gallery** (a
> conversation's array of exhibits) → **Exhibit** (one declaratively-arranged unit, amendable by
> stable `id`) → **Element** (typed building block). Producer is the declarative internal
> `present_exhibit` tool (not fence-scraping) — the same mechanism interactive elements need.
> Visual sibling to the 16.6 Ambassador Agent (which mediates via voice/briefing); this mediates
> visually. Same typed structure doubles as the export/integration payload above.

**Shipped (Slices 1–5, `[v0.21.25]`–`[v0.21.29]`) → [roadmap.md](../../docs-site/src/content/docs/roadmap.md):**
protocol + `present_exhibit` tool, `mermaid`/`choice`/`table`/`citation` elements, `web_search` +
`web_research` citation auto-capture, capability-aware Tavily web tools (search/extract/map/crawl/
research via the `tavily-python` SDK), universal model fallback + bulk/inherited memory-stage models,
and the static client-only conversation **Bibliography** ("Sources" drawer).

**Shipped (multi-modal content slice, `[v0.21.237]`):** `text` element (markdown via the chat
pipeline), media elements `image`/`audio`/`video` (served-blob-only URL validator — the
exfiltration guard), **`grid` layout**, auto-emitted media exhibits (`generate_image` /
`generate_speech` / MCP media passthrough) now persisted as synthetic `present_exhibit` turns
(reload-safe). See Development-Notes → "Multi-modal content protocol".

Open:
- [ ] **Advanced memory visualization** (interactive graph, embedding clusters) as a registered
      element type (the `text` element itself shipped v0.21.237).
- [ ] **Active-citation context-injection** — fold `active` sources' `quote`s back into the agent's
      context (bounded) so it can reference tracked sources later (the "tracked in the chat" payoff).
      Still to do: `memory` citations deep-linking into the Memory drawer fact, and `web_extract` →
      `active` citation promotion.
- [ ] **Per-turn web search/research credit budget** — Tavily burns credits fast. Allot each turn a
      **credit budget** (config `search.credits_per_turn`, e.g. 15) that web tools spend by a
      **weighted cost** (`web_search` ~1, `web_extract` ~2, `web_crawl` ~5, `web_research` ~10 —
      tunable, mirroring real API cost). **Every web tool result returns `credits_remaining`** so the
      model self-rations; once exhausted, calls return a clear "budget exhausted" error instead of
      looping. Track the per-turn tally in the tool loop / internal context (reset each turn).
- [ ] **Truly long `web_research` → background job** — move minutes-long research off the synchronous
      tool path onto the `/api/chat/background` queue so it can't block a turn.
- [ ] **Configure the global default model (UI gap)** — two latent keys exist
      (`preferences.default_model`, `models.defaults.chat`) with **no settings editor**. Live turns
      are safe (agent profiles carry a model = the fallback floor), but background consolidation has
      no floor without one. Add a picker in settings.
- [ ] **`form` (multi-field) interactive element** — multiple inputs submitted together as one
      turn; builds on the `choice` next-turn mechanism.
- [ ] **Richer layouts beyond `grid`** (shipped v0.21.237) + a dedicated browsable **Gallery panel**
      (drawer) listing a conversation's exhibits.
- [ ] **Inline-fence fallback** — also render the model's *native* ` ```mermaid ` fences (no tool
      call) by parsing them into exhibits, for models that under-reach for the tool.
- [ ] **Exhibits in delegation streams** — extend the typed event to `delegation_chunk` so a
      specialist's diagrams surface too.


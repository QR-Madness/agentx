# Translation Quality & Web Search / Delegation

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Translation Quality Overhaul (pluggable `TranslationKit` backend)

> NLLB-200 graded 5/10 — but we just invested in it (GPU accel `[v0.21.6]`, `LanguageLexicon` ISO-code
> bridging), so the move is *pluggable backend*, not rip-and-replace. (Caveat on the eval: Mistral
> grading NLLB output while itself relying on NLLB is a soft/circular benchmark.)

- [ ] **Pluggable translation backend behind `TranslationKit`** — interface so backends swap without
      touching the `LanguageLexicon` code-bridging or call sites.
- [ ] **LLM-provider translation path** — route high-value pairs through the existing model-provider
      stack (reuses the provider abstraction, no new dependency) and keep NLLB-200 as the cheap offline
      fallback.
- [ ] **Evaluate stronger open models** — SeamlessM4T / MADLAD-400 / Tower as alternative offline
      backends; pick on a non-circular eval.

### Web Search & Delegation — shipped + deferred

> Shipped (see plan `~/.claude/plans/i-can-t-do-it-unified-pond.md`): internal `web_search` tool
> (**Tavily** primary + **Brave REST** fallback, in-tool retry + short-TTL cache; Brave MCP server
> auto-connect disabled), `search.*` config + **Settings → Web Search** UI; **parallel fan-out
> delegation** (`alloy.max_parallel_delegations`, reentrant `AlloyExecutor`, queue fan-in in
> `_run_delegations`); **delegatable agent profiles** (`available_for_delegation` flag + filter,
> tool-gating persistence-bug fix, **Settings → Multi-Agent** toggle, **Researcher** preset); profile
> editor **hybrid Tabs+Accordion** UX. SearXNG was dropped in favor of Tavily (no proxy/blacklist ops).

Deferred — **Search Router** subsystem ("browsing on autopilot"); a delegatable Researcher already
covers ~80% of this via its own tool loop:
- [ ] **`fetch_page` tool** — trafilatura (static) now, Playwright (JS-heavy) later — lets a
      Researcher read full pages, not just snippets.
- [ ] **Autonomous browse loop** — search→fetch→follow→synthesize with confidence/termination
      heuristics (a `research` tool or ReAct-derived `ResearchAgent` via `reasoning/orchestrator.py`).
- [ ] **Group-based tool gating** — consume the latent `groups` field (`mcp/server_registry.py`) to
      route a set of web tools into a managed lane (today's per-profile `allowed_tools` suffices).
- [ ] **Router lifecycle subsystem** — per-tool rate limiting, session state, shared cache, backend
      rotation beyond the in-tool Tavily→Brave fallback.
- [ ] **SearXNG self-hosted backend** — optional fully-self-hosted `web_search` backend (needs
      residential/ISP proxy in `settings.yml`); slots behind the existing pluggable tool.


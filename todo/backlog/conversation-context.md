# Conversation Context, Checkpoints & Memory Area UX

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.

---

### Conversation Context & Checkpoints

- [x] **Include prior conversation context every turn (near-verbatim)** — shipped `[v0.21.30]`. The
      in-memory `SessionManager` is now **rehydrated** from the durable `conversation_logs` transcript
      on a cold session (`agent/conversation_history.py`, before the new turn), so resumed/restored
      conversations keep their history. Per-turn context is assembled by
      `ContextManager.assemble_turn_context` — SYSTEM preamble + recent **verbatim** transcript up to
      `context.verbatim_budget_ratio` (0.7) of the model's real window, oldest overflow covered by the
      rolling summary. The memory recall's old current-conversation turn-dump (a band-aid) is dropped
      to avoid double-injection. Tests: `ConversationContextTest`.
- [x] **Context-window-based summary/compression triggering** — shipped `[v0.21.30]`. The rolling
      summary (what fired "early" on a fixed message count) is now **token-triggered**:
      `SessionManager.maybe_update_summary` summarizes aged-out turns only when the verbatim transcript
      crosses `verbatim_budget_ratio` of the window (keeping a `recent_floor`), and the summary is
      **persisted** in Redis so it survives a cold rebuild. (The model-authored `checkpoint` tool has
      no auto-trigger — also hardened: anchor-preserving eviction + a `replace` mode.)
- [ ] **Redis/Postgres-backed live session store** — rehydrate-from-logs (shipped) re-reads the DB on
      a cold session; a durable session store would survive restarts without the per-turn read and
      across workers.
- [ ] **Rolling summary as a first-class `conversations` column** (vs. the current Redis TTL) for
      durability beyond 30 days.
- [ ] **Hydrate the Alloy / background-chat paths** too — this slice rehydrates the main streaming
      chat; the multi-agent + queued-chat paths build their own context.
- [ ] **Stable memory core (kill transient memory injection)** — today the injected memory is
      **transient**: `views.py` calls `agent.memory.remember(query=message)` **every turn**, so the
      facts/entities re-rank against the current message and shift turn-to-turn (the agent "sees" a
      fact one turn, not the next). Inject a **stable, high-salience core** (durable facts/entities for
      this user/channel) as a persistent preamble block consistent every turn, with query-specific
      recall as a small *supplement* on top. Goal: minimal transient context. Slots into the same
      `assemble_turn_context` SYSTEM-preamble budget (and is exactly what the Context Inspector would
      surface).

### Memory Area UX Cleanup

- [~] **Redesign the Memory area** — Memory moved from a cramped right-side `xxl` drawer to a
      **full-screen modal** (`type:'modal', size:'full'`, mirroring Settings/Toolkit; own
      `.memory-modal-content` host with a definite height so the panel fills the dialog and scrolls
      internally). Also fixed the procedure "When when …" render doubling (shared `procedureHeadline`/
      `_prefix_trigger` helpers, render-only). **Still TODO:** the *drastic cleanup* +
      **document every feature in-UI** (per-control abstract descriptions) — deferred from this pass.
- [ ] **Memory-mending agent (memory janitor)** — an agent that actively explores the memory graph
      and *repairs* it: find orphaned facts (no `[:ABOUT]`), broken/dangling links, duplicate or
      contradictory entities/facts, stale context, and weakly-connected clusters; propose/apply mends
      (link, merge, supersede, prune). Build on the new manual fact↔entity link + the existing
      lifecycle ops (`dedupe_entities`, `link_facts_to_entities`, `check_contradictions`,
      `promote_to_global`) and the Fact→Entity surfacing. Likely an **Agent Alloy specialist**
      ("Memory" agent) so it reuses delegation + can run on a schedule; surface proposed mends in the
      Memory explorer for review/approve. (Requested follow-up to the Memory Explorer pass.)


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
- [x] **One compaction target everywhere + budget-anchored triggers** — shipped `[v0.21.203]`. The
      post-turn pre-warm had drifted (still fed the legacy prose summary → stale state digest on
      big-window models; never fired on small ones); all three call sites (JIT backstop, post-turn
      pre-warm, background `Agent.chat`) now route through `session.compaction_uses_state`, triggers
      anchor to the ledger's real history budget, the JIT check projects the post-registration
      coverage blocks, and the tool loop's in-turn ceiling derives from the model's real window
      (retired the flat 32k `MAX_INPUT_TOKENS` cap; optional `context.max_input_tokens` spend guard).
      See Decisions.md INV-CTX-1 rules (f)–(h).
- [x] **Conversation Context settings section** — shipped `[v0.21.203]`. One home
      (Settings → Memory → Conversation Context) for the verbatim window, state/digest compaction,
      compaction summarizer, trajectory compression (moved out of Consolidation), tool-output
      compression, episodic leads, rehydration bounds; ratios moved out of Model Limits. Backed by
      allowlisted `/config/update` sections + the settings manifest.
- [ ] **Redis/Postgres-backed live session store** — rehydrate-from-logs (shipped) re-reads the DB on
      a cold session; a durable session store would survive restarts without the per-turn read and
      across workers.
- [x] **Durable conversation state** — shipped `[v0.21.205]`. The state object (digest + slots)
      gained a durable Postgres tier (`conversation_state` table, Alembic 0006) under the Redis hot
      cache: write-through on save, read-through + re-warm on a Redis miss — compaction coverage
      survives the 30-day TTL and a Redis wipe.
- [ ] **Hydrate the Alloy / plan-executor paths** too — the streaming chat and the background
      `Agent.chat` path now rehydrate + budget-fit; Alloy specialists are task-scoped by contract
      (deliberate), but the plan-resume path still builds its own context.
- [x] **Digest expandability + >rehydrate-cap coverage** — shipped `[v0.21.205]`. The digest is a
      summary, not the record: `read_thread(conversation_id="current", center_turn=N)` now pulls the
      verbatim turns behind it — served from the durable transcript (`conversation_logs` via
      `load_turn_window`), not episodic memory, so it works even with the memory system off; no
      `center_turn` returns the earliest (aged-out) turns (ambient-context resolved; anchor line in
      the digest render, coaching layer v2, and the overflow notice all teach it). With the durable
      digest above, coverage OR retrievability always exists — including beyond the rehydration cap.
- [x] **Stable memory core (kill transient memory injection)** — shipped (Foundation #3): the
      salient core is the prio-70 maintained-not-searched block; `remember(query)` is the prio-30
      supplement, deduped against the core. (Record fixed — this had shipped but was never checked
      off here.)

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


# Retrieval Quality, MCP Tools & Extraction

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.
> Research: [extraction survey](../research/2026-07-extraction-research.md) ·
> [recall survey](../research/2026-07-memory-recall-research.md) (archived 2026-07-05, code-verified;
> the detailed designs live in Memory-Roadmap §2.10/§2.11 — checkboxes here, detail there).

---

### Retrieval Quality Enhancements (migrated from docs/future-feature-pool)
- [x] **Two-stage recall — candidate pool + cross-encoder stage** `[v0.21.155]` —
      `recall_candidate_pool=50` over-fetch + the cross-encoder relocated to a RecallLayer
      **post-fusion** stage with bounded demotion (`recall_ce_max_demotion=2`); gate result
      **MRR +20.3pp / r@1 +29.3pp / p95 ~450ms** → `cross_encoder_enabled` default-ON. The
      first-person guard shipped **default-OFF** (0.0 abstention under CE at −3pp MRR). Absorbs
      the former [misc.md](misc.md) "Cross-encoder reranking model" one-liner. Verbatim-turn
      fuse-vs-parallel decision still parked in §2.11 (turns stay a documented parallel path;
      the CE stage reranks them in their own list). Detail: Memory-Roadmap §2.11/§2.7.
- [ ] Working Memory Scratchpad — always prepend a structured scratchpad (current topic/task, active entities, recent corrections, open questions) to context for coherence/orientation
- [ ] Conversation Summarization — maintain rolling per-session and per-topic summaries; retrieval becomes `recent_turns + relevant_summaries + relevant_facts`
- [ ] Query Intent Classification — classify query before retrieval (follow-up → recency, callback → older history, new topic → broad semantic, factual recall → entities/facts); rule-based or lightweight LLM
- [ ] Negative/Correction Tracking — when `correction_detection_enabled`, mark superseded facts `temporal_context: "past"`, link corrections to originals, prioritize corrections in retrieval
- [ ] Fact Staleness Detection — add `expected_stability: transient|stable|permanent` and surface staleness warnings (relates to Fact Transience above)
- [ ] Multi-hop Entity Traversal — add a lightweight path-finding retrieval mode over the entity graph (e.g. User → works_at → Company → has_project → Project → uses_tool → Tool)
- [x] Memory graph relationship connections — shipped as a Fact↔Entity link tool. Backend
      `link_fact_to_entity`/`unlink_fact_from_entity` (user-scoped ABOUT MERGE/DELETE) +
      `POST/DELETE /api/memory/facts/{id}/entities`; `FactDetail` "Mentioned entities" section has a
      searchable, channel-scoped entity picker (link) + per-chip unlink, optimistically updating the
      fact. (Deferred: drag-to-connect directly on the ReactFlow canvas.)

### MCP Tools (migrated from docs/future-feature-pool)
- [ ] Conversation MCP Tool — expose memory as MCP tools for external agents: `memory_recall(query, filters?)`, `memory_store(fact)`, `conversation_summary(conversation_id?)`

### Extraction Improvements (migrated from docs/future-feature-pool)

> **Extraction v2 umbrella** — designs in [Memory-Roadmap §2.10](../../Memory-Roadmap.md); each
> sub-slice eval-gated by `eval_recall`/`eval_consolidation`.

- [x] **Slice 1 — conversation-windowed extraction + honest entity resolution** `[v0.21.154]` —
      windowed multi-turn extraction (registry + overview + split/truncation retry + per-fact
      `source_turn_id` provenance), within-batch entity dedup, write-time semantic linking band
      (auto ≥0.90 / log-only 0.75–0.90), entity-embedding backfill, relationship endpoint
      recovery, shared node-merge util + `dedupe_entities --semantic`, eval_consolidation
      graph-honesty + multi-contact business cases. **19/19 PASS** with
      `openrouter:nvidia/nemotron-3-ultra-550b-a55b` (before, per-turn: dedup/edges/provenance
      all failing). Detail: Memory-Roadmap §2.10.

- [ ] Canonical Fact/Entity/Edge schema + **structured outputs** — LM Studio `response_format`
      json_schema plumbed through `ModelProvider.complete`; Pydantic validate + one retry; retire
      the regex/truncated-JSON repair paths
- [ ] Claude Sonnet for Extraction — now the **conditional routing rule** (was: wholesale switch):
      only high-surprise/hard turns, and only if constrained local extraction stays <90%
      schema+semantic validity on the golden set
- [ ] Write-time salience score — 1–10 importance rated at consolidation (replaces the constant
      0.5 fact salience); feeds the salient core + spaced-repetition half-life
- [ ] Atomic-proposition normalization before pgvector embedding (Dense X; eval-gated)
- [ ] NLI entailment gate — verify each candidate fact against its source turn; drop/downgrade
      non-entailed (demote to sampled audit if rejection <2%)
- [ ] ADD/UPDATE/DELETE/NOOP merge vocabulary for the consolidation update phase, novelty-gated by
      the surprise score (Memory-Roadmap §3.3); DELETE = tombstone
- [ ] Gleaning second pass ("did you miss anything?") on high-surprise turns
- [ ] Date fidelity — preserve surface date strings in extracted facts (facts augment, never
      replace, the verbatim turn store)
- [ ] Improved Extraction Prompts — few-shot examples, better schema definitions, domain-specific tuning; version via `pipeline_version` so calibration/thresholds re-key automatically


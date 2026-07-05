# Retrieval Quality, MCP Tools & Extraction

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.
> Research: [extraction survey](../research/2026-07-extraction-research.md) ·
> [recall survey](../research/2026-07-memory-recall-research.md) (archived 2026-07-05, code-verified;
> the detailed designs live in Memory-Roadmap §2.10/§2.11 — checkboxes here, detail there).

---

### Retrieval Quality Enhancements (migrated from docs/future-feature-pool)
- [ ] **Two-stage recall — candidate pool + cross-encoder stage** (→ [Memory-Roadmap §2.11](../../Memory-Roadmap.md)):
      widen the hybrid over-fetch from `top_k*2` (=20/arm) to a configurable ~50–100 pool, then
      relocate/enable the shipped-but-off cross-encoder (`cross_encoder_enabled`) as a RecallLayer
      **post-RRF** stage; eval-gated (≥+5pp MRR on `eval_recall`). Absorbs the former
      [misc.md](misc.md) "Cross-encoder reranking model" one-liner. Includes the verbatim-turn
      fusion review (raw turns currently bypass RRF).
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


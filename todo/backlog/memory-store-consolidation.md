# Memory Store Consolidation — Neo4j → PostgreSQL

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
>
> **Status: RATIFIED (2026-11-12).** Decisions D1–D5 were accepted as recommended and are recorded
> as ADR-21 in [Decisions.md](../../Decisions.md). Built so far: Slice 0. Next: Slice 1.
> Supersedes the audit in [phase-19 §19.5](../phases/phase-19-cloud-operation.md), whose numbers
> were partly wrong (corrected under [What Neo4j does today](#what-neo4j-does-today)).

---

### The call

Move the whole memory store into PostgreSQL — plain tables, pgvector and built-in text search —
behind the existing memory classes, prove it with the live test suite and a recall-parity gate, and
retire Neo4j after a deprecation window. No graph extension (Apache AGE) and no new abstraction
layer.

### What Neo4j does today

Measured on master `6c647966`, from the schema a fresh database ends up with and a scan of every
Cypher string in `api/agentx_ai/` (tests excluded).

| Job | Evidence | PostgreSQL equivalent |
|-----|----------|-----------------------|
| **Node storage** | 11 labels with a uniqueness constraint (`User`, `Conversation`, `Turn`, `Entity`, `Fact`, `Goal`, `Procedure`, `Strategy`, `Document`, `Artifact`, `AgentParticipant`), 5 more used in queries (`Tool`, `ToolInvocation`, `Outcome`, `Topic`, `Preference`); 37 range indexes | One table per type, real columns for what is filtered or indexed, `jsonb` for the long tail |
| **Vector search** | 6 vector indexes (`turn`, `entity`, `fact`, `procedure`, `strategy`, `document` embeddings); 8 `db.index.vector.queryNodes` sites | `vector(1024)` columns with HNSW indexes |
| **Full-text search** | 3 indexes (`fact_search`, `entity_search`, `document_search`); **1** query site (`recall.py`, `fact_search`) | Generated `tsvector` column + GIN, as conversation search already does (Alembic `0008`) |
| **Relationships** | 25 types. All are followed one hop at a time except one traversal: entity neighbourhood expansion, depth capped at 3 (`semantic.py:816`) | Foreign keys for the ownership edges (`HAS_*`, `SUBGOAL_OF`, `FOLLOWED_BY`); one `memory_edges` table for the many-to-many ones (`ABOUT`, `RELATES_TO`, `MENTIONS`, …); a recursive CTE for the one traversal |
| **APOC** | 4 sites: `apoc.text.replace` ×2 (entity-name slug match), `apoc.create.relationship` ×2 (entity merge) | `regexp_replace`; plain `INSERT` |
| Graph algorithms | 0 `shortestPath`, 0 GDS calls | — |

**Volume:** 188 Cypher strings in 22 files. Only 61 sit in the memory classes
(`memory/semantic.py` 28, `procedural.py` 16, `episodic.py` 8, `goal.py` 6, `recall.py` 2,
`interface.py` 1). The other 125 reach around them: `consolidation/jobs.py` 40, `views.py` 20,
`portability/` 35, management commands 26, `maintenance/entity_merge.py` 3,
`agent/profiles.py` 1. (The last 2 are the filter builder, `query_utils.py`.)

**Corrections to the 2026-09-20 audit (phase-19 §19.5):**
- Memory recall's vectors are **in Neo4j**. PostgreSQL's pgvector serves Document RAG
  (`kit/workspaces/retrieval.py`) and the conversation-log mirror, not semantic, episodic or
  procedural recall.
- There is one variable-length traversal, not zero; and 25 relationship types in queries, not 20.
- 188 Cypher strings, not 173 `MATCH` sites.

PostgreSQL today: 15 tables (conversation logs and state, audit log, timeline, procedure
candidates, tool invocations, usage events, workspaces and document chunks, auth). The repo already
has a working pgvector retrieval path and a working `tsvector` search to model on.

### Why move — ranked

1. **Tenant isolation.** Neo4j Community cannot enforce it in the engine, so isolation is an audit
   of 188 query sites, forever, and `CypherFilterBuilder` scoping is opt-in (a falsy `user_id`
   widens the query). PostgreSQL row-level security makes it structural. This is the prerequisite
   for the shared stack (19.4) and is far cheaper to build once, here, than to retrofit on Cypher.
2. **Search filters after ranking.** `db.index.vector.queryNodes` returns the top *k* across every
   user and channel, then the query filters by scope; the code over-fetches a fixed 2×
   (`k=top_k * 2`). A scoped query silently returns fewer results as other channels grow, and in a
   shared store other tenants' vectors crowd out yours. pgvector applies the filter during the index
   scan (iterative scan, pgvector ≥ 0.8; the compose image ships 0.8.2).
3. **One stateful service fewer.** Neo4j holds 0.87 GB resident for a 33 MB store, Community has no
   online backup, and managed Neo4j starts at about $65/month. PostgreSQL is already required.
4. **One transaction.** Memory writes and their audit mirror live in different stores today;
   Extract's verify-then-wipe has to coordinate two databases (ADR-14). In one database it is one
   transaction.
5. **The graph features that would justify a graph engine are unbuilt and don't need one.**
   Personalized PageRank and Leiden (Memory-Roadmap §3.6/§3.7) run in-process at per-user scale; the
   PPR spike failed on graph *density* (median entity degree 1), not on storage
   ([memory-recall.md](memory-recall.md)).

### What it costs

- **About 3–5 weeks of focused work** (slice estimates below; the range is honest uncertainty, not
  padding).
- **Tests coupled to Cypher.** `tests_memory.py` has about 90 references to the Neo4j connection
  or a mocked session, and about 27 assertions on call arguments, many of them on query text. They
  move to asserting on store calls and results.
- **Recall-parity risk.** Same vectors, different index and ranking behaviour (HNSW parameters,
  filtered scans, Lucene BM25 vs `ts_rank` on the one keyword query). Gated by `eval_recall`, not by
  inspection.
- **Opportunity cost.** It delays feature work. Nothing here is user-visible until the default flips.

### Decisions

Ratified 2026-11-12 as recommended.

| # | Decision | Ruling |
|---|----------|----------------|
| D1 | End state | **PostgreSQL only.** Neo4j stays selectable for a deprecation window, then is removed. Keeping two backends forever doubles every memory change. |
| D2 | Graph representation | **Plain tables + SQL.** AGE would preserve Cypher syntax for the relationship queries only; it provides neither vector nor full-text search (most of what Neo4j does here), stores properties as `agtype` (awkward for RLS and indexes), needs a custom self-hosted image, and limits managed providers. Neon does not list it (2026-11-04; private builds on request only). Snowflake Postgres does. |
| D3 | Provider | **Stay neutral.** Depend only on `vector` and `pg_trgm`. Then Neon, Snowflake Postgres, Supabase, RDS and a self-hosted container are all candidates, and the choice is a pricing decision made later. |
| D4 | Existing installs | **A direct copy command that carries the vectors** (id-preserving, verified by counts and content hashes, resumable, dry-run first). The text-only portability envelope re-embeds on import, which is slow and makes parity harder to judge. INV-4 is untouched: the envelope stays text-only, and the copy command is a one-time move between two stores on the same embedder, not an export format. |
| D5 | Sequencing | **Slice 1 first, whatever else is decided.** It changes no behaviour and is worth having even if the move stops there. The release does not need to wait for any of this. |

### Design

- **No new layer.** `AgentMemory` already delegates to `EpisodicMemory`, `SemanticMemory`,
  `ProceduralMemory` and the goal store. Those classes become the only door to the store
  (Slice 1); PostgreSQL twins implement the same public methods (Slice 3). `AgentMemory` picks the
  set from one boot-frozen setting, `memory_store = neo4j | postgres`, read at startup like
  `embedding_provider`.
- **Ids are preserved.** Every node already has a unique `id`; PostgreSQL rows keep it, so the
  audit mirror, exports and client links stay valid across the move.
- **Schema.** `memory_entities`, `memory_facts`, `memory_goals`, `memory_procedures`,
  `memory_strategies`, `memory_turns` (or `conversation_logs`, see open question 1),
  `memory_edges (src_type, src_id, rel, dst_type, dst_id, props)`. Each has `user_id` and
  `channel`, an `embedding vector(n)` with an HNSW cosine index where it is searched, and a
  generated `tsvector` where it is keyword-searched. Via Alembic revisions; the baseline stays frozen
  (ADR-9). The vector width follows the configured embedder, substituted at migration time as
  `0002_workspaces` already does.
- **Row-level security.** `ENABLE` + `FORCE ROW LEVEL SECURITY` on every memory table, with the
  policy `user_id = current_setting('agentx.user_id')`. The session helper sets it with
  `SET LOCAL` per transaction; a query that never set it sees nothing. Channel scope stays an
  application filter. Jobs that legitimately cross users (consolidation discovery, admin and eval
  commands) run under one explicit, named system scope — a separate database role — and nothing
  else may use it.
- **Search.** Vector: cosine over HNSW with `hnsw.iterative_scan` and the scope in `WHERE`.
  Keyword: `websearch_to_tsquery('simple', …)` with `ts_rank`. Fusion and reranking are already
  store-independent Python and do not change.
- **API contract.** The memory endpoints and the graph view keep their response shapes; the store
  returns the same pydantic models (`models.py`).

### Slices

One PR per slice, or per sub-slice where one is named. From Slice 3 on, CI's `backend-live` job
runs the suite against both backends.

- [x] **Slice 0 — live tests in CI** (PR #96, 2026-11-11). The database-backed tests now run
      against real services on every PR. Without this, nothing below could be verified.
- [ ] **Slice 1 — one door** (4–6 days). Move the 125 Cypher strings outside the memory classes
      behind methods on them: 1a consolidation jobs, 1b `views.py`, 1c portability, 1d commands and
      maintenance. No behaviour change; tests move from query text to store calls. Schema-init and
      migration commands stay backend-specific. This is
      [Memory-Roadmap §1.5](../../Memory-Roadmap.md) (the repository layer) with a tighter line —
      the memory classes, not the whole kit — and it ships that section's guard: a boundary test
      that fails on a Cypher string or a Neo4j session outside the store modules, with a short
      allowlist that only shrinks.
- [ ] **Slice 2 — schema and RLS** (1–2 days). Alembic revisions for the tables, indexes and
      policies. Empty tables; nothing reads them yet.
- [ ] **Slice 3 — PostgreSQL twins** (6–10 days). One memory class per PR, behind `memory_store`.
      The contract is the existing suite, run on both backends.
- [ ] **Slice 4 — parity gate** (1–2 days). `eval_recall` on both backends over the golden set, same
      vectors: recall@k and MRR within an agreed tolerance, p95 latency no worse. A miss is tuned or
      explained before going on.
- [ ] **Slice 5 — copy command** (2–3 days). `manage.py copy_memory_store`: Neo4j → PostgreSQL with
      vectors, id-preserving, verified, resumable, with a dry-run. Run on every known install.
- [ ] **Slice 6 — flip the default** (1–2 days). New installs start on PostgreSQL; existing ones
      upgrade through the copy command; Neo4j moves to an opt-in compose profile. Docs, measured
      footprint, release note.
- [ ] **Slice 7 — removal.** Delete the Neo4j classes, image and APOC after the deprecation window.

### Provider notes

Checked 2026-11-11; prices are list, on demand, and will drift.

- **Neon** — supported-extensions page (updated 2026-11-04): pgvector 0.8.x, `pg_trgm`, pgrouting;
  **no `age` row**. Unlisted extensions are private builds requested through support, with no
  guarantee.
- **Snowflake Postgres** — lists Apache AGE and pgvector. From the credit consumption table dated
  2026-11-06 (AWS US East, Standard edition, $2.00 per credit): `BURST_XS` 0.0068 credits/hour ≈
  **$9.93/month** always-on, `BURST_S` ≈ $19.86, `BURST_M` ≈ $39.71, `STANDARD_M` ≈ $51.98; high
  availability doubles it; Enterprise edition is 1.5×. A suspended instance uses no compute credits
  (per-second billing, one-minute minimum on resume). Storage $117.76/TB-month (≈ $0.12/GB). The
  compute table is footnoted as Preview. **Unknown:** whether instances suspend and resume
  automatically — that decides the cost of a dormant tenant.

### Confidence

High on the direction and on the slice order. Lower on two things: the Slice 3 estimate (the
consolidation job is the largest and least uniform consumer), and parity for filtered vector search
and the one keyword query, which only Slice 4 can settle. If Slice 4 shows a gap that tuning does
not close, the fallback is to stop after Slice 1 with Neo4j still in place and nothing lost.

### Parked

- **Apache AGE** — see D2. Revisit only if a feature needs real multi-hop traversal that SQL
  expresses badly.
- **Kuzu** — archived upstream.
- **PPR / Leiden** — blocked on extraction density, independent of where the graph is stored.

### Open questions

None of these blocks Slice 1; each names the slice that has to settle it.

1. **Turns** (before Slice 2). `Turn` nodes (with embeddings and consolidation flags) duplicate much
   of `conversation_logs`. One table or two?
2. **Parity tolerance** (Slice 4). What difference in recall@k counts as "the same"? Propose: within
   the run-to-run noise of the current backend, measured first.
3. **Deprecation window** (Slice 6). How many releases does the Neo4j backend stay selectable?
4. **Managed target** (with phase-19 §19.4). Per-tenant instances that suspend, or one shared
   instance with RLS? The schema supports both; the pricing differs by an order of magnitude.

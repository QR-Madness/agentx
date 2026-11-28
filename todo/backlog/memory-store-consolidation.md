# Memory Store Consolidation — Neo4j → PostgreSQL

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
>
> **Status: IN PROGRESS.** Ratified 2026-11-12 (decisions D1–D5, recorded as ADR-21 in
> [Decisions.md](../../Decisions.md)); amended 2026-11-28 by ADR-22, which makes the conversion a
> hard cut (memory starts empty on the new store; no copy command, no deprecation window).
> Built so far: Slices 0, 1 and 2. Next: Slice 3, the PostgreSQL twins.
> Supersedes the audit in [phase-19 §19.5](../phases/phase-19-cloud-operation.md), whose numbers
> were partly wrong (corrected under [What Neo4j does today](#what-neo4j-does-today)).

---

### The call

Move the whole memory store into PostgreSQL — plain tables, pgvector and built-in text search —
behind the existing memory classes, prove it with the live test suite and a recall-parity gate, and
then cut over in one breaking release that removes Neo4j. No graph extension (Apache AGE) and no
new abstraction layer.

The cut resets memory: nothing is copied out of Neo4j (ADR-22). The project has no users with
saved memory, so the schema is designed for where the store is going, not around what the
prototype's graph happens to hold. Transcripts (`conversation_logs`) are already in PostgreSQL and
are not touched; anyone who wants to carry memory across can export before upgrading and import
after, since the envelope is text and re-embeds on import (INV-4).

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

- **About 3–5 weeks of focused work** as estimated at ratification (slice estimates below; the
  range is honest uncertainty, not padding). After Slice 2, with the copy command withdrawn, what
  remains is 9–15 days: the twins, the parity gate and the cut.
- **Tests coupled to Cypher.** `tests_memory.py` has about 90 references to the Neo4j connection
  or a mocked session, and about 27 assertions on call arguments, many of them on query text. They
  move to asserting on store calls and results.
- **Recall-parity risk.** Same vectors, different index and ranking behaviour (HNSW parameters,
  filtered scans, Lucene BM25 vs `ts_rank` on the one keyword query). Gated by `eval_recall`, not by
  inspection.
- **Opportunity cost.** It delays feature work. Nothing here is user-visible until the cut.
- **The cut resets memory** (ADR-22). Acceptable now, while nobody has memory saved; it would not
  be once somebody does, so the cut should not be left to drift behind a release with users.

### Decisions

Ratified 2026-11-12 as recommended. D1 and D4 were amended on 2026-11-28 (ADR-22): the user ruled
that the conversion may be a breaking, hard-cut release, because nobody has saved memory yet and
data-migration machinery would be complication without a beneficiary.

| # | Decision | Ruling |
|---|----------|----------------|
| D1 | End state | **PostgreSQL only.** ~~Neo4j stays selectable for a deprecation window, then is removed.~~ **Amended (ADR-22):** Neo4j is removed in the same release that makes PostgreSQL the store. Until that release the backend switch exists only so the twins can land one at a time and the suite can run on both. |
| D2 | Graph representation | **Plain tables + SQL.** AGE would preserve Cypher syntax for the relationship queries only; it provides neither vector nor full-text search (most of what Neo4j does here), stores properties as `agtype` (awkward for RLS and indexes), needs a custom self-hosted image, and limits managed providers. Neon does not list it (2026-11-04; private builds on request only). Snowflake Postgres does. |
| D3 | Provider | **Stay neutral.** Depend only on `vector` and `pg_trgm`. Then Neon, Snowflake Postgres, Supabase, RDS and a self-hosted container are all candidates, and the choice is a pricing decision made later. |
| D4 | Existing installs | ~~A direct copy command that carries the vectors.~~ **Amended (ADR-22): none.** Memory starts empty on the new store. Export before the upgrade and import after it is the route for anyone who wants to keep theirs; that is the existing text envelope (INV-4), not new machinery. The parity gate (Slice 4) seeds both stores from the same corpus instead of copying one into the other. |
| D5 | Sequencing | **Slice 1 first, whatever else is decided.** It changes no behaviour and is worth having even if the move stops there. The release does not need to wait for any of this. |

### Design

- **No new layer.** `AgentMemory` already delegates to `EpisodicMemory`, `SemanticMemory`,
  `ProceduralMemory` and the goal store. Those classes become the only door to the store
  (Slice 1); PostgreSQL twins implement the same public methods (Slice 3). `AgentMemory` picks the
  set from one boot-frozen setting, `memory_store = neo4j | postgres`, read at startup like
  `embedding_provider`.
- **Ids belong to a user.** A row's identity is `(user_id, id)`: the ids the memory classes hand
  out stay what they are (so the audit mirror, exports and client links keep working), but two
  users may hold the same id, which is what an imported pack needs.
- **Schema.** Ten tables, built in Slice 2 ([the schema](#slice-2--the-schema)): one per kind of
  node that has a life of its own, and `memory_edges` for the many-to-many relationships. Each has
  `user_id` and `channel`, an `embedding vector(n)` with an HNSW cosine index where it is searched,
  and a generated `tsvector` where it is keyword-searched. Via Alembic revisions; the baseline
  stays frozen (ADR-9). The vector width follows the configured embedder, substituted at migration
  time as `0002_workspaces` already does.
- **Row-level security.** `ENABLE` + `FORCE ROW LEVEL SECURITY` on every memory table, with the
  policy `user_id = current_setting('agentx.user_id')`. The session helper sets it with
  `SET LOCAL` per transaction; a query that never set it sees nothing. Channel scope stays an
  application filter. Jobs that legitimately cross users (consolidation discovery, admin and eval
  commands) run under one explicit, named system scope — a separate database role — and nothing
  else may use it. Slice 2 found that this needs a second role, for ordinary queries
  ([row-level security](#slice-2--row-level-security)).
- **Cross-user operations have their own memory class** (decided 2026-11-17, Slice 1):
  `memory/cross_user.py::CrossUserStore`. Everything else in the memory classes works within a user;
  anything that spans users (cluster counts, wipes and snapshots, agent renames that reach every
  user's copy, and as Slice 1 proceeds consolidation discovery and the admin and eval commands) is a
  method there. Its PostgreSQL twin is then the one and only user of the system role, which makes
  "nothing else may use it" checkable rather than a convention.
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
- [x] **Slice 1 — one door** (4–6 days; done 2026-11-20, PRs #116 and #119–#125). Move the 125 Cypher strings outside the memory classes
      behind methods on them: 1a consolidation jobs, 1b `views.py`, 1c portability, 1d commands and
      maintenance. No behaviour change; tests move from query text to store calls. Schema-init and
      migration commands stay backend-specific. This is
      [Memory-Roadmap §1.5](../../Memory-Roadmap.md) (the repository layer) with a tighter line —
      the memory classes, not the whole kit — and it ships that section's guard: a boundary test
      that fails on a Cypher string or a Neo4j session outside the store modules, with a short
      allowlist that only shrinks.
  - [x] **The guard** (2026-11-17): `tests_memory.MemoryStoreBoundaryTest`. A ratchet of
        (Cypher strings, `Neo4jConnection.session`/`get_driver` openings) per module outside the
        memory classes, counted from tokens so comments don't count. Baseline: **127 strings and
        37 openings in 14 modules** (the earlier 125 used a looser regex). A new module with Cypher
        fails it; so does a lowered count that wasn't written back, so gains can't be spent again.
        Schema-init and migration commands are exempt (backend-specific by design).
  - [x] **The cross-user store** (2026-11-17): `CrossUserStore` with `count_conversations`,
        `list_user_ids`, `wipe_all` and `rename_agent_entities`; `portability/cluster.py` and the
        profile manager's rename propagation now call it (guard rows for both removed; 123 strings
        and 33 openings remain).
  - [x] **1c — portability** (2026-11-17): `memory/portability_store.py::PortabilityStore` holds
        every graph query of export, import and Extract (34 strings, 4 openings): `read_collections`,
        `write_collections` (one transaction: the replace-mode wipe, every family, the turn chains),
        `count_channels` and `wipe_channels`. Like the cross-user store it is a memory class of its
        own, because these operations move a whole user's memory against one envelope rather than
        one record at a time; its PostgreSQL twin reads and writes the same envelope. The exporter,
        importer and Extract keep envelope assembly, embedding regeneration, the PostgreSQL mirror,
        artifacts and receipts. Every moved query is unchanged apart from whitespace (checked
        string by string), and the full live suite passed against real databases (1,643 tests).
        89 strings and 29 openings remain, in 9 modules.
  - [x] **1b — `views.py`** (2026-11-17): the five memory handlers that queried Neo4j now call
        the store (21 queries, moved unchanged). The channel list, channel-existence check, channel
        delete and storage-size estimate went to `CrossUserStore`, because they span users today;
        conversation delete went to `EpisodicMemory.delete_conversation`; the stats view's counts went
        to a small `memory/overview.py::MemoryOverview`. 69 strings and 22 openings remain, in 8
        modules (the consolidation jobs and the management commands).
        **Found while moving, not fixed here (behaviour-neutral slice):** `GET/POST
        /api/memory/channels` and `DELETE /api/memory/channels/{name}` span every user's memory
        (the list, the existence check and the delete have no user filter), and the channel delete
        leaves the channel's procedures behind; `DELETE /api/memory/conversations/{id}` doesn't
        check the conversation's owner. All single-user today; each becomes a real leak with a
        second user. Scope them when RLS lands (Slice 2/3), or sooner if multi-user comes first.
  - [x] **1a, part 1 — the background jobs** (2026-11-19): decay, cleanup, reset, promotion to
        `_global`, pattern detection, the conversation-owner lookup and the entity-link backfill
        (with its embedding backfill) now call `CrossUserStore`, because each one spans every
        user; the backfill's per-scope name index reads `SemanticMemory.entity_name_rows`.
        25 queries moved unchanged apart from whitespace (checked string by string); the jobs
        keep their names, return dicts and log lines. The store's writes now consume their
        results: with one session per call, an error arriving after `run()` returns (a failed
        auto-commit) would otherwise be swallowed when the session closes. 44 strings and 14
        openings remain, in 8 modules: the consolidation sweep (1a part 2, whose helpers share
        one session) and the management commands (1d).
        **Gap for Slice 3:** no live test covers these methods. They span every user and some
        destroy data, so they can't run against a developer's live store. A one-off smoke on a
        throwaway stack passed (11 checks); the PostgreSQL twin needs them as live contract
        tests gated on a sterile store.
  - [x] **1a, part 2 — the sweep** (2026-11-20): `consolidate_episodic_to_semantic` and its
        helpers no longer touch Neo4j. Each phase opens a
        `memory/consolidation_store.py::ConsolidationStore` and threads it through the helpers
        as `store` (typed, so a leftover raw query fails type-checking); it holds the 14
        per-conversation queries (duplicate and contradiction checks, entity, relationship and
        provenance writes, the turn and conversation markers). Discovery (the census and the
        pending user and assistant conversations) spans every user, so it went to
        `CrossUserStore`. 17 queries moved unchanged apart from whitespace (checked string by
        string, the marker f-string included); `mark_turns` now refuses any marker but the two
        the sweep stamps; the writes consume their results, as in part 1. `jobs.py` holds no
        Cypher and opens no sessions, so its guard row is gone: 27 strings and 12 openings
        remain, in 7 modules (the management commands and `maintenance/entity_merge.py`, 1d).
        Same gap as part 1: no live test runs the sweep's queries; a store-level smoke on a
        throwaway stack passed (9 check groups, every moved query).
  - [x] **1d, part 1 — entity maintenance** (2026-11-20): `dedupe_entities`, the shared
        node merge (`maintenance/entity_merge.py`) and `backfill_agent_attribution` no longer
        touch Neo4j. Dedupe and merge run in one transaction held by
        `memory/entity_maintenance.py::EntityMaintenance`, which callers reach only through
        `CrossUserStore().entity_maintenance()`: its scans span every user unless one is given,
        and the cross-user store stays the one way into such work. Leaving that block without a
        commit, an exception included, rolls back, which is what a dry run relies on. The merge
        keeps its folding rules in `entity_merge.py` and calls the store for its three queries.
        The backfill's channel scan and claim rewrite went to `CrossUserStore`. 11 queries moved
        unchanged apart from whitespace; a throwaway-stack smoke passed (dry-run rollback,
        in-channel, cross-channel and semantic merges, the backfill). 16 strings and 10 openings
        remain, in 4 modules: the eval and debug harnesses and `bootstrap` (1d, part 2).
  - [x] **1d, part 2 — the harnesses and bootstrap** (2026-11-20): `eval_consolidation`,
        `eval_recall` and `debug_attribution` seed, read back and wipe through
        `memory/eval_harness.py::EvalHarness`, reachable only via `CrossUserStore().eval_harness()`
        (prefix wipes, channel readouts and user-less seeds span users). 16 queries moved
        unchanged apart from whitespace, but for one deliberate merge: the two copies of the
        user wipe (`$u` and `$uid`) are now one method. `bootstrap`'s readiness check reads the
        schema version and index states through `migrate_schema.neo4j_schema_state()`, beside the
        migrations, instead of opening a session itself. A throwaway-stack smoke passed (7 checks).
        **The guard's baseline is now empty: no module outside the memory classes holds Cypher
        or opens a Neo4j session**, schema tooling aside. Slice 1 is done.
- [x] **Slice 2 — schema and RLS** (done 2026-11-28). Alembic `0009_memory_store_tables` and
      `0010_memory_store_rls`: ten empty tables, their indexes, two roles and forced row-level
      security. Nothing reads them yet. Open question 1 (turns) is settled: two tables. The design
      and what it leaves for Slice 3 are recorded under [the schema](#slice-2--the-schema) and
      [row-level security](#slice-2--row-level-security).
- [ ] **Slice 3 — PostgreSQL twins** (6–10 days). One memory class per PR, behind `memory_store`,
      each built on `memory/scope.py`. The contract is the existing suite, run on both backends,
      except where [Slice 3 inherits](#what-slice-3-inherits) says the twin does better on
      purpose. Since the cut resets memory (ADR-22), a twin owes the graph's behaviour, not the
      graph's leftovers.
- [ ] **Slice 4 — parity gate** (1–2 days). `eval_recall` on both backends over the golden set,
      each store seeded from the same corpus by the same embedder: recall@k and MRR within an agreed
      tolerance, p95 latency no worse. A miss is tuned or explained before going on. This is the
      last point at which stopping leaves Neo4j in place and nothing lost.
- [ ] **Slice 5 — the cut** (2–3 days; was Slices 5–7). One breaking release: PostgreSQL is the
      memory store, memory starts empty, and Neo4j goes in the same change (its classes, the
      `memory_store` switch, the image and APOC, `queries/neo4j_*`, `migrate_schema`). With it go
      the duplicates the two-store design needed: `conversation_logs.embedding` and its index (the
      vector then lives once, in `memory_turns`), the `tool_invocations` audit table
      (`memory_tool_invocations` carries its columns), and the never-written `memory_timeline`.
      Docs, measured footprint, and a release note that says plainly that memory resets and how to
      carry it over (export before, import after).

~~Slice 5 — copy command~~ and ~~Slice 7 — removal after a deprecation window~~ are withdrawn
(ADR-22): there is nothing to copy and no window to wait out.

### Slice 2 — the schema

**Open question 1, settled: two tables.** `conversation_logs` stays the transcript and
`memory_turns` is memory's own projection of a turn. Reasons, from the code as it stands:

- **They are two contracts already.** The transcript is written even when the embedder or the
  graph fails (`views.py` falls back to a log-only write so a conversation can always be
  restored), and it is what history, restore, conversation search and the Ambassador read. A
  memory turn exists only once it has a vector, and is what recall searches and consolidation
  sweeps. "No Memorization" is a transcript row with no projection (ADR-6).
- **The transcript is not a memory table.** It has no `user_id`, and about forty call sites
  outside the memory classes read or write it unscoped. Forcing row-level security on it would
  change all of them at once; tenant isolation for transcripts is real work, but it belongs to
  the multi-user track, not to a store swap.
- **This is the direction already resolved** in Memory-Roadmap §1.1: the log is the record, and
  everything memory holds is a projection that can be rebuilt from it.

What makes the projection rebuildable rather than a second copy to keep in step:

- `memory_turns` is unique on `(user_id, conversation_id, turn_index)`, the transcript's own key.
  Re-storing a turn replaces it (the graph appended a duplicate), and a projection row maps to
  exactly one transcript row.
- It keeps its own text, so recall and consolidation depend on neither the transcript's shape nor
  its lifetime; the text is small next to the vector.
- The vector lives in the projection only. `conversation_logs.embedding` is written today and
  read by nothing; it and its index go at the cut.

**The tables** (Alembic `0009_memory_store_tables`). Column names are the model's field names, so
a row maps onto the pydantic models without a translation table; the two exceptions are
`memory_turns.turn_index` (the model's `index`, named as the transcript names it) and
`memory_procedures.channel` (the model's `scope`; the graph stored both, always equal).

| Table | Holds | Notes |
|-------|-------|-------|
| `memory_conversations` | A conversation as memory sees it: scope, consolidation markers, archive flag | Also the outcome (`outcome_success`, `outcome_task_type`) that was an `Outcome` node |
| `memory_turns` | The projection of a turn: text, vector, consolidation markers, archive flag | Cascades with its conversation. Partial indexes hold only the turns still to consolidate |
| `memory_entities` | Entities | HNSW; `(user_id, lower(name))` for the name lookup |
| `memory_facts` | Facts | HNSW; generated `claim_tsv` (`simple`) with GIN; `(user_id, claim_hash)` for the duplicate check |
| `memory_goals` | Goals | `parent_goal_id` is a foreign key that clears when the parent goes. Vector stored, not searched, so no index |
| `memory_strategies` | Strategies | HNSW. `tool_sequence` is the array; there are no `USES_TOOL` rows |
| `memory_procedures` | Procedures | HNSW |
| `memory_tools` | A user's running statistics for a tool | Per user; the graph's `Tool` nodes were shared by every user |
| `memory_tool_invocations` | Tool calls | Cascades with its conversation. Carries the audit table's columns so `tool_invocations` can retire at the cut |
| `memory_edges` | Many-to-many relationships | `(src_type, src_id, rel, label, dst_type, dst_id)`, with `confidence`, `props` and an optional `channel` |

Every embedded table also has `embedder`, the model that produced the vector (Memory-Roadmap
§1.3), so a change of model can find the rows it has to redo.

**Kinds of node with no table**, and why:

- `User`: it is the `user_id` column.
- `AgentParticipant`: who took part in a conversation is a `SELECT DISTINCT agent_id` over its
  turns, and the latest non-empty `agent_name` is the display name, which is what the node held.
- `Outcome`: two columns on the conversation. Only the eval harness writes one today.
- `Preference` and `Topic`: read by `get_user_context`, written by nothing. The twin returns the
  empty shape.
- `Document` and `Artifact`: declared by a Neo4j migration, used by no query.

**Relationships.** Ownership and parentage are columns; the rest are rows in `memory_edges`.

| In the graph | In PostgreSQL |
|--------------|---------------|
| `HAS_CONVERSATION`, `HAS_ENTITY`, `HAS_FACT`, `HAS_GOAL`, `HAS_STRATEGY`, `HAS_PROCEDURE` | The `user_id` column |
| `HAS_TURN`, `USED_TOOL`, `PARTICIPATED_IN`, `RESULTED_IN` | `conversation_id`, as a foreign key where there is a row to point from |
| `FOLLOWED_BY` | Nothing stored: it is `turn_index + 1` in the same conversation |
| `SUBGOAL_OF` | `memory_goals.parent_goal_id` |
| `INVOKED`, `USES_TOOL` | `tool_name`; `tool_sequence` |
| `DERIVED_FROM` a turn | `memory_facts.source_turn_id` (not a foreign key: a fact outlives its turn) |
| `ABOUT`, `RELATES_TO` and any typed entity relationship, `MENTIONS`, `DERIVED_FROM` a conversation, `DISTILLED_FROM`, `SUCCEEDED_IN`, `FAILED_IN`, `SUPERSEDES` | `memory_edges` |

Two things the edge table has to get right, both from how the graph is written:

- **Two entities can be related in more than one way.** Consolidation merges on
  `RELATES_TO {type}`, so the meaning ("works for", "part of") is part of the key. It is the
  `label` column, and it is in the primary key.
- **A deleted node takes its edges with it** (`DETACH DELETE`). A polymorphic table cannot have
  foreign keys, so a trigger does it (`memory_detach_edges`), for the five kinds of node an edge
  may join; a check constraint keeps any other kind out.

**References stay inside a user.** Foreign keys are composite, `(user_id, …)`. Row-level security
does not cover this on its own, because foreign-key checks bypass it: without the composite key a
turn could be attached to another user's conversation by anyone who knew its id.

### Slice 2 — row-level security

Alembic `0010_memory_store_rls`; the app enters a scope only through
`kit/agent_memory/memory/scope.py`.

**Finding: one role is not enough.** The plan named one system role. But the bundled stack logs
in to PostgreSQL as the image's superuser, and a superuser bypasses row-level security even when
it is forced (verified: a superuser session with no scope sees every user's rows). So ordinary
queries need a role of their own to drop into:

- `agentx_memory`: what every memory query runs as. `user_scope(user_id)` does
  `SET LOCAL ROLE agentx_memory` and names the user in `agentx.user_id`.
- `agentx_memory_system`: every user's rows. `system_scope()`, for `CrossUserStore` only.

Neither can log in. Both are entered with `SET LOCAL`, so they end with the transaction and work
behind a transaction-mode pooler. A scope re-enters at the start of every transaction its
session begins, so a commit part-way through does not leave the next statement running as the
login role. Where the login role is not a superuser, `FORCE` binds it as the
tables' owner and it gets nothing by owning them (verified by running the whole migration chain
and the tests as such a role).

**The policies**, on every table:

- `memory_tenant`: `user_id` equals the named user. The setting is read through `NULLIF(…, '')`,
  because a custom setting reads as an empty string, not as unset, once an earlier transaction on
  the connection has set it; `user_id` may never be empty (a check constraint), so an empty
  setting matches nothing. It sits in a scalar sub-select so it is evaluated once per statement.
- `memory_system`: everything, granted `TO agentx_memory_system`. Granting it to the role keeps
  it out of an ordinary query's filter, which stays a plain `user_id = …` that an index can serve.
  It also tests `current_user`, because a policy granted to a role reaches every role that
  inherits from it: a login role given membership with inheritance would otherwise see everything.

Measured on PostgreSQL 16.14 with pgvector 0.8.2, 4,000 entities across two users: a lookup by id
is an index scan on `(user_id, id)`, and a nearest-neighbour query is an HNSW index scan with
`user_id` as its filter, returning only the user's rows. That is the filtered search the move was
for (reason 2 above); how it behaves at scale is Slice 4's to measure.

**Guards.**

- `tests_memory.MemoryScopeBoundaryTest`: no module but `scope.py` spells out the roles, the
  setting or a role switch, and only `memory/cross_user.py` may name the system scope (its twin's
  module joins that list in Slice 3).
- `tests_memory.MemoryStoreSchemaLiveTest`, in CI's live job: every table in
  `scope.MEMORY_TABLES` forces row-level security with both policies, and no other table holds
  memory rows; a transaction sees only the user it named, nothing if it named nobody, everything
  as the system role; a user cannot write into or move a row to another user; a reference cannot
  leave its user; a deleted node's edges go with it. Each test rolls back, so it is safe on a
  developer's database.

**Tenancy gaps this closes, once the twins are in:** the channel endpoints that span every user
and the conversation delete that doesn't check its owner (found in Slice 1b) cannot leak or
destroy another user's rows through a user scope. Their handlers still have to stop asking the
cross-user store for what is one user's business.

### What Slice 3 inherits

Things the twins should do on purpose, not reproduce from the graph. Each is a decision to
confirm when its class is built.

1. **Every row has an owner.** The graph has rows without one: `_global` entities and facts
   minted by promotion (no `user_id`, no vector), and strategies minted by pattern detection (no
   `user_id`, no channel). The tables refuse them. The twin gives a promoted copy its source's
   owner, and a detected strategy the owner of the conversation it was detected in.
2. **The turn write becomes log-first** (Memory-Roadmap §1.1). Transcript row and projection in
   one transaction, the transcript first; a turn that cannot be embedded is still written and
   projected later. For a rebuild to keep ids stable the transcript needs to carry `turn_id` and
   `user_id` (§1.1, step 0), which is an additive revision of its own; the fallback in
   `views.py` then has nothing left to do.
3. **Tool invocations are recorded for a conversation's first turn too.** The graph matches the
   conversation and silently records nothing when it does not exist yet, which is every tool
   call of a first turn. The twin creates the conversation row first.
4. **Tool statistics are per user.** `record_invocation` needs the user it is recording for.
5. **Neighbourhood expansion says what it follows.** The graph's `(e)-[*1..3]-(related:Entity)`
   follows any relationship through any node, the user node included, so at depth 2 every entity
   a user has is "related" to every other. The twin's recursive query should follow entity
   relationships, and co-mention through facts if Slice 4 shows recall needs it.
6. **`embedder` is written with every vector.**
7. **pgvector 0.8 or later**, for iterative index scans. The bundled image has it; the twin
   should say so clearly at startup if a database does not.
8. **The cross-user maintenance methods get live tests** gated on a sterile store (the Slice 1
   gap), now that `system_scope()` gives them one way in.

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
not close, the fallback is to stop before the cut with Neo4j still in place and nothing lost.

### Parked

- **Apache AGE** — see D2. Revisit only if a feature needs real multi-hop traversal that SQL
  expresses badly.
- **Kuzu** — archived upstream.
- **PPR / Leiden** — blocked on extraction density, independent of where the graph is stored.

### Open questions

Each names the slice that has to settle it.

1. ~~**Turns** (before Slice 2). `Turn` nodes (with embeddings and consolidation flags) duplicate
   much of `conversation_logs`. One table or two?~~ **Settled 2026-11-28: two**, the transcript
   and a projection keyed to it ([the schema](#slice-2--the-schema)).
2. **Parity tolerance** (Slice 4). What difference in recall@k counts as "the same"? Propose: within
   the run-to-run noise of the current backend, measured first.
3. ~~**Deprecation window** (Slice 6). How many releases does the Neo4j backend stay
   selectable?~~ **Withdrawn 2026-11-28 (ADR-22): none.** Neo4j goes at the cut.
4. **Managed target** (with phase-19 §19.4). Per-tenant instances that suspend, or one shared
   instance with RLS? The schema supports both; the pricing differs by an order of magnitude.
5. **The login role** (before the shared stack, phase-19 §19.4). The bundled stack connects as
   the database superuser, which is why memory queries drop to `agentx_memory`. A shared
   deployment should connect as a role that is not a superuser, so that a query that forgets its
   scope sees nothing rather than everything; the migrations and the tests already run that way.

# Known Future Issues & Blockers

> Part of the AgentX TODO — index: [Todo.md](../Todo.md)

---

## Known Future Issues

> Architectural concerns that may need addressing at scale

**~~First-boot init hang after model download~~** — RESOLVED `[v0.21.143]`, root cause removed
`[v0.21.145]`. On a fresh cluster (empty HF cache), `init_memory_schema` downloaded BAAI/bge-m3
mid-init, printed success, but the process never exited (non-exiting download threads keep the
interpreter alive — observed with the hf-xet backend AND with `HF_HUB_DISABLE_XET=1`), so the
entrypoint never reached uvicorn. v0.21.143 added a **post-success watchdog**; v0.21.145 removed
the cause: schema init no longer loads the embedder at all (`--validate-embedder` is opt-in), the
model download moved to the explicit `warmup_embeddings` step run only when the boot's
`manage.py bootstrap` reports the model uncached, and the watchdog now wraps only that warmup
(`AGENTX_INIT_EXIT_GRACE`, default 15s). Root-causing which library leaks the thread remains
open groundskeeping.

**Tenant scoping is opt-in, not enforced** — blocks Phase 19.4 (shared multi-tenant stack)
- `CypherFilterBuilder.add_user_filter()` (`kit/agent_memory/query_utils.py`) appends the
  `user_id` predicate only `if user_id:` — a falsy/missing id silently widens the query to every
  tenant instead of failing. Scoping is also opt-in by construction: 19 `CypherFilterBuilder(`
  constructions vs 12 `add_user_filter(` calls, and `portability/{exporter,extract,importer}.py`
  scope by channel only. Separately, **173 raw `MATCH (` queries** in the kit bypass the builder.
- Impact: **NONE today** — single-user-per-cluster means an unscoped query has nothing to leak.
  **HIGH the moment two tenants share a Neo4j instance**, where each gap is a silent cross-tenant
  read, not an error.
- Fix direction: make scoping fail-closed rather than remembered — require an explicit tenant
  context at the session/driver seam so an unscoped query raises instead of widening; treat raw
  `MATCH` sites as the migration surface. Channel scoping is *not* a substitute: channel names
  (`_self_{agent_id}`, `_project_{ws_id}`) are not tenant-unique.
- Chose to record rather than fix: this is only activated by a deployment shape that does not
  exist yet, and the fix belongs with 19.4's design, not ahead of it.

**Distributed Transaction Support**
- Dual-write to Neo4j + PostgreSQL has no transaction coordination
- Impact: LOW for single-user; HIGH for multi-user deployment

**Connection Timeout Configuration**
- Neo4j and PostgreSQL queries have no explicit statement timeouts
- Fix: Add `statement_timeout` to connection config

**Retry Logic for Transient Failures**
- No exponential backoff on transient database failures
- Fix: Add retry decorator with backoff for critical operations

**Rate Limiting on Memory Operations**
- No protection against rapid-fire memory operations
- Fix: Add per-user rate limits in AgentMemory

**Encryption at Rest**
- Conversation history and facts stored unencrypted
- Fix: Enable database-level or app-level encryption

**~~Query Embedding Caching~~** — RESOLVED `[v0.21.6]`
- Identical queries now hit an LRU+TTL cache (`EmbeddingCache`, keyed `(provider:model, text)`) in
  front of the queue (`kit/agent_memory/embedding_queue.py`). Tunable via `EMBEDDING_CACHE_*`.

**~~Embedding Request Queue / Serialization~~** — RESOLVED `[v0.21.6]`
- All embedding calls funnel through one process-wide daemon worker (`EmbeddingDispatcher` →
  `_EmbeddingQueue`, `kit/agent_memory/embedding_queue.py`): serialized so the thread-unsafe local
  model never runs concurrently, with opportunistic batching, bounded-queue backpressure, and
  exponential-backoff retry on transient (remote) failures. The public `embed`/`embed_single` API is
  unchanged, so all ~40 call sites were untouched. Lazy-started; bypassable via
  `EMBEDDING_QUEUE_ENABLED=false`. Covered by `EmbeddingQueueTest`.

---

## Blockers

None currently.

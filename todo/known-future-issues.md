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

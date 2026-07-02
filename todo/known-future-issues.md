# Known Future Issues & Blockers

> Part of the AgentX TODO — index: [Todo.md](../Todo.md)

---

## Known Future Issues

> Architectural concerns that may need addressing at scale

**~~First-boot init hang after model download~~** — RESOLVED `[v0.21.143]`. On a fresh cluster
(empty HF cache), `init_memory_schema` downloads BAAI/bge-m3 mid-init, prints success, but the
process never exits (non-exiting download threads keep the interpreter alive — observed with the
hf-xet backend AND with `HF_HUB_DISABLE_XET=1`), so the entrypoint never reached uvicorn. Fixed by
a **post-success watchdog** in `docker/entrypoint.sh` (marker seen + no exit within
`AGENTX_INIT_EXIT_GRACE`, default 60s → reap, continue boot); `HF_HUB_DISABLE_XET=1` also stays
baked in the image. Verified on a real empty-cache first boot: watchdog fired, API healthy ~6 min
after create. Root-causing which library leaks the thread (and moving the model download out of
schema init into an explicit warmup step) remains open groundskeeping.

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

# Workspaces & Document RAG

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### ⭐ Workspaces & Document RAG (major foundation gap)

> **Slices 1–2 shipped (v0.21.103–104)** — Slice 1: backend foundation
> (`workspaces`/`documents`/`document_chunks` schema via Alembic `0002`, content-addressed blob store,
> CRUD + multipart upload API, background ingestion parse→chunk→embed→auto tag+summary, quotas). Slice 2:
> two-tier retrieval (`retrieval.py`: manifest catalog + pgvector semantic), three agent tools
> (`workspace_search`/`document_query`/`read_document`, workspace-scoped via the internal-tool context),
> auto `doc` citations, and a stable **workspace-manifest ledger block** so the agent knows its corpus.
> Verified end-to-end by `api/agentx_ai/scripts/rag_e2e.py`. **Slice 3 shipped (v0.21.105)** — client UX:
> Workspaces drawer (`components/workspaces/WorkspacesPanel.tsx`, surface `workspaces`) with CRUD,
> drag-drop/picker upload + live ingest status, and conversation→workspace attach
> (`conversationMeta.workspaceId`); `lib/api/workspaces.ts` (+ FormData support in `core.ts`); the chat
> stream sends `workspace_id`. v1 complete. See `kit/workspaces/` + Development-Notes.
>
> ⭐ **Agent shells spawn *inside* the workspace — shipped v0.21.108** (`kit/shell/`): opt-in
> (`shell.enabled`), bubblewrap-jailed `run_command` + path-jailed `write_file`/`read_file`/`list_files`,
> with the workspace materialized into a per-conversation work dir as the CWD (network off, FS jailed, env
> scrubbed). See Development-Notes → "Agent Shells". **Shell follow-ups:** sync agent edits back into the
> workspace; client Settings toggle + per-agent "Allow shell" checkbox + terminal-style rendering;
> seccomp/egress allow-list; per-command confirmation. **Workspace follow-ups:** code-aware/AST chunking,
> folder/repo ingestion, `web_crawl → workspace`, cross-workspace search, Alloy-team shared workspaces,
> sha256 snapshots/restore.

> Confirmed absent: no upload endpoint, no file/document store, no workspace, no ingestion. Agents can
> RAG over their *learned memory* (Neo4j + pgvector) and the *web* (Tavily), but a user can't hand them
> a PDF / codebase / folder of docs. **Retrieval is trivial on our stack** (reuse chunker + embeddings
> + pgvector) — the real design is the **pattern**: a persistent *workspace* with a searchable
> **manifest**, and conversation→workspace tagging that **injects the file list** so the agent is
> *aware* of its corpus (a mini data-warehouse) rather than blindly semantic-searching a blob.

- [ ] **Workspace as a first-class entity** — a named, persistent container of files + metadata (CRUD,
      like agent profiles), **not** per-conversation. A **conversation is tagged to a workspace** (a
      field on the conversation/session); attach/switch is a UI action. (A workspace can later be shared
      by an Alloy team as a common knowledge base.)
- [ ] **Workspace manifest (the catalog / "data-warehouse index")** — per file `{filename, type, size,
      auto-generated tags, short summary}`, kept queryable. Retrieval is **two-tier**: (1) **manifest
      search** by filename/tag/summary → the right *file*; (2) **semantic chunk search** → the right
      *passage*. Mirrors the shipped `tool_output_section` (list → fetch) + `tool_output_query`
      (semantic) pattern, just persisted.
- [ ] **Manifest injected into context (stably)** — the tagged workspace's **file list** rides the
      Slice-6 `assemble_turn_context` preamble as a **stable** system block (names + tags only, bounded;
      aligns with the "stable core, minimal transient" principle), so the agent always knows *what it
      has* before retrieving. This awareness is what makes it a workspace, not just a vector store.
- [ ] **Ingestion (reuse, don't rebuild)** — parse (pdf/text/md/code) → **chunk**
      (`agent/tool_output_chunker.py`) → **auto-tag + summarize** (reuse the extraction/LLM infra) →
      **embed** (`kit/agent_memory/embedding_queue.py` + provider) → **pgvector** (`document_chunks`) +
      a manifest row. Upload endpoint (multipart) + durable file store + composer drop-zone + Workspace drawer.
- [ ] **Retrieval tools + citations** — `workspace_search` (manifest: name/tag), `document_query`
      (semantic chunks), `read_document` (paginated) — registered like the existing stored-output tools;
      hits auto-capture a `citation` exhibit (`source_type: "doc"`) → the conversation **Bibliography**.
- [ ] **Storage backend + quota (Docker)** — **three-store separation** (don't put bytes in Postgres):
      **(1) bytes → a blob store**, content-addressed by **sha256** (free dedup + integrity) — local
      disk at `${AGENTX_DB_DIR:-./data}/workspaces/{workspace_id}/{sha256}` for dev (matches the
      Neo4j/PG/Redis bind-mount pattern), swappable for a **MinIO (S3-compatible) container in
      `docker-compose`** on the production / multi-cluster path (Phase 17 `prod:*`/`cluster:*`).
      **(2) manifest/metadata → Postgres** (`workspaces` + `documents`: `filename, content_type,
      size_bytes, sha256, storage_key, tags[], summary, status`). **(3) vectors → Postgres + pgvector**
      (`document_chunks`, `vector(N)` + HNSW, mirroring `init_memory_schema`'s `fact_embeddings`). Join
      key = `document.id`/`storage_key`. **Per-workspace + per-user quotas** enforced at upload
      (`SUM(size_bytes)` vs a configurable byte budget; reject + notify), plus per-file size/type
      allow-lists. Wire into `task db:init` (create the dir) + `db:status`.
- [ ] *(later)* code-aware/AST chunking, folder/repo ingestion, `web_crawl` → workspace (crawl a site
      *into* a workspace), cross-workspace search.
- [ ] *(future)* **Content-addressed store doubles as snapshots / restore points** — because blobs are
      keyed by sha256 (Git-like), a "snapshot" is just a manifest of hashes and **incremental snapshots
      are cheap** (only changed blobs written). Generalizes today's bespoke snapshot/restore
      (`eval_consolidation --snapshot`, memory export/import) into one uniform primitive across **config
      + memory graph + workspaces + genomes**. This is the **safety net for the meta-layer**: it's what
      makes the autonomy envelope / evolution *reversible* (the parents can experiment because a bad
      mutation rolls back). Ties [[autonomy-envelope]] ↔ restore.


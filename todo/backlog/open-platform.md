# Open Platform — De-walling the Garden

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companion: [Memory-Roadmap.md](../../Memory-Roadmap.md) — the memory-system improvement roadmap.

---

### Open Platform — De-walling the Garden

> AgentX is today an MCP *consumer* (it eats external tool servers). This track makes it
> interoperable in the other direction: scriptable data I/O, an outward-facing contract, and
> egress. All items carry a schema-versioned envelope (`schema_version` + `agent_id`/channel
> scoping) to honor the v0.20 "migratable across platforms" rule. Cross-references the content-part
> protocol below (an export payload is the same typed structure).

- [x] **Scriptable memory import/export (JSON)** — shipped `[v0.21.22]`. Round-trippable full-graph
      export (conversations/turns, facts/entities, strategies, goals, tool-invocations + PG audit
      mirror) and idempotent re-import that `MERGE`-es every node on its stable id. New
      `kit/agent_memory/portability/` (`schema`/`exporter`/`importer`); `AgentMemory.export_memory`/
      `import_memory` delegators; `POST /api/memory/{export,import}`; `task memory:export|import`
      commands; **merge** + **replace** (scoped `DETACH DELETE`) modes. Exports are **text-only** —
      embeddings are regenerated from text on import (`[v0.21.24]`), so files are small (~9× smaller),
      deterministic, git-diffable, and portable across embedding models (the *Memory-as-VCS* basis is
      now the default behavior). `MemoryPortabilityTest` covers round-trip, idempotency, replace-wipe,
      text-only-export + recompute, and schema-version rejection. Client: Export/Import buttons in the
      Memory drawer (`lib/fileTransfer.ts` browser Blob/FileReader helper — first file I/O in the
      client). Unblocks the 18.6 eval snapshot/restore. Supersedes the old "Memory export/import
      (JSON/SQLite backup)" item.
- [ ] **Import dry-run / preview-diff** — show the graph delta before committing (mirrors the existing
      `POST /api/memory/consolidate/preview` pattern) so a bad hand-edit can't silently nuke the graph.
      *Natural next follow-up to the shipped import/export above.*
- [ ] **Verify-on-import (default on, opt-out)** — route imported facts through the three-layer
      `check_contradictions` pipeline rather than trusting JSON verbatim; opt-out flag for trusted/raw
      restores. *Natural next follow-up to the shipped import/export above.*
- [ ] **Memory-as-VCS (diffable, hand-editable snapshots)** — the text-only export + idempotent
      MERGE-on-id import (`[v0.21.24]`) already gives the core loop: export → commit/hand-edit →
      import re-applies, re-embedding from text → branchable/restorable memory states. Remaining
      polish: canonical/sorted-key JSON (+ optional NDJSON) for clean `git diff`; a
      `memory:snapshot`/`memory:restore` task pair; per-node content hash to recompute only changed
      embeddings (today import re-embeds every node); a "memory log" of snapshots.
- [ ] **Import conflict policy** — skip / overwrite / rename strategies for cross-instance id
      collisions (merge currently overwrites — importing another instance's export reassigns shared-id
      nodes to the importing user).
- [ ] **Recompute PG audit-mirror embeddings on import** — import leaves
      `conversation_logs.embedding` NULL (recall uses the recomputed Neo4j `turn_embeddings`); fill it
      from content (cheap via the embedding cache) if anything starts querying the PG vector column.
- [ ] **Per-user export** — export/import is single-user (`DEFAULT_USER_ID = "default"`) until auth
      lands; scope by authenticated user when multi-user ships.
- [ ] **Expose AgentX outward** — publish its own memory/agents as an MCP *server* + promote
      `OpenApi.yaml` to a stable public REST contract. (Subsumes the "Conversation MCP Tool" item under
      MCP Tools below — `memory_recall` / `memory_store` / `conversation_summary`.)
- [ ] **Egress / webhooks** — outbound events so external systems can react to agent activity
      (run lifecycle, new facts, goal completion).


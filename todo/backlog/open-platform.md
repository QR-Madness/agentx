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
- [x] **Web/PWA shell + connection links** — shipped `[v0.21.169]`. Same React app builds as an
      installable web PWA (iOS/Android-friendly, auto-updating, no store) alongside the Tauri
      desktop app via a compile-time `__IS_TAURI__` gate + `src/platform/` capability façade
      (import-boundary test keeps Tauri bytes out of the web bundle). Shareable `#connect=<base64url>`
      links carry a server URL + optional gateway token so a recipient connects by opening the link
      and entering only a password (`lib/connectionString.ts` + `ConnectGate` confirm). Headless-
      tested via `.claude/launch.json` (`web`). See Development-Notes → Client Surface Map.
- [x] **Cloud-deploy the PWA (Cloudflare Pages)** — manual, Taskfile-driven `[v0.21.170]`. `task
      client:deploy:pages` builds `client/dist` and `wrangler pages deploy`s it (project `agentx`,
      custom domain `agx.thejpnet.net`); `client/.env.production` sets `VITE_PUBLIC_APP_URL` so
      desktop-issued share links resolve to the hosted app; `_redirects`/`_headers` for SPA fallback
      + SW freshness. A redeploy ships a PWA update via the existing `onNeedRefresh` prompt.
- [ ] **Hosted API posture + automated rollout** — the remaining PaaS bits: the API a PWA connects to
      must allow the PWA origin (`AGENTX_PUBLIC_HOST`/`CORS_ALLOWED_ORIGINS` + `AGENTX_AUTH_ENABLED=true`,
      reachable via tunnel); and move the manual `client:deploy:pages` to CI (`CLOUDFLARE_API_TOKEN`)
      with staged/targeted rollout. Pairs with provisioning + embeddings decisions (OpenRouter vs Fly GPU).

### MCP client — remote OAuth & lifecycle

> ⭐ **Remote OAuth 2.1 shipped v0.21.163** — `mcp.client.auth.OAuthClientProvider` wired into the
> SSE/streamable-HTTP transports (PKCE, RFC 7591 dynamic registration, RFC 9728 discovery, token
> refresh); per-server token store `data/mcp_oauth/` (`FileTokenStorage`); interactive connect
> state machine (`connect_interactive` → 202 `auth_required` + consent URL → PUBLIC state-validated
> `GET /api/mcp/oauth/callback`); `auth` config block (pre-registered creds for non-DCR providers —
> what a future Google Drive import will use); Toolkit UI (auth section, waiting-poll, Reset auth).
> See Development-Notes → "MCP Remote OAuth".

- [ ] **`tools/list_changed` re-discovery** — pass a `message_handler` to `ClientSession` and re-run
      `discover_tools` on the notification (today discovery happens once at connect; refresh = reconnect).
- [ ] **Make `auto_reconnect` real** — the flag is persisted but no runtime loop acts on transient
      failures; expired-token OAuth sessions also want a refresh-then-reconnect pass.
- [ ] **Tauri deep-link OAuth callback** — for Phase-19 cloud mode where the API isn't on the user's
      localhost; the loopback callback covers desktop + browser today (`AGENTX_OAUTH_REDIRECT_URL`).


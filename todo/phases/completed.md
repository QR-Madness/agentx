# Completed / Archived Phases (11–14, 17)

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

## Phase 11: Deferred Items

> Remaining items from the memory system that didn't make the cut

- [ ] Optional LLM disambiguation for ambiguous entity matches (11.12.3)
- [ ] LLM timeout enforcement (requires async/sync architecture fix)
- [ ] Calibration factors: source, recency, corroboration, contradiction
- [ ] Negative reinforcement for corrected facts
- [ ] UI: "Where did I learn this?" — show original conversation from `source_turn_id`

---

## Phase 12: Documentation

> **Priority**: LOW

- [ ] Auto-generate API docs from OpenAPI
- [ ] Document contribution guidelines
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout Python code
- [ ] Document complex algorithms/flows

---

## Phase 13: UX Overhaul — Immersive AgentX (Complete)

> Moved to [roadmap.md](docs-site/src/content/docs/roadmap.md). All items complete:
> - 13.6 AgentXPage with ConversationContext + ChatPanel
> - 13.10 Removed old tabs dir, renamed to panels, keyboard shortcuts (Ctrl+T/W/K), CSS pass
> - 13.11 Consolidation settings UI + TopBar integration

---

## Phase 14: Context Gating (Complete)

> All 14.1-14.7 complete. See [roadmap.md](docs-site/src/content/docs/roadmap.md) for details.


---

## Phase 17: Seamless Server Management (Complete)

> **Goal**: Solid self-hosted server deployment + client connection tools.
> Complete (17.1–17.5). Moved to [roadmap.md](docs-site/src/content/docs/roadmap.md): authoritative
> `ConfigManager` config, optional session auth (bcrypt + Redis sessions, `AGENTX_AUTH_ENABLED`),
> Docker production stack (`task prod:*`), multi-cluster deployment (`task cluster:*`), and strict
> client/API version matching (`protocol_version` + `VersionMismatchPage`). Android Tauri mobile
> shipped; rate-limiting + request-audit logging dropped in favor of the edge Gateway.

### 17.6 Open

- [ ] Cloudflare Tunnel deployment documentation (pending)

### 17.7 Release & Packaging

- [x] **Client release CI** — `.github/workflows/client-release.yml`: manual-dispatch matrix
      (Windows nsis/msi + Linux deb/AppImage/rpm) building Tauri installers + SHA-256 checksums;
      version flows through `versions.yaml`.
- [x] **Server packaging — local vs isolated clusters**: single-source compose
      (`docker-compose.yml` pulls the image; `docker-compose.build.yml` overlay builds locally;
      `docker-compose.cluster.yml` → `docker-compose.gateway.yml`); self-initializing API
      entrypoint (`docker/entrypoint.sh`) + in-image `agentx` ops CLI (`docker/agentx`);
      `.github/workflows/api-release.yml` publishes `qrmadness/agentx-api` to Docker Hub;
      `task deploy:bundle` generates the isolated deployment bundle; docs at
      `deployment/self-hosting`.
- [ ] **Image groundskeeping** — the API image is ~12 GB uncompressed (~4.4 GB compressed on
      Docker Hub). Slim it: multi-stage build, drop build toolchain/caches from the final layer,
      reconsider the bundled CUDA torch wheel (CPU-only base + optional CUDA variant), and trim
      the nvm/Node install. Heavy pull for self-hosters today.
- [ ] arm64 / multi-arch API image (amd64-only today; QEMU build deferred).
- [ ] Offline "models-baked" image variant (first run downloads ~5 GB to the HF cache volume).
- [x] **GitHub Releases Automation** — the `release` job in `.github/workflows/client-release.yml`
      drafts a `client-v{version}` GitHub Release (draft for manual publish; `-suffix` → prerelease)
      with SHA-256 checksums, supported-server notes (protocol/min-client/api version from
      `versions.yaml`), and the installers attached. Download links on `deployment/self-hosting`.
- [ ] Shared-infra local clusters (one DB stack, namespaced) — deferred per the isolation-axis design.
- [ ] **Cloudflare gateway for isolated clusters** — the gateway overlay (Nginx + cloudflared tunnel,
      `docker-compose.gateway.yml` + `clusters/template/cloudflared/config.yml.example`) is wired only
      for **local** clusters (`cluster:up` includes it when `nginx.conf` exists; documented in
      `deployment/clusters`). The **isolated** bundle (`task deploy:bundle` → ships only
      `docker-compose.yml` + `docker-compose.gpu.yml`) has **no** gateway/cloudflared option, and
      `deploy/.env.example`/README never mention public exposure — yet isolated is the production path.
      Decide: ship a `docker-compose.gateway.yml` + a `cloudflared/config.yml.example` in the bundle
      (image-based, no `build.yml` overlay) and document `AGENTX_PUBLIC_HOST`/`AGENTX_GATEWAY_TOKEN`
      in `deployment/self-hosting`, or explicitly state isolated = bring-your-own-reverse-proxy.
- [ ] **cloudflared SSE/streaming timeout** — in `clusters/template/cloudflared/config.yml.example`
      the comment credits `noHappyEyeballs: true` with holding streaming (chat SSE) connections open
      past ~100s, but that flag is IPv4/IPv6 connection racing, not a response/idle timeout. Verify
      long SSE streams actually survive the tunnel + nginx; if they get cut, set the real knobs
      (nginx `proxy_read_timeout`/buffering off for SSE; confirm cloudflared has no hard cap) and fix
      the misleading comment.
- [ ] **Isolated-deploy doc accuracy** (from the v0.21.31 isolated-cluster smoke test):
      - `deploy/README.md` + `deploy/dockerhub-overview.md` say first boot downloads "~5 GB" of
        models; observed it's really ~2.3 GB (the `BAAI/bge-m3` embedding model) — translation
        (`NLLB`) is **lazy** (`/api/health` showed `translation: not_loaded`). Correct the figure.
      - Add a note that **Docker Desktop** only bind-mounts host-shared paths, so the bundle must be
        unpacked under a shared dir (e.g. `$HOME`, not `/tmp`) or `compose up` fails with
        "mounts denied". (Native docker engine is unaffected.)


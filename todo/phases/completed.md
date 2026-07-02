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

> Moved to [roadmap.md](../../docs-site/src/content/docs/roadmap.md). All items complete:
> - 13.6 AgentXPage with ConversationContext + ChatPanel
> - 13.10 Removed old tabs dir, renamed to panels, keyboard shortcuts (Ctrl+T/W/K), CSS pass
> - 13.11 Consolidation settings UI + TopBar integration

---

## Phase 14: Context Gating (Complete)

> All 14.1-14.7 complete. See [roadmap.md](../../docs-site/src/content/docs/roadmap.md) for details.


---

## Phase 17: Seamless Server Management (Complete)

> **Goal**: Solid self-hosted server deployment + client connection tools.
> Complete (17.1–17.5). Moved to [roadmap.md](../../docs-site/src/content/docs/roadmap.md): authoritative
> `ConfigManager` config, optional session auth (bcrypt + Redis sessions, `AGENTX_AUTH_ENABLED`),
> Docker production stack (the `production` compose profile), multi-cluster deployment
> (`task cluster:*`), and strict
> client/API version matching (`protocol_version` + `VersionMismatchPage`). Android Tauri mobile
> shipped; rate-limiting + request-audit logging dropped in favor of the edge Gateway.

### 17.6 Open

- [ ] Cloudflare Tunnel deployment documentation (pending)

### 17.7 Release & Packaging

- [x] **Client release CI** — release automation now consolidated in
      `.github/workflows/release.yml` (single `workflow_dispatch`): 3-platform Tauri installer
      matrix (Windows nsis/msi + Linux deb/AppImage/rpm + macOS) + SHA-256 checksums;
      version flows through `versions.yaml`.
- [x] **Server packaging — local vs isolated clusters**: single-source compose
      (`docker-compose.yml` pulls the image; `docker-compose.build.yml` overlay builds locally;
      gateway overlay lives in `docker-compose.gateway.yml`); self-initializing API
      entrypoint (`docker/entrypoint.sh`) + in-image `agentx` ops CLI (`docker/agentx`);
      `.github/workflows/release.yml` publishes `qrmadness/agentx-api` to Docker Hub;
      `task deploy:bundle` generates the isolated deployment bundle; docs at
      `deployment/self-hosting`.
- [ ] **Image groundskeeping** — the API image is ~12 GB uncompressed (~4.4 GB compressed on
      Docker Hub). Slim it: multi-stage build, drop build toolchain/caches from the final layer,
      reconsider the bundled CUDA torch wheel (CPU-only base + optional CUDA variant), and trim
      the nvm/Node install. Heavy pull for self-hosters today.
- [ ] arm64 / multi-arch API image (amd64-only today; QEMU build deferred).
- [ ] Offline "models-baked" image variant (first run downloads ~5 GB to the HF cache volume).
- [x] **GitHub Releases Automation** — the `release` job in `.github/workflows/release.yml`
      publishes a `v{version}` GitHub Release (`-suffix` or the prerelease input → prerelease)
      with SHA-256 checksums, supported-server notes (protocol/min-client/api version from
      `versions.yaml`), the installers, and the deploy bundle attached. Download links on
      `deployment/self-hosting`.
- [ ] Shared-infra local clusters (one DB stack, namespaced) — deferred per the isolation-axis design.
- [x] **Cloudflare gateway for isolated clusters** — resolved: the gateway overlay was split and made
      location-independent (`docker-compose.gateway.yml` = nginx only, mounts via
      `AGENTX_GATEWAY_DIR`, fails closed on empty token; `docker-compose.tunnel.named.yml` = named
      cloudflared; `docker-compose.gateway.expose.yml` = host-port mode). All ship in the deploy
      bundle with `gateway/*.example` templates; documented in `deployment/self-hosting` "Going
      public" (gateway+tunnel is the recommended path). Client-IP trust hardened alongside
      (`AGENTX_TRUST_PROXY`).
- [x] **cloudflared SSE/streaming timeout** — the misleading `noHappyEyeballs` comment in
      `clusters/template/cloudflared/config.yml.example` is fixed (it's IPv4/IPv6 racing, not an
      idle timeout; the real SSE knobs are nginx `proxy_buffering off` + 600s timeouts). Long-SSE
      soak procedure documented in `deployment/self-hosting` troubleshooting.
- [x] **Isolated-deploy doc accuracy** — `deploy/README.md` + `deploy/dockerhub-overview.md` now say
      ~2.3 GB (bge-m3; NLLB is lazy), and the README carries the Docker Desktop shared-path note.
      (Push the refreshed overview text to Docker Hub with the next release.)


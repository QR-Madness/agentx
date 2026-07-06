<!-- release-version: 0.21.154 -->
<!--
  Human-written body for the NEXT release. The release action injects everything
  below the markers verbatim into the GitHub Release notes, between the title and
  the auto-generated "Supported server" / "Downloads" / "Docker image" sections.

  Before releasing:
    1. Bump `release-version` above to match versions.yaml (api.version / client.version).
    2. Replace the body below with the highlights/fixes for this version.
    3. `task release:check` verifies the marker matches versions.yaml.

  KEEP IT TIGHT — release notes, not a changelog. Limits:
    • Each bullet ≤ ~200 chars (one sentence). Lead in **bold**.
    • ≤ ~12 highlights + ≤ ~10 fixes. Consolidate related changes into ONE bullet
      (don't add a new bullet per patch — fold it into the existing one).
    • Whole body should fit on one screen (~2 KB). If it's longer, trim.
-->

AgentX is a self-hostable AI agent platform — Django API + Tauri client.
**Mobile-Ready Alpha**: bring your own server and model providers.

### Highlights

- **Three new themes** — Ugentx, Tango, and Blackhawk join Cosmic/Light/Professional.
- **Workspaces grew into Projects**: files + per-project **instructions** + conversations,
  each project with its own scoped memory.
- **Deployment manager, on by default in the bundle** — web dashboard + CLI; zero-config
  first Start (secrets auto-generate).
- **Hardened self-hosting**: token gateway in the bundle; no-env boots default safe
  (debug off, auth ON).
- **Memory brain upgrade + measuring stick**: windowed extraction (one entity per topic,
  cross-turn relationships, provenance, embedding dedup) + an `eval_recall` harness.
- **Reasoning models think out loud** across all OpenAI-compatible providers.

### Fixes

- **UI foundation restored**: global reset, focus rings, real fonts, one field style.
- **Anthropic models get their full system prompt** (all but one block was dropped).
- **Security**: gateway fails closed on an empty token; `X-Forwarded-For` needs
  `AGENTX_TRUST_PROXY=true`; Cloudflare-free rate limiting.
- **Boots**: warm starts 2+ min → seconds; first-boot hang fixed; `start-manager.bat`.
- **Big documents ingest reliably**, with real failure reasons and a Retry button.
- **Token-limit cutoffs auto-continue**; compression retries down the model-fallback chain.
- **Bundle dashboard**: stopped = *down*, manager excluded from gauges, background polling pauses.

### Migration notes (self-hosters)

- Missing `DJANGO_DEBUG`/`AGENTX_AUTH_ENABLED` in `.env` → safe defaults apply (set explicitly
  to keep old behavior); gateway clusters set `AGENTX_TRUST_PROXY=true`; repo clusters migrate
  once via `task cluster:adopt CLUSTER=<name>`.

### Getting started

[Documentation](https://agentx.thejpnet.net/docs) · [quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) · [self-hosting](https://agentx.thejpnet.net/docs/deployment/self-hosting)

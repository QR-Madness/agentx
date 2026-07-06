<!-- release-version: 0.21.158 -->
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

- **New themes** — Ugentx, Tango, Blackhawk.
- **Workspaces grew into Projects**: files + per-project **instructions** +
  conversations with scoped memory.
- **Deployment manager, on by default in the bundle** — web dashboard + CLI;
  zero-config first Start.
- **Hardened self-hosting**: token gateway; no-env boots default safe.
- **Memory brain upgrade**: windowed extraction (cross-turn links, provenance), an
  `eval_recall` harness, and **two-stage recall** (cross-encoder rerank, +20pp MRR).
- **Reasoning models think out loud** on all OpenAI-compatible providers.
- **Model roles** — one model each for Fast Utility, Deep Reasoning, Summarizer;
  unset feature models follow their role, explicit choices win.

### Fixes

- **UI foundation restored**: reset, focus rings, real fonts, one field style.
- **Anthropic models get their full system prompt** (all but one block was dropped).
- **Security**: gateway fails closed on an empty token; proxy trust is opt-in;
  CDN-free rate limiting.
- **Boots**: warm starts 2+ min → seconds; first-boot hang fixed.
- **Big documents ingest reliably**, with real failure reasons + Retry.
- **Token-limit cutoffs auto-continue**; compression retries down the fallback chain.
- **Memory settings apply live**; corrupt files surfaced, bad values rejected per key.
- **Settings autosave** — sections save as you edit; API keys keep explicit Save.
- **Bundle dashboard**: stopped = *down*; manager out of gauges.

### Migration notes (self-hosters)

- Missing `DJANGO_DEBUG`/`AGENTX_AUTH_ENABLED` → safe defaults; gateway clusters set
  `AGENTX_TRUST_PROXY=true`; repo clusters run `task cluster:adopt` once.

### Getting started

[Documentation](https://agentx.thejpnet.net/docs) · [quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) · [self-hosting](https://agentx.thejpnet.net/docs/deployment/self-hosting)

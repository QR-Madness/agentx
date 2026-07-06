<!-- release-version: 0.21.162 -->
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
- **Agents can now write project files** — `create_document`/`update_document` create
  and revise durable markdown/text docs (indexed + searchable); agents are told what a
  project is, even an empty one.
- **Project files open in the hub** — click any file to preview (markdown rendered,
  images, PDFs), edit markdown/text in place, create new docs, and export to PDF.
- **Deployment manager, on by default in the bundle** — web dashboard + CLI;
  zero-config first Start.
- **Hardened self-hosting**: token gateway; safe no-env boots.
- **Memory brain upgrade**: windowed extraction, an `eval_recall` harness, and
  **two-stage recall** (+20pp MRR).
- **Reasoning models think out loud** on OpenAI-compatible providers.
- **Model roles** — one model each for Fast Utility, Deep Reasoning, Summarizer;
  unset feature models follow their role, explicit choices win.

### Fixes

- **Added HTTPS scheme to Tauri app** - which will invalidate all local data on Windows.
- **UI foundation restored**: reset, real fonts, one field style.
- **Anthropic models get their full system prompt** (blocks were dropped).
- **Security**: gateway fails closed on an empty token; proxy trust is opt-in;
  CDN-free rate limiting.
- **Boots**: warm starts 2+ min → seconds.
- **Big documents ingest reliably**, with failure reasons + Retry.
- **Token-limit cutoffs auto-continue**.
- **Memory settings apply live**; corrupt files surfaced, bad values rejected.
- **Image settings actually save now** (the server silently dropped them before).
- **Settings overhauled** — autosave as you edit (API keys keep explicit Save); new
  Prompts & Memory groups; every prompt override in one place with diff-vs-default.
- **Snappier model picker** — instant search, keyboard nav, recents on top.

### Migration notes (self-hosters)

- Missing `DJANGO_DEBUG`/`AGENTX_AUTH_ENABLED` → safe defaults; gateway clusters set
  `AGENTX_TRUST_PROXY=true`; repo clusters run `task cluster:adopt` once.

### Getting started

[Documentation](https://agentx.thejpnet.net/docs) · [quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) · [self-hosting](https://agentx.thejpnet.net/docs/deployment/self-hosting)

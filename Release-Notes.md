<!-- release-version: 0.21.151 -->
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

AgentX is a self-hostable AI agent platform: a Django API server (MCP client, layered
agent memory, reasoning + drafting, model providers, translation) paired with a
cross-platform Tauri desktop/mobile client. This is the **Mobile-Ready Alpha** — point
the client at your own API server and bring your own model providers.

### Highlights

- **Workspaces grew into Projects.** The hub bundles files, custom **instructions** (followed
  in every chat of the project), and the project's conversations — with sidebar project
  sections, "move to project", and "new chat in this project". Conversations remember their
  project on the server, and **each project keeps its own memory**: knowledge learned inside
  a project stays scoped to it (durable facts still graduate to global memory over time).
- **New: the deployment manager — on by default in the bundle.** Web dashboard + CLI
  (`qrmadness/agentx-manager`, port 12320, loopback + token): live health incl. a first-boot
  *initializing* state, CPU/memory/network gauges (live ↓ MB/s while the model downloads),
  log streaming, config-aware restarts, typed-confirmation destroy. Quick start is now
  zero-config: `cp .env.example .env` → `up -d` → open the GUI → click **Start** — the secret
  key + database passwords auto-generate on first Start (existing databases are never
  re-keyed), and LLM provider keys are added in the app afterwards, not in `.env`.
- **The token gateway ships in the deploy bundle.** Shared-secret + rate limiting from the
  bundle; pick your exposure: token tunnel, named tunnel, or a host port for your own proxy.
- **Safe-by-default settings.** With no env set, the API boots with debug off and auth ON;
  dev `.env` templates opt out explicitly.
- **Reasoning models think out loud now.** OpenRouter/Vercel/OpenAI-compatible reasoning
  tokens stream into the live thinking bubble (like LM Studio) instead of silently burning the
  output budget, and reasoning-capable models get a larger token budget automatically.

### Fixes

- **Inputs and pickers in newer panels no longer look washed out.** Form controls now share
  one field style (sunken background, visible border, focus glow) across the Projects hub,
  Ambassador switchers/composer, and avatar generator.
- **Anthropic models now receive their full system prompt.** A provider bug dropped every
  system block except the last (agent persona, memory, and project instructions were all
  silently lost on Anthropic models); everything now reaches the model.
- **Big documents ingest reliably.** Embedding large uploads no longer times out under the
  chat-recall budget (sliced, background-priority embedding), failed documents show a real
  reason, and a Retry button re-runs ingestion without re-uploading.
- **First boot can no longer hang after the model download.** hf-xet's lingering download
  threads could keep schema init alive after success, so uvicorn never started; the image now
  disables xet (`HF_HUB_DISABLE_XET=1`) and the entrypoint reaps a post-success straggler.
- **Gateway fails closed** on an empty `AGENTX_GATEWAY_TOKEN` (previously that silently
  authorized every request), and proxies to the right port for custom `API_PORT`s.
- **X-Forwarded-For is no longer trusted blindly.** Honored only with `AGENTX_TRUST_PROXY=true`
  (set behind the gateway) — closes a spoofable localhost auth bypass on exposed APIs.
- **Rate limiting works without Cloudflare** (TCP-peer fallback when `CF-Connecting-IP` is
  absent); misleading cloudflared `noHappyEyeballs` SSE comment corrected.
- **Container boots are fast now.** Warm starts collapsed from 2+ minutes to seconds — one
  bootstrap process replaces four, schema init no longer loads the embedding model, and the
  first-boot download runs as an explicit warmup step. `agentx migrate` now also applies
  Alembic migrations. Windows users get a one-click `start-manager.bat` in the bundle
  (needs Docker Desktop's WSL 2 integration; community testing welcome).
- **Answers cut off by the token limit no longer end silently.** The agent auto-continues
  once where it stopped (`chat.auto_continue_on_length`, on by default), and a still-truncated
  answer is flagged in the chat (tag + toast) instead of looking finished.
- **Trajectory/tool-output compression no longer dies silently without Anthropic credits.**
  The compression models now default to your active/global model and retry down the fallback
  chain on runtime failures (a broke provider is also skipped for 30s instead of re-paid).
- **Bundle dashboard reads a stopped stack as *down*** (was *degraded*), and gauges no longer
  count the manager itself — its container shares the bundle's compose project by design and
  is now filtered out of status/usage/restart. Bundle mode is now a plain-language dashboard
  (Start/Stop, component health, resource + live-download tiles); polling pauses in
  background tabs. Repo mode keeps the multi-cluster grid.

### Migration notes (self-hosters)

- `.env` omits `DJANGO_DEBUG` / `AGENTX_AUTH_ENABLED`? The new safe defaults apply (debug off,
  auth on) — set them explicitly to keep old behavior.
- Gateway clusters: also set `AGENTX_TRUST_PROXY=true` (`cluster:up` auto-adds
  `AGENTX_GATEWAY_DIR` to older `.env`s).
- Repo clusters now run under per-cluster compose projects — migrate a running cluster once
  with `task cluster:adopt CLUSTER=<name>` (brief downtime).

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

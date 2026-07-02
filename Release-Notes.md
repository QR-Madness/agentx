<!-- release-version: 0.21.141 -->
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

- **The token gateway now ships in the deploy bundle.** The Nginx shared-secret + rate-limit
  gateway works from the bundle — going public no longer means exposing the API bare.
- **Pick your exposure overlay.** Dashboard token tunnel (Service `http://nginx:80`), named
  tunnel (`tunnel.named.yml`), or a host port for your own proxy (`gateway.expose.yml`).
- **Safe-by-default settings.** With no env set, the API boots with debug off and auth ON;
  dev `.env` templates opt out explicitly.

### Fixes

- **Gateway fails closed.** Compose refuses to start the gateway with an empty
  `AGENTX_GATEWAY_TOKEN` (previously that silently authorized every request).
- **X-Forwarded-For is no longer trusted blindly.** Honored only with `AGENTX_TRUST_PROXY=true`
  (set behind the gateway) — closes a spoofable localhost auth bypass on exposed APIs.
- **Rate limiting works without Cloudflare** (TCP-peer fallback when `CF-Connecting-IP` is
  absent); misleading cloudflared `noHappyEyeballs` SSE comment corrected.

### Migration notes (self-hosters)

- `.env` omits `DJANGO_DEBUG` / `AGENTX_AUTH_ENABLED`? The new safe defaults apply (debug off,
  auth on) — set them explicitly to keep old behavior.
- Gateway clusters: `cluster:up` auto-adds `AGENTX_GATEWAY_DIR` to older `.env`s; also set
  `AGENTX_TRUST_PROXY=true` behind the gateway.

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

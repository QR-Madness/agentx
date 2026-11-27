<!-- release-version: 0.21.305 -->
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
**Mobile-Ready Alpha**: bring your own server and providers.

### Highlights

- **Settings that explain themselves** — search jumps to any setting, an **Overview** says what to set up first; all 210 match the Settings Reference.
- **Link OpenRouter in one click** — sign in once, no key pasting; the card shows balance and spend and repairs renamed models.
- **Model Providers, rebuilt** — connect any OpenAI-compatible endpoint as `yourid:model`. A **supply line** shows where turns go: provider, context, price, fallback.
- **The whole organization on one chart** — **Open Organization** charts managers, teams, independents; drag to arrange, edit teams and roles in place, see what won't work.
- **Extract: move memory out, safely** — evict a channel to the server vault: export → verify → wipe → receipt; nothing is deleted unverified.
- **Agents read the web far more cheaply** — `web_extract` returns only the passages your query needs; per-turn budgets cap *calls and dollars*.
- **The server image is CPU-only, ~85% smaller** — GPU hosts use `latest-gpu` (`docker-compose.gpu.yml` selects it); `docker run --gpus` users must switch tags; external Redis needs 6+.

### Fixes

- **Claude works again via Anthropic** — it no longer sends the temperature current models reject.
- **Settings asks for a default model once a provider connects**; background runs use it, and chat without one links to Model Roles.
- **No false "100% context" after a settings change.**
- **Settings that claimed to save now do** — audio input, speech generation, Ambassador's voice and Web Search source-policy lists (editing one wiped the rest).
- **Memory moves are complete** — exports carry procedures, Replace import resets the PostgreSQL mirror; Consolidate reports dropped relationships and reloads lists.
- **Security updates** — a critical DOMPurify bypass, Tauri's origin check and URL handling, torch loading, Django, PyJWT and more; manager image lockfile-pinned, CLI on PATH.

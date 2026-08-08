<!-- release-version: 0.21.269 -->
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

- **Settings you can navigate** — search finds individual settings, not just section names, and jumps to the control. The new **Overview** searches from where you land, says what to set up first, and describes every screen with a count of what you've changed there.
- **Every setting explains itself** — all 210 of them now say what they do, how they work, when to change them, and when to leave them alone. The same words appear beside the control and in the new Settings Reference, so they cannot disagree.
- **Link your OpenRouter account in one click** — sign in once, no key pasting; the card shows balance and spend, and repairs models OpenRouter renamed (their real context read as 8k, compacting turns far too early).
- **Model Providers, rebuilt** — connect any OpenAI-compatible endpoint (Groq, DeepSeek, Ollama, your own) as `yourid:model`. Backends show if they're reachable, and a **supply line** states where turns really go: the resolving provider, its context and price, the fallback.
- **Extract: move memory out, safely** — evict a channel to the server vault with verified deletion: export → verify → wipe → receipt, and nothing deletes unless it verifies.
- **Agents read the web far more cheaply** — `web_extract` returns only the passages answering your query (3.8% of a real article), sub-questions batch, seen pages return as stubs. Per-turn budgets cap the spend in *calls and dollars*.

### Fixes

- **Settings that claimed to save now do** — **"Enable audio input" and "Enable speech generation" never persisted**, so neither could be turned off; Ambassador's voice pickers were dropped silently; editing one Web Search source-policy list wiped the others.
- **Memory exports carry distilled procedures** — learned rules didn't travel.
- **Replace-mode memory import resets the PostgreSQL mirror** for wiped channels — it cleared only the graph, leaving stale rows.

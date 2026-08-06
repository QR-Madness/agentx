<!-- release-version: 0.21.263 -->
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

- **Settings opens somewhere useful** — a new Overview lists what you've changed from the defaults and offers to put it back, instead of landing on an API-key page. Controls show their default, and **Memory → Recall** explains every knob and when to move it.
- **Link your OpenRouter account in one click** — sign in once, no key pasting; the card shows balance and spend, and repairs models OpenRouter renamed (their real context read as 8k, compacting turns far too early).
- **Model Providers, rebuilt** — connect any OpenAI-compatible endpoint (Groq, DeepSeek, Ollama, your own) as `yourid:model`. Backends show if they're reachable, and a **supply line** states where turns really go: the resolving provider, its real context and price, the fallback.
- **Extract: move memory out, safely** — evict a channel to the server vault with verified deletion (export → verify → wipe → receipt; nothing deletes unless it verifies). Exports go multi-channel.
- **Agents read the web far more cheaply** — `web_extract` returns only the passages answering your query (3.8% of a real article), sub-questions batch, seen pages return as stubs. Brave returns real page content, not a link list.
- **Settings → Web Search is the control plane** — defaults, per-turn budgets in *calls and dollars*, and a **source policy** of preferred/blocked domains (blocked is absolute).

### Fixes

- **Settings that claimed to save now do** — Ambassador's voice pickers were dropped silently; editing one Web Search source-policy list wiped the others. A failing section no longer downs the app.
- **Web-search cost tracking uses the provider's own numbers** where reported, not a depth estimate.
- **Memory exports carry distilled procedures** — learned "how we work" rules didn't travel.
- **Replace-mode memory import resets the PostgreSQL mirror** for wiped channels — it cleared only the graph, leaving stale rows.

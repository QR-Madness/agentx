<!-- release-version: 0.21.261 -->
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

- **Link your OpenRouter account in one click** — sign in once and AgentX gets its own key, no pasting. The card also shows your balance and monthly spend, and offers a one-click fix for models OpenRouter has renamed (their real context window was reading as 8k, compacting turns far too early).
- **Model Providers, rebuilt** — connect any OpenAI-compatible endpoint (Groq, DeepSeek, Ollama, vLLM, your own) as `yourid:model`. Each backend shows whether it's reachable and how many models it opens up, and a **supply line** states where turns really go — which provider resolves your agent's model, its real context window and price, and the fallback behind it.
- **Extract: move memory out, safely** — evict a channel to the server vault with verified deletion (export → verify → wipe → receipt; nothing deletes unless it verifies). Exports also go multi-channel now — an exact channel *set*, the substrate for portable Memory Cores.
- **Agents read the web far more cheaply** — `web_extract` returns only the passages answering your query (3.8% of a real article), sub-questions batch into one call, and seen pages come back as stubs.
- **Brave is a real backend, not a downgrade** — Brave searches return pre-extracted page content instead of a bare link list, and deep research works Brave-primary.
- **Settings → Web Search is the control plane** — defaults, per-turn budgets in *calls and dollars*, grounding limits, and a **source policy** of preferred/blocked domains (blocked is a hard floor).

### Fixes

- **Web-search cost tracking uses the provider's own numbers** where reported, not an estimate from search depth.
- **Memory exports now carry distilled procedures** — learned "how we work" rules didn't travel before.
- **Replace-mode memory import resets the PostgreSQL mirror** for the wiped channels — it wiped only the graph before, leaving stale audit rows.

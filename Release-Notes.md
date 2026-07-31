<!-- release-version: 0.21.257 -->
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

- **Connect any OpenAI-compatible provider** — register Groq, Together, DeepSeek, Ollama, vLLM or a private gateway and use it as `yourid:model`, testing the endpoint first. A new route check reports where a turn *actually* goes when a model is substituted.
- **Extract: move memory out, safely** — evict a channel to the server vault with verified deletion (export → verify → wipe → receipt; nothing deletes unless the artifact verifies). Memory Workbench → Extract, `POST /api/memory/extract`, or `task memory:extract`.
- **Memory exports go multi-channel** — export or replace-import an exact channel *set*, the substrate for portable Memory Cores; and `eval_consolidation --snapshot-name` promotes a snapshot to a reusable Testing Core.
- **Agents read the web far more cheaply** — `web_extract` returns only the passages answering your query (3.8% of a real article), several sub-questions run in one `web_search` call, and pages already seen come back as stubs.
- **Brave is a real backend, not a downgrade** — Brave searches return pre-extracted page content instead of a bare link list, and deep research keeps working on a Brave-primary setup.
- **Settings → Web Search is the control plane** — defaults, per-turn budgets in *calls and dollars*, grounding limits, and a **source policy** of preferred/blocked domains applied to every search (blocked is a hard floor). Plus date windows, country targeting and a "Test connection".

### Fixes

- **Web-search cost tracking uses the provider's own numbers** where reported, instead of estimating from search depth.
- **Memory exports now carry distilled procedures** — learned "how we work" rules previously didn't travel; the v2 envelope round-trips them.
- **Replace-mode memory import now resets the PostgreSQL mirror** for the wiped channel(s) — it previously wiped only the graph, leaving stale audit rows.

<!-- release-version: 0.21.258 -->
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

- **Model Providers, rebuilt** — connect any OpenAI-compatible endpoint (Groq, DeepSeek, Ollama, vLLM, your own) as `yourid:model`; each backend shows whether it's reachable and how many models it opens up; and a **supply line** states where turns really go — the provider resolving your agent's model, its true context window and price, and the fallback behind it. Keys moved server-side.
- **Extract: move memory out, safely** — evict a channel to the server vault with verified deletion (export → verify → wipe → receipt; nothing deletes unless the artifact verifies). Memory Workbench → Extract.
- **Memory exports go multi-channel** — export or replace-import an exact channel *set*, the substrate for portable Memory Cores; `--snapshot-name` promotes an eval snapshot to a Testing Core.
- **Agents read the web far more cheaply** — `web_extract` returns only the passages answering your query (3.8% of a real article), several sub-questions run in one call, and seen pages come back as stubs.
- **Brave is a real backend, not a downgrade** — Brave searches return pre-extracted page content instead of a bare link list, and deep research works Brave-primary.
- **Settings → Web Search is the control plane** — defaults, per-turn budgets in *calls and dollars*, grounding limits, and a **source policy** of preferred/blocked domains applied to every search (blocked is a hard floor).

### Fixes

- **Web-search cost tracking uses the provider's own numbers** where reported, not an estimate from search depth.
- **Memory exports now carry distilled procedures** — learned "how we work" rules previously didn't travel.
- **Replace-mode memory import now resets the PostgreSQL mirror** for the wiped channels — it previously wiped only the graph, leaving stale audit rows.
- **A model an aggregator doesn't list reads as unknown**, not as a small context window that compacts too early.

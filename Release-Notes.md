<!-- release-version: 0.21.255 -->
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

- **Extract: move memory out, safely** — evict a channel to the server vault with verified deletion (export → verify → wipe → receipt; nothing deletes unless the artifact verifies). Memory Workbench → Extract, `POST /api/memory/extract`, or `task memory:extract`.
- **Memory exports go multi-channel** — export or replace-import an exact channel *set* (`channels` in the API/CLI), the substrate for the upcoming portable Memory Cores.
- **Named Testing Cores** — promote an eval snapshot to a reusable fixture with `eval_consolidation --snapshot-name`.
- **Targeted web extraction** — `web_extract` takes a `query` and returns just the passages answering it, not whole pages (measured: 3.8% of a real article).
- **Brave is a real backend, not a downgrade** — Brave searches return pre-extracted page content (search and extract in one call, on a token budget you control) instead of a bare link list, and deep research keeps working on a Brave-primary setup.
- **Parallel search + a spend ceiling** — agents send several sub-questions in one `web_search` call instead of one round each, repeat pages come back as stubs rather than re-sent, and turns can be capped by dollars spent, not just calls made.
- **Richer search controls** — faster depths, date windows, country/language targeting, per-source chunk limits, `web_crawl` path scoping, and a "Test connection" reporting every backend.

### Fixes

- **Web-search cost tracking uses the provider's own numbers** where reported, instead of estimating from search depth.
- **Memory exports now carry distilled procedures** — learned "how we work" rules previously didn't travel; the v2 envelope round-trips them (older builds are asked to upgrade rather than silently dropping them).
- **Replace-mode memory import now resets the PostgreSQL mirror** for the wiped channel(s) — it previously wiped only the graph, leaving stale audit rows.

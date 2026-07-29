<!-- release-version: 0.21.252 -->
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

### Fixes

- **Memory exports now carry distilled procedures** — learned "how we work" rules previously didn't travel; the v2 envelope round-trips them (older builds are asked to upgrade rather than silently dropping them).
- **Replace-mode memory import now resets the PostgreSQL mirror** for the wiped channel(s) — it previously wiped only the graph, leaving stale audit rows.

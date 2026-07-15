<!-- release-version: 0.21.226 -->
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

- **The homepage, reimagined**: an immersive cosmic hero where a *real* recorded agent run streams
  live in a glassbox console (recall → delegate → tools → merge, with real tokens and cost), gold
  energy rising from a true-black starfield, a copy-paste quickstart, and a living system-map.

- **Non-blocking delegation — `delegate_start` work orders**: an agent can now dispatch a task to
  a teammate and *keep working* — it gets a dispatch receipt immediately, and the teammate's
  report is delivered back into the same turn automatically (the turn won't end until every
  work order reports in; Stop cancels them cleanly).
- **The trace console became a Work Console**: master–detail with a work-order rail (tree-ready),
  a focused detail pane with a `Run N / wo·xxxx` breadcrumb, and deep-links — click any Work
  Order card in the chat to open the console focused on it.
- **Work Order cards**: delegations render as compact holographic cards (dispatched → working →
  report delivered) with a real metrics strip — duration, tokens, and honest costs ("Pricing
  unavailable" instead of silently missing); folded reports show as hairline markers in the
  transcript. The Trace chip pulses while work orders run.

- **Agent-ready docs site**: the site now advertises itself to AI agents — RFC 8288 `Link` headers,
  an RFC 9727 API catalog at `/.well-known/api-catalog`, per-page Markdown twins (`<page>.md`) +
  `/llms.txt`, robots.txt content signals, and a WebMCP `search_docs` tool.

### Fixes

- **Removed a no-op Agent Teams toggle**: "Members inherit the lead's tools" never did anything —
  a delegated teammate's tools always come from its own profile. Dropped the dead setting rather
  than leave a switch that lies.
- **Truthful token metering on every provider**: streamed turns on OpenAI, Vercel, and LM Studio now
  report authoritative token counts (hidden reasoning included) instead of a visible-text estimate —
  previously only OpenRouter did, so reasoning models could meter far low.
- **`alloy.delegation_timeout_seconds` is now actually enforced** (both delegation modes) — a
  stuck specialist fails cleanly instead of hanging the turn indefinitely.
- **Late delegation completions no longer drop silently** — cards settle by work-order id even
  after the live stream handle is gone; interrupted background orders read "Cancelled" instead
  of a stuck "streaming" state.
- **Chat auto-scroll behaves**: the tail no longer escapes under rapid output, and scrolling up
  now cleanly disengages it (only *your* scrolls change the follow state; the follower's own
  scrolls are ignored). Scroll back to the bottom — or tap the jump button — to re-engage.
- **Plans + delegation**: a decomposed plan in a conversation with ad-hoc delegation enabled no
  longer fails every subtask ("'NoneType' object has no attribute 'specialists'").
- **Streaming got a contract**: chat streams are golden-tested against real recorded runs —
  immediately fixing three bugs: duplicate `close` events, crash errors arriving after the close
  (invisible to clients), and `<think>` tags leaking into the transcript mid-thought.

<!-- release-version: 0.21.87 -->
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

- **The Ambassador is now a parallel operator.** A dedicated agent that runs *beside* your conversations (never in them). Its panel is an **Inquiry** you ask, brief, or talk to — with read-only tools to read and survey across your conversations.
- **Talk to the Ambassador.** Voice mode is the *same* Inquiry, just hands-free: hold-to-talk and it answers out loud (your spoken Q&A lands right in the thread), and drafts relays you approve before sending.
- **One nameable Inquiry thread.** Briefings + Q&A are a single ordered conversation with live tool chips that survive reload; rename or clear it. CC a turn from the chat to forward it *into* the Inquiry; "Brief this conversation" reads the whole thing.
- **Conversations moved to a proper sidebar.** A dedicated, searchable, resizable rail (collapsible to an avatar strip; a drawer on mobile) — open, running, and history grouped by recency. Pin / archive / icon / color / group, multi-select bulk actions, themed confirms.
- **The command palette is the one command surface.** ⌘K rebuilt on `cmdk`: fuzzy search, grouped sections, Recent, inline theme switching. Mobile got a floating glass dock + a focus-mode exit button.
- **Readable, traceable logging + a live Log panel.** Per-subsystem badges, semantic highlighting, per-turn `run:` tags, startup banner. Stream logs in-app with filtering; daily archives are **encrypted at rest** (AES-256-GCM, envelope-keyed to your password).
- **New "Professional" theme + warmer Light.** A deep monochrome theme where color means emphasis; a system-wide pass moved hardcoded colors onto theme tokens so switches apply everywhere (incl. code highlighting).
- **Durable, layered system prompt + block editor.** The global prompt is a stack of editable blocks that persist across restarts; Settings → Intelligence is a two-pane composer (drag-reorder, autosave, per-layer reset/diff, custom layers, in-place enhance).
- **Redesigned agent-profile editor.** A control-center: hero identity header over control cards, segmented reasoning, temperature gradient, a searchable ~95-icon avatar picker, and a per-agent signature color.
- **Calmer chain-of-thought.** Reasoning streams live then collapses into a "Thinking" affordance once the answer lands; thoughts are process, not persisted.
- **Unified model picker + sturdier plans.** Memory/planner/enhancer settings share the full filterable model picker; plans restore, resume, and terminate cleanly across a cold load.
- **Desktop navbar refresh.** The chat tab is now **Agents** with an animated galaxy and an aurora "expand" on the active item (honors reduced-motion).

### Fixes

- **Public gateway hardening.** Long gateway tokens no longer overflow nginx's hash bucket; CORS preflight passes through (browser/Tauri clients reach tunneled instances); `AGENTX_PUBLIC_HOST` works in Docker; dashboard-managed Cloudflare Tunnel overlay shipped.
- **Ambassador correctness.** Loading a conversation is pure display (no re-speaking/re-billing saved briefings); it names each conversation's own agent; the panel shows the ambassador's own name/avatar; voice shows the ambassador thinking, not your agent.
- **Quieter logs.** A downed Redis backs off instead of flooding; verbose LLM logs show a one-line console summary (full payload in the Log panel); benign voice-cancel `CancelledError` is silenced.
- **Briefings settle cleanly.** No mid-sentence truncation on thinking models, no perpetual "briefing…" spinner after cancel, reliable agent naming on restored turns.
- **Layering + diffs.** Dialogs from inside a full-screen modal layer on top; prompt diffs are colored everywhere (incl. memory prompts); accent-tint contrast fixes across the UI.
- Web search is bounded by a wall-clock cap; plan resume no longer duplicates the user turn or sticks in `running`.
- **Ambassador polish.** One stable header across voice/text (switching modes no longer reshuffles the top bar or strands you — same Inquiry, only the input swaps), a header that holds together when the dock is dragged narrow (the toggle no longer slides off-screen), a redesigned + higher-contrast conversation switcher, an accent mode toggle, and fixed hue-shifting buttons.

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

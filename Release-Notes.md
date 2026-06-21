<!-- release-version: 0.21.109 -->
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
- **Stabler memory in long chats.** A stable high-salience "core" of what the agent knows is now kept in view every turn (no longer re-guessed from each message), with query-specific recall as a supplement. A new context *ledger* budgets the whole preamble by priority, so a growing sidecar can no longer silently crowd out the live transcript.
- **Calmer chain-of-thought.** Reasoning streams live then collapses into a "Thinking" affordance once the answer lands; thoughts are process, not persisted.
- **Unified model picker + sturdier plans.** Memory/planner/enhancer settings share the full filterable model picker; plans restore, resume, and terminate cleanly across a cold load.
- **Desktop navbar refresh.** The chat tab is now **Agents** with an animated galaxy and an aurora "expand" on the active item (honors reduced-motion).

### Fixes

- **PostgreSQL schema now managed by Alembic.** Self-hosters get reliable, incremental DB upgrades — the container applies pending migrations on start (your existing data is adopted in place, never recreated). Neo4j keeps its own migrator. *Upgrading: be on a recent prior version first so your DB is already current before this switch.*
- **Public gateway hardening.** Long gateway tokens no longer overflow nginx's hash bucket; CORS preflight passes through (browser/Tauri clients reach tunneled instances); `AGENTX_PUBLIC_HOST` works in Docker; dashboard-managed Cloudflare Tunnel overlay shipped.
- **Ambassador correctness.** Loading a conversation is pure display (no re-speaking/re-billing saved briefings); it names each conversation's own agent; the panel shows the ambassador's own name/avatar; voice shows the ambassador thinking, not your agent.
- **Quieter logs.** A downed Redis backs off instead of flooding; verbose LLM logs show a one-line console summary (full payload in the Log panel); benign voice-cancel `CancelledError` is silenced.
- **Briefings settle cleanly.** No mid-sentence truncation on thinking models, no perpetual "briefing…" spinner after cancel, reliable agent naming on restored turns.
- **Layering + diffs.** Dialogs from inside a full-screen modal layer on top; prompt diffs are colored everywhere (incl. memory prompts); accent-tint contrast fixes across the UI.
- **Delegated work now counts toward your usage.** Multi-agent (Alloy) delegation cost was computed but dropped; it's now recorded, so specialist spend shows up in usage/cost totals. A unified spend ledger now also captures **Ambassador** activity — every briefing, question, and voice answer — which was previously untracked entirely. **Voice is metered too**: text-to-speech (per character) and speech-to-text (per minute) now carry a cost estimate, with per-model audio rates you can override. **Web search is now metered + budgeted**: each Tavily/Brave call records an estimated credit cost, and a per-turn cap (`search.per_turn_limit`, default 8) stops a runaway tool loop from burning unbounded search credits. The dashboard's Usage & Cost panel adds a **by-source breakdown** so you can see where spend goes — chat vs multi-agent vs ambassador vs voice vs web search.
- **Set a global default model.** Settings → Models now has a default-model picker — the model new agents and ad-hoc requests fall back to when a profile doesn't pin its own.
- **A down or misconfigured model no longer crashes the turn.** Model fallback now covers every feature — chat, reasoning, drafting, the planner, plans, and multi-agent specialists (previously only the Ambassador) — so an unavailable model degrades to a working one instead of failing; a swap surfaces as a status notice. Specialized roles (speculative draft/target, voice TTS/STT) and explicit availability checks stay strict; toggle with `models.fallback_enabled`. Queued/background chats also now resume **warm** — a backgrounded reply rehydrates the conversation's history just like an interactive one.
- **Forgotten facts stay forgotten.** Soft-forgetting a fact now removes it from recall outright (vector, keyword, and entity search) instead of merely lowering its rank — genuinely-past but still-valid facts are unaffected.
- Web search is bounded by a wall-clock cap; plan resume no longer duplicates the user turn or sticks in `running`.
- **More consistent context budgeting.** Chat, memory, the rolling summary, and the context ledger now share one tiktoken-backed token estimator (instead of four divergent chars/4 heuristics), so "what fits the window" is judged the same everywhere; unused legacy context knobs were removed.
- **Hand your agent files (Workspaces & Document RAG).** New workspaces you can upload documents into (PDF + text/markdown/code): each file is parsed, chunked, embedded into pgvector, and auto-tagged + summarized so it becomes searchable. Attach a workspace to a conversation and the agent *sees its file list* every turn, then retrieves with two new tiers — search the catalog (by name/tag) and search inside documents (by meaning) — citing the documents it used. Bytes are stored content-addressed (dedup) with per-file + per-workspace quotas. A **Workspaces drawer** (command palette → "Open Workspaces") manages workspaces (create/rename/delete inline) and click-to-upload (with live ingest status), and you attach a workspace to a conversation from there — attached conversations show the workspace as a chip in their header.
- **Agents can run commands in a workspace (opt-in per workspace, sandboxed).** Flip **"Allow shell" on a workspace** (off by default) and an attached agent can run shell commands and read/write files against a **sandboxed working copy** of it. Every command runs in a bubblewrap jail — **no network, no access to your secrets/`data/`, scrubbed env, time-limited** — so a prompt-injected agent can't exfiltrate or roam the host. Enablement is per-workspace, not a global switch.
- **Browse stored tool outputs.** A debug surface (command palette → "Tool Outputs") lists every large tool result stashed in the cache — filter by tool, read the full body, copy, and prune individually or all at once.
- **Snappier conversation sidebar.** The list no longer rebuilds on every streaming token (stable tab projection + memoized rows), so only the active row updates while a reply streams; pinned/grouped conversations that are also open now show a small "open" dot.
- **Ambassador polish.** One stable header across voice/text (switching modes no longer reshuffles the top bar or strands you — same Inquiry, only the input swaps), a header that holds together at any dock width (controls no longer slide off-screen), and a conversation switcher + mode toggle rebuilt on the app's standard themed components so they read correctly in every theme.

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

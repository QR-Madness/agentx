<!-- release-version: 0.21.76 -->
<!--
  Human-written body for the NEXT release. The release action injects everything
  below the markers verbatim into the GitHub Release notes, between the title and
  the auto-generated "Supported server" / "Downloads" / "Docker image" / checksum
  sections.

  Before releasing:
    1. Bump `release-version` above to match versions.yaml (api.version / client.version).
    2. Replace the body below with the highlights/fixes for this version.
    3. `task release:check` verifies the marker matches versions.yaml.
-->

AgentX is a self-hostable AI agent platform: a Django API server (MCP client,
layered agent memory, reasoning + drafting, model providers, translation) paired
with a cross-platform Tauri desktop/mobile client. This is the **Mobile-Ready
Alpha** — point the client at your own API server and bring your own model
providers (LM Studio, Anthropic, OpenAI, OpenRouter, Vercel).

### Highlights

- **The Ambassador is now a voice call.** Opt an ambassador into voice mode and its panel opens
  into an immersive, Discord-call-style surface — **hold to talk** (mic or Space; or flip on
  tap-to-toggle), and it answers you out loud. It **figures out what you mean**: ask it a question
  and it answers (spoken, and saved to the Text tab); tell it to do something for the agent and it
  drafts that as a message you review and send into the conversation — never sent without you.
  Captions show both sides, talking over it cuts it off (barge-in), and your speech and the
  reply models are picked per-ambassador — now with a **voice dropdown** per speech model instead
  of pasting voice ids. The old text panel is still there as a second **Text** tab for history and
  playback. Built on OpenRouter TTS + speech-to-text (defaults `microsoft/mai-voice-2` +
  `openai/whisper-1`). *(Still to come: a floating mini-player and recording history/cleanup.)*
- **The Ambassador now opens beside your conversation, not over it.** It used to take over the
  screen — a panel that dimmed and blurred everything behind it, so you couldn't watch your agent
  work and talk to the ambassador at the same time. Now it **slides in as a docked column** next
  to the chat: both panels stay live, and you can **drag the edge to resize** it (remembered
  across restarts). On a narrow screen, where there's no room to dock, it still opens as the
  full-screen panel. And since it runs in *parallel* to your conversation, opening it now **waits
  for you to ask** (summarize this, dig into a turn, relay a message) instead of immediately
  briefing the latest turn — so it also works fine on a brand-new, empty conversation.
- **Voice-mode captions are now a real transcript.** Captions used to show only the single
  most recent line from each side — earlier exchanges vanished, and once the ambassador had
  answered once, auto-played briefings stopped captioning at all. Now both sides accumulate
  into a **scrollable caption log** that **auto-scrolls** to the newest line, but **holds its
  place when you scroll back** to re-read, and resets cleanly when you switch conversations.
  (And voice mode no longer shows a blank surface before a conversation has started.)
- **The Ambassador remembers your conversation with it.** Asking it follow-up questions used
  to start from scratch each time — it had no memory of what you'd just asked. Now it carries
  the thread: ask "what did it find?", then "and the second source?", and the second question
  lands with the context of the first. (Its own conversation, still kept entirely separate from
  your agent's transcript.)
- **Conversations moved to a proper sidebar.** The cramped browser-style tab strip at the
  top is gone; the chat page now has a dedicated **Conversations rail** on the left — open
  conversations, anything still running ("Resume"), and your full history grouped by recency,
  all in one well-styled, searchable place. It collapses to a slim avatar rail when you want
  the room (remembered across restarts), and on mobile the same list opens as a drawer, and
  it's **resizable** — drag the edge to the width you like.
- **Conversation management.** Each conversation now has a `⋯` menu to **pin** it (pinned float
  to the top), **archive** it (tucked into a collapsible Archived section), give it a custom
  **icon** and **color**, and sort it into **custom groups** — all remembered per server.
  **Multi-select** lets you act on many at once (bulk pin / archive / delete). And the jarring
  native "are you sure?" delete prompt is replaced by a proper, themed confirm dialog. (Under
  the hood this rides one extensible per-conversation metadata model — room to grow toward
  workspace/file linking later.)
- **Mobile sweep on the new conversation + ambassador surfaces.** The two leftover "active
  conversations" switchers in the top bar are gone — on phones the conversation list now opens
  from a button right in the chat header (and from the ⌘K palette's new **Open conversations**
  command, so it's always one keystroke away even in focus mode). The ambassador's voice and
  text panels got a touch pass too: the voice settings popover no longer clips the screen edge,
  the relay buttons wrap instead of overflowing on narrow phones, and tapping into a text field
  no longer triggers iOS's zoom-on-focus.
- **Command palette + bottom bar, polished for touch.** The ⌘K palette now has a visible **close
  button** (it went full-screen on phones with no way out but a hardware Esc), drops the
  keyboard-only hint bar on mobile, and respects the device's status bar and home indicator.
  Searching "profiles" (and other plurals like "memories" / "translations") now finds the right
  command — and "Edit agent profile" is now just **Agent profiles**. The mobile bottom bar
  became a **floating glass dock**: it lifts off the screen edges with comfortable margins (so it
  clears rounded corners and the home indicator), the search control shrinks to a tidy icon so the
  bar no longer overflows, and in **focus mode** the bar dissolves to a single frosted exit button
  in the corner.
- **Cleaner data tables.** Agent-authored tables now read like real tables — more generous
  spacing, a crisper header with a clear divider, delineated columns that wrap long prose
  instead of crushing it (wide tables scroll), subtler zebra striping and row hover. The
  **Expand** and dialog **close** buttons got proper padding and contrast. Sort,
  expand-to-modal, and the mobile stacked-card view are unchanged.
- **Toolbar cleanup — the command palette is now the one place for everything.** The
  redundant "Workspace" overflow (`⋯`) menu is gone; the ⌘K command palette is the single,
  primary command surface — a labeled **Search…** pill in the toolbar makes it obvious
  (and it's a genuine lifesaver on mobile, where it's the main way to get around). The
  palette was rebuilt on `cmdk`: fuzzy search + ranking, grouped sections (Navigation /
  Conversation / Workspace / Theme / Account), a **Recent** group, **theme switching**
  (Cosmic / Light / Professional / System) inline, keyboard nav with scroll-into-view, and
  a near-fullscreen layout on mobile. Under the hood, the toolbar and palette now open every
  surface through one shared registry, so they can't drift out of sync.
- **Logging overhaul — readable, color-coded, traceable, in-app.** Server logs now
  render with per-subsystem category badges (provider / memory / stream / mcp / …),
  semantic highlighting (model ids, token/cost figures, durations), and a per-turn
  `run:` correlation tag so one chat turn can be followed across providers, streaming
  and memory. Startup shows an ASCII banner + a live status table (version, providers,
  datastores, MCP). The previously enormous LLM request logs are now compact summary
  cards by default (full payload on demand), and secrets are redacted everywhere. A new
  **Log panel** (Workspace menu → Logs) streams the server's logs live into the client
  with level/category/run filtering and a downloadable compressed archive. Everything is
  governed by `AGENTX_LOG_*` flags — all on by default; `AGENTX_LOG_DECORATIONS=false`
  restores plain output for CI/journald. The log API is auth-gated when auth is enabled.
- **Encrypted log archives.** Durable logs are now chunked **by day** and, once you've
  set up a password, **sealed with AES-256-GCM** keyed to that password — history at rest
  is unreadable without it (on top of the existing secret redaction). It uses envelope
  encryption, so changing your password just re-wraps one tiny key instead of rewriting
  every archive (`task logs:rotate-keys`; `--reencrypt` for a full re-encrypt). Encrypted
  segments show a lock in the Log panel and are decrypted on download. With auth disabled
  it transparently falls back to the previous plaintext-gzip behavior. Retention defaults
  to 30 days (`AGENTX_LOG_ARCHIVE_*`). The Log panel's archive drawer now shows a **vault
  status strip** (encrypted/unlocked/locked + sealed/pending counts), lists history as
  friendly **per-day chunks**, and **loads earlier days as you scroll**.
- **Desktop navbar refresh.** The chat tab is now **Agents** with an **animated spiral
  galaxy** — a glowing core, slowly rotating arms, and an orbiting star — and the active
  nav item gets a rich aurora "expand": it widens with a soft glow, a sheen sweeps across
  it, an aurora underline grows in, and the galaxy spins up and lights when its tab is
  active. Honors reduced-motion.
- **New "Professional" theme + a warmer Light.** A deep monochrome graphite theme
  with calm off-white text, where color is reserved for emphasis — feedback
  (success/error/warning) and code stay colored, the chrome is neutral with a single
  restrained slate-blue accent, and the gradients/glows are gone. Feedback tints now
  adapt per theme rather than being hardcoded. The Light theme was refreshed to a warm, neutral white in
  the same restrained spirit. Pick them in **Settings → Appearance** — which is itself
  now properly themed (it previously wasn't). A system-wide enforcement pass moved the
  remaining hardcoded colors across the UI onto theme tokens, so a theme switch (and any
  future palette) actually applies everywhere — even focus rings and code-syntax
  highlighting now follow the theme (One Dark on dark, One Light on Light). Only the
  startup-error screen and the theme-preview swatches keep fixed colors by design.
- **Calmer chain-of-thought.** A model's reasoning now streams live and then tucks
  itself into a collapsed "Thinking" affordance once the answer lands, instead of
  sitting expanded above the reply (which read as duplicated content). Thoughts are
  also **no longer persisted** — they're process, not result, so they're shown in the
  moment and don't clutter a reloaded transcript. Only the result is kept. Agents are
  now told this directly, and nudged to write a genuinely useful thought to their
  scratchpad (or memory tools) rather than lose it in ephemeral reasoning.
- **Unified model picker.** Memory, planner, and prompt-enhancement settings now use
  the same full filterable model picker as the agent-profile editor (provider and
  capability filters, search, context/pricing metadata) — the old inline dropdown
  is gone.
- **Sturdier plan execution.** Plans restore and resume cleanly across a cold
  conversation load, terminate faithfully on Stop (interrupted vs. failed), and
  offer a resume nudge with the remaining lifetime.
- **Ambassador (foundation).** A new dedicated agent that runs *parallel* to a
  conversation and briefs you on a turn — without entering the conversation. Hit
  the new CC button on any reply and a right-side **Ambassador** panel streams a
  plain-language briefing of that turn, persisted per-conversation. The ambassador
  is just an agent profile with an extra "ambassador" section, picked globally in
  Settings → Ambassador. Briefings now **stream in token-by-token**, scale their
  length to the chosen verbosity, and **speak directly to you** — addressing you in
  the second person and naming your agent, instead of narrating "the user asked… the
  assistant replied." Per-turn status (briefing / briefed / cancelled / error) is
  clearer in a refreshed panel. The briefing now **grounds on what the agent actually
  did** that turn — the web searches it ran, the sources it cited, the tables and
  diagrams it built — not just its written reply, so it interprets the turn instead of
  paraphrasing it. And you can now **ask the ambassador anything** about a conversation
  from the panel ("what sources did it use?", "summarize that table") — it answers
  grounded only in what actually happened, streaming its reply. And the relay now goes
  **both ways**: from the panel you can **send a message to the agent** — type it, or let
  the ambassador shape your rough intent into a ready-to-send message ("Refine"), then
  send. It lands as your own message in the conversation (or folds into the running turn).
  The ambassador never speaks into the conversation itself — you're always the author.
  Briefings and Q&A never touch the transcript or the agent's context; future spoken
  briefings slot onto the same seam.

- **Durable, layered global prompt (foundation).** The agent's global system prompt is now
  composed from a **layered stack** of editable blocks rather than one in-memory blob — and
  edits **persist across restarts** (the old global prompt was lost on restart). Built-in
  layers (core principles, citing sources, reasoning-vs-results, structured thinking, concise
  output, safety) ship a default you can override per-layer; untouched layers keep receiving
  release improvements while your edits stay pinned and are never silently overwritten. This
  is the plumbing for the block-based prompt editor — behavior is unchanged by default
  (the default stack reproduces the previous default prompt exactly). The stack is exposed
  over a full REST API (`/api/prompts/layers` — list/create/update/delete/reorder, plus
  per-layer reset and update-acknowledge), with typed client methods.
- **Block-based system-prompt editor.** **Settings → Intelligence → System Prompt** is a new
  two-pane composer: a stack of editable layer cards beside a live composed preview. Drag to
  reorder, toggle layers on/off, and edit any layer inline with **autosave** — changes persist
  immediately. Built-in layers show a default until you override them (marked **● edited**); a
  one-click **Reset** restores the shipped default, and when an update ships a new default
  underneath your edit you get a **▲ update** badge and a **diff view** to **Keep yours**,
  **Adopt the new default**, or **load it into the editor to merge**. Add your own **custom
  layers** too. Your edits are always kept separate from the defaults and never silently
  overwritten. You can also **insert a snippet from the Prompt Library** straight into the
  stack as a new layer, and **enhance any layer in place** — let the prompt enhancer rewrite
  it, with one-click undo.
- **The Ambassador is now its own profile type.** Instead of being a section bolted onto a
  normal agent, an ambassador is a first-class profile *kind* — so you can have several, and
  one is the **default ambassador** that briefings use. Its personality (the "Communications"
  prompt) is customizable, and its functional voices (briefing, Q&A, draft) can each be
  overridden or reset to their shipped default. Ambassadors are kept out of the chat agent
  picker, delegation, and @-mentions — they only ever brief, never join the conversation.
  Existing setups migrate automatically (and a default ambassador is created if you don't have
  one), without ever turning your main agent into an ambassador. **Settings → Ambassador** now
  lets you pick the default ambassador and create or edit one; the profile editor adapts for
  ambassadors — leading with the **Communications** prompt and exposing each functional voice
  (briefing / Q&A / draft) with override, reset, and a diff against the shipped default.
- **Polished per-agent prompt editor.** Editing an agent profile's instructions now uses the
  same refined editor — now with **many more avatar icons** to pick from. An approximate token/char count, in-place **Enhance** (with undo),
  **Insert placeholder** (`{agent_name}`, `{date}`, `{time}` — substituted when the prompt is
  composed, highlighted in the preview), **Insert from library** (which replaces the field), and now **autosaves** as you edit
  (no Save button — your place in the form is preserved). **Memory settings** (recall +
  consolidation) autosave the same way. And a collapsible **effective-prompt
  preview** showing the real composed result — the agent's name, its instructions, and the
  global layer stack together — so you can see exactly what the agent receives.
- **Redesigned agent-profile editor — a control center for your agent.** A hero identity header
  (big avatar, inline name, copyable agent-id, kind/default badges, tags, description) sits over
  a clean grid of control cards (Model, Generation, System Prompt, Memory, Tools, Delegation /
  Ambassador voices) — each with an at-a-glance value in its header. Reasoning strategy is now a
  segmented control with a glyph per strategy, and temperature is a cool→warm gradient with a
  live label (Precise → Wild). The avatars are consolidated into a **searchable, categorized icon
  picker** (now ~95 icons across Tech, Nature, Creatures, Craft, Emblems, Symbols, People) with a
  recently-used row and a live preview — plus a seam for **generated icons** soon. And every agent
  gets a **quiet signature color** derived from its id, threading its avatar aura, nav highlight,
  and badge together. Motion is tasteful and respects reduced-motion.

### Fixes

- **A stopped Redis no longer floods the console.** If the datastore went down
  while the API kept running, the background-chat worker logged a connection
  warning every two seconds forever. It now backs off (up to 30s), logs the
  outage once, and quietly announces when Redis is reachable again.
- **LLM request logs no longer flood the console.** With the verbose
  (`AGENTX_LLM_LOG_LEVEL=full`) setting, a single LLM request used to dump its
  entire payload to the console and blow away your scrollback. The console now
  always shows just a compact one-line summary (model · messages · ~tokens ·
  tools); the full, redacted payload is kept in the in-app **Log panel** (expand
  the row to read it) and the on-disk archive. Summaries log at INFO so they show
  inline in the standard console.
- **Contrast / theming fixes on the Ambassador (and friends).** Several chips, the identity mark,
  the voice orb, and a few hover states were rendering accent text on a clashing solid-pink fill
  (a mix-up between the third accent hue and a subtle accent *tint*) — low-contrast and, in places,
  near-invisible. They now use a proper soft accent tint that reads cleanly in every theme. The
  same mix-up is fixed on the Sources "active" badge and the Memory import-mode toggle. Separately,
  the **inactive** Voice/Text, Ask/Relay, and push-to-talk toggle buttons were greyed out with the
  muted text token; they now use the standard secondary text colour (matching the shared segmented
  control), so the option you're *not* on is still clearly readable.
- Prompt diffs are now properly colored (red removals, green additions) everywhere they
  appear — including the ambassador voices and the memory prompts. The diff view also now
  shows up on the **memory** extraction/relevance prompts, so you can compare your override
  against the shipped default.
- Dialogs opened from inside a full-screen modal now appear correctly. "Insert from library"
  (and the diff and preview popups) in the System Prompt editor were rendering *behind* the
  Settings window and looked like they did nothing — they now layer on top.
- Ambassador briefings no longer truncate mid-sentence on thinking models. The token
  budget now leaves generous headroom for the model's reasoning on top of the (short)
  briefing, so a model like Gemini that reasons before answering isn't cut off — the
  briefing's length is governed by the prompt, not a tight token cap.
- The ambassador now reliably names the agent it's briefing — it resolves the name
  from the turn, the producing profile, or the conversation, so briefings of
  restored/older turns no longer fall back to a generic "your agent."
- Cancelling an ambassador briefing now settles it cleanly — a cancelled briefing
  no longer reopens as a perpetual "briefing…" spinner after a reload.
- Web search is now bounded by a wall-clock cap so a slow provider can't wedge a
  turn or block cancellation.
- Plan resume rebuilds full context and no longer duplicates the user turn or leaves
  a stuck `running` status after an interrupt.
- **Public gateway now starts with a real token.** The Nginx + Cloudflare Tunnel gateway
  failed to start when given a normal-length gateway token (`openssl rand -hex 32` → 64
  chars) — the token overflowed nginx's default map-hash bucket. The bundled `nginx.conf`
  now sizes the bucket for long tokens. The tunnel's tokenless `/__gateway_health` probe
  is also reachable again (the token gate is scoped so it no longer shadows the health
  endpoint).
- **Gateway no longer blocks browser/desktop clients (CORS preflight).** The gateway 401'd
  the CORS `OPTIONS` preflight — which by design can't carry the custom `AgentX-Gateway-Token`
  header — so no browser/Tauri client could reach a tunneled instance. Preflight now passes
  through to the API (which answers it), and `agentx-gateway-token` is an allowed CORS request
  header.
- **`AGENTX_PUBLIC_HOST` now works in Docker.** It's passed through to the API container,
  so setting it actually extends `ALLOWED_HOSTS`/`CORS`/`CSRF` as documented — previously
  a no-op in containerized deployments.
- **Go public with a dashboard-managed Cloudflare Tunnel — no host `cloudflared`.** The
  self-hosted (isolated) bundle now ships a `docker-compose.tunnel.yml` overlay that runs
  `cloudflared` as a container (token from the Cloudflare Zero Trust dashboard): set
  `TUNNEL_TOKEN` + `AGENTX_PUBLIC_HOST`, point the dashboard Service at the API, and you're
  public with no host install and no inbound ports. Shipped in the release bundle and
  documented end-to-end (setup + verify + troubleshooting) in the self-hosting guide.

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

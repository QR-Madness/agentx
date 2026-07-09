<!-- release-version: 0.21.200 -->
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

- **Memory consolidation has a real progress drawer** — the ⚡ icon now opens an animated drawer
  with a genuinely moving progress bar (per-conversation, with a trickle so it never looks frozen),
  live counters (turns · LLM calls · tokens), rotating status messages, surfaced issues, and a clean
  Start. When there's nothing new it says so ("Memory's up to date"), and it re-attaches to a run
  started in the background.
- **Run AgentX from a browser** — an installable **PWA** build (ideal for iPhone/Android
  and anywhere the desktop app can't go, auto-updating with no store), plus shareable
  **connection links**: send someone a link and they connect to your server by opening it
  and entering only a password.
- **Memory got the room it deserves** — the explorer is now a full-screen **workbench**: a top tab
  bar (Overview · Entities · Facts · Strategies · Procedures · Explore · History · Jobs) over a roomy
  list+detail canvas, a **channel filter** to scope every area, an **Overview** home with per-channel
  totals, and a smoother, search-driven **Explore** graph. Editing entities/facts is no longer cramped,
  and it's genuinely usable on mobile.
- **Provider settings tell the truth about capability** — providers are now grouped by tier
  (**OpenRouter = Recommended** and the one that powers images + voice today; **Anthropic / OpenAI /
  Vercel = Beta**; **LM Studio = Local**, and it's disabled with a note when you're on a remote cluster
  where its localhost server can't be reached). New read-only **On-device** tiles show the locked local
  engines — **BGE-M3** embeddings and **NLLB-200** translation — and whether they're running on **CUDA
  or CPU**. The global default model moved out of Model Limits into **Model Roles**.
- **New themes** — Ugentx, Tango, Blackhawk.
- **Workspaces grew into Projects**: files + per-project **instructions** +
  conversations with scoped memory.
- **Agents can now manage project files, not just create them** — beyond
  `create_document`/`update_document`, agents get `list_project_files`, `append_to_document`,
  `edit_document` (targeted find-and-replace — no more resending the whole file), `rename_document`,
  and `delete_document`, with a soft write-lock so two agents editing the same file don't clobber.
  The prompt now frames a project as a home for **any files** (not just documents) and nudges
  agents to lean into it; they're told generated images land in **Home** by default, and those
  images now get a **readable name from their prompt** instead of a bare timestamp.
- **Project files open in the hub** — click any file to preview (markdown rendered,
  images, PDFs), edit markdown/text in place, create new docs, and export to PDF.
  **Click an image to zoom** it full-screen (scroll to zoom, drag to pan, download), and
  **rename files** inline in the list or the preview (the file keeps its identity, so any
  conversation that shows the image still works).
- **Deployment manager, on by default in the bundle** — web dashboard + CLI;
  zero-config first Start.
- **Hardened self-hosting**: token gateway; safe no-env boots.
- **Memory brain upgrade**: windowed extraction, an `eval_recall` harness, and
  **two-stage recall** (+20pp MRR).
- **Conversations keep structured state, and now you can see and edit it** — agents maintain a
  slot-based working record (goals, decisions, open threads, artifacts, plus a freeform narrative)
  that survives context compression; a composer indicator shows it at a glance and opens a drawer to
  add, edit, or remove entries (your edits win, and the agent reads them next turn). When older turns
  age out of the window they're now compacted **into this structured state** (a rolling summary that
  stays bounded) instead of a separate lossy prose blob, so long conversations stay coherent.
  Default-on; opt out in settings.
- **Recall the actual discussion, not just the fact** — when you ask about something you worked
  through before ("when did we decide…", "earlier you said…"), agents now surface lightweight
  **links to the past conversations** those facts came from and can pull the real turns on demand,
  instead of only seeing an atomized fact. Semantic memory and episodic memory, connected.
- **Agents use their memory more deliberately, and more safely** — better in-prompt guidance on
  when to record state, when to recall, and when to pull a past thread; plus a hardening pass so
  instructions hidden in tool or web results can't quietly rewrite an agent's memory (every state
  entry is attributable to you or the agent — never forged).
- **Reasoning models think out loud** on OpenAI-compatible providers.
- **Sign in to remote MCP servers** — OAuth 2.1 with automatic client registration;
  connect opens your browser, tokens are stored and refreshed server-side.
- **Model roles** — one model each for Fast Utility, Deep Reasoning, Summarizer;
  unset feature models follow their role, explicit choices win.
- **Agent Teams** — delegation is now first-class: opt profiles into the team roster
  with a one-line Specialty, and agents see their teammates in-prompt and delegate when
  one is clearly better suited; a per-chat **Solo/Team** toggle keeps any conversation
  single-agent. ("Agent Alloy" workflows are now **Teams** — a lead who delegates to
  members.) Ad-hoc delegation ships enabled, but the roster starts empty (profiles are
  now opt-in, so nothing delegates until you add teammates). The **Team Builder was
  rebuilt**: pick a Lead, then add members from a searchable **＋ Add member** picker
  (no more scrolling every profile), each with its own delegation hint that falls back
  to the agent's profile Specialty — and it's now fully mobile-friendly (master→detail
  flow, thumb-reachable bottom-sheet picker).
- **The Agent Profile editor grew up on mobile** — a master→detail flow (profiles list → tap →
  full-screen editor with a ‹ Back), **drag-to-reorder** profiles (a grip handle; order persists
  server-side), a compact identity header so the settings aren't buried, and the editor now fills the
  screen instead of a tiny box. The icon picker selects on the **first tap** (no more tap-twice), the
  profile's icon shows in the composer agent chip, and the redundant **Done** button is gone (edits
  autosave). Fresh installs seed just **AgentX** (⚛ atom) + a ready **Researcher** (🔭 telescope) with
  a proper web-research delegation prompt.
- **Sources stay out of the way** — cited sources no longer stack as full-width cards mid-chat;
  they collapse into a slim, tool-call-style row (icon + count + a host hint) that expands on
  click, and the Sources drawer got a lighter, icon-led restyle.
- **A sharper chat composer, and Projects that work on a phone** — the composer's Relay hub (new
  glowing **Orbit** icon) is now the single home for image + document + **project** actions (paste
  an image straight in), grows taller as you type, and opens as a thumb-reachable bottom sheet on
  mobile. The Projects hub gets a roomier project list (no more clipped titles), a mobile
  master→detail flow, file cards that stack instead of cramping, a bigger instructions box, and
  full-screen file previews.
- **A clearer, snappier conversation sidebar** — bolder rounded section headers that stick as you
  scroll, open/active chats marked with an accent rail (not a tiny dot), conversation titles that
  wrap to two lines with a longer preview, and a **refresh** button for detached/running chats. The
  ⋯ actions menu opens instantly (dropped the modal machinery that stalled it), and the mobile
  conversations drawer is now a proper full-screen page.
- **Name your chats, fast** — **Auto-title** any conversation (from the ⋯ menu or the Relay) and it
  generates a concise title from the conversation's state + first and last message; **tap the chat
  title** to rename it inline; and a **Close others** button clears every open tab except the current
  one (it glows a deeper red the more tabs you have open).

### Fixes

- **Delegated agents now see the project — and delegations stay visible** — when you delegate inside a
  project, the teammate is now *told* which project it's in (name, instructions, file list) instead of
  only being able to touch its files silently, so it uses project tools and follows your instructions.
  And after a hand-off, the delegating agent can see on later turns that it *delegated* (to whom, for
  what) rather than just "a summary was generated."
- **No more premature "spotty memory" on OpenRouter models** — a model whose provider reports the
  wrong context window (e.g. an OpenRouter `:latest` route, which falls back to ~8k) made the agent
  start summarizing almost immediately and forget most of the conversation. Settings → Model Limits
  now lets you **pin a per-model context window** for any provider, and the model picker flags
  `:latest` routes. New **Context Compaction** controls expose the verbatim-window budget and the
  rolling-summary trigger, so the digest only kicks in near the model's real limit.
- **Image settings copy caught up** — the Images settings no longer says in-conversation image
  generation is "coming soon" (it's here), and the enable toggle now notes it gates the
  `generate_image` tool, not just the avatar "Generate" tab.
- **Projects hub fills the screen, and Home avatars make sense** — the Projects drawer no longer
  stops short with dead space at the bottom (it inherited a centered-dialog height); and in Home,
  each avatar image is badged with the agent that uses it, with a **Delete unused** button to clear
  avatars no profile references.
- **Delegating to an image agent now returns the image** — asking a teammate that's an image-output
  model (e.g. gemini-flash-image / flux) to make a picture produced only a text description, never the
  image: the delegation path ran it through the text tool-loop, which can't carry image bytes. It now
  routes image agents through the same generate→store→render path as a direct image chat, so the
  picture appears under the delegation card (and persists on reload). The delegating agent also gets
  the image's id back, so it can `view_image` the result instead of saying it can't see it.
- **Modals are full-screen on phones again, and menus feel snappier** — a stray tablet breakpoint
  was overriding the phone rule, so dialogs (like the file preview) opened as a small floating box
  instead of filling the screen; and conversation ⋯ menus dropped a heavy backdrop-blur that made
  them stutter to open on the desktop app.
- **Internal utility models now honor your Model Roles** — recall (HyDE / self-query), rolling
  summaries, recaps, and tool/trajectory compression shipped pinned to a hardcoded Anthropic (or
  local LM Studio) model that silently overrode your Fast Utility / Summarizer role, so an
  OpenRouter-only setup still fired Claude Haiku (and local Gemma) behind your back — most visibly a
  Haiku recap on every consolidation run. These now follow the configured role (or your explicit
  override); a guard test keeps any future setting from re-pinning a role.
- **Memory consolidation no longer redoes finished work** — it now tracks what's been consolidated
  per **turn**, so a re-run only processes genuinely new turns instead of re-pouring every
  conversation (and burning LLM calls) on a fixed timer. Idempotent runs are now near-instant; a
  one-time backfill preserves your existing consolidation state, and it survives export/import.
- **Background memory consolidation reliably sees your providers** — the consolidation worker now
  boots the full app stack, so provider keys follow Settings (seeded once from `.env`) instead of
  going stale on rotation, and the worker's own logs are captured like the API's rather than
  vanishing to stdout. Previously a provider key set only in `.env` (e.g. OpenRouter) was invisible
  to the worker and its failures left no trace.
- **Long-conversation coverage is tighter** — when older turns are compacted into the structured
  state, their summary is now guaranteed to appear the same turn even if the cache read behind it
  hiccups, closing a rare window where a just-compacted turn could momentarily go uncovered.
- **Your chosen default model is finally respected** — agent profiles with no model set silently fell
  through to a local LM Studio model instead of the global default you picked in Settings, so turns
  hung or errored if LM Studio wasn't running; the configured default is now in the resolution chain
  everywhere (chat, plan resume, background agent).
- **Memory consolidation now follows your model roles** — its stages shipped pinned
  to a specific model instead of your Fast Utility / Deep Reasoning choice (the ambassador
  aide is now role-aware too); fresh installs inherit the role, and existing installs get a
  one-click **Adopt roles for all consolidation stages**.
- **Delegated agents can show their work** — images and exhibits produced by a teammate
  during delegation now render in the chat (they were silently dropped, leaving empty
  replies and broken image links), survive reloads, and are correctly attributed.
- **Agents no longer forget mid-conversation** — older turns are now summarized *before*
  they leave the context window (previously they could vanish uncovered at 70% usage or
  after a restart), verbatim history runs to ~90% of the window, and the composer shows
  how full the context is at all times (beside the message token estimate, from 0% on a
  fresh chat) with a heads-up when compression kicks in.
- **Added HTTPS scheme to Tauri app** - which will invalidate all local data on Windows.
- **Windows: Settings, Toolkit & profile editor open properly** — they collapsed to a
  thin line on WebView2 (full-screen surfaces were nested inside a centered dialog).
- **UI foundation restored**: reset, real fonts, one field style.
- **Anthropic models get their full system prompt** (blocks were dropped).
- **Security**: gateway fails closed on an empty token; proxy trust is opt-in;
  CDN-free rate limiting.
- **Boots**: warm starts 2+ min → seconds.
- **Big documents ingest reliably**, with failure reasons + Retry.
- **Token-limit cutoffs auto-continue**.
- **Memory settings apply live**; corrupt files surfaced, bad values rejected.
- **Image settings actually save now** (the server silently dropped them before).
- **Settings overhauled** — autosave as you edit (API keys keep explicit Save); new
  Prompts & Memory groups; every prompt override in one place with diff-vs-default.
- **Snappier model picker** — instant search, keyboard nav, recents on top.
- **MCP sign-in fixed & polished** — Connect now reliably opens your browser (with a
  copy-link fallback dialog); a cancelled or denied sign-in no longer shows as "signed
  in"; OAuth connectors reconnect on restart from their stored tokens (refreshing silently
  only when actually expired) instead of asking you to sign in again; the Toolkit forms are
  rebuilt on the shared field controls
  with a simpler URL-first "Add server" flow (advanced options tucked behind a disclosure).
- **Connector sign-in nudge** — starting a new conversation prompts you to connect any
  OAuth tools the active agent still needs, with a one-click Connect.
- **Toggles look right everywhere** — the switch thumb was mis-centered by a global button
  reset; the control is now a self-contained primitive with a clean slide + accent glow.
- **Sign-in screen polish** — password fields now use the shared field styling (reveal toggle +
  Caps-Lock warning), a status-dot connection pill, and a **readable reason** when a server is
  unreachable (was "[object Object]"); the server picker no longer closes while you paste a
  gateway token.
- **Desktop share links resolve to the hosted app** — connection links copied from the desktop app
  now point at the hosted web app (`agx.thejpnet.net`) instead of the unshareable `tauri://localhost`,
  so a recipient can just open the link on their phone and sign in.

### Migration notes (self-hosters)

- Missing `DJANGO_DEBUG`/`AGENTX_AUTH_ENABLED` → safe defaults; gateway clusters set
  `AGENTX_TRUST_PROXY=true`; repo clusters run `task cluster:adopt` once.

### Getting started

[Documentation](https://agentx.thejpnet.net/docs) · [quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) · [self-hosting](https://agentx.thejpnet.net/docs/deployment/self-hosting)

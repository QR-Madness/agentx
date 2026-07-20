<!-- release-version: 0.21.239 -->
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

- **Prompt editing, rethought**: the system-prompt layer stack, a renamed per-agent **Instructions**
  composer, and a decluttered Prompt Library now share one editing language — roomy editors, cleaner
  cards, and live `{placeholder}` highlighting as you type.

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
  `/llms.txt`, robots.txt content signals, and a WebMCP `search_docs` tool. Any page (home page
  included) now answers `Accept: text/markdown` with its Markdown twin and an `x-markdown-tokens`
  estimate; browsers still get HTML.

- **Ambassador v2, closed out**: the spoken router now remembers the exchange you're having
  (a follow-up like "relay that to the agent" routes in context), answers wear your ambassador's
  own avatar, and the load-bearing invariants (SELECT-only tool belt, sidecar-only writes,
  never-raise degrades) are regression-guarded.
- **Ambassador v3 — she can see everything now**: five new reads — a conversation's structured
  state ("where does this stand?"), full-text search across ALL your conversations, live
  runs ("what are my agents doing right now?"), a spend/usage report, and long-term memory
  recall ("what do my agents *know* about X?").
- **Ambassador v3 — conversation management, confirm-first**: ask her (typed or spoken) to
  rename, archive, or delete a conversation — she files a proposal and *you* confirm it on a
  strip (deletes get a full confirm dialog); nothing ever executes on its own. Titles and
  archives are now durable server-side, manual rename/archive in the conversation list rides
  the same store, and archived conversations rest in the Archived section until restored.
- **Deck and Memory became real tabs**: both now live in the top bar on desktop — the surface
  slides in under the bar, the lit pill closes it again, and the Deck pill pulses while your
  agents have runs in flight. On mobile they stay a command-palette reach away.
- **The Ambassador can show what a conversation *produced***: ask "what did that conversation
  actually make?" and she reads its exhibits — tables, images, diagrams, and cited sources with
  their links — outcomes, not just the chat.
- **She can propose dispatches now**: "have the Researcher dig into X" files a dispatch proposal
  *you* confirm — into a fresh conversation or an existing one — and the task appears in its tab
  instantly instead of a blank wait. Deleting a conversation also now clears every sidecar
  (state, summary, Inquiry thread) — nothing strands.
- **The Ambassador speaks in facts now**: digests carry the load-bearing specifics verbatim, she
  drills back into the source when a summary can't answer, tells you what's read-directly vs.
  reported, and leads with the answer. Her voice-command routing is now a fourth editable
  persona in the profile editor.

- **Agents can hear and speak now**: attach an audio clip or record a voice note on any message —
  audio-capable models hear it natively, every other model gets an automatic transcript (cached,
  never re-billed) — and ask an agent to read something aloud: `generate_speech` renders an
  inline player in the conversation.
- **Exhibits went multi-modal**: new audio/video/markdown-text elements plus a side-by-side
  `grid` layout; media an external MCP tool returns (screenshots, clips) now renders as an
  exhibit instead of being silently dropped; and generated media survives reload.
- **Agents can hand media to each other now**: delegations carry images and audio — a supervisor
  passes the attachment's id and the specialist receives it inside its first message, matched to
  what it can handle (vision models see it, audio models hear it, text models get a transcript).
  The team roster now advertises who *sees images*, *hears audio*, or *makes images*, so routing
  media work to the right agent is autonomous.
- **Image editing works end-to-end**: attach a picture and say "restyle this" — to an image-model
  agent directly, through the `generate_image` tool (new source-image support), or via delegation
  to an image specialist. Previously every one of these paths silently dropped the source image.

- **Under the hood, media got honest accounting**: speech became a neutral capability every surface
  shares (one voice config, enforced by an architecture test), context budgeting now sees the true
  weight of attached images and audio, and long sessions stop accumulating media memory in the app.

- **Memory got a face**: a custom glowing-brain mark — pulsing glow, twinkling synapses — replaces
  the database stack in the command palette, the Memory explorer, and Settings.
- **The Start page became a launchpad**: type right on the hero and Enter *is* your first message —
  plus a living agent mark (orbiting star + glow), a starfield backdrop, a time-aware greeting with
  a live status line, glass recent-conversation cards with previews ("Pick up where you left off"),
  and a "While you were away" card that opens the Ambassador's Command Deck.

### Fixes

- **No more leaked thinking**: a reasoning model's `<think>` blocks are now stripped from the
  Ambassador's answers, replays, and speech — previously they could persist into the thread,
  render raw, and even be read aloud by TTS.
- **Deck menus surface properly**: the Inquiry switcher, ⋯ actions, worker picker, and relay
  picker no longer open *behind* the Command Deck (dropdown/popover portals now clear the
  full-screen surfaces).
- **Research reports are dated correctly**: a report is now stamped with today's date instead of a
  date guessed from its sources. (Bundled with internal housekeeping — declared ambassador voice
  config keys, typed the plan-executor's subtask status, and Anthropic cache-token metering.)
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

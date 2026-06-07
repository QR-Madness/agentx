<!-- release-version: 0.21.57 -->
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

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

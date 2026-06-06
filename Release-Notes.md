<!-- release-version: 0.21.47 -->
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

### Fixes

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

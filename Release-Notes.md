<!-- release-version: 0.21.34 -->
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
  paraphrasing it. Briefings never touch the transcript or the agent's context; future
  spoken briefings slot onto the same seam.

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

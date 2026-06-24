<!-- release-version: 0.21.132 -->
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

- **Agents can generate images — avatars and in-conversation.** Give an agent a real face from the avatar picker's **Generate** tab ("a gray-haired strategist…"), and **agents can now generate images mid-chat** ("draw a diagram of…") — the picture renders inline in the conversation. Powered by OpenRouter; the app-wide avatar style lives in **Settings → Images**; cost is tracked (Usage → Images / Avatars). Under the hood: a reusable **image transport** (blob store + serving + a personal **"Home" workspace**) — the foundation the rest of the multi-modal pipeline (vision input next) builds on.
- **The Ambassador is now a parallel operator — with a standalone Command Deck.** A dedicated agent that runs *beside* your conversations (never in them). Its panel is an **Inquiry** you ask, brief, or talk to — with read-only tools to read across your conversations, **survey them all and summarize what your agents have been working on** (including each conversation's goals), and know its **agent roster** (each agent's role, what its model can do — image/audio/vision/tools — and whether it's delegable; groundwork for multi-modal routing). Open it from anywhere as a full-screen **Command Deck** (⌘K → "Open Command Deck") — no conversation required; the deck holds **multiple named Inquiries** (create / switch / rename / delete), Inquiries **auto-title from your first question**, and the ambassador can **retitle its own Inquiry** as its focus sharpens.
- **Ambassador voice now relays into any conversation.** In hands-free voice mode, the relays the Ambassador drafts for your approval can be sent into **any** conversation — the one it's focused on or one you pick in the deck — even when that conversation isn't open, delivered as a real message from you (headless `POST /ambassador/relay`).
- **Stabler memory in long chats.** A stable high-salience "core" of what the agent knows is now kept in view every turn (no longer re-guessed from each message), with query-specific recall as a supplement. A new context *ledger* budgets the whole preamble by priority, so a growing sidecar can no longer silently crowd out the live transcript.

### Fixes

- **PostgreSQL schema now managed by Alembic.** Self-hosters get reliable, incremental DB upgrades — the container applies pending migrations on start (your existing data is adopted in place, never recreated). Neo4j keeps its own migrator. *Upgrading: be on a recent prior version first so your DB is already current before this switch.*
- **Ambassador voice relay reliability.** Voice relay now reaches the active conversation reliably and tells you where it landed (no more silent "Send to agent" that went nowhere); read-only "summarize/explore" voice asks run automatically.
- **Delegated work now counts toward your usage.** Multi-agent (Alloy) delegation cost was computed but dropped; it's now recorded, so specialist spend shows up in usage/cost totals. A unified spend ledger now also captures **Ambassador** activity — every briefing, question, and voice answer — which was previously untracked entirely. **Voice is metered too**: text-to-speech (per character) and speech-to-text (per minute) now carry a cost estimate, with per-model audio rates you can override. **Web search is now metered + budgeted**: each Tavily/Brave call records an estimated credit cost, and a per-turn cap (`search.per_turn_limit`, default 8) stops a runaway tool loop from burning unbounded search credits. The dashboard's Usage & Cost panel adds a **by-source breakdown** so you can see where spend goes — chat vs multi-agent vs ambassador vs voice vs web search.
- **Set a global default model.** Settings → Models now has a default-model picker — the model new agents and ad-hoc requests fall back to when a profile doesn't pin its own.
- **A down or misconfigured model no longer crashes the turn.** Model fallback now covers every feature — chat, reasoning, drafting, the planner, plans, and multi-agent specialists (previously only the Ambassador) — so an unavailable model degrades to a working one instead of failing; a swap surfaces as a status notice. Specialized roles (speculative draft/target, voice TTS/STT) and explicit availability checks stay strict; toggle with `models.fallback_enabled`. Queued/background chats also now resume **warm** — a backgrounded reply rehydrates the conversation's history just like an interactive one.
- **Forgotten facts stay forgotten.** Soft-forgetting a fact now removes it from recall outright (vector, keyword, and entity search) instead of merely lowering its rank — genuinely-past but still-valid facts are unaffected.
- **More consistent context budgeting.** Chat, memory, the rolling summary, and the context ledger now share one tiktoken-backed token estimator (instead of four divergent chars/4 heuristics), so "what fits the window" is judged the same everywhere; unused legacy context knobs were removed.
- **Hand your agent files (Workspaces & Document RAG).** New workspaces you can upload documents into (PDF + text/markdown/code): each file is parsed, chunked, embedded into pgvector, and auto-tagged + summarized so it becomes searchable. Attach a workspace to a conversation and the agent *sees its file list* every turn, then retrieves with two new tiers — search the catalog (by name/tag) and search inside documents (by meaning) — citing the documents it used. Bytes are stored content-addressed (dedup) with per-file + per-workspace quotas. A **Workspaces drawer** (command palette → "Open Workspaces") manages workspaces (create/rename/delete inline) and click-to-upload (with live ingest status), and you attach a workspace to a conversation from there — attached conversations show the workspace as a prominent chip in their header (which also dropped the noisy session id to free up that space).
- **Agents can run commands in a workspace (opt-in per workspace, sandboxed).** Flip **"Allow shell" on a workspace** (off by default) and an attached agent can run shell commands and read/write files against a **sandboxed working copy** of it. Every command runs in a bubblewrap jail — **no network, no access to your secrets/`data/`, scrubbed env, time-limited** — so a prompt-injected agent can't exfiltrate or roam the host. Enablement is per-workspace, not a global switch. Each workspace also picks a **shell backend**: the lightweight bubblewrap jail, or a **persistent per-workspace container** the agent can `pip`/`apt`-install into (network on, files + installs persist) — run in an isolated Docker-in-Docker sidecar, with status/Start/Stop/Reset/Remove controls.
- **Browse stored tool outputs.** A debug surface (command palette → "Tool Outputs") lists every large tool result stashed in the cache — filter by tool, read the full body, copy, and prune individually or all at once.
- **Model-capability awareness.** An agent on a model that can't use tools no longer 404s on its first turn — tools are simply not sent to it. The model picker now shows an **Image gen** badge (image *output*) alongside Vision (image *input*), so you can pick the right model for generation. Generated avatars now render correctly everywhere — full-size, while a reply is *streaming* (not just after it lands), un-squished in the profile-editor hero, and on the Start page / Ambassador / delegation traces too (every spot that previously fell back to the default star for a generated face).
- **Direct mode for bare models — and image models that actually return an image.** New per-agent **Direct mode** (Settings → agent → Advanced) sends the model only your message — no system prompt, no memory, no tools — for a transform-only model (a fast classifier/rewriter), **auto-enabled for image-only models** (e.g. flux). And an image-output model used *as* a direct-mode conversation agent (flux, gemini-flash-image) now **renders the picture inline** instead of returning an empty turn — its image is stored, shown, and survives reload. The profile also now honors its own **memory toggle** in chat (previously only the request-level switch applied).
- **No more straggler dev servers — and the memory worker actually runs in dev.** `task dev` now reaps leftover dev processes before starting (a crashed `uvicorn --reload`/`uv run` supervisor used to keep the API bound to port 12319). New `task dev:kill` (alias `dev:reap`) frees the app ports on demand; `task debug:ports` points you at it. Also fixed the consolidation worker in `task dev` / `task memory:worker` — it ran from the wrong directory and silently failed to import (`No module named 'agentx_ai'`), so procedural/semantic memory wasn't accruing during development. Windows devs are steered to WSL2, where process lifecycle behaves correctly. *(Contributor tooling only.)*

### Getting started

See the [documentation](https://agentx.thejpnet.net/docs) — the
[quickstart](https://agentx.thejpnet.net/docs/getting-started/quickstart) and
[self-hosting guide](https://agentx.thejpnet.net/docs/deployment/self-hosting) cover
standing up the server and connecting the client.

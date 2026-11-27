# Docs Site — Beginner-Friendly, Service-Ready

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

**Goal.** A newcomer who is not a contributor can tell what AgentX is, pick a path, and reach a
first useful conversation without reading architecture pages. The same front door will later
serve JNT service customers, so the docs keep three jobs apart: *use it*, *run your own
server*, and *build on it*.

**Rules carried over from the July overhaul:** pages lead with usage; user guides carry no raw
API endpoints (one pointer to the API reference instead); no UI micro-detail inventories; named
controls get inline lucide icons; examples stay abstract (reasoning, not coding); no emoji.
Never describe a hosted backend before one exists.

### Audit (2026-11-25, read as a newcomer)

- **Docs home is written for developers.** It opens with a jargon-dense sentence and the
  internal architecture diagram, and its quick links go to the API reference, architecture,
  database stack and contributor setup.
- **Get Started leads with contributor pages.** Installation (build from source) and Windows
  Setup (WSL2 development) come before Quick Start.
- **Quick Start is stale and stops early.** It says "fill 3 values" in `.env`, while Self-Hosting
  correctly says the defaults need no edits. It names no prerequisites (Docker, memory, a model).
  It offers only the desktop installers. It ends at "you're chatting", with nothing about the
  first session.
- **Prerequisites are stale in two places.** Installation and Development Setup both list Python
  3.10+ and Node 18+; the project needs Python 3.14, and Vite needs Node 20.19+ or 22.12+.
  The two pages also duplicate each other, which is how both drifted.
- **No concepts page or glossary.** Agent profiles, thinking patterns, Agent Teams, connectors,
  providers, memory channels, Projects, the Ambassador and Cores are each introduced on their
  own feature page and nowhere together.
- **No troubleshooting or FAQ page.** The useful fixes sit in a collapsed block at the end of
  Self-Hosting.
- **Search results were unstyled.** Their scoped CSS never reached rows created by script, and
  the hint put the raw query into `innerHTML`. Fixed in #149.
- **Projects has no page in the nav.** `features/workspaces` covers it under the internal name
  "Workspaces", and the Memory page links "Project" to the Chat page instead.
- **Some feature pages lead with internals.** Agent Teams opens with phase history and API field
  names (`workflow_id`, `available_for_delegation`, `delegate_to`); Reasoning names
  `POST /api/agent/run` in its first section.
- **Extract and Cores are missing from the user docs**, though the release notes headline Extract.
- `docs-site/README.md` still calls the landing page a blank canvas.

### Slices

- [x] **0 · Search dialog** — results and hint styled; the query is treated as text (#149).
- [x] **1 · Get Started, accurate and in newcomer order** (#151)
  - Quick Start leads the section and matches Self-Hosting (no `.env` edits).
  - It gains a prerequisites box and the web-client path (desktop app for a local server; the
    hosted web client once the server has an HTTPS address and allows the origin).
  - Installation becomes a short router page outside the nav, so existing links still land.
  - Development Setup becomes the one build-from-source page, with current prerequisites.
  - Windows Setup moves to Contribute.
- [x] **2 · Concepts page** — `getting-started/concepts`: a sentence or two per core term, each
      linking to its page. In the nav after Quick Start, which points to it.
- [x] **2b · Projects page** (#155) — the Workspaces page retitled to the app's name and
      rewritten usage-first from the client's labels; in Using the Platform; Memory's "Project"
      link points at it.
- [x] **3 · Docs home for newcomers** — a plain-language opener, a "Start here" table of
      paths, and a feature table in plain terms. The component diagram went (Architecture →
      Overview has a fuller one); the technology stack moved there; the stale status list went
      (the Roadmap holds the history). Also fixed: two links to a `main` branch that doesn't
      exist (the License link and one in Docker).
- [x] **4 · Your first session** — `getting-started/first-session`, written from a real
      walkthrough on a scratch instance (OpenRouter, $0.014 in model spend):
  - Connect a provider, then set the Global Default Model.
  - Chat, make an agent of your own, and watch memory recall and consolidation.
  - Nine screenshots (WebP, 364 KB in total), captured at 2× from a headless Chromium driven
    over CDP.
  - The walkthrough corrected the default-model guidance: Quick Start pointed at "Settings";
    Providers, Configuration and `.env.example` said `DEFAULT_MODEL` drives chat (it drives
    only `/api/agent/run`); and `.env.example`'s bare model name would be rejected.
- [x] **5 · Troubleshooting** — `getting-started/troubleshooting`: server setup, connecting a
      client (including web-client CORS and the gateway token), the root password reset, models
      (empty picker, LM Studio in Docker, Claude's temperature), and where the logs are. Also
      fixed: Authentication's opener said auth was off by default; it is on.
- [ ] **5b · Feature pages lead with usage**
  - [x] Thinking Patterns (#157): the user guide first, then a "For developers" section with
        the offline kit, config keys and the drafting schema.
  - [x] Agent Teams: a plain-language intro and Ambassador section. The history, API fields,
        data model, runtime, streaming events, storage and status move to Architecture → Agent
        Teams Internals.
  - [x] Extract on the Memory page: the verify-before-wipe order, the vault, the consolidation
        freeze, and importing a vault file to restore. Checked against ADR-14.
  - [ ] (Low priority) Link Concepts terms on their first use in each feature page. Weigh
        the gain against awkward edits across about ten pages.
- [ ] **6 · Service-facing material (on hold, by the user's decision)** — what a managed Agent Core
      offers (bring your own model key, request metering), a privacy and data statement, and
      the landing page's call to action. Blocked on the launch gates (repo move and licence,
      phase 19 cloud operation) and on the pricing shape (ADR 007, unwritten).

### Decisions

- **The web client's origin is not a CORS default yet** (the user's decision, 2026-11-26). The
  API's default CORS list allows the desktop origins only, so the hosted web client needs
  `CORS_ALLOWED_ORIGINS` before it can talk to a newly self-hosted server, and the docs say so.
  Once the web client moves to its proper JNT domain at launch, that domain becomes a default.
  - Treat it like `_TAURI_ORIGINS` in `settings.py`, configurable with an opt-out.
  - Update the manager Share dialog's CORS check to match.
- **Slice 4's walkthrough runs on OpenRouter** on a scratch instance, keeping model spend under
  $2.50.

### Product issues the walkthrough found

Each one is a code change, so it belongs outside the docs track:

- **A fresh install with only a cloud provider fails its first default conversation.** Chat
  resolves the conversation's model, then the profile's, then the Global Default Model, then
  `lmstudio:llama3.2`. A newcomer who connects OpenRouter and starts chatting gets "Provider
  'lmstudio' not configured". The docs now say to set the global default.
  - Fix: a clear message pointing to Settings → Model Roles, or onboarding that sets the
    default once a provider connects.
- **The Memory panel doesn't reload after Consolidate finishes.** The Facts tab shows "No facts
  found" until you press refresh, while the overview counts are already updated.
- **The context meter reads "100% context" (in warning colour)** on a conversation using
  "Default model". It seems not to know that model's context window.
  - **Done (v0.21.297):** "Default model" was incidental; the Settings save before the turn was
    the cause. Every save reloads the provider registry, which drops the fetched model catalogs,
    and the next turn read its capabilities cold: OpenRouter's and the AI Gateway's 8192-token
    fallback stood in for a 1M window and sized the turn (output and tool-loop budgets) as well as
    the meter. Turns now warm the catalog first (`providers/capabilities.warm_catalog_for`, in the
    stream and in `Agent.chat`), and a fresh "Default model" conversation seeds its chip from the
    Global Default Model. Conversations that hit the bug keep their stored 8192 reading until their
    next turn.
  - **Open, proposed:** send `context_window: null` in `start`/`done` when the provider's catalog
    doesn't list the model, so the meter hides rather than showing a percentage of a fallback. It
    changes the SSE contract (`int` to nullable), and filtered or static catalogs need care: OpenAI
    lists only `gpt-`/`o1-`/`o3-` ids and Anthropic falls back to a built-in list, so an unlisted id
    can still be a real model.
  - **Open, related:** assistant turns record the bare model id (`done.model`), while the model
    catalog's ids are `provider:model`. The chip's backfill for older conversations and
    `MetadataBar`'s cost backfill therefore never match the catalog.
- **Two names for the default memory channel.** Profiles and the docs say `_global`, but turns
  and facts are stored under `_default` (`agent/core.py`), which is what the Memory panel shows.
- **The embeddings card on Model Providers says "Local · Locked"** even when cloud embeddings
  are configured (its provider field then reads `openai`).
- **The Settings Reference says `preferences.default_model` is "Set via
  `/api/config/update`"**, but the registry places it on the Model Roles screen
  (`ui_section="model-roles"`). The generated reference doesn't reflect the screen.

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
- [ ] **2b · Projects page** — retitle the Workspaces page to the app's name, add it to Using
      the Platform, and point Memory's "Project" link at it.
- [x] **3 · Docs home for newcomers** — a plain-language opener, a "Start here" table of
      paths, and a feature table in plain terms. The component diagram went (Architecture →
      Overview has a fuller one); the technology stack moved there; the stale status list went
      (the Roadmap holds the history). Also fixed: two links to a `main` branch that doesn't
      exist (the License link and one in Docker).
- [ ] **4 · Your first session** — add a provider key, pick a model, chat, create a profile,
      watch memory work. Written from a real walkthrough on a scratch instance, with
      screenshots.
- [ ] **5 · Troubleshooting & FAQ** — client can't reach the server, sign-in and gateway token,
      model key errors, Docker Desktop file sharing, the first-boot download, where the logs
      are.
- [ ] **5b · Feature pages lead with usage** — move internals (Agent Teams' history and API
      fields, Reasoning's endpoint) to the architecture pages; link Concepts terms on first use;
      document Extract on the Memory page.
- [ ] **6 · Service-facing material (needs founder decisions)** — what a managed Agent Core
      offers (bring your own model key, request metering), a privacy and data statement, and
      the landing page's call to action. Blocked on the launch gates (repo move and licence,
      phase 19 cloud operation) and on the pricing shape (ADR 007, unwritten).

### Open questions

- **Should a fresh server allow the official web client by default?** The API's default CORS
  list allows the desktop origins only, so the hosted web client at `agx.thejpnet.net` needs
  `CORS_ALLOWED_ORIGINS` set before it can talk to a newly self-hosted server. Adding the
  origin to the deploy bundle's defaults would remove that step. Credentialed CORS makes it a
  security decision, not a docs one.

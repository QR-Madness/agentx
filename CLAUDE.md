# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is the **light index**; deep subsystem internals live in [`Development-Notes.md`](Development-Notes.md).

## Documentation Map

Find the right doc before diving in — they're split deliberately so this file stays small.

| Doc | What it is |
|-----|-----------|
| **[`Todo.md`](Todo.md)** | Roadmap **index** — Progress Tracker + a map into [`todo/`](todo/) |
| **[`todo/phases/`](todo/phases/)** | Per-phase work (`completed.md`, `phase-15/16/18/19-*.md`); **16** holds the live Ambassador planning; **19** is Cloud Operation |
| **[`todo/backlog/`](todo/backlog/)** | Future work by theme — `foundation`, `workspaces`, `memory-recall`, `procedural`, `retrieval-extraction`, `chat-ux`, `genome-advisor`, `open-platform`, … |
| **[`Memory-Roadmap.md`](Memory-Roadmap.md)** | Memory-system hardening & experimental roadmap; pairs with the `memory-*` backlog files |
| **[`Development-Notes.md`](Development-Notes.md)** | Deep subsystem internals + the full API/SSE reference (not auto-loaded — read when working that area) |
| **[`Decisions.md`](Decisions.md)** | Load-bearing invariants + locked decisions (the "don't relitigate" list) — read before changing memory/ambassador/streaming internals |
| **[`Repo-Questions.md`](Repo-Questions.md)** | Open hard design questions parked for a deep answer (Fable answers; resolutions fold into the docs above) |
| **[`Release-Notes.md`](Release-Notes.md)** | Human-written notes for the *next* release (see the version rule below) |
| `OpenApi.yaml` + `docs-site/.../api/endpoints.md` | Authoritative API contract |

> When you change code, update the matching doc in the same change (the backlog file, the
> roadmap, `endpoints.md`/`OpenApi.yaml`, `architecture/memory-capabilities.md`). **`task docs:check`**
> guards the mechanical half — broken inter-doc links, orphaned `todo/` files, version drift, and
> Release-Notes size; it runs inside `task release:check`. [`Decisions.md`](Decisions.md) is the
> judgement half a script can't enforce.

## Development Notices

- The client is cross-platform, thus UI should be highly responsive and use comfortable hit-regions.
- Following API & Client v.0.20 (see `versions.yaml` for the authoritative version), all changes should be migratable for existing platforms.
- **Version + notes travel with the work.** Any notable/user-facing change must **bump the version and update the release notes in the same commit** — bump `versions.yaml` (patch, via `task versions:sync` to propagate to all manifests **and refresh `uv.lock`/`Cargo.lock` — the lockfiles ride the version commit**), bump the `<!-- release-version: X.Y.Z -->` marker in root `Release-Notes.md` to match, and add the change to the `Release-Notes.md` body (it always describes the *next* release). This is a continuous dev habit, not a release-time step: `task release:check` asserts the marker matches `versions.yaml`. See [Build & Release](#build--release).

## Hard-Won Working Rules

Landmines that cost real debugging time — read before touching memory:

1. **Settings overrides:** the memory kit reads settings **live** (zero module snapshots,
   ratchet 0). Temporary overrides use ONE mechanism (see Development-Notes):
   `with pin_memory_settings(override):` — never `save_memory_settings()` (it writes
   `data/memory_settings.json` for a live server).
2. **Two extraction paths:** windowed (default) + legacy per-turn behind
   `extraction_windowing_enabled` — behavior changes must edit both or gate on the flag.
3. **Eval harnesses:** `eval_consolidation` is GLOBAL (needs a sterile cluster or `--snapshot`;
   its drain loop exists because discovery sweeps 10 conversations at a time — keep it).
   `eval_recall` is user/channel-scoped and safe on a live cluster. Extend their CASES/corpus —
   don't write one-off eval scripts. Both need `task db:up`.
4. **Red means you:** the test suite and pyright baseline (`api/.pyright-baseline` = 0, gated in
   CI) are green. Any failure or new type error is yours — never rationalize it as pre-existing.
5. **Harness artifact ≠ model failure:** an eval case scoring "stored nothing" is usually
   plumbing (unconsolidated conversation, truncated output, channel mismatch) — check
   `c.consolidated` and the metrics counters before touching prompts.
6. **Prompt templates** (`prompts/system_prompts.yaml`) use `{var}` substitution — literal JSON
   braces must be doubled `{{ }}`; the render test in `tests_memory.py` guards this.

## Project Overview

AgentX is an AI Agent Platform combining:
- **Backend**: Django REST API (`api/`) on port 12319 — translation, agent memory, MCP client, model providers, drafting, reasoning
- **Frontend**: Tauri v2 desktop app (`client/`) with React 19, TypeScript, Vite
- **Data Layer**: Neo4j (graphs), PostgreSQL + pgvector (vectors), Redis (cache) — all via Docker

## Terminology

| Term | Meaning |
|------|---------|
| **Agent Profile** / **Profile** | Configuration defining an agent's identity, behavior, settings. The primary configuration entity — produces "agents" when used. |
| **Global Settings** | Apply across all agents (server connections, API keys, MCP tools) |
| **Profile Settings** | Per-agent (model, temperature, system prompt, reasoning strategy, memory channel) |

Agent profiles configure: name, avatar, default model, temperature, system prompt, reasoning strategy, memory enable/channel, tool enable. The global prompt has no agent name — agent names are injected from the profile during prompt composition.

## Architecture

```
Tauri Client (React 19 + Vite)          Django API (port 12319)
  TopBar → Start, Dashboard, AgentX       Agent Core (planner, session, context)
  ConversationTabs (browser-style)        ├── MCP Client (consume external tool servers)
  Drawers: Settings, Memory, Connectors   ├── Reasoning (CoT, ToT, ReAct, Reflection)
  Modals: Translation, Prompt Library     ├── Drafting (speculative, pipeline, candidate)
     ↕ HTTP                               ├── Model Providers (LM Studio, Anthropic, OpenAI, OpenRouter, Vercel)
                                          ├── Context Gating (compression, chunking, retrieval)
                                          ├── Translation Kit (NLLB-200, 200+ languages)
                                          └── Agent Memory (episodic, semantic, procedural, working)
                                                ↕
                                          Neo4j │ PostgreSQL (pgvector) │ Redis
```

### Key Backend Modules (`api/agentx_ai/`)

One-liners for orientation. **Deep internals for the starred subsystems are in
[`Development-Notes.md`](Development-Notes.md).**

- `kit/translation.py` — `TranslationKit` (NLLB-200) + `LanguageLexicon` (ISO 639 detection↔translation code bridging)
- `kit/agent_memory/` — memory system, lazy-loaded connections (`interface.py` → `connections.py` → impls); `RecallLayer` = 5 retrieval techniques (hybrid, entity-centric, query expansion, HyDE, self-query) + a post-fusion cross-encoder rerank stage (50-pool, bounded demotion; default-ON). ★
- `kit/shell/` — Agent Shells: **opt-in per-workspace** (`workspaces.allow_shell`, off by default) sandboxed command execution — bubblewrap jail default, per-workspace Docker-container backend optional. Internals + threat model ★. e2e: `scripts/shell_e2e.py`.
- `kit/workspaces/` — File Workspaces & Document RAG, surfaced as **Projects** (instructions ride every turn; durable conversation membership; `_project_{ws_id}` memory channels; `ws_home` is never a project). Agent tools `list_project_files`/`project_search` (legacy alias `workspace_search`)/`document_query`/`read_document` + write tools `create_document`/`update_document`/`append_to_document`/`edit_document`/`rename_document`/`delete_document` (partial edits use an `expected_sha256` soft write-lock). Internals ★. e2e: `scripts/rag_e2e.py`.
- `mcp/` — MCP client manager, server registry, tool executor, transports, remote OAuth 2.1 (`auth_state` reports `expired`/`refreshable`), registry-search proxy; `mcp_servers.json`; `media_passthrough.py` stores image/audio blocks external tools return (capped, untrusted) and surfaces them as exhibits instead of dropping them. Client surface: **Connectors & Tools** (connector catalog + registry search; internal name stays `toolkit`). ★
- `content_blocks.py` — multi-modal payload vocabulary mirroring MCP/ACP ContentBlocks (text/image/audio/resource_link/resource, base64 `data`+`mimeType`); the seam shared by providers (`StreamChunk.media`), the MCP executor, and exhibits. Audio in/out rides it: `MediaRef` refs + capability-gated `input_audio` vs STT fallback; `agent/audio_gen.py` + `generate_speech` = the audio twin of `image_gen.py`/`generate_image`. ★
- `kit/speech.py` — neutral TTS/STT seam (resolution, hygiene, `SpeechUnavailable`, metering); the Ambassador keeps only profile-precedence wrappers, chat consumes directly. **ADR-11**: capabilities live in neutral modules, surfaces consume — enforced by `tests.CapabilitySeamBoundaryTest` (core packages must not import the ambassador family). `providers/capabilities.py` = the one warm-once modality probe.
- `providers/` — abstract `ModelProvider` (LM Studio/Anthropic/OpenAI/OpenRouter/Vercel); `models.yaml` configs + defaults, `pricing.py` cost. Resolution/fallback ★
- `config.py` — `ConfigManager` singleton; persists `data/config.json`, dot-notation access + env-var fallback
- `drafting/` — speculative decoding, multi-stage pipelines, N-best candidates; `drafting_strategies.yaml`
- `reasoning/` — **Thinking Patterns**: chat patterns compiled into the streaming turn (`chat_patterns.py` + `streaming/thinking_exec.py` — native/cot/step_back/reflection/deep_reflection/self_consistency; `selection.py` = the shared auto brain) + the offline CoT/ToT/ReAct/Reflection kit for `/agent/run`. ★
- `agent/` — `Agent` orchestrates reasoning + drafting + tools; `TaskPlanner` decomposes (chat path composes plans with the main agent model; legacy `plan()` for non-chat callers ★); `SessionManager` for conversations.
- `agent/profiles.py` — `ProfileManager` CRUD (`data/agent_profiles.yaml`); Docker-style `agent_id` + `self_channel`; ships seeded defaults (AgentX, Researcher, **Deluxe Image Creator** — one-time `seeded_defaults` markers, deletions stick; sync with `api/defaults/agent_profiles.yaml`). **Rule:** `kind` ∈ `agent`|`ambassador`, and ambassadors are **excluded from chat** (default/routing/`delegate_to` filter `kind=='agent'`). ★
- `agent/skills.py` — **Agent Skills**: named instruction packs, progressively disclosed — compact index in the chat prompt (`views._skills_block`), bodies load via the `use_skill` internal tool; `data/skills.yaml` (ProfileManager-style seeding, sync with `api/defaults/skills.yaml`); access via `allowed_agent_ids`. UI: Connectors & Tools → Skills. ★
- `alloy/` — **Agent Teams** (user-facing name; internals/routes/config keep `alloy` — Workspaces→Projects precedent): Team (workflow) CRUD (`data/workflows.yaml`), `delegate_to` tool (+ per-dispatch `effort` tiers quick/standard/deep/marathon → tool-round budgets, `alloy.effort_tiers`) + `AlloyExecutor`; supervisor prompt in workflows, **soft ad-hoc roster block** in normal chats (opt-in `available_for_delegation` + `delegation_hint`; per-conversation `disable_delegation`). ★
- `agent/tool_output_compressor.py` / `tool_output_chunker.py` — task-aware LLM compression for oversized tool outputs (section detection, JSON-path, semantic search)
- `streaming/trajectory_compression.py` — Focus-style intra-trajectory compression for multi-round tool loops
- `prompts/` — `PromptManager` + durable layered system-prompt stack (`LayerStore`). ★
- `agent/context.py` — per-turn `assemble_turn_context` (verbatim budget + conversation-state digest compaction + checkpoints/scratchpad); knobs in Settings → Memory → Conversation Context. ★
- `agent/ambassador.py` (+ `ambassador_storage.py`, `ambassador_tools.py`, `aide_swarm.py`, `conversation_meta.py`) — the **Ambassador**: parallel conversational operator with persistent "Inquiry" threads + the standalone **Command Deck**; a tool belt that **never executes a write** — reads auto-run (read/survey conversations, structured state, what a conversation *produced* (exhibits/sources), full-text search, live runs, usage, memory recall, roster; aide digests), while conversation-meta writes (rename/archive/delete) AND task dispatches (`dispatch_task`) are **proposal-only** (confirm strip → `PATCH /memory/conversations/{id}/meta` / `POST …/dispatch`; new-or-existing conversation, instant task echo client-side), the write-side landing only as *your* user turns (relay/dispatch); sidecar-only, never pollutes the transcript. ★
- `logging_kit/` — centralized logging (queue handler → console/ring-buffer/`/api/logs` + daily encrypted archives), `AGENTX_LOG_*` flags. ★

★ Plus the **memory subsystems** (extraction pipeline, procedural memory, context gating, web-research tools, model resolution & fallback, multi-agent attribution) and the **full API + chat-stream SSE reference** — all in [`Development-Notes.md`](Development-Notes.md).

### Key Client Patterns (`client/src/`)

The full **client surface map** (chat page anatomy, Ambassador Deck, Projects hub, profile/prompt
editors, theme internals) lives in [`Development-Notes.md`](Development-Notes.md) — read it before
touching a surface. The rules that must not drift:

- 3 primary pages (Start, Dashboard, AgentX) routed via `RootLayout` + `TopBar`; gate pages `AuthPage` (`AGENTX_AUTH_ENABLED`) + `VersionMismatchPage`. TopBar also carries desktop **surface pills** (Deck, Memory) — no new `PageId`s: selected state derives from `useModal().isOpen`, the surfaces sit below the 56px bar, and pills are palette-only on mobile.
- **Command palette is the primary command surface** — `components/common/CommandPalette.tsx` over the `hooks/useCommands.tsx` registry; palette + TopBar icons share `lib/surfaces.ts` (`SURFACES.*`) so they can't drift.
- Destructive actions use `ui/ConfirmDialog` (`useConfirm()`), never native `confirm`.
- **Settings sections** build on `components/settings/fields/` + `useSettingsAutosave`
  (see `PlannerSection`); **secrets keep explicit Save**.
- Multi-server: `ServerContext` app-wide; `lib/api` typed client facade; `lib/hooks.ts` data hooks on the `useApi<T>` factory; `AgentProfileContext` for profiles.
- **Add a theme = one entry in `THEMES`** (`lib/theme.ts`) — pickers iterate the registry; a vitest enforces cross-theme token parity; glow tokens use a transparent shadow, never bare `none`.
- API errors: `ApiError` carries a status-derived `kind`; use `apiErrorMessage(err)`/`toApiError(err)`; surface via `useNotify().notifyError(err)` (toasts); inline errors only for form-field validation.
- **Two shells (desktop + web/PWA)** — one React app, gated by compile-time `__IS_TAURI__`. **Rule:** `@tauri-apps/*` is imported **only** under `src/platform/` (capability façade `platform.opener`/`platform.window`; `importBoundary.test.ts` fails on any stray import, keeping the web bundle Tauri-free). PWA shell in `src/pwa/`; share-a-server **connection links** in `lib/connectionString.ts` (`#connect=…`) → `ConnectGate`. Full detail: [`Development-Notes.md`](Development-Notes.md) → Client Surface Map.

#### Styling (Tailwind v4 + design tokens)

- **Tailwind v4** via `@tailwindcss/vite`. CSS entry `src/App.css` imports only the `theme` + `utilities` layers — **Preflight is intentionally disabled** so it doesn't clobber `styles/base.css` (imported into `base`; utilities out-rank it, unlayered per-component CSS out-ranks utilities).
- **Design tokens** in `lib/theme.ts`, injected at runtime by `ThemeProvider` as CSS vars; `App.css` bridges them via `@theme inline`. Use **semantic utilities**, not raw palette: `bg-surface-base|raised|overlay|sunken|hover`, `text-fg|fg-secondary|fg-muted|fg-inverse`, `border-line|line-strong`, `text-accent|bg-accent(-secondary|-tertiary)`, feedback `text-error|success|warning|info`. Spacing tokens `--space-*` for hand-written CSS. Brand shadows stay `var(--shadow-md)`.
- **Components**: prefer Tailwind for new/shared UI; keep per-feature CSS for complex panels. Shared primitives in `components/ui/` follow shadcn (CVA + `cn()` in `lib/utils.ts`, exported from `components/ui/index.ts`). Radix enter/exit animations from `tw-animate-css`. **Form controls must use the field primitives** — `Input`/`Textarea` (`ax-field`, `--sm` variant, `icon` slot), `FieldTrigger` for select-like dropdown triggers, `.ax-fieldwrap` for composer wrappers hosting a transparent textarea. **Icon-only buttons use `IconButton`** (md/sm/xs, `tone="danger|accent"`, `active`); status pips use `StatusDot`. Never hand-roll `bg-surface-raised border-line` fields or ghost text-button pickers (washed-out regressions). Preflight is OFF but `base.css` resets button background/color — still give intentional-transparent buttons an explicit `bg-transparent`. Kit scale via `@theme static`: `text-2xs…4xl`, `tracking-caps` (eyebrow labels = `text-2xs font-semibold uppercase tracking-caps text-fg-muted`), `rounded-sm..2xl/pill` (6/8/10/12/16/999px), `border-line-subtle`, `font-mono` for metadata.

## Development Commands

All commands use [Task](https://taskfile.dev/) (`Taskfile.yml`). Run `task --list-all` for the complete list.

```bash
# Setup & dev
task setup              # First-time: install deps, init DB dirs, verify env
task dev                # Start Docker + API + Client concurrently (alias: d)
task dev:down           # Stop it all: reap dev processes + stop DBs (alias: dd)
task dev:api            # API only (assumes Docker running)
task dev:client         # Tauri client only (assumes API running)
task dev:web            # Client in browser mode (port 1420, no Tauri)
task install            # Install all deps (uv sync + bun install)

# Database (Docker) — Neo4j, PostgreSQL, Redis
task db:up / db:down    # Start / stop services
task db:init:schemas    # Init schemas: PG via Alembic (upgrade head) + Neo4j/Redis
task db:migrate         # Apply pending migrations: PG (Alembic) + Neo4j
task db:revision -- "msg"  # New Alembic (PostgreSQL) revision — see alembic/README
task db:shell:postgres  # psql / db:shell:redis / db:shell:neo4j

# Django
task api:run            # Dev server; also api:migrate / api:makemigrations / api:shell

# Deployment manager (manager/ — owns cluster lifecycle; ADR-10)
task manager:serve      # Web GUI on http://127.0.0.1:12320 (bearer token in .manager-token)
task cluster:up/down/restart/destroy/adopt/status/list CLUSTER=x   # over `agentx-manager`
```

> **Schema migrations:** the memory **PostgreSQL** schema is managed by **Alembic**
> (`alembic/`; baseline frozen, single-head — gated in `task docs:check`). **Neo4j** stays on the
> home-grown runner (`manage.py migrate_schema`, Neo4j-only). Django's own SQLite ORM is separate
> (`api:migrate`). New PG change = an Alembic revision (never edit `alembic/baseline.sql`); see
> [Decisions.md](Decisions.md) ADR-9 + `alembic/README`.

### Testing

```bash
task test               # All backend tests (slow — loads translation models)
task test:quick         # Tests not needing model loading (HealthCheck, MCP)

# Single test class or method:
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest -v2
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest.test_translate_to_french -v2
```

Test files: `tests.py` (translation, health, MCP, extraction, providers, voice) and
`tests_memory.py` (the memory system — mock-based, fast; docker-gated integration classes
auto-skip without `task db:up`). `task test:memory` runs the memory suite (pass a class via
`task test:memory -- agentx_ai.tests_memory.SomeTest`). Model-heavy classes
(`TranslationKitTest`) are slow on first run; everything degrades to skips without
Docker/API keys. `DJANGO_SETTINGS_MODULE` defaults inside `manage.py` — bare invocations work.

### Linting, Formatting & Static Analysis

```bash
task lint               # All linters (Python + Client)
task lint:python        # ruff check api/  (lint:python:fix to auto-fix)
task format:python      # ruff format api/
task check:static       # All static analysis (lint + types + build)
task check:types        # All type checkers (python: pyright; client: tsc)
task check:build        # Verify both API and client build
task api:spec:lint      # Lint OpenApi.yaml with Redocly
```

The root `OpenApi.yaml` mirrors the docs-site API reference (`docs-site/src/content/docs/api/endpoints.md`). When endpoints change, update both and run `task api:spec:lint`. When a **memory capability** changes, update `docs-site/src/content/docs/architecture/memory-capabilities.md` (matrix row + section, and the interconnection diagram if a new edge appears) in the same change.

### Build & Release

```bash
task client:build       # Build Tauri app for production
task release:check      # Verify release readiness (clean tree, tests, TS compile, notes-vs-versions)
task models:download    # Pre-download HuggingFace models (NLLB-200, language detection)
```

**Releasing** is one headless action: `.github/workflows/release.yml` (`workflow_dispatch` with a single `version` input) builds the desktop installers (3-platform matrix) **and** publishes the API Docker image (`qrmadness/agentx-api:{version}` + `:latest`), then publishes a single GitHub Release (tag `v{version}`). Human-written notes live in root **`Release-Notes.md`** — its body is injected verbatim into the annotated template; the `<!-- release-version: X.Y.Z -->` marker is asserted against the baked version (workflow + `task release:check`). Version bumps are **bake-only** (not committed back — bump the repo separately via `task versions:sync`).

## API Endpoints

Base URL: `http://localhost:12319/api/`. The **full endpoint table + the `/api/agent/chat/stream`
SSE contract** live in [`Development-Notes.md`](Development-Notes.md); the authoritative spec is
`OpenApi.yaml` + `docs-site/.../api/endpoints.md`. Headline groups: `/health`, `/tools/*`
(translate/detect), `/mcp/*`, `/providers/*`, `/agent/{run,chat,chat/stream,status,profiles,plans,
ambassador/*}`, `/alloy/workflows`, `/prompts/*` (layers/profiles/templates/enhance), `/memory/*`,
`/metrics/usage`, `/jobs/*`, `/config*`, `/logs/*`, `/auth/*`, `/conversations`.

## Environment Configuration

Copy `.env.example` to `.env`. Key vars: `NEO4J_PASSWORD`, `POSTGRES_PASSWORD` (docker-compose + agent_memory), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMBEDDING_PROVIDER` (`local` sentence-transformers, or `openai`). MCP servers in `mcp_servers.json` (see `.example`).

## Important Technical Details

- **Translation models load eagerly** at `TranslationKit` init (first request downloads ~600MB); the **memory system is lazy** (DB connections on first use; pydantic-settings from `.env`).
- **Docker data is bind-mounted** to `./data/` (not Docker volumes); `task db:init` creates the structure.
- **Python managed by uv**, client packages by **bun**; Tauri dev = Vite on port 1420 (HMR 1421).

## Agent Memory Interface

`AgentMemory` (`kit/agent_memory/memory/interface.py`) is the unified API: `store_turn`,
`remember(query, top_k)`, `learn_fact`, `upsert_entity`, `record_tool_usage`, `reflect`, and
goal tracking (`add_goal`/`complete_goal`/`get_active_goals`). Full method semantics + the
extraction, procedural, context-gating, web-research, model-resolution, and attribution
subsystems are in [`Development-Notes.md`](Development-Notes.md).

## Project Status

The Progress Tracker in [`Todo.md`](Todo.md) is authoritative (live: **16.7 Ambassador v2**
planning; **Phase 19 Cloud Operation** queued). Current version: `versions.yaml`; memory
direction: [`Memory-Roadmap.md`](Memory-Roadmap.md).

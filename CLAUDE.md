# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is the **light index**; deep subsystem internals live in [`Development-Notes.md`](Development-Notes.md).

## Documentation Map

Find the right doc before diving in — they're split deliberately so this file stays small.

| Doc | What it is |
|-----|-----------|
| **[`Todo.md`](Todo.md)** | Roadmap **index** — Progress Tracker + a map into [`todo/`](todo/) |
| **[`todo/phases/`](todo/phases/)** | Per-phase work (`completed.md`, `phase-15/16/18-*.md`); **16** holds the live Ambassador planning |
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
- **Version + notes travel with the work.** Any notable/user-facing change must **bump the version and update the release notes in the same commit** — bump `versions.yaml` (patch, via `task versions:sync` to propagate to all manifests), bump the `<!-- release-version: X.Y.Z -->` marker in root `Release-Notes.md` to match, and add the change to the `Release-Notes.md` body (it always describes the *next* release). This is a continuous dev habit, not a release-time step: `task release:check` asserts the marker matches `versions.yaml`. See [Build & Release](#build--release).

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
  Drawers: Settings, Memory, Tools        ├── Reasoning (CoT, ToT, ReAct, Reflection)
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
- `kit/agent_memory/` — memory system, lazy-loaded connections (`interface.py` → `connections.py` → impls); `RecallLayer` = 5 retrieval techniques (hybrid, entity-centric, query expansion, HyDE, self-query). ★
- `mcp/` — MCP client manager, server registry, tool executor, transports (stdio, SSE); configure via `mcp_servers.json`
- `providers/` — abstract `ModelProvider` (LM Studio/Anthropic/OpenAI/OpenRouter/Vercel); `models.yaml` configs + defaults, `pricing.py` cost. Resolution/fallback ★
- `config.py` — `ConfigManager` singleton; persists `data/config.json`, dot-notation access + env-var fallback
- `drafting/` — speculative decoding, multi-stage pipelines, N-best candidates; `drafting_strategies.yaml`
- `reasoning/` — CoT, ToT (BFS/DFS/beam), ReAct, Reflection; `orchestrator.py` picks strategy by task type
- `agent/` — `Agent` orchestrates reasoning + drafting + tools. `TaskPlanner` decomposes; the chat path composes the plan with the **main agent model** via `compose_with_model` (structured JSON; opts out with `{"plan": null}`), gated by `_assess_complexity`. Legacy `plan()` + `SUBTASK`-regex path for non-chat callers. `SessionManager` for conversations.
- `agent/profiles.py` — `ProfileManager` CRUD (`data/agent_profiles.yaml`). Each profile has a Docker-style `agent_id` + `self_channel` (`_self_{agent_id}`). `kind` ∈ `agent`|`ambassador` with **separate defaults**; ambassadors are **excluded from chat** (default/routing/`delegate_to` all filter `kind=='agent'`). `_ensure_ambassador_defaults()` seeds/migrates without ever converting the default agent.
- `agent/tool_output_compressor.py` / `tool_output_chunker.py` — task-aware LLM compression for oversized tool outputs (section detection, JSON-path, semantic search)
- `streaming/trajectory_compression.py` — Focus-style intra-trajectory compression for multi-round tool loops
- `prompts/` — `PromptManager` + durable layered system-prompt stack (`LayerStore`). ★
- `agent/context.py` — per-turn `assemble_turn_context` (verbatim budget + rolling summary + checkpoints/scratchpad). ★
- `agent/ambassador.py` (+ `ambassador_storage.py`) — parallel briefer/relay; sidecar-only, never pollutes the transcript. ★
- `logging_kit/` — centralized logging (queue handler → console/ring-buffer/`/api/logs` + daily encrypted archives), `AGENTX_LOG_*` flags. ★

★ Plus the **memory subsystems** (extraction pipeline, procedural memory, context gating, web-research tools, model resolution & fallback, multi-agent attribution) and the **full API + chat-stream SSE reference** — all in [`Development-Notes.md`](Development-Notes.md).

### Key Client Patterns (`client/src/`)

- 3 primary pages (Start, Dashboard, AgentX) routed via `RootLayout` + `TopBar`; gate pages `AuthPage` (`AGENTX_AUTH_ENABLED`) + `VersionMismatchPage`.
- Chat page (`pages/AgentXPage.tsx`) = left `ConversationSidebar` rail + `ChatPanel` (collapsible/resizable, presentation over `ConversationContext`). Shared list logic in `hooks/useConversationList.ts` + `components/chat/ConversationList.tsx`/`ConversationRow.tsx`, reused by the mobile Conversations drawer. Per-conversation meta (pin/archive/icon/color/group/bulk) in `lib/conversationMeta.ts` (reserves `workspaceId`/`fileRefs`). Destructive actions use `ui/ConfirmDialog` (`useConfirm()`), not native `confirm`.
- Settings, Tools, **Memory** open as full-screen modals (`type:'modal', size:'full'`); Plans/Sources are right-side drawers.
- **Command palette is the primary command surface** — `components/common/CommandPalette.tsx` (thin `cmdk`) over the `hooks/useCommands.tsx` registry; palette + TopBar icons share `lib/surfaces.ts` (`SURFACES.*`) so they can't drift. `⌘K` dispatches `agentx:toggle-command-palette` (RootLayout owns it).
- Multi-server: per-server settings in localStorage (`agentx:servers`/`…:meta`/`activeServer`); `ServerContext` app-wide; `lib/api` typed client (facade over `lib/api/`); `lib/hooks.ts` data hooks on the `useApi<T>` factory. `AgentProfileContext` manages profiles.
- **Agent-profile editor** (`unified-profile-editor/`): hero identity header over a `ControlCard` grid; avatars in `common/AvatarPicker` (`lib/avatars.ts`); signature color from `lib/agentAccent.ts`. Primitives: `ui/SegmentedControl`, `ui/CopyChip`, `common/ControlCard`.
- **Prompt Stack editor** (Settings → Intelligence → "System Prompt"): two-pane block composer over `/api/prompts/layers` — `@dnd-kit` reorder, debounced autosave, reset/diff, live preview via `lib/promptStack.ts::composeStack`. Library snippets insert as custom layers; the enhancer (`/api/prompts/enhance`) rewrites a layer in place.
- Themes (Cosmic dark, warm Light, monochrome Professional) are token-driven `ThemeDefinition`s in `lib/theme.ts` (~80 CSS vars) applied by `ThemeProvider`. Add a theme = new definition + register in `THEMES` + extend `ThemeName`.
- API errors: `ApiError` carries a status-derived `kind`; use `apiErrorMessage(err)`/`toApiError(err)`; surface via `useNotify().notifyError(err)` (toasts); inline errors only for form-field validation.

#### Styling (Tailwind v4 + design tokens)

- **Tailwind v4** via `@tailwindcss/vite`. CSS entry `src/App.css` imports only the `theme` + `utilities` layers — **Preflight is intentionally disabled** so it doesn't clobber `styles/base.css` (imported into `base`; utilities out-rank it, unlayered per-component CSS out-ranks utilities).
- **Design tokens** in `lib/theme.ts`, injected at runtime by `ThemeProvider` as CSS vars; `App.css` bridges them via `@theme inline`. Use **semantic utilities**, not raw palette: `bg-surface-base|raised|overlay|sunken|hover`, `text-fg|fg-secondary|fg-muted|fg-inverse`, `border-line|line-strong`, `text-accent|bg-accent(-secondary|-tertiary)`, feedback `text-error|success|warning|info`. Spacing tokens `--space-*` for hand-written CSS. Brand shadows stay `var(--shadow-md)`.
- **Components**: prefer Tailwind for new/shared UI; keep per-feature CSS for complex panels. Shared primitives in `components/ui/` follow shadcn (CVA + `cn()` in `lib/utils.ts`, exported from `components/ui/index.ts`). Radix enter/exit animations from `tw-animate-css`.

## Development Commands

All commands use [Task](https://taskfile.dev/) (`Taskfile.yml`). Run `task --list-all` for the complete list.

```bash
# Setup & dev
task setup              # First-time: install deps, init DB dirs, verify env
task dev                # Start Docker + API + Client concurrently (full stack)
task dev:api            # API only (assumes Docker running)
task dev:client         # Tauri client only (assumes API running)
task dev:web            # Client in browser mode (port 1420, no Tauri)
task install            # Install all deps (uv sync + bun install)

# Database (Docker) — Neo4j, PostgreSQL, Redis
task db:up / db:down    # Start / stop services (aliases: runners / teardown)
task db:status
task db:init            # Create local data dirs (data/neo4j, data/postgres, data/redis)
task db:init:schemas    # Init schemas: PG via Alembic (upgrade head) + Neo4j/Redis
task db:verify:schemas  # Read-only schema check
task db:migrate         # Apply pending migrations: PG (Alembic) + Neo4j
task db:migrate:pg      # PG only (alembic upgrade head); db:migrate:pg:status to inspect
task db:revision -- "msg"  # New Alembic (PostgreSQL) revision — see alembic/README
task db:shell:postgres  # psql / db:shell:redis (redis-cli) / db:shell:neo4j (cypher-shell)

# Django
task api:run            # Dev server (alias: api:runserver)
task api:migrate / api:makemigrations / api:shell
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

Test files: `tests.py` (TranslationKit, HealthCheck, MCP, Extraction), `tests_memory.py` (80 memory-system tests). Key categories: `TranslationKitTest` (needs HuggingFace models, slow first run), `HealthCheckTest` (auto-skips without Docker), `MCPClientTest`/`MCPServerRegistryTest` (no external deps), `ExtractionPipelineTest` (skips without API keys), `ToolOutputCompressorTest`/`ToolOutputChunkerTest`, `IntentAwareRetrievalTest`/`TrajectoryCompressionTest`/`ContextGateTest`, `AgentSelfMemoryTest`, `FactVerificationPipelineTest`. Memory integration tests skip gracefully without Docker or on embedding-dimension mismatch. `DJANGO_SETTINGS_MODULE` is set automatically by the Taskfile.

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

- **Translation models load eagerly** at `TranslationKit` init. First request downloads ~600MB (NLLB-200) from HuggingFace.
- **Memory system is lazy**: DB connections (Neo4j, PostgreSQL, Redis) created on first use. Config via pydantic-settings from `.env`.
- **Docker data is bind-mounted** to `./data/` (not Docker volumes). Run `task db:init` to create the structure.
- **Tauri dev server** runs Vite on port 1420, HMR on 1421. Window config in `client/src-tauri/tauri.conf.json`.
- **Python managed by uv**, client packages by **bun**. `task dev` uses globally-installed `concurrently` (from `task install`).

## Agent Memory Interface

`AgentMemory` (`kit/agent_memory/memory/interface.py`) is the unified API. **Core**: `store_turn`,
`remember(query, top_k)`, `learn_fact`, `upsert_entity`, `record_tool_usage`, `reflect` (async
consolidation). **Goal tracking**: `add_goal`/`get_goal`/`complete_goal`/`get_active_goals`
(`TaskPlanner.plan()` opens a goal, `Agent.run()` closes it). Full method semantics + the extraction,
procedural, context-gating, web-research, model-resolution, and attribution subsystems are documented
in [`Development-Notes.md`](Development-Notes.md).

## Project Status

Phases 1–14 + 17 complete. Phase 15 (Plan Execution) core complete. **Phase 16 (Multi-Agent) ~72%**
— Agent Alloy v1, attribution, routing, tool isolation, ad-hoc + @-mention delegation, and the
Ambassador (foundation + TTS/STT voice) shipped; **16.7 Ambassador v2** is the live planning. Phase 18
(UX + Memory Tuning) ~98%. Current version: see `versions.yaml`. **Detailed tracking:** [`Todo.md`](Todo.md)
→ [`todo/`](todo/); memory direction in [`Memory-Roadmap.md`](Memory-Roadmap.md).

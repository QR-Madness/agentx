# CLAUDE.md

The **light index** for working in this repo; deep subsystem internals live in [`Development-Notes.md`](Development-Notes.md).

## Documentation Map

Find the right doc before diving in:

| Doc | What it is |
|-----|-----------|
| **[`Todo.md`](Todo.md)** | Roadmap **index** — Progress Tracker + a map into [`todo/`](todo/) |
| **[`todo/phases/`](todo/phases/)** | Per-phase work; **16** = multi-agent/Ambassador (live), **19** = Cloud Operation |
| **[`todo/backlog/`](todo/backlog/)** | Future work by theme (`foundation`, `workspaces`, `memory-recall`, …) |
| **[`Memory-Roadmap.md`](Memory-Roadmap.md)** | Memory-system hardening & experimental roadmap; pairs with the `memory-*` backlog files |
| **[`Development-Notes.md`](Development-Notes.md)** | Deep subsystem internals + the full API/SSE reference — read when working that area |
| **[`Decisions.md`](Decisions.md)** | Load-bearing invariants + locked decisions (the "don't relitigate" list) — read before changing memory/ambassador/streaming internals |
| **[`Repo-Questions.md`](Repo-Questions.md)** | Open hard design questions parked for a deep answer |
| **[`Release-Notes.md`](Release-Notes.md)** | Human-written notes for the *next* release (see the version rule below) |
| `OpenApi.yaml` + `docs-site/.../api/endpoints.md` | Authoritative API contract |

> When you change code, update the matching doc in the same change. **`task docs:check`** guards
> the mechanical half (links, orphans, version drift, notes size; runs inside
> `task release:check`); [`Decisions.md`](Decisions.md) is the judgement half.

## Development Notices

- The client is cross-platform — UI must be highly responsive with comfortable hit-regions.
- Post-v0.20, all changes must be migratable for existing platforms (`versions.yaml` is authoritative).
- **Version + notes travel with the work.** Any notable change bumps both **in the same commit**: `versions.yaml` via `task versions:sync` (propagates manifests + lockfiles), and root `Release-Notes.md` — its `<!-- release-version -->` marker *and* body (always the *next* release). A habit, not a release step; `task release:check` asserts it. See [Build & Release](#build--release).

## Hard-Won Working Rules

Landmines that cost real debugging time:

0. **Adding a setting:** declare it in `settings_registry` + help in
   `settings_help.yaml`, then `task docs:gen:settings`. Undeclared = read-only by
   design (first place to look when a save "does nothing"). Never hand-write a
   `config_update` handler. A new settings *screen* also needs an `ALL_SECTIONS`
   entry + a `sections:` blurb — gated, not optional. ADR-17 ★
1. **Settings overrides:** the memory kit reads settings **live** (zero module snapshots,
   ratchet 0). Temporary overrides use ONE mechanism:
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

AgentX is an AI Agent Platform: a Django REST API (`api/`, port 12319) + a Tauri v2 desktop app (`client/`, React 19/TypeScript/Vite) over Neo4j (graphs), PostgreSQL + pgvector (vectors), and Redis (cache), all via Docker.

## Terminology

| Term | Meaning |
|------|---------|
| **Agent Profile** / **Profile** | The primary configuration entity — defines an agent's identity, behavior, settings; produces "agents" when used. |
| **Global vs Profile Settings** | Global = all agents (servers, API keys, MCP tools); Profile = per-agent (model, temperature, system prompt, reasoning, memory channel) |

The global prompt has no agent name — agent names are injected from the profile during prompt composition.

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

One-liners for orientation; ★ = deep internals in [`Development-Notes.md`](Development-Notes.md).

- `kit/translation.py` — `TranslationKit` (NLLB-200) + `LanguageLexicon` (ISO 639 code bridging)
- `kit/agent_memory/` — memory system, lazy connections (`interface.py` → `connections.py` → impls); `RecallLayer` = 5 retrieval techniques (hybrid, entity-centric, query expansion, HyDE, self-query) + a cross-encoder rerank stage (default-ON). ★
- `kit/shell/` — Agent Shells: **opt-in per-workspace** (`workspaces.allow_shell`, off by default) sandboxed command execution — bubblewrap jail default, Docker-container backend optional. Internals + threat model ★. e2e: `scripts/shell_e2e.py`.
- `kit/workspaces/` — File Workspaces & Document RAG, surfaced as **Projects** (instructions ride every turn; durable conversation membership; `_project_{ws_id}` memory channels; `ws_home` is never a project). Read + write agent tools (partial edits take an `expected_sha256` soft write-lock); roster + internals ★. e2e: `scripts/rag_e2e.py`.
- `mcp/` — MCP client manager, server registry, tool executor, transports, remote OAuth 2.1, registry-search proxy; `mcp_servers.json`; `media_passthrough.py` surfaces returned image/audio blocks as exhibits (capped, untrusted). Client surface: **Connectors & Tools** (internally `toolkit`). ★
- `content_blocks.py` — multi-modal payload vocabulary mirroring MCP/ACP ContentBlocks; the seam shared by providers (`StreamChunk.media`), the MCP executor, and exhibits. Audio in/out rides it (`agent/audio_gen.py` = the audio twin of `image_gen.py`). ★
- `kit/speech.py` — neutral TTS/STT seam; the Ambassador keeps only profile-precedence wrappers, chat consumes directly. **ADR-11**: capabilities live in neutral modules, surfaces consume — enforced by `tests.CapabilitySeamBoundaryTest`. `providers/capabilities.py` = the one warm-once modality probe.
- `providers/` — `ModelProvider` over the five built-ins **+ user-registered OpenAI-compatible endpoints** (`catalog.py`; `egress.py` guards caller-supplied URLs). `models.yaml`, `pricing.py`. Resolution/fallback + the catalog ★ · ADR-15
- `config.py` — `ConfigManager` singleton; persists `data/config.json`, dot-notation access + env-var fallback
- `settings_registry.py` — **the one declaration** behind the config write surface; `config_update`, the settings manifest, and the generated docs reference all derive from it. ★ · ADR-17
- `drafting/` — speculative decoding, multi-stage pipelines, N-best candidates; `drafting_strategies.yaml`
- `reasoning/` — **Thinking Patterns**: chat patterns compiled into the streaming turn (`chat_patterns.py` + `streaming/thinking_exec.py`; `selection.py` = the shared auto brain) + the offline reasoning kit for `/agent/run`. ★
- `agent/` — `Agent` orchestrates reasoning + drafting + tools; `TaskPlanner` decomposes (chat path composes plans with the main agent model ★); `SessionManager` for conversations.
- `agent/profiles.py` — `ProfileManager` CRUD (`data/agent_profiles.yaml`); Docker-style `agent_id` + `self_channel`; seeded default profiles (one-time markers, deletions stick ★). **Rule:** `kind` ∈ `agent`|`ambassador`; ambassadors are **excluded from chat** (default/routing/`delegate_to` filter `kind=='agent'`). ★
- `agent/skills.py` — **Agent Skills**: named instruction packs, progressively disclosed — compact index in the chat prompt, bodies load via `use_skill`; `data/skills.yaml`; per-agent access. UI: Connectors & Tools → Skills. ★
- `alloy/` — **Agent Teams** (user-facing name; internals/routes/config keep `alloy`): Team (workflow) CRUD (`data/workflows.yaml`), `delegate_to` (per-dispatch `effort` tiers → round budgets, `alloy.effort_tiers`) + `AlloyExecutor`; supervisor prompt in workflows, **soft ad-hoc roster block** in normal chats (opt-in `available_for_delegation`; per-conversation `disable_delegation`). ★
- `agent/tool_output_compressor.py` / `tool_output_chunker.py` — task-aware LLM compression for oversized tool outputs
- `streaming/trajectory_compression.py` — Focus-style intra-trajectory compression for multi-round tool loops
- `prompts/` — `PromptManager` + durable layered system-prompt stack (`LayerStore`). ★
- `agent/context.py` — per-turn `assemble_turn_context` (verbatim budget + digest compaction + checkpoints/scratchpad); knobs in Settings → Memory → Conversation Context. ★
- `agent/ambassador.py` (+ `_storage`/`_tools`/`aide_swarm`/`conversation_meta` siblings) — the **Ambassador**: parallel conversational operator (persistent "Inquiry" threads + the **Command Deck**). **Rule:** the tool belt **never executes a write** — reads auto-run; conversation-meta writes and `dispatch_task` are **proposal-only** (client confirm strip), the write side landing only as *your* user turns (relay/dispatch); sidecar-only, never pollutes the transcript. ★
- `logging_kit/` — central logging (queue handler → console/ring-buffer/`/api/logs` + daily encrypted archives), `AGENTX_LOG_*` flags. ★

★ Plus the **memory subsystems** and the **full API + chat-stream SSE reference** — all in [`Development-Notes.md`](Development-Notes.md) (see its Contents list).

### Key Client Patterns (`client/src/`)

The full **client surface map** lives in [`Development-Notes.md`](Development-Notes.md) — read it
before touching a surface. The rules that must not drift:

- 3 primary pages (Start, Dashboard, AgentX) routed via `RootLayout` + `TopBar`; gate pages `AuthPage` (`AGENTX_AUTH_ENABLED`) + `VersionMismatchPage`. TopBar's desktop **surface pills** (Deck, Memory, Projects) add no new `PageId`s — selected state derives from `useModal().isOpen`; pills are palette-only on mobile.
- **Command palette is the primary command surface** — `components/common/CommandPalette.tsx` over the `hooks/useCommands.tsx` registry; palette + TopBar icons share `lib/surfaces.ts` so they can't drift.
- Destructive actions use `ui/ConfirmDialog` (`useConfirm()`), never native `confirm`.
- **Settings sections** build on `components/settings/fields/` + `useSettingsAutosave`; **secrets keep explicit Save**.
- Multi-server: `ServerContext` app-wide; `lib/api` typed client facade; `lib/hooks.ts` data hooks on the `useApi<T>` factory; `AgentProfileContext` for profiles.
- **Add a theme = one entry in `THEMES`** (`lib/theme.ts`) — pickers iterate the registry; a vitest enforces cross-theme token parity; glow tokens use a transparent shadow, never bare `none`.
- API errors: `ApiError` carries a status-derived `kind`; use `apiErrorMessage(err)`/`toApiError(err)`; surface via `useNotify().notifyError(err)` (toasts); inline errors only for form-field validation.
- **Two shells (desktop + web/PWA)** — one React app, gated by compile-time `__IS_TAURI__`. **Rule:** `@tauri-apps/*` is imported **only** under `src/platform/` (`importBoundary.test.ts` fails on a stray import, keeping the web bundle Tauri-free). PWA shell in `src/pwa/`; connection links in `lib/connectionString.ts`. Detail ★ → Client Surface Map.

#### Styling (Tailwind v4 + design tokens)

- **Tailwind v4** via `@tailwindcss/vite`. CSS entry `src/App.css` imports only the `theme` + `utilities` layers — **Preflight is intentionally disabled** (it would clobber `styles/base.css`; utilities out-rank base, unlayered per-component CSS out-ranks utilities).
- **Design tokens** in `lib/theme.ts`, injected at runtime by `ThemeProvider` as CSS vars; `App.css` bridges them via `@theme inline`. Use **semantic utilities**, not raw palette: `bg-surface-base|raised|overlay|sunken|hover`, `text-fg|fg-secondary|fg-muted|fg-inverse`, `border-line|line-strong`, `text-accent|bg-accent(-secondary|-tertiary)`, feedback `text-error|success|warning|info`. Spacing `--space-*` in hand-written CSS; shadows stay `var(--shadow-md)`.
- **Components**: prefer Tailwind for new/shared UI; per-feature CSS for complex panels. Shared primitives in `components/ui/` (shadcn-style); Radix enter/exit animations from `tw-animate-css`. **Form controls must use the field primitives** — `Input`/`Textarea` (`ax-field`), `FieldTrigger` for select-like triggers, `.ax-fieldwrap` for composer wrappers. **Icon-only buttons use `IconButton`**; status pips use `StatusDot`. Never hand-roll `bg-surface-raised border-line` fields or ghost text-button pickers (washed-out). `base.css` resets button background/color — intentional-transparent buttons need an explicit `bg-transparent`. Kit scale via `@theme static` (`text-2xs…4xl`, `tracking-caps`, `rounded-sm..2xl/pill`, `border-line-subtle`, `font-mono`; variants ★ → Styling addenda).

## Development Commands

All commands use [Task](https://taskfile.dev/) (`Taskfile.yml`); `task --list-all` shows everything.

```bash
# Setup & dev
task setup              # First-time: install deps, init DB dirs, verify env
task dev                # Start Docker + API + Client concurrently (alias: d)
task dev:down           # Stop it all: reap dev processes + stop DBs (alias: dd)
task dev:api / dev:client   # One side only (API / Tauri client)
task dev:web            # Client in browser mode (port 1420, no Tauri)

# Database (Docker) — Neo4j, PostgreSQL, Redis
task db:up / db:down    # Start / stop services
task db:init:schemas    # Init schemas: PG via Alembic (upgrade head) + Neo4j/Redis
task db:migrate         # Apply pending migrations: PG (Alembic) + Neo4j
task db:shell:postgres  # psql / db:shell:redis / db:shell:neo4j

# Django
task api:run            # Dev server; also api:migrate / api:makemigrations / api:shell

# Deployment manager (manager/ — owns cluster lifecycle; ADR-10)
task manager:serve      # Web GUI :12320; cluster lifecycle via task cluster:* CLUSTER=x
```

> **Schema migrations:** the memory **PostgreSQL** schema is managed by **Alembic**
> (`alembic/`; baseline frozen, single-head — gated in `task docs:check`). **Neo4j** stays on the
> home-grown runner (`manage.py migrate_schema`). Django's own SQLite ORM is separate
> (`api:migrate`). New PG change = an Alembic revision via `task db:revision -- "msg"` (never edit
> `alembic/baseline.sql`); see [Decisions.md](Decisions.md) ADR-9 + `alembic/README`.

### Testing

```bash
task test               # All backend tests (slow — loads translation models)
task test:quick         # No model loading (HealthCheck, MCP)
task test:sterile       # Sterile config — a test must never read live config.json ★

# Single test class (or .test_method):
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest -v2
```

Test files: `tests.py` (non-memory) and `tests_memory.py` (memory — mock-based, fast;
docker-gated classes auto-skip without `task db:up`). `task test:memory` runs it (accepts a class).
Everything degrades to skips without Docker/API keys; `DJANGO_SETTINGS_MODULE` defaults inside
`manage.py` — bare invocations work.

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

The root `OpenApi.yaml` mirrors the docs-site API reference (`docs-site/.../api/endpoints.md`) — update both when endpoints change and run `task api:spec:lint`. A **memory capability** change also updates `docs-site/.../architecture/memory-capabilities.md` (matrix row + section + diagram) in the same change.

### Build & Release

```bash
task client:build       # Build Tauri app for production
task release:check      # Verify release readiness (clean tree, tests, TS compile, notes-vs-versions)
task models:download    # Pre-download HuggingFace models (NLLB-200, language detection)
```

**Releasing** is one headless action: `.github/workflows/release.yml` (`workflow_dispatch`, single `version` input) builds the 3-platform installers **and** publishes the API Docker image (`qrmadness/agentx-api`), then one GitHub Release (tag `v{version}`). The **`Release-Notes.md`** body is injected verbatim; its `<!-- release-version -->` marker is asserted against the baked version. Version bumps are **bake-only** (bump the repo via `task versions:sync`).

## API Endpoints

Base URL: `http://localhost:12319/api/`. The **full endpoint table + the `/api/agent/chat/stream`
SSE contract** live in [`Development-Notes.md`](Development-Notes.md); the authoritative spec is
`OpenApi.yaml` + `docs-site/.../api/endpoints.md`. Headline groups: `/health`, `/tools/*`,
`/mcp/*`, `/providers/*`, `/agent/*`, `/alloy/workflows`, `/prompts/*`, `/memory/*`,
`/metrics/usage`, `/jobs/*`, `/config*`, `/logs/*`, `/auth/*`, `/conversations`.

## Environment Configuration

Copy `.env.example` to `.env`. Key vars: `NEO4J_PASSWORD`, `POSTGRES_PASSWORD`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMBEDDING_PROVIDER` (`local` or `openai`). MCP servers in `mcp_servers.json` (see `.example`).

## Important Technical Details

- **Translation models load eagerly** at `TranslationKit` init (~600MB on first request); the **memory system is lazy** (DB connections on first use; pydantic-settings from `.env`).
- **Docker data is bind-mounted** to `./data/` (not Docker volumes); `task db:init` creates the structure.
- **Python managed by uv**, client packages by **bun**; Tauri dev = Vite on port 1420 (HMR 1421).

## Agent Memory Interface

`AgentMemory` (`kit/agent_memory/memory/interface.py`) is the unified API: `store_turn`,
`remember(query, top_k)`, `learn_fact`, `upsert_entity`, `record_tool_usage`, `reflect`, and
goal tracking. Full method semantics + subsystem internals: [`Development-Notes.md`](Development-Notes.md).

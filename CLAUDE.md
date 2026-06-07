# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

- `kit/translation.py` — `TranslationKit` (NLLB-200) and `LanguageLexicon` (ISO 639 bridging between Level I detection codes and Level II translation codes)
- `kit/agent_memory/` — Memory system with lazy-loaded connections (`interface.py` → `connections.py` → implementations). `RecallLayer` provides 5 retrieval techniques (hybrid, entity-centric, query expansion, HyDE, self-query)
- `mcp/` — MCP client manager, server registry, tool executor, transports (stdio, SSE). Configure via `mcp_servers.json`
- `providers/` — Abstract `ModelProvider` with LM Studio, Anthropic, OpenAI, OpenRouter, Vercel AI Gateway impls. Provider configs + default models in `models.yaml` (capabilities fetched dynamically); cost estimation in `pricing.py`
- `config.py` — `ConfigManager` singleton for runtime settings. Persists to `data/config.json` with dot-notation access and env var fallback
- `drafting/` — Speculative decoding, multi-stage pipelines, N-best candidate generation. Strategies in `drafting_strategies.yaml`
- `reasoning/` — CoT, ToT (BFS/DFS/beam), ReAct, Reflection. `orchestrator.py` selects strategy by task type
- `agent/` — `Agent` orchestrates reasoning + drafting + tools. `TaskPlanner` decomposes (with goal tracking); the chat path composes the plan with the **main agent model** via `compose_with_model` (full turn context + structured JSON; model opts out with `{"plan": null}`), gated by a cheap `_assess_complexity` heuristic. Legacy `plan()` + `SUBTASK`-regex path remains for non-chat callers. `SessionManager` for conversations.
- `agent/profiles.py` — `ProfileManager` for agent profile CRUD (stored in `data/agent_profiles.yaml`). Each profile has a Docker-style `agent_id` (e.g. "bold-cosmic-falcon") and `self_channel` (`_self_{agent_id}`). A profile's **`kind`** is `agent` or `ambassador` with **separate defaults** (`is_default`/`get_default_profile`; `is_default_ambassador`/`get_default_ambassador`). Ambassadors are **excluded from chat** — `get_default_profile`, routing lookups (`get_profile_by_agent_id`/`get_profile_by_name`), and `delegate_to` all filter to `kind=='agent'`. `_ensure_ambassador_defaults()` seeds a default ambassador and migrates legacy ones **without ever converting the default agent**.
- `agent/tool_output_compressor.py`, `agent/tool_output_chunker.py` — LLM task-aware compression for oversized tool outputs; section detection, JSON path queries, semantic search over stored outputs
- `streaming/trajectory_compression.py` — Focus-style intra-trajectory compression for multi-round tool loops
- `logging_kit/` — centralized logging. `configure_logging()` (called from `settings.py` with `LOGGING_CONFIG=None`) installs a root `QueueHandler` → `QueueListener` feeding console (rich color + category badge + run tag via `handler.py`/`highlighters.py`/`categories.py`), an in-memory `RingBufferHandler` (backs `/api/logs`), and a gzip `archive.py` rotating file. `context.py` stamps a per-turn `run_id` (reuses `streaming/status.py::current_run_id`); `redaction.py` scrubs secrets at capture; `llm_cards.py` renders compact/full LLM request logs; `banner.py` is the startup banner. All behavior is governed by `AGENTX_LOG_*` flags (`flags.py`; decorations on by default, off → historical plain output). The client mirrors categories in `lib/logCategories.ts`.

#### Prompts (`prompts/`)

`PromptManager` singleton composes prompts. The **conversational global system prompt** comes from a durable **layered stack** (`layers.py`: `LayerStore` over `ConfigManager` key `prompts.layers`; `BUILTIN_LAYERS` ship a versioned `default` overlaid by the user's `override` — `effective = override ?? default`; `update_available` diffs a released default bump against `base_version`). `compose_prompt` sources global content from `LayerStore.compose()` (edits persist across restarts) and only attaches a prompt-profile's sections for an explicit non-`is_default` selection (default profile's sections fold into the stack — no double-injection). Legacy `/prompts/global*` are back-compat shims. This stack governs **only** the conversational persona — internal *feature* prompts (reasoning, planner `decompose`, extraction, compression) come from the separate `SystemPromptLoader` (`loader.py`; optional `data/system_prompts.yaml`). **Placeholders**: `placeholders.py::substitute_placeholders` replaces a whitelist (`{agent_name}`/`{date}`/`{time}`) at the end of `compose_system_prompt`; client mirrors it in `lib/promptPlaceholders.ts`.

#### Conversation Context (`agent/context.py`)

Per-turn context assembled by `ContextManager.assemble_turn_context`: keep the SYSTEM preamble + as much **recent verbatim transcript** as fits `context.verbatim_budget_ratio` (0.7) of the model's real window; drop oldest overflow (covered by the rolling summary). In-memory `SessionManager` is **rehydrated** from durable `conversation_logs` on a cold session (`conversation_history.py::hydrate_session_from_history`). The rolling summary is **context-window-triggered** (`maybe_update_summary`, token-based) and **persisted** in Redis (`conversation_summary_storage.py`). Checkpoints/scratchpad (`checkpoint_storage.py`) are Redis-keyed per conversation, re-injected each turn; eviction is anchor-preserving (keeps the first), and the `checkpoint` tool has a `replace` mode. After a turn, `views.py::generate_sse` persists via pure builders in `streaming/persistence.py` (user → tool → steer → assistant Turns); a hard Stop (`GeneratorExit`) persists the **partial** turn (`metadata.interrupted`); folded steers persist as `user` turns (`metadata.steered`). **Thinking/CoT is process, not result**: it streams live (shown collapsed in the client) but is **never persisted** to `conversation_logs`. Old turns persisted before this still restore stored `metadata.thinking`.

#### Ambassador (`agent/ambassador.py` + `ambassador_storage.py`, Phase 16.6)

A dedicated agent running *parallel* to a conversation that briefs the user on a turn **without polluting the main transcript** (the load-bearing invariant). `AmbassadorService.brief_turn` resolves the default ambassador (`get_default_ambassador()`; an `AgentProfile` with `kind='ambassador'`, hidden from chat). The profile's `system_prompt` is the personality voice; functional personas come from `AmbassadorConfig.{briefing,qa,draft}_persona` overrides ?? code defaults. Grounds **read-only** via `load_recent_turns`, runs `resolve_with_fallback` + token-streams via `provider.stream` (never raises; cancel settles sidecar to `cancelled`), emits namespaced `ambassador_*` SSE. The persona **speaks TO the reader** (second person, names the agent). The briefing grounds on the turn's substance via client-gathered `artifacts` (tool calls / sources / exhibits, `lib/ambassadorTurn.ts::gatherTurnContext`). Also answers **free-form questions** (`answer_question`, shared `_stream_and_settle` core, `qa:` key family) and does **outbound relay** (`draft_relay_message`: shapes a rough intent into a first-person message the user reviews and relays into the conversation as a real **user** turn via `ConversationContext.relayToConversation` — the ambassador never speaks into the transcript as itself). Output persists ONLY to the Redis **sidecar** (`ambassador:` prefix) — never `conversation_logs`/`conv_summary:`. Runs ride detached-run infra via `start_chat_run(indexed=False)`.

### Key Client Patterns (`client/src/`)

- 3 primary pages: Start, Dashboard, AgentX — routed via `RootLayout` + `TopBar`; plus gate pages `AuthPage` (when `AGENTX_AUTH_ENABLED`) and `VersionMismatchPage`
- Conversations: the chat page (`pages/AgentXPage.tsx`) is a **left `ConversationSidebar` rail + `ChatPanel`** (the old TopBar tab strip was removed). The rail is collapsible (`agentx:conv-sidebar-collapsed`) and **resizable** (`agentx:conv-sidebar-width`), presentation over `ConversationContext` (the tab/session/streaming model is unchanged — "open conversations" are the top rows). Logic is shared via **`hooks/useConversationList.ts`** (normalizes tabs + server conversations into one `ConversationItem[]`, partitions by meta, owns selection/bulk) + **`components/chat/ConversationList.tsx`** + **`ConversationRow.tsx`** (the `⋯` actions menu), reused by the mobile **Conversations drawer** (`SURFACES.conversations`). Per-conversation management (pin/archive/icon/color/group + multi-select bulk) is stored client-side per-server in **`lib/conversationMeta.ts`** (an extensible store that folded in the old title-override shim and reserves `workspaceId`/`fileRefs` for future workspace/file linking; reactive via `useConversationMeta`/`useSyncExternalStore`). Destructive actions use the themed **`ui/ConfirmDialog`** (`useConfirm()` + `ConfirmProvider` at the app root) — the replacement for native `window.confirm`.
- Settings, Tools, **Memory** open as full-screen modals (`type:'modal', size:'full'`); Plans/Sources remain right-side drawers
- **Command palette is the primary command surface** (the TopBar "Workspace" overflow was removed). `components/common/CommandPalette.tsx` is a thin `cmdk` renderer over the **`hooks/useCommands.tsx`** registry (grouped: Navigation/Conversation/Workspace/Theme/Account; theme switching is the only dynamic group; `lib/recentCommands.ts` MRU). Both the palette and the TopBar's live icons open drawers/modals through one shared **`lib/surfaces.ts`** (`SURFACES.*` `openModal` descriptors) so the two can't drift. The toolbar's `⌘K` is a labeled "Search…" pill that dispatches `agentx:toggle-command-palette` (RootLayout owns the global key listener + open state)
- Multi-server support: per-server settings in localStorage (`agentx:servers`, `agentx:server:{id}:meta`, `agentx:activeServer`). `ServerContext` provides app-wide server state; `lib/api` is the typed API client (facade over domain modules in `lib/api/`); `lib/hooks.ts` has React data hooks built on the `useApi<T>` factory
- `AgentProfileContext` manages agent profiles
- **Agent-profile editor** (`unified-profile-editor/`): control-center with a hero identity header over a `ControlCard` grid per tab. Avatars in `common/AvatarPicker` (searchable modal over `lib/avatars.ts`). Deterministic signature color from `lib/agentAccent.ts::agentAccent(agent_id)`. Shared primitives: `ui/SegmentedControl`, `ui/CopyChip`, `common/ControlCard`. Field logic + `useProfileEditorState` (autosave) unchanged — redesign is presentation only.
- **Prompt Stack editor** (Settings → Intelligence → "System Prompt", `unified-settings/sections/SystemPromptSection.tsx` + `prompt-stack/{LayerCard,ComposedPreview,LayerDiffModal}`): two-pane block composer over `/api/prompts/layers` — `@dnd-kit` drag-reorder, debounced-autosave inline edit, reset/diff for built-ins, live preview via `lib/promptStack.ts::composeStack` (client mirror of the backend join). Prompt Library snippets insert as custom layers; the prompt **enhancer** (`/api/prompts/enhance`) rewrites a layer in place (with undo).
- Themes (Cosmic dark, warm Light, monochrome Professional) are token-driven: each a `ThemeDefinition` in `lib/theme.ts` (~80 CSS vars) applied to `:root` by `ThemeProvider`. Add a theme = new `ThemeDefinition` + register in `THEMES` + extend `ThemeName`. Glassmorphism + Lucide-react icons.
- API errors: `ApiError` carries a status-derived `kind`; use `apiErrorMessage(err)`/`toApiError(err)`. Surface failures with `useNotify().notifyError(err)` (toasts via `contexts/NotificationContext` + `ui/Toaster`); keep inline errors for form-field validation.

#### Styling (Tailwind v4 + design tokens)

- **Tailwind v4** via `@tailwindcss/vite`. CSS entry `src/App.css` imports only the `theme` + `utilities` layers — **Preflight is intentionally disabled** so it doesn't clobber `styles/base.css` (imported into `base`; utilities out-rank it, unlayered per-component CSS out-ranks utilities).
- **Design tokens** in `lib/theme.ts`, injected at runtime by `ThemeProvider` as CSS vars; `App.css` bridges them into Tailwind via `@theme inline`. Use **semantic utilities**, not raw palette: `bg-surface-base|raised|overlay|sunken|hover`, `text-fg|fg-secondary|fg-muted|fg-inverse`, `border-line|line-strong`, `text-accent|bg-accent(-secondary|-tertiary)`, feedback `text-error|success|warning|info`. Spacing tokens `--space-*` for hand-written CSS. Brand shadows stay `var(--shadow-md)` (not bridged).
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
task db:init:schemas    # Init memory schemas (Neo4j indexes, PG tables)
task db:verify:schemas  # Read-only schema check
task db:shell:postgres  # psql / db:shell:redis (redis-cli) / db:shell:neo4j (cypher-shell)

# Django
task api:run            # Dev server (alias: api:runserver)
task api:migrate / api:makemigrations / api:shell
```

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

Base URL: `http://localhost:12319/api/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (add `?include_memory=true` for DB status) |
| `/api/tools/language-detect-20` | GET/POST | Detect language of text |
| `/api/tools/translate` | POST | Translate (`{"text": "...", "targetLanguage": "fra_Latn"}`) |
| `/api/mcp/servers` `/api/mcp/tools` `/api/mcp/resources` | GET | List MCP servers / tools / resources (filter `?server=`) |
| `/api/mcp/connect` `/api/mcp/disconnect` | POST | Connect/disconnect (`{"server": "name"}` or `{"all": true}`) |
| `/api/providers` `/api/providers/models` `/api/providers/health` | GET | List providers / models (`?provider=`) / health |
| `/api/agent/run` | POST | Execute a task with the agent |
| `/api/agent/chat` | POST | Conversational interaction with session |
| `/api/agent/chat/stream` | POST | Streaming chat via SSE (see below) |
| `/api/agent/chat/stream/attach` | GET | Re-attach to a detached run (`?run_id=`): replays buffered events + follows live; `run_missing` if buffer expired |
| `/api/agent/chat/runs` | GET | List the caller's detached chat runs (newest first) for recovery surfaces |
| `/api/agent/chat/runs/{run_id}/cancel` | POST | Cooperatively cancel a detached chat run |
| `/api/agent/chat/runs/{run_id}/steer` | POST | Live-steer a running turn (`{message, mode?}`): queued and folded in at the next safe boundary as a fresh user turn; owner-only; echoes a `steer` event; persisted as a `user` turn with `metadata.steered` |
| `/api/agent/ambassador/brief-turn` | POST | Start a parallel briefing of one turn (`{conversation_id, message_id, assistant_text, user_text?, agent_name?, artifacts?}`) → `{run_id}`. Detached + `indexed=False`; writes only to the `ambassador:` sidecar |
| `/api/agent/ambassador/ask` | POST | Free-form question (`{conversation_id, qa_id, question, agent_name?, artifacts?}`) → `{run_id, qa_id}`; persists under `qa:` |
| `/api/agent/ambassador/draft` | POST | Outbound-relay draft (`{conversation_id, intent, agent_name?, artifacts?}`) → `{draft}`; client relays as a real user turn |
| `/api/agent/ambassador/stream` | GET | Tail a briefing/Q&A run (`?run_id=`): `ambassador_start`/`_chunk`/`_done`/`_error` SSE |
| `/api/agent/ambassador/{conversation_id}` | GET | Replay persisted briefings + Q&A (`{briefings, qa}`) |
| `/api/agent/status` | GET | Current agent status |
| `/api/prompts/profiles` `/api/prompts/profiles/{id}` | GET | List prompt profiles / detail with composed preview |
| `/api/prompts/global` `/api/prompts/global/update` | GET/POST | Back-compat shims over the layer stack |
| `/api/prompts/layers` | GET/POST | List the stack (`{layers, composed}`) or create a custom layer (`{title, content?}`) |
| `/api/prompts/layers/reorder` | POST | Reorder (`{order: [id, …]}`) |
| `/api/prompts/layers/{id}` | PATCH/DELETE | Update (`{content?, title?, enabled?}`, content sets override) or delete a custom layer |
| `/api/prompts/layers/{id}/reset` `/acknowledge` | POST | Reset a built-in's override / mark a bumped default seen |
| `/api/prompts/sections` `/compose` `/mcp-tools` | GET | List sections / preview composed prompt / auto MCP-tools prompt |
| `/api/memory/channels` `/entities` `/entities/graph` | GET | List channels / entities (`?channel=`,`?type=`) / entity graph |
| `/api/memory/facts` | GET | List facts (`?channel=`,`?entity_id=`); each carries `entities[]` for its ABOUT'd entities |
| `/api/memory/facts/{id}/entities` | POST/DELETE | Link/unlink an entity (`{entity_id}`, ABOUT edge) |
| `/api/memory/strategies` `/procedures` `/stats` | GET | Procedural strategies / distilled procedures / stats |
| `/api/memory/checkpoints` | GET/DELETE | List/clear a conversation's checkpoints (`?conversation_id=`) |
| `/api/memory/user-history` | POST | Browse the user's past turns + top facts (`{topic?, limit?, channel?}`) |
| `/api/memory/recall-settings` `/settings` | GET/POST | Get/update recall-layer / memory settings |
| `/api/memory/consolidate` `/reset` | POST | Trigger consolidation / reset (with confirmation) |
| `/api/memory/export` `/import` | POST | Round-trippable JSON export / idempotent import (`{data, mode: merge\|replace, channel?}`); also `task memory:export`/`import` |
| `/api/metrics/usage` | GET | Aggregated token/cost/latency from `conversation_logs` (`?days=` 1–90, default 14) |
| `/api/jobs` `/jobs/{id}` | GET | List background jobs / detail |
| `/api/jobs/clear-stuck` `/jobs/{id}/run` `/jobs/{id}/toggle` | POST | Clear stuck / run / enable-disable |
| `/api/config` `/config/update` `/config/context-limits` | GET/POST/GET | Runtime config (secrets redacted) / update (`{"key": "dot.path", "value": ...}`) / per-model limits |
| `/api/logs` `/logs/stream` `/logs/categories` | GET | Ring-buffer log records (filters `?level=&category=&run_id=&search=&since=&limit=`) / SSE live tail / category registry. Gated by `AGENTX_LOG_API_ENABLED`; auth-gated when `AGENTX_AUTH_ENABLED` |
| `/api/logs/archive` `/logs/archive/{name}` | GET | List / download compressed archive segments (`data/logs/*.gz`) |

Additional endpoint groups (added since v0.18 — see `urls.py` for the full set):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/{status,setup,login,logout,session,change-password}` | GET/POST | Optional session auth (Phase 17, gated by `AGENTX_AUTH_ENABLED`) |
| `/api/agent/profiles`, `/api/agent/profiles/{id}` | GET/PATCH/DELETE | Profile CRUD; `.../set-default` (agent), `.../set-default-ambassador`. Profiles carry `kind` (`agent`\|`ambassador`) |
| `/api/alloy/workflows`, `/api/alloy/workflows/{id}` | GET/POST/PATCH/DELETE | Multi-agent (Agent Alloy) workflow CRUD |
| `/api/chat/background`, `/api/chat/background/{job_id}` | POST/GET | Queue + poll background conversations |
| `/api/tool-outputs`, `/api/tool-outputs/{key}` | GET/DELETE | Stored tool-output retrieval |
| `/api/prompts/templates*`, `/api/prompts/enhance` | GET/POST/PUT/DELETE | Prompt template CRUD + LLM prompt enhancer |
| `/api/conversations`, `/api/conversations/{id}/messages` | GET | Conversation history |
| `/api/agent/plans/cancel` | POST | Cancel active plan execution |
| `/api/agent/plans/{plan_id}/status` | GET | Read Redis plan state (`?session_id=`); `resumable` flags an `active`/`interrupted` plan with work left; `ttl_seconds` = resumable lifetime. On load the client auto-appends a `choice` exhibit (`exh_resume_{plan_id}`) nudging the model to continue (a normal turn, not a `PlanExecutor` re-run) |
| `/api/agent/plans/{plan_id}/resume` | POST | Resume an interrupted plan (`{session_id, agent_profile_id?, model?}`): rebuilds from Redis and streams only not-yet-terminal subtasks (SSE, first event `plan_resumed`) as a detached run. Single-agent only; 404 when not resumable |

### `/api/agent/chat/stream` (SSE)

The main streaming chat path. Runs **detached server-side** (survives client disconnect); first event `run_started` carries `run_id`. Events: `run_started, start, chunk, status, steer, tool_call, tool_result, exhibit, done, error, close`.

- **Routing**: optional `target_agent_id` routes to a specific agent by `agent_id` (priority `workflow_id > target_agent_id > agent_profile_id > default`). An inline `@agent-id`/`@name` in the message overrides the selection (suppressed in workflows); the client composer offers `@`-autocomplete. With `alloy.allow_adhoc_delegation`, the agent gets a `delegate_to` tool (depth-limited, no self-delegation; emits `delegation_*` events).
- **Exhibits**: the internal `present_exhibit` tool emits a typed `exhibit` event (declarative Gallery→Exhibit→Element tree) instead of tool cards. Element types: `mermaid`, `choice` (clicking submits as the next user turn), `table` (sortable/scrollable/expand-to-modal), `citation` (active foldable sources vs passive links). A successful `web_search` auto-emits a passive `citation` exhibit (deduped by URL; id `exh_src_<tool_call_id>`; toggle `citations.auto_capture_web_search`).
- **Status events** (`{phase, label, detail?, group?, progress?}`): a coarse per-phase activity feed (`recalling`/`composing`/`thinking`/`running_tool`/`reading`). Rides the run's Redis event bus (not generator yields) via an ambient `emit_status()` (`streaming/status.py`, run resolved from a `ContextVar`), so it replays on re-attach.
- **Steer**: a live `steer` event carries a user message folded into the running turn.

**Plan termination & cancellation.** `PlanExecutor` wraps its subtask loop in `try/finally`: a hard Stop (`GeneratorExit`) mid-subtask resets the in-flight subtask to `pending` and marks the plan **`interrupted`** (resumable) instead of stuck `running`; cooperative cancel marks `cancelled` + `clear_cancel`s the flag. The Stop handler in `views.py` persists the interrupted plan card faithfully. Cancellation is **cooperative** (GeneratorExit lands at a `yield`), made prompt by capping tool wall-clock (`web_search` bounded by `search.timeout`, default 15s). Tool execution stays **synchronous** (off-thread/`asyncio` deadlocked Stop).

## Environment Configuration

Copy `.env.example` to `.env`. Key vars: `NEO4J_PASSWORD`, `POSTGRES_PASSWORD` (docker-compose + agent_memory), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `EMBEDDING_PROVIDER` (`local` sentence-transformers, or `openai`). MCP servers in `mcp_servers.json` (see `.example`).

## Important Technical Details

- **Translation models load eagerly** at `TranslationKit` init. First request downloads ~600MB (NLLB-200) from HuggingFace.
- **Memory system is lazy**: DB connections (Neo4j, PostgreSQL, Redis) created on first use. Config via pydantic-settings from `.env`.
- **Docker data is bind-mounted** to `./data/` (not Docker volumes). Run `task db:init` to create the structure.
- **Tauri dev server** runs Vite on port 1420, HMR on 1421. Window config in `client/src-tauri/tauri.conf.json`.
- **Python managed by uv**, client packages by **bun**. `task dev` uses globally-installed `concurrently` (from `task install`).

## Agent Memory Interface

`AgentMemory` (`kit/agent_memory/memory/interface.py`) is the unified API.

**Core**: `store_turn(turn)` (episodic + working), `remember(query, top_k)` (turns/facts/entities/strategies), `learn_fact(claim, source, confidence)`, `upsert_entity(entity)`, `record_tool_usage(...)`, `reflect(outcome)` (async consolidation).

**Goal tracking**: `add_goal`, `get_goal`, `complete_goal(goal_id, status, result)` (`completed`/`abandoned`/`blocked`), `get_active_goals`. `TaskPlanner.plan()` accepts an optional `memory` param: creates a `Goal` for the main task, stores it on `TaskPlan.goal_id`; `Agent.run()` calls `complete_goal()` on success/failure.

### Extraction Pipeline (`extraction/service.py`)

LLM-based extraction. `check_relevance_and_extract(text)` combines relevance check + extraction in one call (~75% fewer calls); `check_relevance_and_extract_assistant(text)` does self-knowledge extraction (certainty: definitive/analytical/speculative). Both accept a `roster` (`[{agent_id, name}]`) + `addressed_agent_id` so facts attribute to a *specific* agent: the LLM names the agent in prose (`subject_agent`), `_resolve_agent_attribution` resolves name → `agent_id` (durable truth, stored transiently as `subject_agent_id`); unknown names demote to `third_party` (never fabricate an agent_id). `check_contradictions(claim, facts)` is three-layer fact verification: hash gate → semantic duplicate → entity-scoped candidates → LLM adjudication. Default extraction model `nvidia/nemotron-3-nano` (`combined_extraction_model` setting). Consolidation jobs (`consolidation/jobs.py`) call extractors every 15 min. Degrades to empty results with no provider.

### Procedural Memory (encode → distill → reflex-core)

Learns the "how we work here" **delta** (not baseline behavior a capable model already does). Three loops:
- **Encode** (Slice 0, every turn, cheap): stages `procedure_candidates` PG rows — `correction` (from persisted steers) + `explicit_rule` (`procedural.detect_explicit_rule`, heuristic). `status='pending'`.
- **Distill** (Slice 1, consolidation): the async `distill_procedures` job reads pending candidates, groups by derived scope (corrections with `agent_id` → `_self_{agent_id}`; explicit rules stay on their channel), runs `ExtractionService.distill_procedure` (baseline-deviation filter), and **strengthens** a cosine-similar existing `Procedure` (`procedural_dedupe_threshold`) instead of duplicating. Writes via `learn_procedure`/`reinforce_procedure` on Neo4j `Procedure` nodes (`trigger`/`body`/`rationale`/`scope`/`strength`/`evidence_refs`; `procedure_embeddings` vector index).
- **Activate — reflex core** (Slice 1): `ProceduralMemory.get_reflex_procedures` (top-`strength` over recall channels, *maintained not searched*) attached at `interface.remember()` and rendered by `MemoryBundle.to_context_string` ("Learned Procedures"). Gated by `reflex_core_enabled`/`reflex_core_limit`.
- **Worker**: `ConsolidationWorker.run()` awaits coroutine jobs; `task dev` runs it; manual `task memory:distill-procedures`. Inspect via `GET /api/memory/procedures` + `/api/memory/stats`.

### Context Gating

Large tool outputs (>12K chars) compressed + stored for retrieval: `ToolOutputCompressor` (task-aware LLM compression with structure indexing), `tool_output_chunker.py` (section detection, JSON path, semantic search). Internal MCP tools: `read_stored_output`, `tool_output_query`, `tool_output_section`, `tool_output_path`. Trajectory compression (Focus-style) at 75% context threshold. Retrieval-tool bypass prevents re-storage loops.

### Web Research Tools (`mcp/internal_tools.py`)

**Capability-aware advertisement**: backends differ (Tavily = research suite via `tavily-python`: search/extract/map/crawl/research; Brave = search-only via httpx). A pre-check (`resolve_active_search_backend` + `build_tool_schema`/`build_tool_description`) inventories the active backend at tool-listing time and advertises only its real tools + knobs. `web_extract`/`web_map`/`web_crawl`/`web_research` advertised only when Tavily is active but stay registered (self-guard on stale call). `SEARCH_CAPABILITIES` is the single source of truth. `web_search` keeps results compact (under oversize threshold, parseable for auto-capture); `web_crawl`/`web_research` ride the stored-output handler; `web_research` is agentic/slow (~120s timeout + `web_research.enabled`). Config: `search.backend`, `search.{tavily,brave}_api_key`, `search.fallback_enabled`. Successful searches auto-capture sources as a passive `citation` exhibit (shared `citation_exhibit_from_web_search` helper also backs the conversation **Bibliography**: `lib/bibliography.ts` → `SourcesPanel`).

### Model Resolution & Fallback (`providers/registry.py`)

Every LLM feature resolves `provider:model` through the registry. `resolve_with_fallback(model, *, preferred_fallback=)` (sync) and `complete_with_fallback(...)` (async, execution-retry) make a feature **never hard-fail the turn**: an unconfigured/unreachable provider falls back to the caller's active agent-profile model (`preferred_fallback`) → global default (`preferences.default_model` / `models.defaults.chat`) → first healthy provider. Kill-switch `models.fallback_enabled`; best-effort health cache (fed by `health_check`). The **main chat path stays strict** (`get_provider_for_model`) — the agent's chosen model is the floor. **Memory stages inherit**: each stage resolves `explicit override → consolidation.feature_default_model (bulk) → default chat` (`_resolve_stage_model`); stage defaults stay `lmstudio:…` so local users stay cheap and cloud-only users auto-fall-back.

### Agent Identity & Multi-Agent Attribution (Phase 16)

Each profile has a Docker-style `agent_id` (auto-generated, immutable, adj-adj-noun, ~83K combos). `self_channel` = `_self_{agent_id}`; recall searches `[active_channel, _self_{agent_id}, _global]`. Attribution is **agent-specific** (`agent_id` is the durable key, display name is prose only):
- **Agents are first-class entities**: consolidation `_ensure_agent_entities` upserts `Entity(type="Agent")` per agent in `_global`, id `agent:{user_id}:{agent_id}`, canonical key `properties.agent_id`, name + prior names as `aliases`. Facts link via normal `[:ABOUT]`.
- **Name stamping**: assistant turns carry the display name in `Turn.metadata["agent_name"]` (set by writers in `views.py`/`core.py`/`alloy/executor.py`); `episodic.store_turn` stamps `Turn`/`AgentParticipant` nodes so the kit builds a roster (`get_conversation_roster`) without importing `ProfileManager`.
- **Per-agent routing**: `_resolve_subject_channel` honors `subject_agent_id` → `_self_{that_agent}`; assistant self-extraction routes each turn by its own producing `agent_id`.
- **Rename-safety**: `update_profile` propagates a name change to the Agent entity's `aliases`; `dedupe_entities` never merges `Agent` nodes (keyed by `agent_id`).
- **Backfill / debug**: `task memory:backfill-agent-attribution[:apply]` rewrites legacy generic self-facts deterministically; `task memory:debug-attribution -- --scenario <directive|cross-agent|mixed> --agents "..."` drives a scripted multi-agent conversation through real consolidation (non-destructive by default; `--isolate` for a sterile read).

## Project Status

Phases 1-14 and 17 (Server Management) complete. Phase 15 (Plan Execution) ~80%. Phase 16 (Multi-Agent) ~45% — Agent Alloy v1, 16.1 attribution, 16.2 routing, 16.3 per-agent tool isolation, 16.4 ad-hoc delegation, 16.5 @-mention routing + `AgentParticipant` nodes, multi-agent attribution, and 16.6 Ambassador foundation shipped. Phase 18 (UX + Memory Tuning) in progress. Current version: see `versions.yaml`. See `Todo.md` for detailed tracking.

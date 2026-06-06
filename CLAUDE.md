# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Notices

- The client is cross-platform, thus UI should be highly responsive and use comfortable hit-regions.
- Following API & Client v.0.20 (see versions.yaml for the authoritative version), all changes should be migratable for existing platforms.
- **Version + notes travel with the work.** Any notable/user-facing change must **bump the version and update the release notes in the same commit** — bump `versions.yaml` (patch, via `task versions:sync` to propagate to all manifests), bump the `<!-- release-version: X.Y.Z -->` marker in root `Release-Notes.md` to match, and add the change to the `Release-Notes.md` body (it always describes the *next* release). This is a continuous dev habit, not a release-time step: `task release:check` asserts the marker matches `versions.yaml`, so the repo stays release-ready at all times. See [Build & Release](#build--release).


## Project Overview

AgentX is an AI Agent Platform combining:
- **Backend**: Django REST API (`api/`) on port 12319 — translation, agent memory, MCP client, model providers, drafting, reasoning
- **Frontend**: Tauri v2 desktop app (`client/`) with React 19, TypeScript, Vite
- **Data Layer**: Neo4j (graphs), PostgreSQL + pgvector (vectors), Redis (cache) — all via Docker

## Terminology

| Term | Meaning |
|------|---------|
| **Agent Profile** / **Profile** | Configuration that defines an agent's identity, behavior, and settings. These are the primary configuration entity — they produce "agents" when used. |
| **Global Settings** | Settings that apply across all agents (e.g., server connections, API keys, MCP tools) |
| **Profile Settings** | Per-agent settings (e.g., model, temperature, system prompt, reasoning strategy, memory channel) |

Agent profiles configure: name, avatar, default model, temperature, system prompt, reasoning strategy, memory enable/channel, and tool enable. The global prompt has no agent name — agent names are injected from the profile during prompt composition.

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

- `kit/translation.py` — `TranslationKit` (NLLB-200 translation) and `LanguageLexicon` (ISO 639 code bridging between Level I detection codes and Level II translation codes)
- `kit/agent_memory/` — Full memory system with lazy-loaded connections (`interface.py` → `connections.py` → memory implementations). `RecallLayer` provides 5 retrieval techniques (hybrid, entity-centric, query expansion, HyDE, self-query)
- `mcp/` — MCP client manager, server registry, tool executor, transports (stdio, SSE). Configure via `mcp_servers.json`
- `providers/` — Abstract `ModelProvider` with LM Studio, Anthropic, OpenAI, OpenRouter, and Vercel AI Gateway implementations. Provider configs + default models in `models.yaml` (capabilities fetched dynamically); cost estimation in `pricing.py`
- `prompts/` — `PromptManager` singleton for prompt composition. The **conversational global system prompt** is composed from a durable **layered stack** (`prompts/layers.py`: `LayerStore` over `ConfigManager` key `prompts.layers`; `BUILTIN_LAYERS` ship a versioned `default` sidecar overlaid by the user's `override` — `effective = override ?? default`, with `update_available` diffing a released default bump against `base_version`). `PromptManager.compose_prompt` sources the global content from `LayerStore.compose()` (so edits persist across restarts, fixing the old in-memory-only loss) and only attaches a prompt-profile's sections for an explicit non-`is_default` selection (the default profile's sections are folded into the stack — no double-injection). `_ensure_layers_migrated` one-time-imports any customized legacy global. Legacy `/prompts/global*` are back-compat shims over the store. This stack governs **only** the conversational persona/behavior prompt — internal *feature* prompts (reasoning CoT/ToT/ReAct/Reflection, planner `decompose`, extraction, compression) come from the separate `SystemPromptLoader` (`prompts/loader.py`; optional `data/system_prompts.yaml`). `models.py` defines PromptLayer, PromptProfile, PromptSection, GlobalPrompt
- `config.py` — `ConfigManager` singleton for runtime settings. Persists to `data/config.json` with dot-notation access and env var fallback
- `drafting/` — Speculative decoding, multi-stage pipelines, N-best candidate generation. Strategies in `drafting_strategies.yaml`
- `reasoning/` — CoT, ToT (BFS/DFS/beam), ReAct, Reflection. `orchestrator.py` selects strategy by task type
- `agent/` — `Agent` class orchestrating reasoning + drafting + tools. `TaskPlanner` for decomposition (with goal tracking) — the chat path composes the plan with the **main agent model** via `TaskPlanner.compose_with_model` (full turn context + structured JSON, model opts out with `{"plan": null}`), gated by a cheap `_assess_complexity` heuristic; the legacy separate-call `plan()` + `SUBTASK`-regex path remains for non-chat callers. `SessionManager` for conversations
- **Conversation context** — the per-turn context is assembled by `ContextManager.assemble_turn_context` (`agent/context.py`): keep the SYSTEM preamble + as much **recent verbatim transcript** as fits `context.verbatim_budget_ratio` (0.7) of the model's real window, drop the oldest overflow (covered by the rolling summary). The in-memory `SessionManager` is **rehydrated** from the durable `conversation_logs` transcript on a cold session (`agent/conversation_history.py::hydrate_session_from_history`, before the new turn) so a resumed/restored conversation keeps its history. The rolling summary is **context-window-triggered** (`SessionManager.maybe_update_summary`, token-based, not a message count) and **persisted** in Redis (`agent/conversation_summary_storage.py`) so it survives a cold rebuild. Checkpoints/scratchpad (`agent/checkpoint_storage.py`) are Redis-keyed per conversation, re-injected each turn; checkpoint eviction is anchor-preserving (keeps the first) and the `checkpoint` tool has a `replace` mode. After a turn, `views.py::generate_sse` persists it to `conversation_logs` via pure builders in `streaming/persistence.py` (user → tool → steer → assistant Turns); a **hard-stop** (Stop button → `GeneratorExit`) persists the **partial** turn up to the stop (`metadata.interrupted`), and folded steers persist as `user` turns (`metadata.steered`). **Thinking/CoT is process, not result**: it streams live (and shows in the client, collapsed once the turn lands) but is **never persisted** to `conversation_logs` — thoughts are free to regenerate; only results are kept (mirrors persisting a plan's summary card, not its ephemeral step scratchpad). Old turns persisted before this still restore their stored `metadata.thinking`
- `agent/profiles.py` — `ProfileManager` for agent profile CRUD. Profiles stored in `data/agent_profiles.yaml`. Each profile has a Docker-style `agent_id` (e.g., "bold-cosmic-falcon") for identity and a `self_channel` (`_self_{agent_id}`) for self-knowledge
- `agent/tool_output_compressor.py` — LLM-based task-aware compression for oversized tool outputs
- `agent/ambassador.py` + `agent/ambassador_storage.py` — **Ambassador** (Phase 16.6): a dedicated agent that runs *parallel* to a conversation and briefs the user on a turn without polluting the main transcript. `AmbassadorService.brief_turn` resolves the global ambassador profile (`config.ambassador.profile_id` → default profile; an ambassador is any `AgentProfile` with an `ambassador` section), grounds **read-only** via `load_recent_turns`, runs `resolve_with_fallback` + **token-streams** via `provider.stream` (graceful empty-provider degradation, never raises; a cancel/`GeneratorExit` settles the sidecar to `cancelled` — never stuck on `streaming`), and emits namespaced `ambassador_*` SSE per delta. The **persona speaks TO the reader** (second person, names the agent — not "the user … the assistant"), output budget scales with verbosity (`_VERBOSITY_TOKENS`, capped by `ambassador.max_tokens`). The briefing is **grounded on the turn's substance**, not just the reply: the client gathers the turn's tool calls / cited sources / table+diagram exhibits (`client/src/lib/ambassadorTurn.ts::gatherTurnContext`, compact + capped) and posts them as `artifacts`; `_render_artifacts` weaves them into the prompt ("what the agent actually did"). Beyond per-turn briefings, the ambassador answers **free-form questions** about the conversation (`AmbassadorService.answer_question`): same streaming + sidecar machinery (shared `_stream_and_settle` core), grounded on a wider transcript window + optional latest-turn artifacts, persisted under the disjoint `qa:` key family. It also does the **outbound relay** (you → agent): `AmbassadorService.draft_relay_message` shapes a rough intent into a clean *first-person* message (the ambassador as ghostwriter, not speaker), which the user reviews/edits and **relays into the conversation as a real user turn** (or steers the running turn) via the client `ConversationContext.relayToConversation` seam (ChatPanel registers a tab's send/steer handler). The relayed text is the *user's* message — the ambassador never speaks into the transcript as itself, preserving the invariant. Output persists ONLY to the Redis **sidecar** under the `ambassador:` key prefix — never `conversation_logs`/`conv_summary:` (the load-bearing no-pollution invariant). Runs ride the detached-run infra via `start_chat_run(indexed=False)` so they stay out of the `/api/agent/chat/runs` recovery list. Endpoints: `/api/agent/ambassador/{brief-turn,ask,stream,<conversation_id>}`
- `agent/tool_output_chunker.py` — Section detection, JSON path queries, semantic search over stored tool outputs
- `streaming/trajectory_compression.py` — Focus-style intra-trajectory compression for multi-round tool loops

### Key Client Patterns (`client/src/`)

- 3 primary pages: Start, Dashboard, AgentX — routed via `RootLayout` + `TopBar`; plus gate pages `AuthPage` (when `AGENTX_AUTH_ENABLED`) and `VersionMismatchPage` (protocol mismatch)
- Browser-style conversation tabs via `ConversationContext` (add, close, switch, rename, reorder)
- Former tabs → toolbar-icon surfaces: Settings, Tools, and **Memory** open as full-screen modals (`type:'modal', size:'full'`); Plans/Sources remain right-side drawers
- Multi-server support: per-server settings in localStorage (`agentx:servers`, `agentx:server:{id}:meta`, `agentx:activeServer`)
- `ServerContext` provides app-wide server state; `lib/api` is the typed API client (a facade over domain modules in `lib/api/`); `lib/hooks.ts` has React data hooks
- `AgentProfileContext` manages agent profiles (name, model, temperature, reasoning strategy, memory channel)
- Themes (Cosmic dark, warm Light, monochrome Professional) are token-driven: each is a `ThemeDefinition` in `lib/theme.ts` (a map of ~80 CSS vars) applied to `:root` by `ThemeProvider`; components read semantic tokens, never raw palette colors. Add a theme = new `ThemeDefinition` + register in `THEMES` + extend `ThemeName`. Glassmorphism effects and Lucide-react icons
- API errors: `ApiError` carries a status-derived `kind`; use `apiErrorMessage(err)`/`toApiError(err)` (`lib/api`). Surface failures with `useNotify().notifyError(err)` (toasts via `contexts/NotificationContext` + `ui/Toaster`); keep inline errors for form-field validation. Read hooks are built on the `useApi<T>` factory in `lib/hooks.ts`.

#### Styling (Tailwind v4 + design tokens)

- **Tailwind v4** is enabled via `@tailwindcss/vite` (`vite.config.ts`). The CSS entry `src/App.css` imports only the `theme` + `utilities` layers — **Preflight is intentionally disabled** so it doesn't clobber the resets/element styles in `styles/base.css` (which is imported into the `base` layer; utilities out-rank it, per-component CSS files imported unlayered out-rank utilities).
- **Design tokens** live in `lib/theme.ts` and are injected at runtime by `ThemeProvider` as CSS vars (`--surface-base`, `--text-primary`, …). `App.css` bridges them into Tailwind via `@theme inline` so utilities follow theme switches. Use the **semantic utilities**, not raw palette colors: `bg-surface-base|raised|overlay|sunken|hover`, `text-fg|fg-secondary|fg-muted|fg-inverse`, `border-line|line-strong`, `text-accent|bg-accent(-secondary|-tertiary)`, feedback `text-error|success|warning|info`. Spacing tokens `--space-*` exist for hand-written CSS (Tailwind's default spacing scale handles utilities). Brand shadows stay as `var(--shadow-md)` (not bridged — name collision).
- **Components**: prefer Tailwind utilities for new/shared UI; keep per-feature CSS files for complex panels. Shared primitives in `components/ui/` (Button, Badge, Card, Input/Textarea, SectionHeader, Dialog, DropdownMenu, Tooltip, …) follow the shadcn pattern — CVA + `cn()` (`lib/utils.ts`), exported from `components/ui/index.ts`. Radix enter/exit animations come from `tw-animate-css`.

## Development Commands

All commands use [Task](https://taskfile.dev/) (see `Taskfile.yml`). Run `task --list-all` for the complete list.

### Setup & Development

```bash
task setup              # First-time: install deps, init DB dirs, verify env
task check              # Verify environment is ready
task dev                # Start Docker + API + Client concurrently (full stack)
task dev:api            # API server only (assumes Docker running)
task dev:client         # Tauri client only (assumes API running)
task dev:web            # Client in browser mode (port 1420, no Tauri)
task install            # Install all deps (uv sync + bun install)
```

### Database Services (Docker)

```bash
task db:up              # Start Neo4j, PostgreSQL, Redis (aliases: runners)
task db:down            # Stop services (aliases: teardown)
task db:status          # Show container status
task db:init            # Create local data directories (data/neo4j, data/postgres, data/redis)
task db:init:schemas    # Initialize memory system schemas (Neo4j indexes, PostgreSQL tables)
task db:verify:schemas  # Verify schemas exist (read-only check)
task db:shell:postgres  # psql shell into agent-postgres
task db:shell:redis     # redis-cli into agent-redis
task db:shell:neo4j     # cypher-shell into agent-neo4j
```

### Testing

```bash
task test               # Run all backend tests (slow — loads translation models)
task test:quick         # Run tests that don't require model loading (HealthCheck, MCP tests)

# Run a specific test class or method:
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest -v2
uv run python api/manage.py test agentx_ai.tests.TranslationKitTest.test_translate_to_french -v2
```

Test files:
- `tests.py` — Core tests (TranslationKit, HealthCheck, MCP, Extraction)
- `tests_memory.py` — Phase 11.8+ memory system tests (80 tests covering security, integration, edge cases)

Test categories:
- `TranslationKitTest` — Requires HuggingFace models (slow first run)
- `HealthCheckTest` — API structure tests; auto-skips if Docker not running
- `MCPClientTest`, `MCPServerRegistryTest` — MCP infrastructure (no external dependencies)
- `ExtractionPipelineTest` — Memory extraction; skips without API keys
- `ToolOutputCompressorTest`, `ToolOutputChunkerTest` — Context gating (no external dependencies)
- `IntentAwareRetrievalTest`, `TrajectoryCompressionTest`, `ContextGateTest` — Compression pipeline
- `AgentSelfMemoryTest` — Agent identity, self-channel, confidence calibration
- `FactVerificationPipelineTest` — Three-layer fact verification (temporal progression, config, metrics)
- Memory integration tests — Skip gracefully when Docker not running or embedding dimensions mismatch

The `DJANGO_SETTINGS_MODULE` env var is set automatically by the Taskfile (`agentx_api.settings`).

### Linting & Formatting

```bash
task lint               # Run all linters (Python + Client)
task lint:python        # Lint Python with ruff: uv run ruff check api/
task lint:python:fix    # Auto-fix Python lint issues
task format             # Format all code
task format:python      # Format Python with ruff: uv run ruff format api/
```

### Static Analysis & Type Checking

```bash
task check:static       # Run all static analysis (lint + types + build)
task check:types        # Run all type checkers (Python + TypeScript)
task check:types:python # Type check Python with pyright
task check:types:client # Type check TypeScript
task check:build        # Verify both API and client build successfully
task check:build:api    # Verify Python imports and Django configuration
task check:build:client # Build client web assets (TypeScript + Vite)
task api:spec:lint       # Lint OpenApi.yaml (full API spec) with Redocly
```

The root `OpenApi.yaml` is the machine-readable mirror of the docs-site API reference (`docs-site/src/content/docs/api/endpoints.md`). When endpoints change, update both and run `task api:spec:lint`.

When a **memory capability** is added or changed, update the capability manifest `docs-site/src/content/docs/architecture/memory-capabilities.md` (matrix row + section, and the interconnection diagram if a new edge appears) in the same change.

### Django Commands

```bash
task api:run            # Run dev server (aliases: api:runserver)
task api:migrate        # Apply migrations
task api:makemigrations # Create new migrations
task api:shell          # Django interactive shell
```

### Build & Release

```bash
task client:build       # Build Tauri app for production
task client:build:web   # Build web assets only (TypeScript check + Vite build)
task release:check      # Verify release readiness (clean tree, tests, TS compile, Release-Notes.md matches versions.yaml)
task models:download    # Pre-download HuggingFace models (NLLB-200, language detection)
```

**Releasing** is one headless action: `.github/workflows/release.yml` (`workflow_dispatch`
with a single `version` input) builds the desktop installers (3-platform matrix) **and**
publishes the API Docker image (`qrmadness/agentx-api:{version}` + `:latest`), then
publishes a single GitHub Release (tag `v{version}`). The human-written notes live in the
root **`Release-Notes.md`** — its body is injected verbatim into the annotated
supported-server / downloads / Docker / SHA-256-checksum template. A `<!-- release-version:
X.Y.Z -->` marker on its first line is asserted against the baked version (both by the
workflow and by `task release:check`), so a stale notes file can't ship. Version bumps are
**bake-only** (carried into artifacts/image, not committed back — bump the repo separately
via `task versions:sync`).

> **Release hygiene:** the repo is kept release-ready continuously — see the
> version+notes rule in [Development Notices](#development-notices). Because every
> notable change already bumped `versions.yaml` + `Release-Notes.md` (marker and
> body), cutting a release is genuinely one click and never ships a stale version or
> notes file.

## API Endpoints

Base URL: `http://localhost:12319/api/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (add `?include_memory=true` for DB status) |
| `/api/tools/language-detect-20` | GET/POST | Detect language of text |
| `/api/tools/translate` | POST | Translate text (`{"text": "...", "targetLanguage": "fra_Latn"}`) |
| `/api/mcp/servers` | GET | List configured MCP servers and connection status |
| `/api/mcp/tools` | GET | List available MCP tools (filter: `?server=name`) |
| `/api/mcp/resources` | GET | List available MCP resources (filter: `?server=name`) |
| `/api/mcp/connect` | POST | Connect to MCP server (`{"server": "name"}` or `{"all": true}`) |
| `/api/mcp/disconnect` | POST | Disconnect MCP server (`{"server": "name"}` or `{"all": true}`) |
| `/api/providers` | GET | List configured model providers |
| `/api/providers/models` | GET | List available models (filter: `?provider=openai`) |
| `/api/providers/health` | GET | Check health of all providers |
| `/api/agent/run` | POST | Execute a task with the agent |
| `/api/agent/chat` | POST | Conversational interaction with session |
| `/api/agent/chat/stream` | POST | Streaming chat via SSE. Optional `target_agent_id` routes the turn to a specific agent by Docker-style `agent_id` (priority `workflow_id > target_agent_id > agent_profile_id > default`). When `alloy.allow_adhoc_delegation` is on, the agent gets a `delegate_to` tool for ad-hoc agent-to-agent delegation to any other profile (depth-limited, no self-delegation; emits `delegation_*` events). An inline `@agent-id`/`@name` in the message routes that turn to the named agent (overrides the selected agent; suppressed in workflows) — the client composer offers an `@`-autocomplete that inserts the routable agent_id slug. When the agent calls the internal `present_exhibit` tool, the turn emits a typed `exhibit` event (a declarative Gallery→Exhibit→Element tree) instead of a `tool_call`/`tool_result` card — the client renders it via its element registry. Element types: `mermaid` (diagram), `choice` (interactive options; clicking one submits it as the user's next turn), `table` (sortable/scrollable/responsive/expand-to-modal), and `citation` (active foldable sources with a quote vs passive record-keeping links). A successful `web_search` auto-emits a passive `citation` exhibit (one source per result, deduped by URL; id `exh_src_<tool_call_id>`; toggle `citations.auto_capture_web_search`) so sources surface without the agent presenting them. During a turn the backend also emits typed `status` events (`{phase, label, detail?, group?, progress?}`) — a coarse per-phase activity feed (`recalling`/`composing`/`thinking`/`running_tool`/`reading`) the client renders as a live line instead of a silent "thinking". Status rides the run's Redis event bus (not generator yields) via an ambient `emit_status()` (`streaming/status.py`, run resolved from a `ContextVar` set in `chat_run._drive_run`), so it replays on re-attach and is the drop-in seam for future deep sub-phases (embedding/reranking/reasoning-step) that fire inside blocked/cross-thread calls. Runs detached server-side (survives client disconnect); first event `run_started` carries `run_id`. A live `steer` event (echoed by the steer endpoint) carries a user message folded into the running turn. Events: run_started, start, chunk, status, steer, tool_call, tool_result, exhibit, done, error, close |
| `/api/agent/chat/stream/attach` | GET | Re-attach to a detached run (`?run_id=`): replays buffered events + follows live. Emits `run_missing` if the buffer expired |
| `/api/agent/chat/runs` | GET | List the caller's detached chat runs (newest first) for recovery surfaces (Relay inbox, conversation selector) |
| `/api/agent/chat/runs/{run_id}/cancel` | POST | Cooperatively cancel a detached chat run |
| `/api/agent/ambassador/brief-turn` | POST | **Ambassador** (16.6): start a parallel briefing of one turn (`{conversation_id, message_id, assistant_text, user_text?, agent_name?, artifacts?}` — `artifacts` carries the turn's tools/sources/exhibits so the briefing grounds on what the agent *did*) → `{run_id}`. Detached + `indexed=False` (stays out of `/chat/runs`); writes only to the `ambassador:` sidecar (never the transcript). Cancel via `/chat/runs/{run_id}/cancel` |
| `/api/agent/ambassador/ask` | POST | **Ambassador** (16.6): ask a free-form question about a conversation (`{conversation_id, qa_id, question, agent_name?, artifacts?}`) → `{run_id, qa_id}`. Same detached + `indexed=False` + sidecar machinery as brief-turn; persists under the `qa:` family; tail via `/ambassador/stream` (shared `ambassador_*` SSE, keyed by `qa_id`) |
| `/api/agent/ambassador/draft` | POST | **Ambassador** (16.6): outbound-relay draft (`{conversation_id, intent, agent_name?, artifacts?}`) → `{draft}`. Shapes a rough intent into a ready-to-send *first-person* message; the client relays it into the conversation as a real user turn (no transcript write by this endpoint itself) |
| `/api/agent/ambassador/stream` | GET | Tail a briefing or Q&A run (`?run_id=`): namespaced `ambassador_start`/`ambassador_chunk`/`ambassador_done`/`ambassador_error` SSE; `run_missing` on buffer expiry |
| `/api/agent/ambassador/{conversation_id}` | GET | Replay a conversation's persisted briefings **and Q&A** from the sidecar (`{briefings, qa}`) (cold-open / reload / tab-switch) |
| `/api/agent/chat/runs/{run_id}/steer` | POST | Live-steer a running turn (`{message, mode?}`): queues the message; the tool loop drains it at the next safe boundary (after a tool round, or instead of ending) and folds it in as a fresh user turn so the agent course-corrects without stopping. Owner-only; echoes a `steer` event onto the run bus so all clients show the steer bubble inline. The steer is **persisted** as a `user` turn with `metadata.steered` (+`steer_round`/`after_tools`/`phase`) — a procedural-memory signal |
| `/api/agent/status` | GET | Get current agent status |
| `/api/prompts/profiles` | GET | List all prompt profiles |
| `/api/prompts/profiles/{id}` | GET | Get profile detail with composed preview |
| `/api/prompts/global` | GET | Get global prompt |
| `/api/prompts/global/update` | POST | Update global prompt |
| `/api/prompts/sections` | GET | List all prompt sections |
| `/api/prompts/compose` | GET | Preview composed system prompt |
| `/api/prompts/mcp-tools` | GET | Get auto-generated MCP tools prompt |
| `/api/memory/channels` | GET | List memory channels |
| `/api/memory/entities` | GET | List entities (filter: `?channel=`, `?type=`) |
| `/api/memory/entities/graph` | GET | Get entity relationship graph |
| `/api/memory/facts` | GET | List facts (filter: `?channel=`, `?entity_id=`); each fact carries `entities[]` ({id,name,type}) for its ABOUT'd entities |
| `/api/memory/facts/{id}/entities` | POST/DELETE | Link/unlink an entity to a fact (`{entity_id}`, ABOUT edge); returns the fact's updated entity list |
| `/api/memory/strategies` | GET | List procedural strategies |
| `/api/memory/procedures` | GET | List distilled procedures (Slice 1 — trigger/body/scope/strength) |
| `/api/memory/stats` | GET | Memory system statistics |
| `/api/metrics/usage` | GET | Aggregated token/cost/latency usage from `conversation_logs` (`?days=` 1–90, default 14): totals, by-model, daily series |
| `/api/memory/checkpoints` | GET/DELETE | List or clear a conversation's model-authored checkpoints (`?conversation_id=`) |
| `/api/memory/user-history` | POST | Browse the user's past turns + top facts (`{topic?, limit?, channel?}`) |
| `/api/memory/recall-settings` | GET/POST | Get or update recall layer settings |
| `/api/memory/consolidate` | POST | Trigger manual consolidation |
| `/api/memory/reset` | POST | Reset memory data (with confirmation) |
| `/api/memory/export` | POST | Export the user's memory graph to a round-trippable JSON envelope (`{channel?, include_embeddings?}`); also `task memory:export` |
| `/api/memory/import` | POST | Import a memory export idempotently (`{data, mode: merge\|replace, channel?}`); also `task memory:import` |
| `/api/memory/settings` | GET/POST | Get or update memory settings |
| `/api/jobs` | GET | List background jobs |
| `/api/jobs/clear-stuck` | POST | Clear stuck jobs |
| `/api/jobs/{id}` | GET | Get job detail |
| `/api/jobs/{id}/run` | POST | Manually run a job |
| `/api/jobs/{id}/toggle` | POST | Enable/disable a job |
| `/api/config` | GET | Get runtime configuration (secrets redacted) |
| `/api/config/update` | POST | Update runtime config (`{"key": "dot.path", "value": ...}`) |
| `/api/config/context-limits` | GET | Per-model context-window limits |

Additional endpoint groups (added since v0.18 — see `urls.py` for the full set):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/{status,setup,login,logout,session,change-password}` | GET/POST | Optional session auth (Phase 17, gated by `AGENTX_AUTH_ENABLED`) |
| `/api/agent/profiles`, `/api/agent/profiles/{id}` | GET/PATCH/DELETE | Agent profile CRUD; `.../set-default` (POST) |
| `/api/alloy/workflows`, `/api/alloy/workflows/{id}` | GET/POST/PATCH/DELETE | Multi-agent (Agent Alloy) workflow CRUD |
| `/api/chat/background`, `/api/chat/background/{job_id}` | POST/GET | Queue + poll background conversations |
| `/api/tool-outputs`, `/api/tool-outputs/{key}` | GET/DELETE | Stored tool-output retrieval |
| `/api/prompts/templates*`, `/api/prompts/enhance` | GET/POST/PUT/DELETE | Prompt template CRUD + LLM prompt enhancer |
| `/api/conversations`, `/api/conversations/{id}/messages` | GET | Conversation history |
| `/api/agent/plans/cancel` | POST | Cancel active plan execution |
| `/api/agent/plans/{plan_id}/status` | GET | Read Redis-tracked plan state (`?session_id=`); `{found:false}` on TTL expiry; `resumable` flags an `active`/`interrupted` plan with work left; `ttl_seconds` = how long the snapshot stays resumable. On conversation load the client auto-appends a `choice` exhibit (`exh_resume_{plan_id}`, shows the remaining lifetime) that **nudges the model** to continue the interrupted plan (a normal turn with the done/remaining steps — not a `PlanExecutor` re-run) |

**Plan termination & cancellation.** `PlanExecutor` wraps its subtask loop in `try/finally`: a hard Stop (`GeneratorExit`) mid-subtask resets the in-flight subtask to `pending` and marks the plan **`interrupted`** (a resumable status alongside `active`) instead of a stuck `running`; cooperative cancel marks `cancelled` + `clear_cancel`s the flag. The Stop handler in `views.py` persists the interrupted plan card (top-level `status` in the `plan` metadata; the client `mapServerMessages` maps it faithfully rather than rewriting to `failed`). Cancellation is **cooperative** (GeneratorExit lands at a `yield`), made prompt by capping tool wall-clock — `web_search` is bounded by `search.timeout` (default 15s; Tavily otherwise defaulted to ~60s). Tool execution stays **synchronous** (no off-thread/`asyncio` surgery — that deadlocked Stop).
| `/api/agent/plans/{plan_id}/resume` | POST | Resume an interrupted plan (`{session_id, agent_profile_id?, model?}`): rebuilds the plan from Redis (`PlanStateStore.load_plan`) and streams only its not-yet-terminal subtasks (SSE, first event `plan_resumed`) as a detached run; persists the synthesis (no duplicate user turn). Single-agent only (alloy resume deferred); 404 when not resumable |

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:

```bash
NEO4J_PASSWORD=...          # Used by docker-compose and agent_memory
POSTGRES_PASSWORD=...       # Used by docker-compose and agent_memory
OPENAI_API_KEY=sk-...       # For OpenAI provider
ANTHROPIC_API_KEY=sk-ant-...# For Anthropic provider
EMBEDDING_PROVIDER=local    # "openai" or "local" (sentence-transformers)
```

MCP servers are configured in `mcp_servers.json` (see `mcp_servers.json.example`).

## Important Technical Details

- **Translation models load eagerly** at `TranslationKit` init, not lazily. First request downloads models from HuggingFace (~600MB for NLLB-200).
- **Memory system is lazy**: Database connections (Neo4j, PostgreSQL, Redis) are created on first use, not at import time. Config via pydantic-settings from `.env`.
- **Docker data is bind-mounted** to `./data/` (not Docker volumes). Run `task db:init` to create the directory structure.
- **Tauri dev server** runs Vite on port 1420 with HMR on port 1421. Main window config in `client/src-tauri/tauri.conf.json`.
- **Python managed by uv**, client packages by **bun**. The `task dev` command uses globally-installed `concurrently` (installed via `task install`).

## Agent Memory Interface

The `AgentMemory` class (`kit/agent_memory/memory/interface.py`) provides a unified API for memory operations:

### Core Methods
| Method | Description |
|--------|-------------|
| `store_turn(turn)` | Store conversation turn in episodic + working memory |
| `remember(query, top_k)` | Retrieve relevant memories (turns, facts, entities, strategies) |
| `learn_fact(claim, source, confidence)` | Store factual knowledge in semantic memory |
| `upsert_entity(entity)` | Create/update entity in semantic memory |
| `record_tool_usage(...)` | Record tool invocation for procedural learning |
| `reflect(outcome)` | Trigger async consolidation job |

### Goal Tracking Methods
| Method | Description |
|--------|-------------|
| `add_goal(goal)` | Create a goal linked to user |
| `get_goal(goal_id)` | Retrieve goal by ID |
| `complete_goal(goal_id, status, result)` | Update goal status ('completed', 'abandoned', 'blocked') |
| `get_active_goals()` | Get all active goals for user |

### TaskPlanner Integration
`TaskPlanner.plan()` accepts an optional `memory` parameter. When provided:
- Creates a `Goal` for the main task on plan creation
- `TaskPlan.goal_id` stores the linked goal ID
- `Agent.run()` calls `complete_goal()` on task success/failure

### Extraction Pipeline
The `ExtractionService` (`kit/agent_memory/extraction/service.py`) provides LLM-based extraction:
- `check_relevance_and_extract(text)` — Combined relevance check + extraction in single LLM call (~75% fewer calls)
- `check_relevance_and_extract_assistant(text)` — Self-knowledge extraction from agent's own responses (certainty: definitive/analytical/speculative)
- Both accept a `roster` (`[{agent_id, name}]`) + `addressed_agent_id` so facts are attributed to a *specific* agent: the LLM names the agent in prose (`subject_agent`), and `_resolve_agent_attribution` resolves that name → `agent_id` (the durable source of truth), stored transiently as `subject_agent_id`. Unknown names are demoted to `third_party` (never fabricate an agent_id)
- `check_contradictions(claim, facts)` — Three-layer fact verification: hash gate → semantic duplicate → entity-scoped candidates → LLM adjudication
- Uses `nvidia/nemotron-3-nano` by default for extraction (configurable via `combined_extraction_model` setting)
- Consolidation jobs (`consolidation/jobs.py`) call extractors every 15 minutes
- Gracefully degrades to empty results when no provider is configured

### Procedural Memory (encode → distill → reflex-core)
Learns the project/user/domain "how we work here" **delta** (not baseline behavior a capable model already does). Three loops:
- **Encode (Slice 0, every turn, cheap):** stages high-signal `procedure_candidates` PG rows — `correction` (from persisted steers) + `explicit_rule` (`procedural.detect_explicit_rule`, heuristic). `status='pending'`.
- **Distill (Slice 1, consolidation):** the async `distill_procedures` job (`consolidation/jobs.py`, registered in the registry default pipeline + the autonomous worker) reads pending candidates, groups by derived scope (corrections with an `agent_id` → `_self_{agent_id}`; explicit rules stay on their channel), runs `ExtractionService.distill_procedure` (baseline-deviation filter with explicit-rule deference), and **strengthens** a cosine-similar existing `Procedure` (`procedural_dedupe_threshold`) instead of duplicating. Candidates flip to `distilled`/`discarded` (+`distilled_into`). Repurposes the (dead) `learn_strategy`/`reinforce_strategy` write pattern as `learn_procedure`/`reinforce_procedure` on a new Neo4j `Procedure` node (`models.Procedure`: `trigger`/`body`/`rationale`/`scope`/`strength`/`evidence_refs`; `procedure_embeddings` vector index).
- **Activate — reflex core (Slice 1):** `ProceduralMemory.get_reflex_procedures` (top-`strength` over recall channels, *maintained not searched*) is attached at the `interface.remember()` boundary and rendered by `MemoryBundle.to_context_string` ("Learned Procedures"). Gated by `reflex_core_enabled`/`reflex_core_limit`.
- **Worker:** `ConsolidationWorker.run()` now awaits coroutine jobs (it was silently dropping the async `consolidate`); `task dev` runs the worker; manual `task memory:distill-procedures`. Inspect via `GET /api/memory/procedures` + `procedures` on `/api/memory/stats`.
- Deferred: trigger-indexed activation (activation-nerve/point-of-action/deliberate `recall_procedures`), tool-sequence procedures (needs internal-tool recording + a live Outcome signal), ReflectiveReasoner correction-reflection.

### Context Gating
Large tool outputs (>12K chars) are compressed and stored for retrieval:
- `ToolOutputCompressor` — task-aware LLM compression with structure indexing
- `tool_output_chunker.py` — section detection, JSON path queries, semantic search
- Internal MCP tools: `read_stored_output`, `tool_output_query`, `tool_output_section`, `tool_output_path`
- Trajectory compression — Focus-style intra-trajectory compression at 75% context threshold
- Retrieval tool bypass prevents re-storage loops

### Web Research Tools (`mcp/internal_tools.py`)
- **Capability-aware advertisement**: the backends differ (Tavily = a research suite via the official `tavily-python` SDK: search/extract/map/crawl/research; Brave = search-only via httpx). A pre-check (`resolve_active_search_backend` + `build_tool_schema`/`build_tool_description`) inventories the *active* backend at tool-listing time (`get_internal_tools`) and advertises only its real tools + knobs to the model. `web_extract`/`web_map`/`web_crawl`/`web_research` are advertised only when Tavily is active, but stay registered (executable) so a stale call self-guards. `SEARCH_CAPABILITIES` is the single source of truth.
- `web_search` (both backends) keeps results compact (no raw page content → stays under the oversize threshold + parseable for auto-capture); `web_crawl`/`web_research` ride the oversize/stored-output handler. `web_research` is agentic/slow → bounded ~120 s timeout + `web_research.enabled` flag. Config: `search.backend`, `search.{tavily,brave}_api_key`, `search.fallback_enabled`, `web_research.enabled`.
- A successful `web_search`/`web_research` auto-captures sources as a passive `citation` exhibit (see the `/api/agent/chat/stream` row); the shared `citation_exhibit_from_web_search` dedupe helper also backs the conversation **Bibliography** (client `lib/bibliography.ts` → `SourcesPanel`, a right-side "Sources" drawer that dedupes/numbers a conversation's cited sources).

### Model Resolution & Fallback (`providers/registry.py`)
- Every LLM-using feature resolves its `provider:model` through the registry. `resolve_with_fallback(model, *, preferred_fallback=)` (sync) and `complete_with_fallback(...)` (async, execution-retry) make a feature **never hard-fail the turn**: an unconfigured *or* unreachable provider falls back to the caller's active **agent-profile model** (`preferred_fallback`) → global default (`preferences.default_model` / `models.defaults.chat`) → first healthy provider. Kill-switch `models.fallback_enabled`; best-effort provider health cache (fed by `health_check`). The **main chat path stays strict** (`get_provider_for_model`) — the agent's chosen model is the floor.
- **Memory stage models inherit**: each consolidation/recall stage resolves `explicit stage override → consolidation.feature_default_model (bulk) → default chat model` (`ExtractionService._resolve_stage_model`). Set `feature_default_model` (Consolidation settings → "Default model for all memory stages" + "Apply to all stages") to point all of memory at one model in a single write; stage defaults stay `lmstudio:…` so local users stay cheap and cloud-only users auto-fall-back.

### Agent Identity
Each agent profile has a Docker-style `agent_id` (e.g., "bold-cosmic-falcon"):
- Auto-generated, immutable, adjective-adjective-noun format (~83K combinations)
- `self_channel` property: `_self_{agent_id}` for agent's self-knowledge
- Recall searches: `[active_channel, _self_{agent_id}, _global]`
- Agent self-extraction during consolidation stores knowledge to self-channel

### Multi-Agent Attribution (Phase 16)
Attribution is agent-specific, not a singleton "agent". `agent_id` is the durable key; the display name is prose only.
- **Agents are first-class entities**: consolidation `_ensure_agent_entities` upserts an `Entity(type="Agent")` per participating agent, in `_global` (so it serves every channel via `find_entity_by_name_or_alias`), id `agent:{user_id}:{agent_id}` (one-per-user, no cross-user bleed), canonical key in `properties.agent_id`, name + prior names as `aliases`. Facts about an agent link to it via the normal `[:ABOUT]` entity machinery
- **Name stamping**: assistant turns carry the agent's display name in `Turn.metadata["agent_name"]` (set by the writers in `views.py`/`core.py`/`alloy/executor.py`); `episodic.store_turn` stamps it onto the `Turn`/`AgentParticipant` nodes so the kit can build a roster (`get_conversation_roster`) without importing `ProfileManager`
- **Per-agent routing**: `_resolve_subject_channel` honors a fact's `subject_agent_id` → `_self_{that_agent}` (a directive aimed at Mobius lands in Mobius's memory, not the producer's). The user-turn fetch resolves "you" per turn via the responding agent (`FOLLOWED_BY`); assistant self-extraction routes **each turn by its own producing `agent_id`** so multi-agent conversations don't funnel into the first agent's channel
- **Rename-safety**: `ProfileManager.update_profile` propagates a display-name change to the Agent entity's `aliases` (`_propagate_agent_rename`); `dedupe_entities` never merges `Agent` nodes (keyed by `agent_id`, not name)
- **Backfill**: `task memory:backfill-agent-attribution[:apply]` (mgmt command `backfill_agent_attribution`) deterministically rewrites legacy generic "Agent …" self-facts to the agent's name + adds the `[:ABOUT]` link (channel encodes which agent → no LLM needed)
- **Debug harness**: `task memory:debug-attribution -- --scenario <directive|cross-agent|mixed> --agents "Mobius,Jeff"` (mgmt command `debug_attribution`) drives a scripted multi-agent conversation through *real* consolidation and reports per-channel attribution + `[:ABOUT]` links with ✅/❌ expectations. Non-destructive by default (throwaway user, scoped cleanup); `--isolate` snapshots+wipes+restores for a sterile read

## Project Status

Phases 1-14 and 17 (Server Management: auth, Docker production stack, multi-cluster, version matching) complete. Phase 15 (Plan Execution) ~80%. Phase 16 (Multi-Agent Conversations) ~45% — Agent Alloy v1 shipped (supervisor + specialist delegation); 16.1 per-turn attribution, 16.2 explicit agent routing, 16.3 per-agent tool isolation, 16.4 ad-hoc agent-to-agent delegation, and 16.5 @-mention routing + `AgentParticipant` graph nodes (with client `@`-autocomplete) shipped; multi-agent attribution (per-agent self-channel routing, agents as first-class entities, name-stamping + roster-aware extraction, rename-safety, legacy backfill) shipped; 16.6 Ambassador foundation (parallel non-polluting per-turn briefing) shipped. Phase 18 (UX Improvements + Memory Tuning) in progress. Current version: 0.21.32 ("Mobile-Ready Alpha"). See `Todo.md` for detailed tracking.

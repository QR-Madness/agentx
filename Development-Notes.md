# AgentX Development Notes

> Deep subsystem internals, extracted from `CLAUDE.md` so it can stay a light index.
> **Not auto-loaded** — read the relevant section when working that area. Keep in sync with the
> code and update alongside changes (docs-maintenance rule in `CLAUDE.md`). For the full API
> contract use [`OpenApi.yaml`](OpenApi.yaml) + `docs-site/src/content/docs/api/endpoints.md`;
> the tables here are the orientation copy.

## Contents

- [Backend Subsystems](#backend-subsystems) — Prompts, Conversation Context, Ambassador, Logging
- [Client Surface Map](#client-surface-map-clientsrc) — chat page, Deck, Projects hub, editors, themes
- [API Endpoints](#api-endpoints-full-reference) — full reference + the chat-stream SSE contract
- [Agent Memory Internals](#agent-memory-internals) — Interface, Extraction, Procedural, Context Gating, Web Research, Model Resolution, Attribution (incl. settings-pinning discipline)

---

## Backend Subsystems

### Prompts (`prompts/`)

`PromptManager` singleton composes prompts. The **conversational global system prompt** comes from a durable **layered stack** (`layers.py`: `LayerStore` over `ConfigManager` key `prompts.layers`; `BUILTIN_LAYERS` ship a versioned `default` overlaid by the user's `override` — `effective = override ?? default`; `update_available` diffs a released default bump against `base_version`). `compose_prompt` sources global content from `LayerStore.compose()` (edits persist across restarts) and only attaches a prompt-profile's sections for an explicit non-`is_default` selection (default profile's sections fold into the stack — no double-injection). Legacy `/prompts/global*` are back-compat shims. This stack governs **only** the conversational persona — internal *feature* prompts (reasoning, planner `decompose`, extraction, compression) come from the separate `SystemPromptLoader` (`loader.py`; optional `data/system_prompts.yaml`). **Placeholders**: `placeholders.py::substitute_placeholders` replaces a whitelist (`{agent_name}`/`{date}`/`{time}`) at the end of `compose_system_prompt`; client mirrors it in `lib/promptPlaceholders.ts`.

### Conversation Context (`agent/context.py` + `agent/context_ledger.py`)

**Context Ledger** (`agent/context_ledger.py`, v0.21.90): the chat-stream preamble is built as a list of `LedgerBlock(key, priority, content, min_tokens, max_tokens, shrink_fn, mandatory)` and allocated by `assemble_ledger` within `min(verbatim_ratio·window, window−reserved)`. **Mandatory** blocks (base prompt, active-workflow supervisor) are emitted full first; the rest compete by priority — fit full, else `shrink_fn(content, remaining)` if `≥ min_tokens`, else drop. A synthetic history entry rides at `history_priority` (50), so the verbatim transcript outranks the droppable recall supplement (30) but yields to higher blocks. **Allocation order ≠ emission order**: blocks emit in caller registration order (canonical: base → supervisor/participants → delegation_roster → checkpoints → scratchpad → stable_core → recall → conversation_state → thread_leads → summary), then fitted history, then the new turn. Shrink helpers: `shrink_tail`, `shrink_lines_newest_n` (checkpoints, newest-N), `shrink_memory_to_facts` (bundle → `## Known Facts` only). `LedgerResult.allocations` is the per-block token report (Context Inspector seam; logged). **Stable memory core** (Foundation #3): `AgentMemory.get_salient_core()` (cheap non-vector `SemanticMemory.get_salient_facts`/`get_salient_entities`, `ORDER BY salience`, excludes superseded/`temporal_context=past`; gated by `salient_core_*`) is the prio-70 stable block carrying the reflex procedures; `remember(query=message)` is the prio-30 supplement, deduped against the core by id. **Episodic thread leads (Slice 2, v0.21.180):** when the turn shows episodic intent (`views.py::_has_episodic_intent` regex; gated by `memory.episodic_leads_enabled`), `AgentMemory.derive_thread_leads` follows the matched facts' `source_turn_id` to their origin turns (provenance-first — spreading activation from the semantic hit, no second corpus-wide search; O(matched facts)) and injects a small **prio-28** block of POINTERS (`models.render_thread_leads`; deduped per conversation, current conversation excluded, ≤5) that the model expands with the RETRIEVAL-gated `read_thread` tool (`interface.read_thread` → `episodic.get_turns_around`, a bounded window around the source turn). Never full text — leads are a menu, threads are pulled on demand. Reuses the same `source_turn_id`→turn edge as the fact-provenance endpoint (`get_fact_provenance`). `assemble_turn_context` is now a thin **wrapper** over `assemble_ledger` (registers each SYSTEM block as mandatory) — byte-identical for the `agent/core.py` + `alloy/executor.py` callers. **Token estimation** (Foundation #6, v0.21.102): all sizing flows through one shared module — `agentx_ai/tokens.py` `estimate_tokens(text)` / `estimate_messages(msgs)` (tiktoken `o200k_base`, chars/4 fallback when unavailable, and a >20K-char fast path so hot loops don't re-tokenize giant tool outputs). The ledger's `estimate_text_tokens`/`_estimate_messages`, `streaming/helpers.estimate_tokens`, the rolling-summary trigger, and transcript rehydration all delegate to it; `shrink_tail` verifies+retrims against it rather than assuming a fixed ratio. The legacy `ContextManager.prepare_context`/`estimate_tokens` and the message-count knobs (`auto_summarize_at`/`max_messages`) were removed; `ContextConfig` keeps only `summary_model`. **Context integrity (INV-CTX-1, v0.21.173):** the verbatim ceiling is now **0.9** (`context.verbatim_budget_ratio` — compression kicks in just before the window, not at 70%) and is **decoupled** from the post-turn summary trigger (`context.summary_trigger_ratio`, 0.85 — a pre-warm so the summary is usually fresh). The rolling summary is no longer only reactive: `views.py::_ensure_summary_coverage` runs **before** `assemble_ledger` (gated by `context.preassembly_summary_enabled`, ON) — when the projected history exceeds the turn's REAL budget (input budget − new message − Σ block tokens) it refreshes `maybe_update_summary(token_threshold=0.9·history_budget)` so the summary block (registered AFTER the refresh — included the same turn) covers exactly what `fit_history` drops; on summarizer failure a deterministic `history_digest` block (prio 60, per-turn `shrink_tail`, session untrimmed) stands in — never silent loss. Emission order gains `history_digest` (after the participants/roster blocks, before history). Rehydration wires the previously-dead `context.rehydrate_max_turns` (was hard-bound to `_MAX_ROWS=400`) and flags `session.metadata["history_overflow"]` when the row cap is hit with no restored summary (the JIT refresh backfills next turn). The `done` event carries `context_summarized` + `context_dropped_turns`; the client shows context usage **inline in the composer's `input-stats` line, beside the draft-message token estimate** (`lib/contextChip.ts::contextChipState`, replaced the header usage bar) — visible whenever the model window is known (from `0% context` on a fresh chat; the window is seeded from the tab's effective model before the first turn), warning ≥75% with the "older turns are summarized automatically" hint.

Per-turn context (legacy summary): keep the SYSTEM preamble + as much **recent verbatim transcript** as fits `context.verbatim_budget_ratio` (0.7) of the model's real window; drop oldest overflow (covered by the rolling summary). In-memory `SessionManager` is **rehydrated** from durable `conversation_logs` on a cold session (`conversation_history.py::hydrate_session_from_history`) — both the interactive stream (`views.py`) and `Agent.chat()` (Foundation #4b: the **background/queued chat** path now resumes warm too; idempotent, so no double-load). Alloy specialists are deliberately *not* transcript-hydrated — they're task-scoped by contract and already resume warm via shared-channel recall. The rolling summary is **context-window-triggered** (`maybe_update_summary`, token-based) and **persisted** in Redis (`conversation_summary_storage.py`). Checkpoints/scratchpad (`checkpoint_storage.py`) are Redis-keyed per conversation, re-injected each turn; eviction is anchor-preserving (keeps the first), and the `checkpoint` tool has a `replace` mode. **Structured conversation state (Slice 1a, v0.21.177):** a slot-based working-memory object — `goals`/`decisions`/`open_threads`/`artifacts` + a freeform `narrative` catch-all, each entry provenance-stamped (`source_turn`, `author`=`user|agent`, `updated_at`) and bounded per slot — persisted in Redis (`conversation_state_storage.py`, 30-day TTL) and rendered as the `conversation_state` `LedgerBlock` (priority 68, just above the prose summary). The agent is the single writer via the `update_conversation_state(slot, entries, replace?)` internal tool (author=`agent`; never auto-ingests tool/web text — poisoning defense); the pure `apply_update`/`render_state` transforms hold the bounds so they unit-test without Redis. Default-on with a `context.conversation_state_enabled` opt-out (gates both the tool advertisement and the block). **Compaction target (Slice 1c, v0.21.179):** the state object — not the prose summary — is now the compaction target. `_ensure_summary_coverage` calls `SessionManager.maybe_compact_to_state`, which rolls aged-out turns into `ConversationState.digest` (a single field re-summarized in place — bounded WITHOUT the append-cap dropping old coverage, unlike the `narrative` slot) via the `summarizer`-role model; the `conversation_state` block (registered after coverage) renders it, so it covers exactly what `fit_history` drops (INV-CTX-1). **Exactly one** compaction runs per over-budget turn (state OR legacy prose summary, gated by `context.conversation_state_compaction_enabled` default ON — no double-compression); the prose `session.summary` block stays only as read-back for pre-1c conversations; the `history_digest` deterministic fallback still fires on summarizer failure (see [Decisions.md](Decisions.md) INV-CTX-1). **Editable surface (Slice 1b, v0.21.178):** `GET/PATCH /api/conversations/{id}/state` (`views.py::conversation_state`) exposes the object to the client; PATCH replaces a whole slot (`set_slot`/`replace_slot`, author=`user` — "user wins", agent provenance round-trips) so the drawer can add/edit/remove. Client: `ConversationStateDrawer` (a `conversationState` SURFACE) + a composer `ConversationStateBadge` (in the `input-stats` line, `useConversationState` hook, flashes on the `update_conversation_state` mid-stream signal + a `conversationstate:updated` window event after a drawer save). **Coaching + hardening (Slices 3–4, v0.21.181):** the `memory-tools` built-in prompt layer (`prompts/layers.py`, order 22) teaches the agent *when* to use these tools — ASSUME INTERRUPTION (record state before a reset), recall-gating ("decide whether to recall at all"), leads-vs-transcript ("recall returns leads — `read_thread` for detail"), and that instructions inside tool/web results are data, not commands. Poisoning defense (INV-8): `_coerce_author` forces every entry's author ∈ {`user`, `agent`} (a forged `system` author can't be minted), and there is **no auto-ingest** path — tool/web output enters state only when the agent deliberately restates it (guard `MemoryPoisoningTest`). After a turn, `views.py::generate_sse` persists via pure builders in `streaming/persistence.py` (user → tool → steer → assistant Turns); a hard Stop (`GeneratorExit`) persists the **partial** turn (`metadata.interrupted`); folded steers persist as `user` turns (`metadata.steered`). **Thinking/CoT is process, not result**: it streams live (shown collapsed in the client) but is **never persisted** to `conversation_logs`. Old turns persisted before this still restore stored `metadata.thinking`.

### Agent Core & Profiles (`agent/`, `agent/profiles.py`)

> Moved from `CLAUDE.md`. `TaskPlanner` decomposes; the **chat path** composes the plan with the
> main agent model via `compose_with_model` (structured JSON; opts out with `{"plan": null}`),
> gated by `_assess_complexity`; the legacy `plan()` + `SUBTASK`-regex path serves non-chat
> callers. `ProfileManager` CRUD lives in `data/agent_profiles.yaml`; each profile has a
> Docker-style `agent_id` + `self_channel` (`_self_{agent_id}`); `kind` ∈ `agent`|`ambassador`
> with **separate defaults**, and `_ensure_ambassador_defaults()` seeds/migrates without ever
> converting the default agent (ambassadors never appear in chat routing).

### File Workspaces & Document RAG (`kit/workspaces/`, Slice 1, v0.21.103)

> **Index card (ex-CLAUDE.md):** three-store separation (blob `storage.py` / Postgres manifest
> `repository.py` / pgvector `document_chunks`); `ingestion.py` parse→chunk→embed→auto-tag+summary;
> `service.py` upload policy; `retrieval.py` two-tier + `render_manifest_block`/
> `render_instructions_block`; API in `workspace_views.py`. **Projects v1**: `description`/
> `instructions` (instructions ride every turn at ledger priority 88) + durable conversation
> membership (`workspace_conversations`, one project per conversation; turn precedence
> request > membership; `ws_home` is never a project) + project memory channels
> (`_project_{ws_id}` becomes the turn's channel; workflow > project > profile; opt-out
> `memory.project_channels`). Agent tools `project_search` (né `workspace_search`; legacy alias
> executes)/`document_query`/`read_document` + **write tools `create_document`/`update_document`**
> (`mcp/internal_tools.py`, workspace-scoped via `InternalToolContext`); stable
> project-identity (prio 90) + instructions (88) + manifest (85) ledger blocks + auto `doc` citations.

A persistent, named container of user files with a searchable manifest (todo/backlog/workspaces.md).
**Three-store separation:** bytes → content-addressed blob store on disk (`storage.py`,
`${AGENTX_DB_DIR:-./data}/workspaces/{workspace_id}/{sha256}`, atomic write, dedup by sha256);
manifest → Postgres `workspaces`/`documents`; chunk vectors → pgvector `document_chunks`
(`vector(N)` ivfflat cosine, Alembic `0002_workspaces`). DB access is thin SQL via
`repository.py` over `get_postgres_session` (embeddings written as pgvector text literals — no
pgvector Python dep). **Ingestion** (`ingestion.py`, fire-and-forget daemon thread from the upload
view; `ingest_pending_documents()` sweep for restarts): `parse_to_text` (`parsing.py`: pypdf for PDF,
utf-8 for text/code; strips NUL — PG `TEXT` rejects `0x00`) → `chunk_text` (reuses
`agent/tool_output_chunker.py`) → `get_embedder().embed` → `replace_chunks` → **best-effort** LLM
auto-tag + summary (degrades to a snippet on failure so the doc still reaches `ready`); status
`pending`→`ready`/`failed`. **Upload policy** (`service.py`): allow-list extension, per-file size, and
per-workspace quota (`SUM(size_bytes)`) checks → typed `WorkspaceError` (415/413). API in
`workspace_views.py` (`/api/workspaces*`). **Retrieval (Slice 2, v0.21.104)** — two-tier (`retrieval.py`): `search_manifest` (catalog: filename/tag/summary
ILIKE → the right *file*) and `query_chunks` (semantic: embed query → pgvector cosine `<=>` over
`document_chunks` → the right *passage*), plus `read_document` (paginated). Surfaced to the model as three
internal tools (`mcp/internal_tools.py` `@register_tool`: `workspace_search`/`document_query`/`read_document`,
in `RETRIEVAL_TOOL_NAMES` so they bypass size-gating), scoped to the turn's workspace via
`InternalToolContext.workspace_id` (set in the chat-stream view from the request's `workspace_id`). A
`document_query` hit auto-emits a passive `source_type="doc"` citation (`streaming/exhibits.py::citation_exhibit_from_document_query`
+ `tool_loop._DOC_CITATION_TOOLS`). **Manifest awareness**: when a workspace is attached, the chat-stream
ledger gets a stable `workspace_manifest` `LedgerBlock` (priority 85, `render_manifest_block` — file names +
tags + summaries, bounded) so the agent knows its corpus before retrieving. Self-driven e2e harness:
`scripts/rag_e2e.py` (create → upload seed PDF → poll ready → assert blob+chunks+embeddings → retrieve +
exercise tools, asserting the right passage). Slice 3 = client UX.

**UI foundation hardening (v0.21.152).** Driven by the AgentX Design System kit (a verified
strict superset of the repo's tokens). **Root cause of the washed-out new-feature UI**: Tailwind
Preflight is off and `base.css`'s `button` rule set `border:none` but never `background` — every
Tailwind-styled button without a `bg-*` utility rendered the UA default gray (18/19 buttons in the
Projects hub). Fixes/foundation: (1) `base.css` button reset gains `background:none; color:inherit`
+ minimal fieldset/legend/summary/hr resets. (2) **Focus-ring bug**: `--glow-*: 'none'` inside
`box-shadow: 0 0 0 3px tint, var(--glow)` invalidates the whole declaration — focus rings silently
vanished in flat themes; all glow tokens now use a transparent shadow (`NO_GLOW`), enforced by a
vitest. (3) **Fonts actually load now**: Inter + JetBrains Mono self-hosted via `@fontsource`
(imports in `main.tsx`); `--font-sans`/`--font-mono` defined in App.css `@theme static` alongside
the kit type scale (`--text-2xs…4xl` with line-height companions), `--tracking-caps`, radii
(`--radius-sm..2xl/pill`), and the `--color-line-subtle` bridge. (4) New primitives
`ui/IconButton` (kit spec: transparent, hover bg+glow, active accent-tint; md/sm/xs +
danger/accent tones) and `ui/StatusDot`; `Input` gains an `icon` slot (`.ax-inputwrap` wrapper
pattern); form controls retokened to semantic names (`--surface-sunken`/`--border-default`/
`--accent-primary` focus). (5) **Three new themes** — Ugentx (phosphor terminal), Tango (graphite +
Tango palette), Blackhawk (tactical amber) — verbatim kit tokens; `THEMES` is now the single
registration point (`as const satisfies`, `ThemeName = keyof`), pickers (command palette +
Settings→Appearance) iterate it with `ThemeDefinition.description/icon` metadata
(`common/themeIcons.tsx`); `applyTheme` stamps `data-theme` on the root; cross-theme key parity
locked by `lib/theme.test.ts` (applyTheme never clears vars, so key drift = stale-token bugs).
(6) `styles/expression.css` (unlayered, `[data-theme]`-scoped) gives themes surface voice:
Ugentx scanlines, Blackhawk dot-grid + bezel, Cosmic explicit glass blur; chat-voice depth
(gradient titles, prompt prefixes) deliberately parked. (7) Projects hub rebuilt on the
primitives (IconButton/Button/Badge/SegmentedControl, eyebrow labels, accent-tint selected rail
rows, kit dropzone); remaining bare-background buttons swept in ambassador/AvatarPicker/
CitationElement (`AvatarPicker`'s local `IconButton` render-fn renamed `AvatarCell`).

**Projects v1 (v0.21.149).** Workspaces surface to the user as **Projects** (Claude-Projects-style;
internal naming stays `workspace`). Alembic `0005_workspace_projects` adds `workspaces.description`
(cap 500) + `workspaces.instructions` (cap 8000; both enforced with a 400 at PATCH) and the
**`workspace_conversations`** membership table (`conversation_id UUID PRIMARY KEY` → one project per
conversation; `workspace_id` FK `ON DELETE CASCADE`; no FK to `conversation_logs` — membership can
precede/outlive log rows). Repository: `set_description`/`set_instructions`,
`link_conversation` (upsert that *moves*; `DO UPDATE … WHERE IS DISTINCT FROM` avoids a dead tuple
per chat turn; **refuses `ws_home`** — the client auto-attaches Home for generated media, and an
unguarded upsert would steal conversations out of real projects), `unlink_conversation`,
`get_conversation_workspace`, `delete_conversation_links` (called from conversation delete).
**Turn precedence** (`views._resolve_turn_workspace`): explicit request `workspace_id` wins and
self-heals membership (idempotent link) > stored membership (server resolves it and re-emits the
existing `workspace_attached` SSE event so the client badge re-syncs) > none. This also fixes the
pre-session orphan: meta written under `tab.id` migrates to the session id on first assignment
(client `useTabMessages.setSessionId` → `conversationMeta.migrateMeta`). **Instructions injection**:
`retrieval.render_instructions_block` (defensive re-truncation at 8000) rides
`_append_corpus_awareness_blocks` as a stable `workspace_instructions` `LedgerBlock` at **priority 88**
— above the manifest's 85, so user guidance outlives the file catalog under budget pressure.
Membership API: `GET /workspaces/{id}/conversations` (reuses `views._conversation_summaries`, the
extracted `/api/conversations` query which now LEFT JOINs membership → rows carry `workspace_id`),
`PUT`/`DELETE /workspaces/{id}/conversations/{conv_id}` (PUT 400s for non-UUID or `ws_home`).
Client: the Workspaces drawer became the **Projects hub** (`WorkspacesPanel.tsx` — description +
debounced-autosave instructions editors, files pane, project conversations list + "New chat in this
project"; Home renders as a fixed personal-media entry with files only); sidebar **project sections**
(`useConversationList` partition `archived > project > group > pinned > open/past`, sections from
server membership with meta fallback) + a "Move to project" row menu; one-time localStorage→server
membership sync (`lib/projectSync.ts`, per-server done-flag, runs before the history fetch).

**Project memory channels + agent-facing terminology (v0.21.150).** A conversation in a project
now stores/recalls on **`_project_{workspace_id}`** — the "project" tier of the INV-7 scope
hierarchy, realized ahead of the Roadmap §1.4 `ChannelRef` sweep (which will type it; the wire
format here is what `kind="project"` will parse). Resolution
(`views._resolve_project_channel_workspace`): explicit request `workspace_id`, else durable
membership via the request's `session_id` (== conversation id); applied where the workflow
override lives, with precedence **workflow shared channel > project channel > profile channel**;
`ws_home` never scopes; opt-out `memory.project_channels` (default on); best-effort — scoping
can never fail a turn. Recall keeps the INV-7 set `[_project_X, _self_{agent}, _global]`
automatically (project = active channel), and `promote_to_global` lifts durable facts out as
usual. The ad-hoc Alloy executor inherits the project channel, so delegated specialists record
into the project too. **Agents now speak "project"**: the manifest ledger block header
("Project files — this conversation belongs to a project…"), the
`workspace_search`/`document_query`/`read_document`/`view_image` + shell tool descriptions and
their no-workspace errors all use project terminology (tool *names* unchanged — the wire API is
stable). **Ingestion robustness follow-through**: `apps.py ready()` now launches a delayed
`workspace-ingest-sweep` daemon that runs `ingest_pending_documents()` on startup — a restart
(deploys, the dev autoreloader) kills in-flight ingestion threads, and stranded `pending`
uploads previously stayed stuck forever. Documents also embed **one at a time**
(`ingestion._EMBED_LOCK`, slices of 16, 600s background timeout): a CPU embedder is serial
anyway, and concurrent uploads round-robining slices multiplied every slice's queue wait past
any sane timeout (observed with 4 simultaneous PDFs).

**Agent-writable project documents + project prompting (v0.21.161).** Agents can now
*manipulate* project files, not just read them. **Write path** (`service.py`):
`create_text_document` / `update_text_document` — shared by the REST endpoints
(`POST /workspaces/{id}/documents/text`, `PUT /workspaces/{id}/documents/{doc}/text`) and two new
internal tools **`create_document`** / **`update_document`** (advertised by default; opt-out
`workspace_agent_write_tools`; agent-writable extensions capped to
`workspace_agent_writable_extensions` = md/markdown/txt). Semantics: create refuses filename
collisions with a typed `conflict` (409 + existing `document_id` — update is an explicit act, never
a silent overwrite); update is full-content replace with an optional `expected_sha256` ETag
(hub editor sends it; agent tool is last-write-wins), a same-sha **no-op short-circuit**, quota math
`used − old + new`, and re-ingestion (`status→pending`, chunks/tags/summary refresh for free).
**Blob refcounting** (`release_blob_if_unreferenced`): content-addressing means two doc rows can
share one blob — update *and* the document-delete path (previously deleted unconditionally: a real
dedup bug) only remove the file when no other row references the `storage_key`. **Stale-ingest
guard** (`ingestion._superseded`): rapid successive updates race daemon threads; ingest re-checks
the row's `sha256` before writing chunks/status and bails if a newer update won. **Project
prompting** (the "agent doesn't know what a project is" fix — three stacked gaps): (1) new builtin
prompt layer **`project-collaboration`** (`prompts/layers.py`, order 25) teaches the concept, the
tool vocabulary, "create durable things as project documents, keep them current", and
prefer-native-over-filesystem-MCP; (2) an **always-on `project_identity` ledger block** (priority
90 > instructions 88 > manifest 85; `retrieval.render_project_identity_block`) so even an *empty*
project announces itself (name, description, doc count, how to add files) — previously an empty
project was invisible; (3) the model-visible rename is finished: **`workspace_search` →
`project_search`** with `_TOOL_ALIASES` (legacy name still resolves/executes for old conversations
+ procedural records; `legacy_names_for` keeps per-profile `_internal.workspace_search`
allow/block entries matching in `Agent._get_tools_for_provider`), and shell tool text now says
"temporary shell working copy — NOT project documents; use create_document" (the exact confusion
that sent an agent to a filesystem MCP). Tests: `WorkspaceWriteServiceTest`,
`WorkspaceTextEndpointTest`, `ProjectPromptingTest`, `ToolGatingTest.test_legacy_alias…`;
`rag_e2e.py` §7 drives create→409→ready→update→refcount→ETag-409 + both tools end-to-end.

**Anthropic system-prompt drop (fixed, v0.21.150).** `anthropic_provider._convert_messages`
assigned `system_prompt = msg.content` per SYSTEM message — each one *overwrote* the last. A
ledger-assembled turn carries several (base prompt → project instructions → memory blocks →
token-budget header appended last), so **Anthropic models received only the budget header** as
their system prompt; everything else was silently dropped. Now all SYSTEM messages join with
`\n\n` into the single `system` param (`AnthropicSystemPromptJoinTest`). Verified live: project
instructions are followed by claude-haiku only after this fix.

**Binary media + serving (v0.21.123).** The blob store also holds non-text media (images): `service.store_media`
mirrors `upload_document` but validates an image content-type (png/jpeg/webp/gif), **skips ingestion**
(no parse/chunk/embed), and writes the doc `status="ready"`. **`GET /api/workspaces/{id}/documents/{doc}/raw`**
(`workspace_views.workspace_document_raw`) serves the stored bytes with their content-type — the **stable image
URL** (clients fetch it via the authed API client and object-URL it, so it works under auth). A reserved,
visible **"Home"** workspace (`repository.ensure_home_workspace`, id `ws_home`) is the user's personal space —
generated avatars land there under an `avatars/` filename prefix (a flat-list naming convention; `temp/` reserved
for scratch). This is the image-transport foundation for the broader multi-modal pipeline.

**Avatar generation (v0.21.124).** `POST /api/agent/avatar/generate {subject_prompt, agent_profile_id?, style_prompt?, model?}`
(`views.avatar_generate`) composes the app-level **style** prompt (`config.images.avatar_style_prompt`, Settings →
Images) + the per-request **subject** prompt, resolves the image model (`config.images.default_model`, default
`flux.2-klein-4b`) via `registry.resolve_with_fallback`, calls `provider.generate_image`, stores the bytes in **Home**
via `store_media`, records cost (`usage_ledger` source `image`, `pricing.estimate_image_cost`), and returns the served
`…/raw` URL. Degrades to 422 (disabled / unconfigured / non-image model). Client: `AvatarPicker`'s Generate tab calls
it and sets `profile.avatar = media:{ws}/{doc}`; `common/AgentAvatar` renders image avatars (resolved to an authed
object URL via `lib/avatarImage.ts`, mirroring the TTS blob pattern) and falls back to the lucide icon. Config defaults
that the feature needs (`DEFAULT_IMAGE_MODEL`/`DEFAULT_AVATAR_STYLE_PROMPT`) are module constants in `config.py` —
`_load` doesn't merge new keys into a pre-existing `config.json`, so callers fall back to the constants.

**Image generation in a conversation (v0.21.127).** A `generate_image` **internal tool**
(`mcp/internal_tools.py`, `@register_tool`) lets an agent make an image mid-chat: it self-gates on
`config.images.enabled`, resolves the image model, bridges the async `provider.generate_image` from the
sync tool via `utils/async_bridge.run_coro_sync`, stores the bytes via `store_media` into the
conversation's attached workspace (else the user's **Home**) under a `generated/` prefix, records cost
(source `image`), and returns `{success, url, prompt}` (never the bytes — vision input is separate). The
result renders inline as a new **`image` exhibit**: `streaming/exhibits.py` adds an `ImageElement`
(`{type:"image", url, alt}`) + `image_exhibit_from_generate`; `tool_loop.py`'s `_emit_image_exhibit(tm)`
auto-emits it (parsing the tool result, mirroring the citation auto-capture) so the chat shows the picture
without the model re-describing it. Client: `lib/exhibits.ts` gains the `image` element; the
`components/chat/exhibits/ImageElement` renderer (registered in `elementRegistry`) resolves the served-blob
URL to an authed object URL via `lib/mediaImage.ts::resolveMediaImage` (the shared helper avatars also use).

**Direct mode (v0.21.129).** A per-profile `AgentProfile.direct_mode` (mirrored on `AgentConfig`) makes a
turn **bare**: the model receives only the user message — no system prompt, no memory (core + recall), no
tools, no token-budget header, no prior history. The chat-stream generator (`views.agent_chat_stream`)
resolves it via the module helper `_resolve_direct_mode(agent.config, caps)`, which returns
`(direct_mode, image_only)` — on when the profile flag is set **or** the resolved model is *image-only*
(`caps.output_modalities` has `image` but not `text`, e.g. flux), so an image model that can't act on a
harness can't be misconfigured. Direct mode is one consolidated `if`-branch over the ledger/tools/budget
section (kept to a single decision point so the giant generator stays within pyright's path-analysis
budget; the delegation-tool descriptor moved to the `_resolve_delegation_tool` helper for the same reason).
**Separately**, the chat-stream path now honors the profile's own `enable_memory` (it previously read only
the request-level `use_memory`, so a memory-off profile still got recall). Client: a "Direct mode" card in
the profile editor's Advanced tab (`ProfileContent.tsx`), locked-on + info-noted when the selected model is
image-only.

**Image-output models in conversation (v0.21.131).** When a **direct-mode** agent's model can
*output images* (`caps.output_modalities` has `image` — flux, gemini-flash-image), a chat turn
makes a *picture*, not text: its completion carries the image in `message.images`, which the
streaming text loop ignores (→ "empty completion"). `views.agent_chat_stream` detects this via
`_model_outputs_image(provider, model_id, caps)` (warms the catalog once when caps look cold,
mirroring `core._model_supports_tools`; `False` on any probe error) and, instead of
`streaming_tool_loop`, runs `_run_image_generation` → the shared `agent/image_gen.py::
generate_and_store_image` (proven non-streaming `provider.generate_image` → `store_media` into the
attached workspace else **Home**, under `generated/`, + `record_usage(source="image")`). The result
is emitted as an **`image` exhibit** and **persisted as a synthetic `present_exhibit` tool turn** so
the existing reload path (`mapServerMessages` `present_exhibit` branch) rebuilds it (exhibits have no
persistence of their own). The same `generate_and_store_image` helper backs the `generate_image`
**tool** (DRY). Gated on direct mode so a text+image model still chats normally; non-direct
inline-image capture in the streaming path is out of scope. Planning is extracted to
`_compose_plan_if_complex` (skipped for an image turn) to keep the generator within pyright's
path-analysis budget — same discipline as `_resolve_direct_mode`/`_resolve_delegation_tool`.

**Vision input — image *input* (v0.21.134).** The reverse of image-output: a user attaches an image a
vision-capable model *sees*. Client uploads the file via `POST /api/agent/chat/images`
(`views.chat_image_upload` → `store_media` into **Home** under `uploads/`, no ingestion), gets back a
**ref** `{workspace_id, doc_id, media_type}`, and sends refs in the chat body's `images[]` (not bytes — the
endpoint stays pure-JSON). One canonical ref shape rides the wire → `providers.base.Message.images:
list[ImageRef]` → the persisted user-turn `metadata.images`. The two provider chokepoints build image blocks
**only when `msg.images`** (text-only stays byte-identical): `convert_messages_to_openai_format` emits
`{type:"image_url", image_url:{url: data-URI}}` (OpenAI/OpenRouter/Vercel), Anthropic `_convert_messages`
emits `{type:"image", source:{base64}}`. The data is inlined as **base64** via the shared
`base.resolve_image_data(ref)` (`read_blob` + b64; never raises — a missing/mismatched/stale ref is logged +
dropped) because the blob URL is auth-gated and an external provider can't fetch it. **Gating:**
`_model_accepts_vision(provider, model_id, caps)` (`caps.supports_vision` or `input_modalities` has `image`;
warms a cold catalog, mirroring `_model_outputs_image`) — a non-vision model (or `config.vision.enabled` off)
has its images **stripped** from the outgoing list (incl. a defensive non-mutating strip over history, so a
mid-conversation model switch can't leak blocks) with a `status` notice, while the refs still **persist +
render** on the bubble. **Persistence/reload:** `build_user_turn(..., metadata={"images":[...]})` →
`conversation_logs.metadata`; `conversations_messages` already returns metadata → `mapServerMessages` maps it
to `UserMessage.images`. **Multi-turn re-feed (bounded):** `conversation_history._default_reader` now returns
`(role, content, metadata)` and `load_recent_turns` repopulates `Message.images` for only the most-recent
**K** image-bearing user turns (`config.vision.refeed_recent_turns`, default 2) — re-feeding base64 every
turn is expensive, and a ref-only Message under-estimates the ledger budget, so the K-cap is the real guard.
Back-compat: the reader tolerates 2-tuple custom readers (ambassador, tests). Client: `MessageImages`
(thumbnails via `resolveMediaImage`), image attach via the **Relay menu** (`relay/RelayMenu.tsx` "Attach
image" — the standalone composer image button was folded in, v0.21.188) plus **paste-to-attach** on the
composer textarea (click-to-browse + paste only — Tauri HTML5 drag-drop is broken), a pre-warning when the
active model lacks `supports_vision` (shared `fetchModelsOnce`), and a Settings → Images "vision input"
opt-out. Out of scope: Alloy/ambassador vision (no composer entry), backend downscale.

**On-demand image viewing — `view_image` (v0.21.135).** Vision input above only covers *composer*
attachments; this lets an agent *see* an image already in the conversation/workspace (a generated image,
or a `.png`/`.jpg` a user uploaded) — `read_document` returns text only, so without this the agent could
list `generated/*.jpg` but never view it. The `view_image(document_id)` internal tool
(`mcp/internal_tools.py`) validates the doc is an image and is in the **attached workspace or the user's
Home** (where generated images land — no cross-workspace peeking), and returns the ref; it deliberately
returns *no pixels*. The streaming tool loop (`streaming/tool_loop.py::_view_image_messages`, invoked from
`_execute_and_emit_tools` with `vision_capable` threaded from the view's `accepts_vision`) turns a successful
result into a **user-role `Message` carrying the image block** for the *next* round (reusing the vision-input
converters) — or, on a non-vision model, a short "can't show it" note instead. **Awareness (so the agent
knows to look, never auto-shoved):** `retrieval.render_manifest_block` marks image files 🖼 with their
`document_id` + a `view_image` hint; `conversation_history.render_conversation_images_block` (backed by
`list_conversation_images`, which scans the conversation's persisted `generate_image` tool-results) injects a
prio-60 droppable ledger block cataloguing images made earlier in the conversation. Both are folded into
`views._append_corpus_awareness_blocks` (with the workspace manifest) to keep the chat-stream generator within
pyright's path budget. Key principle (per product direction): images are surfaced as a **text catalog** the
agent reads and chooses from — pixels enter context only on an explicit `view_image` call, never every turn.

### Agent Shells (`kit/shell/`, v0.21.108)

> **Index card (ex-CLAUDE.md):** two backends via `dispatch.py` (per-workspace `shell_backend`):
> `bubblewrap` (default, locked-down jail) or `container` (`container.py` — persistent
> per-workspace Docker container, installs + network, via `docker` CLI → dind sidecar in prod;
> status/lifecycle at `/api/workspaces/{id}/shell/container`). `sandbox.py` `BubblewrapSandbox`
> jail — network off, FS=work dir only, env scrubbed; bare subprocess only behind
> `allow_unsandboxed`. `workdir.py` materializes the workspace into `data/shell/{ws}/{conv}/`
> (+ GC + path-jail); `policy.py` deny-list; tools `run_command`/`write_file`/`read_file`/
> `list_files` in `mcp/internal_tools.py`.

**Opt-in per-workspace** (`workspaces.allow_shell`, **off by default** — LLM-driven arbitrary code
execution is a deliberate exception to "experimental ships ON"; enablement lives on the *workspace*, not a
global flag). Lets an agent run commands against a **sandboxed working copy of the attached workspace**. **Threat model:** the LLM is the actor and can be prompt-injected (malicious doc
/ web result / MCP output) → the "lethal trifecta" (secrets in `data/`, untrusted instructions, network
egress). So v1 runs every command in a **bubblewrap jail**: `--unshare-all` (network OFF; `--share-net`
only when `shell.allow_network`), `--clearenv` + scrubbed `minimal_env` (no API keys/passwords),
`--new-session` (blocks TIOCSTI), read-only merged-usr rootfs, and **only the conversation work dir bound
rw** — so it can't read `data/config.json` or reach the network. `Sandbox` is a protocol
(`sandbox.py`); `BubblewrapSandbox` is default, `LocalSubprocessSandbox` is used ONLY behind
`shell.allow_unsandboxed` when bwrap is missing (else `run_command` errors — fails safe). `get_sandbox`
probes once that bwrap+bind-set actually run a command. Execution is synchronous + timeout-bounded (ADR-1),
same shape as web_search. **Work dir:** `${AGENTX_DB_DIR:-./data}/shell/{workspace_id|_scratch}/{conversation_id}/`,
materialized from the workspace's ready docs (`filename → bytes`, idempotent via a manifest marker, size-capped
by `shell.max_materialize_bytes`); agent edits persist across commands in the conversation (not synced back to
blobs in v1); stale dirs GC'd by mtime (`shell.workdir_cleanup_days`). Tools (`mcp/internal_tools.py`,
gated out of `get_internal_tools` unless the turn's attached workspace has `allow_shell=true` — resolved
from `InternalToolContext.workspace_id`; runtime-rechecked in the tools too — and still subject to
per-profile `allowed_tools`): `run_command` (jailed `sh -lc`), plus path-jailed `write_file`/`read_file`/`list_files`
(safe structured subset, no subprocess). `policy.py` deny-list is secondary defense (the jail is primary).
Every command is audit-logged (cmd, cwd, workspace/conversation ids, exit, timed_out, sandbox). `bubblewrap`
is installed in the API `Dockerfile`. Self-driven harness: `scripts/shell_e2e.py`.

**Backends (v2, per-workspace `shell_backend`):** the shell tools route through `kit/shell/dispatch.py`
to one of two backends chosen on the workspace — **`bubblewrap`** (default; the locked-down jail above) or
**`container`** (`kit/shell/container.py`): a **persistent per-workspace Docker container** the agent can
`pip`/`apt`-install into, with **network on** (its own bridge — internet yes, AgentX DBs/secrets no). The
container is clean (no host mounts, only its `/workspace` volume), so network+root inside are safe; files
move in via `docker cp` (bind paths don't cross the dind boundary). The API drives Docker via the `docker`
**CLI** against `DOCKER_HOST` — **dev** uses the host daemon; **prod** uses a **dind sidecar**
(`docker-compose.shell.yml`; the host socket is never mounted). Lifecycle: lazy `ensure_container` (image
**pre-pulled** on backend-select so first use doesn't block; `provisioning` state otherwise), persistent
named volume, idle-GC + removal on workspace delete; **status/stats** (`docker stats`/`ps -s`) +
`start`/`stop`/`reset`(drop installs, keep files)/`remove` exposed at `/api/workspaces/{id}/shell/container`
for the UI resource card. Config `shell.docker.*` (off unless enabled + a daemon reachable). Self-driven
harness: `scripts/shell_container_e2e.py`. **Weight:** ~1 dind sidecar per cluster when enabled (opt-in;
bwrap stays free) — see `todo/backlog`/plan for the cluster cost note.

### Ambassador (`agent/ambassador.py` + `ambassador_storage.py`, Phase 16.6)

A dedicated agent running *parallel* to a conversation that briefs the user on a turn **without polluting the main transcript** (the load-bearing invariant). `AmbassadorService.brief_turn` resolves the default ambassador (`get_default_ambassador()`; an `AgentProfile` with `kind='ambassador'`, hidden from chat). The profile's `system_prompt` is the personality voice; functional personas come from `AmbassadorConfig.{briefing,qa,draft}_persona` overrides ?? code defaults. Grounds **read-only** via `load_recent_turns`, runs `resolve_with_fallback` + token-streams via `provider.stream` (never raises; cancel settles sidecar to `cancelled`), emits namespaced `ambassador_*` SSE. The persona **speaks TO the reader** (second person, names the agent). The briefing grounds on the turn's substance via client-gathered `artifacts` (tool calls / sources / exhibits, `lib/ambassadorTurn.ts::gatherTurnContext`). Free-form questions (`answer_question`) and spoken questions (`route_voice_command`'s `answer` action) share **one agentic answer core** (`_agentic_answer`): a single streaming `provider.stream` loop that offers a **read-only tool belt** (`agent/ambassador_tools.py`: `summarize_conversation`/`explore_conversation`/`read_conversation`/`list_conversations` — SELECT-only — plus `survey_conversations`, a digest-rich cross-conversation view (each session's own rolling summary via `conversation_summary_storage.get_summary` when present, else the first/last snippet, plus a best-effort `goals:` line per conversation via `AgentMemory.get_goals_for_conversation` — goals are now stamped with their `conversation_id` on the `(:Goal)` node, Neo4j index `goal_conversation`/migration `0004`; the goal read hits Neo4j so it's wrapped to degrade to no line when Neo4j is down/disabled; model-free — the ambassador synthesizes the app-wide summary itself), and `list_agents`, the **agent roster** (`kind=='agent'` profiles only) surfacing each agent's role tags, delegation availability, role blurb, and its model's **live provider capabilities** (input/output modalities + tools/vision/speech/transcription flags, via `registry.resolve_with_fallback` → `provider.get_capabilities`, degrading **per-agent**) — the input for multi-modal routing; `execute_tool` never raises) — a round carrying `tool_calls` runs the tool (emits `ambassador_tool_call`/`_result` SSE) and loops; a content-only round **is** the streamed answer. Bounded by `_MAX_TOOL_ROUNDS` (last round forces tools off); never-raises, degrading to a grounded one-shot. **Tools are always on** (no gate — experimental app). **Grounding is tool-first**: the prompt pre-loads only `_LEAN_GROUNDING_TURNS` recent turns, so the ambassador *reads* the conversation (or surveys others via `list_conversations` → `read_conversation`) for real depth — this is why the tools actually fire. **Conversational memory**: prior settled Q&A is fed back as dialogue turns (`_thread_history`, read-only over `qa:`) so follow-ups have context (text **and** voice — the ambassador's *own conversation*, never mixed into the main transcript). Provider resolution is DRY'd through `_resolve_answerer`; the answer persona (`_build_answer_persona` + `_answer_persona` + `_TOOLS_NOTE`) states its capabilities so "what can you do?" is answered right. Tool reads name **each conversation by its own producing agent** (`conversation_history.load_recent_labeled_turns` / the survey's `agents` column read `metadata->>'agent_name'`), so a cross-conversation survey never mislabels one session's agent with the active one. The client panel header shows the **ambassador profile's own name + avatar** (`AmbassadorPanel`'s `AmbassadorMark` takes the profile `avatar`), since the ambassador is a customizable profile. The panel's **focus is independent of the chat tab** (`focusedConversationId`, snapshotted from the active tab on open, then "stays put"; an `AmbassadorConversationSwitcher` over the open tabs + a "current conversation" jump change it) — and it's told the **active conversation** as ambient context (`active_conversation` {id,title} on ask/voice → `_active_conversation_note`), so it knows where the person is *now* even when focused elsewhere. **Loading a conversation is pure display** — `AmbassadorPanel`'s voice auto-speak fires only for items seen `streaming` *this session* (`locallyStreamedRef`), so switching/reopening never re-synthesizes TTS or speaks history (the earlier bug billed for it). The tool belt's "this conversation" is the **focused** one (`execute_tool(focused_conversation_id=…)`); the answer persona is **multi-agent** (names each agent from its conversation, not one global "active agent"). `logging_kit/async_noise.py::install` (called from the client-abort-prone ambassador async views) downgrades Python 3.14's benign `CancelledError exception in shielded future` (a client aborting an in-flight TTS/SSE request) to debug while delegating all other loop errors. Also does **outbound relay** (`draft_relay_message`: shapes a rough intent into a first-person message the user reviews and relays into the conversation as a real **user** turn via `ConversationContext.relayToConversation` — the ambassador never speaks into the transcript as itself). Relay reaches **any** conversation, not just the open tab: the client `planRelay` (`lib/ambassadorRelay.ts`) routes to the live in-tab handler when the target is the **active** tab (steer-aware, instant) and otherwise to **`POST /api/agent/ambassador/relay`** `{conversation_id, text}`, which runs the target conversation's agent **headless** via `enqueue_background_chat(session_id=…, agent_profile_id=…)` (profile recovered from the conversation's last stamped turn via `conversation_history.latest_agent_name` → `get_profile_by_name`, else the default) — appending the user turn + the agent's reply to the durable history. The deck (no focused conversation) picks a target from `listConversations`. Output persists ONLY to the Redis **sidecar** — never `conversation_logs`/`conv_summary:`. **Slice 1b** unified briefings + Q&A into one ordered **thread** ("Inquiry") under the `amb_thread:{thread_id}` family (entry-oriented — one record per ambassador turn carrying its optional `question`; ordered by `created_at`; `thread_id` defaults to the conversation id) carrying its own `title`. The pre-1b per-family public API (`create_qa`/`set_qa_answer`/`set_summary`/`get_qa`/`list_briefings`/…) is preserved as **thin projections** over the entry store, and legacy `ambassador:` records still replay (one-place fold in `list_thread`). Tool-call chips are **persisted on the entry** (`set_entry_tool_calls`, called inside `_agentic_answer`) so they survive a reload. `GET/PATCH /api/agent/ambassador/thread/{thread_id}` replays (`thread_payload` → `{thread_id, title, entries}`) / renames (`set_thread_title`); the `{conversation_id}` endpoint is a back-compat shim. A brand-new Inquiry **auto-titles from its first question** (`AmbassadorService._maybe_autotitle` → `ambassador_storage.derive_title`, model-free, text + voice); a `title_auto` flag on the meta guards a manual rename from being overwritten. **Multiple standalone Inquiries** (the command deck) are tracked by a per-user registry — a Redis ZSET `amb_user:{user}:threads` (`register_thread`/`unregister_thread`/`list_user_threads`, self-healing on aged-out meta) written from the views (storage has no user dimension). `GET/POST /api/agent/ambassador/threads` list/mint standalone Inquiries (`deck:{user}` home thread pinned + minted `inq:{user}:{uuid}`); PATCH/DELETE on `/thread/{id}` also register/unregister. The ambassador can retitle **its own** current Inquiry via the `rename_inquiry` tool — the tool belt's **lone write**, self-scoped to its thread meta (never `conversation_logs`). Client: `AmbassadorContext.threadFor` renders one **Inquiry** stream (briefings as their own turns via `BriefingItem`), with `titleFor`/`renameThread`/`clearThread` driving the switcher rename + a `⋯` menu (brief / rename / clear). Runs ride detached-run infra via `start_chat_run(indexed=False)`. **The ambassador is a parallel operator, not a turn-by-turn briefer** (the UI is deliberately *boringly stable*): `AmbassadorPanel` is **de-coupled from turns** — no Turns strip. The panel is conversation-level — **"Brief this conversation"** (`briefConversation` → an `ask` with a summary intent) + starter chips + free-form ask/relay; the ambassador scopes to conversations through its read-only tools (it never receives per-turn `artifacts`). Per-turn work is **CC**: the chat's `MessageActions` button (`ChatPanel.handleAmbassador`) forwards a turn *into* the Inquiry as an `ask` ("brief me on this turn: …") — like an email into the thread — and opens the panel. The header is **one stable compact command bar in both modes** (voice only adds a subtle accent tint — never a layout morph, so the Voice/Text toggle never relocates and can't strand you); the body **auto-scrolls** (`useStickyScroll` + jump-to-latest); answers carry **copy** + relative timestamps. **Voice and text share one body** — the Inquiry stream *is* the transcript (a spoken question persists as a `qa:` entry), so only the footer differs: the text composer ↔ `components/ambassador/VoiceBar.tsx` (PTT mic + `voiceCommand` → spoken answer/relay-confirm + settings popover). The old full-screen `VoiceSurface` (orb + separate caption log) and `lib/voiceCaptions.ts` were retired. **Voice (TTS):** `AmbassadorService.synthesize` speaks a briefing/answer via the profile's `AmbassadorConfig.{voice_mode,speech_model,voice,speech_speed}` block — `ModelProvider.synthesize_speech` (only `OpenRouterProvider` implements it, via OpenAI-compatible `/audio/speech` → MP3 bytes; base raises `NotImplementedError`); resolved **strictly** (no chat fallback) with precedence override → profile → `config.ambassador.*` → shipped default `openrouter:microsoft/mai-voice-2`, degrading to a typed `SpeechUnavailable` (→ `422`) when unconfigured. The client plays it through a framework-agnostic `lib/audio.ts::SpeechPlayer` singleton (blob-URL cache + playback queue + autoplay-unlock) behind `hooks/useSpeech.ts`; `AmbassadorPanel` adds per-item speaker buttons; a voice-enabled ambassador (`ambassador.voice_mode`) **leads with a Voice tab** ([Voice | Text]) — the immersive `components/ambassador/VoiceSurface.tsx` (Discord-call layout: PTT mic + captions + settings popover). **STT (voice input):** `AmbassadorService.transcribe` → `ModelProvider.transcribe_speech` (OpenRouter `/audio/transcriptions`, base64; default `openrouter:openai/whisper-1`); client `lib/audioRecorder.ts` captures **raw PCM via Web Audio → WAV** (NOT MediaRecorder — webkit2gtk's is broken) behind `hooks/useDictation.ts`. **Voice command routing:** a transcript → `route_voice_command` first **classifies intent** (`_voice_command_persona`, strict JSON `{action,text}`, defensive parse → `answer` fallback); an `answer` is then produced by the shared agentic core (`_answer_to_text` → `_agentic_answer`, so spoken questions get the **same tools + continuity** as typed ones, persisted `qa:`), while a `relay` returns the drafted message the user confirms in a strip and sends via `ConversationContext.relayToConversation` (never auto-sent). VoiceSurface: hold-to-talk default + tap-to-toggle (localStorage `agentx:voice:pttMode`), barge-in (speak stops on record), instant transcript echo, captions both ways, toggle max-duration safety. Per-model **voice dropdowns** via `lib/voiceCatalog.ts` + `components/common/VoicePicker.tsx` (curated set + free-text). Tauri mic perms: CSP `media-src`, macOS `Info.plist`/`entitlements.plist`, Linux webkit2gtk hook `src-tauri/src/lib.rs::enable_webview_microphone` (enables `enable-media-stream` + auto-grants user-media). **Command Deck (Slice 4, `0.21.116`):** the same `AmbassadorPanel` also mounts as a **standalone, app-wide full-screen surface** (`SURFACES.ambassadorDeck` → `stubs.AmbassadorDeckContent`, opened from ⌘K / TopBar) in a conversation-less **deck mode** — pass a `deckThreadId` prop and the panel binds to a single persistent per-user thread (`lib/ambassadorDeck.ts::deckThreadId` → `deck:{user_id|default}`) instead of a conversation. The minted id rides the existing `conversation_id` seam as the thread key, so **no backend change**: ask/voice/thread all accept it, grounding/`_thread_history` degrade to empty (no real conversation), and the conversation-agnostic tools (`survey_conversations`/`list_agents`/`list_conversations`) carry it; the deck never writes `conversation_logs`, so it can't appear in its own survey. In deck mode the conversation-coupled affordances are gated off (`isDeck`): brief-this-conversation, the relay mode + composer (relay needs an open tab — voice/text relay return a friendly note), and the conversation switcher (a plain renamable title stands in); starter chips become `DECK_STARTERS` (survey + roster). **Deferred:** multiple named standalone Inquiries (needs a per-user thread registry + `GET /ambassador/threads`) and deck→conversation relay (needs the deferred `POST /ambassador/relay`). **Deferred** (Todo §16.6): floating CC sticky player, recording persistence/history/GC, streaming answers, cross-agent delegation (relay JSON is target-extensible). **Aide swarm (Slice, `0.21.137`):** so a cross-conversation survey doesn't pull raw transcripts into the ambassador's own context, the read tools fan out to cheap **aides** (`agent/aide_swarm.py::AideService`, mirroring `ToolOutputCompressor`) — each condenses ONE conversation read-only into a short digest (map-reduce: aides *map*, the ambassador *reduces*). `survey_conversations` digests only **un-summarized** conversations (summarized ones keep their zero-extra-cost rolling summary); `summarize_conversation`/`explore_conversation` return a digest instead of dumping the transcript; `read_conversation` stays raw. Bounded (`asyncio.Semaphore(max_parallel)` + per-aide `asyncio.wait_for` + `max_per_survey`), **never-raise** (a bad/timed-out aide → snippet/raw fallback), and OFF (`ambassador.aide.enabled`) ⇒ today's behavior. Digests are cached in the sidecar (`ambassador_storage.get/set_aide_digest`, key `amb_aide:{cid}:{focus}`, fingerprinted on message_count+last_at so a grown conversation re-digests — never `conv_summary:`, so INV-2 holds). Cheap tier defaults to the haiku floor; spend is metered under usage source `aide`. **Dispatch (write-side, Slice, `0.21.138`):** the ambassador can **hand a task to a chosen worker** — `POST /api/agent/ambassador/dispatch` `{agent_id, text}` resolves the worker via `get_profile_by_agent_id` (agents only → an unknown/ambassador id is `400`), mints a **brand-new** `conversation_id` (uuid), and runs that worker **headless** on the task as its first **user** turn via `enqueue_background_chat(session_id=<new>, agent_profile_id=…)` — so INV-2 holds (you authored it; the ambassador writes nothing as itself). Returns `{ok, conversation_id, job_id}`. The draft seam grows a `fresh` flag (`draft_relay_message(fresh=True)`) framing a **self-contained task** for a worker to start cold (no grounding). Client: a **Dispatch** composer mode in `AmbassadorPanel` (worker picker over `AgentProfileContext` agents, Refine, send) — v1 is confirm-first. Since the worker runs async, the new conversation 404s until its first turn lands, so the client **polls `getConversationMessages` then `restoreConversation`** (open-on-ready). Gated by `config.ambassador.dispatch.enabled` (default on; Settings → Ambassador opt-out). **Deferred:** autonomous dispatch (the ambassador picking the worker + `AlloyExecutor.delegate` start/steer), dispatch into an existing conversation, instant task echo.

### MCP Remote OAuth (`mcp/oauth_storage.py` + `mcp/oauth_flow.py`, v0.21.163)

Remote MCP servers (`sse`/`streamable_http`) can require **OAuth 2.1**. We wire the python-sdk's
built-in machinery (`mcp.client.auth.OAuthClientProvider` — RFC 9728 protected-resource-metadata
discovery, RFC 7591 dynamic client registration, PKCE, token refresh) into both transports
(`sse_client(..., auth=)` / `streamablehttp_client(..., auth=)`); the provider is built in
`MCPClientManager._build_oauth_provider` when the server config carries
`auth: {"type": "oauth", "scope"?, "client_id"?, "client_secret"?}` (`ServerConfig.auth`;
`${VAR}` values env-expand via `resolve_auth`). **Storage**: `FileTokenStorage` implements the
SDK's `TokenStorage` protocol over per-server JSON at `data/mcp_oauth/{server}.json` (0600,
atomic write) holding tokens + the dynamic registration; pre-registered `client_id`/`client_secret`
in the config seed `get_client_info` for providers without DCR (Google). **Interactive flow**
(`oauth_flow.py`): the SDK's `redirect_handler`/`callback_handler` run on the manager's background
loop; a per-server `PendingFlow` (single-flight, TTL 600s) holds an `asyncio.Future` — the redirect
handler parses the OAuth `state` from the authorization URL and indexes the flow by it; the Django
view `GET /api/mcp/oauth/callback` (PUBLIC — exempted in `AgentXAuthMiddleware.PUBLIC_ROUTES`, the
browser carries no token; strict state validation instead) resolves the future thread-safely.
**Connect state machine**: `manager.connect_interactive(name)` — stored-tokens-still-good connects
synchronously; consent-needed returns `{"status": "auth_required", "authorization_url"}` (endpoint
→ HTTP 202) while the connect task keeps running; on eventual success it persists `auto_connect`.
Headless paths (startup restore, `connect_all`) build the provider with raising handlers → a clear
"connect it from the Toolkit page" error instead of a hang. `POST /api/mcp/servers/{name}/auth/reset`
forgets tokens + registration. Server payloads carry `auth` + `auth_state`
(`authorized`/`pending`/`error` — `views._serialize_auth_state`; sticky per-server `last_error`).
Client: `ServerForm` auth section (None | OAuth 2.1, optional scope + pre-registered creds);
`ToolkitPage` opens the consent URL on 202, shows "waiting for authorization…" with a 2.5s poll
(5-min cap + cancel), an OAuth status line, and a "Reset auth" action. Redirect URI defaults to
`http://localhost:12319/api/mcp/oauth/callback`; override `AGENTX_OAUTH_REDIRECT_URL` (Tauri
deep-link for remote-API cloud mode is a Phase-19 consideration — backlog). Tests: `MCPOAuthTest`.
**Not yet**: `tools/list_changed` re-discovery and runtime `auto_reconnect` (backlog).

### Logging (`logging_kit/`)

Centralized logging. `configure_logging()` (called from `settings.py` with `LOGGING_CONFIG=None`) installs a root `QueueHandler` → `QueueListener` feeding console (rich color + category badge + run tag via `handler.py`/`highlighters.py`/`categories.py`), an in-memory `RingBufferHandler` (backs `/api/logs`), and a **daily-rotating** gzip `archive.py` file (`DailyArchiveHandler`, midnight/UTC → `agentx-YYYY-MM-DD.log.gz`). When auth is set up, completed days are **sealed** with envelope AES-256-GCM by `archive_crypto.py`: a random DEK (encrypts archives) wrapped by a Scrypt KEK from the login password, stored in `data/logs/keyring.json`. The DEK is unwrapped + cached in process memory on login (sealing is lazy — the hot path never needs the key; days that roll while locked stay redacted-plaintext `.gz` until the next `seal_pending`). Password change re-wraps the DEK (`rewrap_dek`, O(1)); `reencrypt_all` is the deep-rotation path. Retention (`prune_old`, default 30 days) is ours, not `backupCount`. Auth-disabled → plaintext-gzip fallback. Ops: `task logs:keys:status|seal|rotate-keys|rotate-keys:deep` (cmd `rotate_log_keys`). `context.py` stamps a per-turn `run_id` (reuses `streaming/status.py::current_run_id`); `redaction.py` scrubs secrets at capture; `llm_cards.py` renders compact/full LLM request logs; `banner.py` is the startup banner. All behavior is governed by `AGENTX_LOG_*` flags (`flags.py`; decorations on by default, off → historical plain output). The client mirrors categories in `lib/logCategories.ts`.

### Container boot (`docker/entrypoint.sh` + `manage.py bootstrap`)

The api image self-initializes through **one** Python process: `manage.py bootstrap` runs Django
migrations, the memory-PG Alembic upgrade **in-process** (`alembic.command.upgrade`; ini via
`AGENTX_ALEMBIC_INI` or walk-up discovery), a **memory-schema stamp fast path** (Neo4j
`_SchemaMeta.version` vs latest `queries/neo4j_migrations/NNNN_*`, vector indexes ONLINE, Redis
PING — full `init_memory_schema` only on a miss or `--full`), an embedding-**warmup signal**
(HF-cache probe via `huggingface_hub.try_to_load_from_cache` — never imports torch), and the auth
hint. Stdout contract (greppable, `BOOTSTRAP <phase>=<state>` … `BOOTSTRAP_RESULT ok|failed`; exit
2 = config error, no retry). Schema init itself is **model-free** (`--validate-embedder` opt-in);
the model download happens only when bootstrap reports `warmup=needed` — the entrypoint then runs
`warmup_embeddings` under a post-success watchdog (`AGENTX_INIT_EXIT_GRACE`, default 15s) that
reaps the known non-exiting download threads. Warm boots skip warmup entirely, so the container
reaches uvicorn in seconds. `agentx migrate` (in-image ops CLI) calls the same `bootstrap`
(previously it silently skipped Alembic). Disable everything with `AGENTX_AUTO_INIT=false`.

---

## Client Surface Map (`client/src/`)

> Moved from `CLAUDE.md` (which keeps only the drift-proof rules). Read the relevant bullet
> before touching a surface.

- **Chat page** (`pages/AgentXPage.tsx`) = left `ConversationSidebar` rail + `ChatPanel` (collapsible/resizable, presentation over `ConversationContext`). Shared list logic in `hooks/useConversationList.ts` + `components/chat/ConversationList.tsx`/`ConversationRow.tsx`, reused by the mobile Conversations drawer. Per-conversation meta (pin/archive/icon/color/group/bulk) in `lib/conversationMeta.ts` (reserves `workspaceId`/`fileRefs`).
- **Surfaces**: Settings, Tools, **Memory**, and the **Ambassador Command Deck** (`SURFACES.ambassadorDeck` — `AmbassadorPanel` in conversation-less `deckThreadId` mode; holds multiple named Inquiries via `AmbassadorInquirySwitcher` + the per-user registry; relay routes via `lib/ambassadorRelay.ts::planRelay` — live in-tab for the active tab, else headless `POST /ambassador/relay`) open as full-screen modals (`type:'modal', size:'full'`); Plans/Sources/**Projects** are right-side drawers.
- **Projects hub** (`components/workspaces/WorkspacesPanel.tsx`, surface `workspaces`; internal naming stays `workspace`): CRUD + upload + ingest status + description/instructions editors + the project's conversation list + "new chat in project"; Home is a fixed personal-media entry. **Document preview/editor (v0.21.162)**: file rows click-open `DocumentPreviewModal.tsx` — markdown renders through the shared chat renderer (`chat/MessageContent`), txt/code as mono text, images/PDFs from an authed object URL (`workspacesApi.fetchDocumentBlob`, sha-cache-busted since `/raw` content changes on edit; revoked on close); md/txt get an Edit mode with explicit Save/Cancel using the `expected_sha256` ETag (409 → "changed elsewhere" toast) and an **Export PDF** button (`window.print()` + a print-visibility stylesheet in `DocumentPreviewModal.css`; known no-op on macOS WKWebView — Linux/Windows/browser fine, Download is the fallback). A **"New document"** action creates a scaffolded `.md` via `POST /documents/text` and opens it in edit mode. Attach = `conversationMeta.workspaceId` (fast path; chat stream sends `workspace_id`) + durable server membership (`PUT /workspaces/{id}/conversations/{conv}`; the sidebar's project sections derive from it, with a one-time localStorage sync in `lib/projectSync.ts`); `lib/api/workspaces.ts`.
- **Memory Workbench** (`components/memory/MemoryWorkbench.tsx`, surface `memory`): full-screen immersive explorer — in `ModalPortal` `FULLSCREEN_SURFACES` + `SELF_CLOSING`, so it renders **bare** and owns its own backdrop + ESC/✕ close (the `UnifiedSettings` pattern; avoids the Chromium thin-strip bug). A horizontally-scrollable **top tab bar** (`MemoryArea` in `memory/types.ts`: Overview · Entities · Facts · Strategies · Procedures · Explore · History · Jobs) sits over a list+detail canvas. A header **channel picker** (`useMemoryChannels`) scopes every area (previously hardcoded `_all`); **Overview** (`OverviewPanel`) is the landing tab (totals + per-channel table over `useMemoryStats`); **Procedures** (`ProcedureListView`/`ProcedureDetail`, read-only) surfaces `useMemoryProcedures`; **Explore** (`MemoryGraphView`) is search/topic-driven with static (non-animated) edges + `onlyRenderVisibleElements` for smooth panning. The `EntityDetail`/`FactDetail` editors are reused, re-hosted in a roomy detail pane and migrated to the field primitives (Input/Textarea/Slider/Select). Shell/tabs/Overview CSS is `styles/MemoryWorkbench.css`; the reused list/detail/graph classes stay in `styles/MemoryPanel.css` (also shared by the memory **settings** sections — don't gut it). Mobile (`useIsMobile`) swaps master↔detail with a "‹ Back" button and stacks the list rows.
- **Agent-profile editor** (`unified-profile-editor/`): hero identity header over a `ControlCard` grid; avatars in `common/AvatarPicker` (`lib/avatars.ts`) — an icon id **or** a generated image (the Generate tab → `POST /api/agent/avatar/generate`, stored as `media:{ws}/{doc}` on `profile.avatar`); render via `common/AgentAvatar` (image avatars resolve to an authed object URL through `lib/avatarImage.ts`, icon otherwise). Signature color from `lib/agentAccent.ts`. Primitives: `ui/SegmentedControl`, `ui/CopyChip`, `common/ControlCard`.
- **Settings** (`unified-settings/`, registry `sections/index.tsx::SECTION_HIERARCHY`): six nav groups — Infrastructure (Providers, Model Limits, **Model Roles** — per-role picker + member effective-model chips over `GET /api/models/roles`, Web Search, Images), Intelligence (Planner, Multi-Agent, Ambassador), **Prompts** (System Prompt, Prompt Enhancement, **Template Library** — the shared `prompt-library/PromptLibraryBrowser` also backing the in-chat modal, **Feature Prompts** — all four overridable feature prompts with diff/reset via `GET /api/prompts/feature-defaults`), **Memory** (Overview, Recall — incl. the Two-Stage Rerank + Advanced blocks, Consolidation — with per-stage role chips), Tools, Interface. Sections build on the settings field kit + `useSettingsAutosave` (CLAUDE.md client rules); search matches section `keywords`.
- **Prompt Stack editor** (Settings → Prompts → "System Prompt"): two-pane block composer over `/api/prompts/layers` — `@dnd-kit` reorder, debounced autosave, reset/diff, live preview via `lib/promptStack.ts::composeStack`. Library snippets insert as custom layers; the enhancer (`/api/prompts/enhance`) rewrites a layer in place.
- **Themes**: six token-driven `ThemeDefinition`s in `lib/theme.ts` (~110 CSS vars + picker metadata `description`/`icon`), applied by `ThemeProvider` (stamps `data-theme` on the root); icons map in `common/themeIcons.tsx`; a vitest enforces cross-theme key parity (glow tokens use a transparent shadow, never bare `none` — `none` in a shadow list kills the whole declaration). Per-theme surface decorations (scanlines/dot-grid/glass) live in `styles/expression.css` (`[data-theme]`-scoped, unlayered). Fonts: Inter + JetBrains Mono self-hosted via `@fontsource` (imported in `main.tsx`; stacks in App.css `@theme static` → `--font-sans`/`--font-mono`).
- **Multi-server**: per-server settings in localStorage (`agentx:servers`/`…:meta`/`activeServer`).
- **Two shells (desktop + web/PWA)**: the same app builds as the Tauri desktop app and an installable web PWA. The split is a compile-time gate — `__IS_TAURI__` (vite `define`, keyed off `TAURI_ENV_PLATFORM`) over dynamic `import()` in `src/platform/` capability triples (`opener`, `window`); `@tauri-apps/*` is confined there (guarded by `platform/importBoundary.test.ts`) so Rollup strips it from the web bundle. Runtime chrome guards stay in `lib/platform.ts`. `src/pwa/` holds `registerPwa` (SW via `vite-plugin-pwa` `registerType:'prompt'` + `vite:preloadError` stale-chunk reload) and `installPrompt` (Chromium `beforeinstallprompt` + iOS "Add to Home Screen" hint), bridged to toasts by `PwaToasts`; the plugin is `disable: isTauriBuild`, manifest + icons in `client/public/`. **Connection links** (`lib/connectionString.ts`): `#connect=<base64url>` share URLs carry server URL + optional gateway token in the fragment; `consumeConnectFragment()` (boot, `main.tsx`) → `ConnectGate` confirm → `addServer`; copy-link affordance in `ServerSelector`. Headless-testable via `.claude/launch.json` (`web`). Remote use needs the PWA origin in CORS + `AGENTX_AUTH_ENABLED=true`; Cloudflare Pages deployment is deferred (see `todo/backlog/open-platform.md`).

## API Endpoints (full reference)

Base URL: `http://localhost:12319/api/`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check (add `?include_memory=true` for DB status) |
| `/api/tools/language-detect-20` | GET/POST | Detect language of text |
| `/api/tools/translate` | POST | Translate (`{"text": "...", "targetLanguage": "fra_Latn"}`) |
| `/api/mcp/servers` `/api/mcp/tools` `/api/mcp/resources` | GET | List MCP servers / tools / resources (filter `?server=`) |
| `/api/mcp/connect` `/api/mcp/disconnect` | POST | Connect/disconnect (`{"server": "name"}` or `{"all": true}`); OAuth servers needing consent → 202 `auth_required` + `authorization_url` |
| `/api/mcp/oauth/callback` | GET | OAuth redirect target (PUBLIC — state-validated; middleware-exempt) |
| `/api/mcp/servers/{name}/auth/reset` | POST | Forget a server's OAuth tokens + registration |
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
| `/api/agent/ambassador/voice-command` | POST | Voice-mode intent routing (`{conversation_id, transcript, agent_name?, artifacts?}`) → `{action: answer\|relay, text, qa_id?}`; ambassador answers (persists `qa:`) or drafts a relay; never fails (degrades to a spoken notice) |
| `/api/agent/ambassador/speak` | POST | TTS: synthesize a briefing/answer (`{text, agent_profile_id?, voice?, model?}`) → raw MP3 (`audio/mpeg`) via OpenRouter `/audio/speech`; strict speech-model resolve, graceful `422` when unconfigured |
| `/api/agent/ambassador/transcribe` | POST | STT: transcribe push-to-talk audio (`{audio: base64, format?, agent_profile_id?, model?, language?}`) → `{text}` via OpenRouter `/audio/transcriptions`; strict STT-model resolve, graceful `422`; transcript fills the reviewable input (never auto-sent) |
| `/api/agent/ambassador/stream` | GET | Tail a briefing/Q&A run (`?run_id=`): `ambassador_start`/`_chunk`/`_done`/`_error` SSE |
| `/api/agent/ambassador/thread/{thread_id}` | GET/PATCH/DELETE | Replay one unified ambassador thread ("Inquiry") `{thread_id, title, entries}` (briefings + Q&A as one ordered list, persisted tool chips) / rename (`{title}`) / clear it. `thread_id` defaults to the conversation id |
| `/api/agent/ambassador/{conversation_id}` | GET | Back-compat shim: replay as `{briefings, qa}` (now projected from the unified thread) |
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
| `/api/logs/archive` `/logs/archive/status` `/logs/archive/{name}` | GET | List daily archive segments (`data/logs/agentx-YYYY-MM-DD.log.gz[.enc]`; sealed `.enc` carry `encrypted:true`) / vault state (keyring present, `unlocked`, sealed/pending counts, retention) / download a segment (sealed ones decrypt on the fly; `423` when locked, `?raw=true` for ciphertext) |

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

The main streaming chat path. Runs **detached server-side** (survives client disconnect); first event `run_started` carries `run_id`. Events: `run_started, start, chunk, status, info, steer, tool_call, tool_result, exhibit, workspace_attached, done, error, close`. (`info` carries loop-internal notices, e.g. `{type: 'trajectory_compressed'}`; the client currently ignores it.)

- **Routing**: optional `target_agent_id` routes to a specific agent by `agent_id` (priority `workflow_id > target_agent_id > agent_profile_id > default`). An inline `@agent-id`/`@name` in the message overrides the selection (suppressed in workflows); the client composer offers `@`-autocomplete. With `alloy.allow_adhoc_delegation` (default **ON** since the Agent Teams work), the agent gets a `delegate_to` tool (depth-limited, no self-delegation; emits `delegation_*` events).
- **Ad-hoc delegation roster** ("Agent Teams" — the user-facing name for Alloy; strings-only rename, Workspaces→Projects precedent — internal module/routes/config keys keep `alloy`, and `WorkflowRoute` stays schema-only under the new name): outside a workflow, when the ad-hoc executor is attached, a `LedgerBlock(key="delegation_roster", priority=85, shrink_fn=shrink_tail, non-mandatory)` is injected (`views.py::_build_delegation_roster_block` → `alloy/prompts.py::build_adhoc_roster_prompt`) listing each opted-in teammate (`available_for_delegation`, **opt-in**, model default `False`) with its specialty (`AgentProfile.delegation_hint`, falling back to `description`). Roster + tool enum share one source (`delegation_tool.py::list_adhoc_delegation_targets` — filters: has agent_id, not self, opted in, `kind=='agent'`), so they can't drift. The nudge is deliberately **soft** ("delegation is an option, not an obligation" — a tests-guarded tone invariant), unlike the workflow supervisor's "default to delegation". Never rendered under a workflow (supervisor block, prio 95, owns the framing); may coexist with `participants` (prio 90 — who has spoken vs. whom you may delegate to). Request flag `disable_delegation` (per-conversation "Solo" chip / palette toggle client-side) short-circuits the executor attach via `_adhoc_delegation_enabled()` — suppressing tool + roster together; ignored when `workflow_id` is set (a team run IS delegation).
- **Exhibits**: the internal `present_exhibit` tool emits a typed `exhibit` event (declarative Gallery→Exhibit→Element tree) instead of tool cards. Element types: `mermaid`, `choice` (clicking submits as the next user turn), `table` (sortable/scrollable/expand-to-modal), `citation` (active foldable sources vs passive links). A successful `web_search` auto-emits a passive `citation` exhibit (deduped by URL; id `exh_src_<tool_call_id>`; toggle `citations.auto_capture_web_search`).
- **Exhibits under delegation**: a specialist's `exhibit` + `workspace_attached` events pass through the executor's re-wrapper **top-level, unchanged** (`alloy/executor.py` — the client renders/attaches them exactly like the main loop; the card lands under the streaming delegation card). `status`/`info` events are still dropped inside delegation. The wires also ride `delegation_complete.exhibits` (cap 5): `_run_delegations` appends an "already displayed — don't write image URLs" note to the supervisor's TOOL message (ONLY there — previews/memory/persisted raw_content stay clean; kills hallucinated `/api/workspaces/ws_home/...` links), and `_execute_and_emit_tools` strips the wires from the persisted delegation metadata + persists each as a **synthetic `present_exhibit` tool_call turn** (same seam as the direct image flow) so reload rebuilds the cards. The specialist loop runs under a **specialist-scoped `InternalToolContext`** (user/conversation/workspace inherited from the outer binding; `agent_id`/`channel` re-scoped to specialist + alloy channel; try/finally reset) — fixes tool-usage attribution and ambient-less callers. Specialist `AgentConfig` carries the profile's `allowed_tools`/`blocked_tools` (Phase 18.2 parity). The tool-loop empty-final guard covers ANY empty final (was round-0 only); after delegations it falls back to the last specialist's output instead of an empty bubble. `alloy.specialist_inherits_supervisor_tools` is **unwired** (defined + settable, never read) — backlog: wire or remove.
- **Status events** (`{phase, label, detail?, group?, progress?}`): a coarse per-phase activity feed (`recalling`/`composing`/`thinking`/`running_tool`/`reading`/`model_fallback`/`truncated`). Rides the run's Redis event bus (not generator yields) via an ambient `emit_status()` (`streaming/status.py`, run resolved from a `ContextVar`), so it replays on re-attach.
- **Truncation surfacing** (v0.21.148): the tool loop tracks each round's `finish_reason`; `done` carries `finish_reason` + `truncated` (true when the final round hit max_tokens), persisted as `metadata.truncated` on the assistant turn (client renders a "truncated" tag + warning toast). On a final-round `length` the loop **auto-continues once** ("continue where you stopped"; config `chat.auto_continue_on_length`, default on) before flagging. Reasoning tokens from OpenRouter/Vercel/OpenAI-compatible streams (`delta.reasoning`/`reasoning_content`) now stream as `<think>` content like LM Studio (live ThinkingBubble, stripped before persistence); reasoning-capable models (`ModelCapabilities.supports_reasoning`, from the OpenRouter/Vercel catalogs) get a larger output budget (`REASONING_DEFAULT_OUTPUT_TOKENS`/`REASONING_MIN_OUTPUT_TOKENS`) so thinking doesn't starve the visible answer.
- **Steer**: a live `steer` event carries a user message folded into the running turn.
- **Workspace auto-attach**: when a `generate_image` result lands in a workspace, a `workspace_attached` event (`{workspace_id}`) follows the `image` exhibit. A conversation with no workspace falls back to the personal **Home** store; the client durably attaches it (`conversationMeta.workspaceId`) and notifies once, so the agent keeps seeing that media on later turns. (Uploaded vision images attach the same way via the `POST /agent/chat/images` response.)

**Plan termination & cancellation.** `PlanExecutor` wraps its subtask loop in `try/finally`: a hard Stop (`GeneratorExit`) mid-subtask resets the in-flight subtask to `pending` and marks the plan **`interrupted`** (resumable) instead of stuck `running`; cooperative cancel marks `cancelled` + `clear_cancel`s the flag. The Stop handler in `views.py` persists the interrupted plan card faithfully. Cancellation is **cooperative** (GeneratorExit lands at a `yield`), made prompt by capping tool wall-clock (`web_search` bounded by `search.timeout`, default 15s). Tool execution stays **synchronous** (off-thread/`asyncio` deadlocked Stop).

---

## Agent Memory Internals

`AgentMemory` (`kit/agent_memory/memory/interface.py`) is the unified API.

**Core**: `store_turn(turn)` (episodic + working), `remember(query, top_k)` (turns/facts/entities/strategies), `learn_fact(claim, source, confidence)`, `upsert_entity(entity)`, `record_tool_usage(...)`, `reflect(outcome)` (async consolidation).

**Goal tracking**: `add_goal`, `get_goal`, `complete_goal(goal_id, status, result)` (`completed`/`abandoned`/`blocked`), `get_active_goals`. `TaskPlanner.plan()` accepts an optional `memory` param: creates a `Goal` for the main task, stores it on `TaskPlan.goal_id`; `Agent.run()` calls `complete_goal()` on success/failure.

### Extraction Pipeline (`extraction/service.py`)

LLM-based extraction. **User turns extract per conversation WINDOW** (§2.10 Slice 1, v0.21.154, `extraction_windowing_enabled` default-ON): `_extract_windowed` pre-filters pleasantries (`is_heuristic_skip`) *before* window assembly, batches turns via `_assemble_windows` (greedy, `extraction_window_max_tokens`/`_max_turns`, never splits a turn), and calls `check_relevance_and_extract_window(turns, …)` once per window — numbered `[Tn | addressed: …]` turns, a rolling **registry** of prior-window extractions ("reuse the exact name"), and a conversation-**overview** block (rolling summary, lazy best-effort read of `conv_summary:{id}`). Each fact's LLM-emitted `source_turn` maps to the turn id → **per-fact `source_turn_id`** (user facts stored None before). Window output gets its own cap (`extraction_window_max_output_tokens`, floor over the stage max); a parse failure or repaired-truncated multi-turn response splits the window and retries once (`metrics.window_retries`; truncation eats the trailing relationships array, so salvage is kept only for single-turn windows). Flag-off falls back to the legacy per-turn `check_relevance_and_extract(text)`; the assistant self-knowledge flow (`check_relevance_and_extract_assistant`, certainty: definitive/analytical/speculative) remains per-turn. Both accept a `roster` (`[{agent_id, name}]`) so facts attribute to a *specific* agent: the LLM names the agent in prose (`subject_agent`), `_resolve_agent_attribution` resolves name → `agent_id` (durable truth, transiently `subject_agent_id`) — windowed facts resolve "you" against their *source turn's* addressed agent; unknown names demote to `third_party` (never fabricate an agent_id). `check_contradictions(claim, facts)` is three-layer fact verification: hash gate → semantic duplicate → entity-scoped candidates → LLM adjudication. **Entity resolution** (`_resolve_and_prepare_entities`): LLM `existing_entity_id` → batch-local pending index (name/alias/`entity_slug` — a within-conversation mention can't mint twice) → exact `find_entity_by_name_or_alias` → **semantic band** (`semantic_entity_linking_enabled` default-ON: entities embed at store time; `vector_search_entities` top-3, same-type + never-Agent; ≥ `entity_linking_auto_threshold` (0.90) auto-link + alias fold, ≥ 0.75 log-only gray zone for a later adjudicator) → mint (reusing the band-lookup vector). The `entity_linking` job also backfills missing entity embeddings (`entity_embedding_backfill_batch` per run); relationship endpoints missing from the batch map recover via the same resolver (irrecoverable drops are logged + counted in `relationships_dropped` — previously silent). Node-merge machinery lives in `maintenance/entity_merge.py` (shared with `dedupe_entities`, incl. its `--semantic` embedding-pair mode). Metric semantics under windowing: `extraction_calls` counts *window* calls; `turns_relevant` counts turns of relevant windows. Default extraction model `nvidia/nemotron-3-nano` (`combined_extraction_model` setting). Consolidation jobs (`consolidation/jobs.py`) run on the worker sweep cadence (`job_consolidate_interval`). **Idempotency is turn-level** (v0.21.185, migration 0005): discovery selects only user turns with `t.consolidated IS NULL` (assistant self-extraction: `t.self_consolidated IS NULL`) and collects *only* those; every turn poured over cleanly (relevant **or** irrelevant) is stamped `t.consolidated`/`t.self_consolidated` via `_mark_turns_consolidated` (a failed window's turns stay unmarked → retry), so a turn is extracted **exactly once, ever** — an idempotent re-sweep is a no-op, never re-pouring old turns (dedup is *not* the safety net). `c.consolidated`/`c.self_consolidated` remain coarse "last-touched" markers, no longer the gate; `reset_consolidation()` clears the turn markers too, and portability round-trips them so a re-imported bundle resumes rather than re-consolidating. Degrades to empty results with no provider.

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

Every LLM feature resolves `provider:model` through the registry. `resolve_with_fallback(model, *, preferred_fallback=)` (sync) and `complete_with_fallback(...)` (async, execution-retry) make a feature **never hard-fail the turn**: an unconfigured/unreachable provider falls back to the caller's active agent-profile model (`preferred_fallback`) → global default (`preferences.default_model` / `models.defaults.chat`) → first healthy provider. Kill-switch `models.fallback_enabled`; best-effort health cache (fed by `health_check` + observed `complete_with_fallback` failures; 30s TTL). `complete_with_fallback` defers cached-unhealthy candidates to last resort, so a keyed-but-failing provider (e.g. no credits) isn't re-paid on every utility call inside the TTL. **Compression paths are fallback-hardened** (v0.21.148): trajectory compression + the tool-output compressor default to **no configured model** (`trajectory_compression.model`/`compression.model` unset ⇒ chain = active turn model → global default) and execute via `complete_with_fallback(preferred_fallback=<active model>)` — previously they hardcoded Anthropic model heads and raw `complete()`, silently failing every time on a keyed-but-broke Anthropic account. **Foundation #4** extended this from the Ambassador to **every feature site** — the main chat path (`views.py`, surfacing a substitution as a `model_fallback` status notice), reasoning (CoT/ReAct/Reflection/ToT), drafting candidates/pipeline stages, the planner, plan execution, conversation summarization, the prompt enhancer, and Alloy specialists. Sites resolving the agent's own model fall through to the global default; reasoning/drafting sub-models do likewise (they hold no agent-model handle). **Agent-creation model chain (v0.21.178 fix):** the request-handling views build the agent's `default_model` as `request model → profile model → preferences.default_model (Settings) → AgentConfig dataclass default (lmstudio:llama3.2, offline-first last resort)` (`views.py` chat-stream + `agent_plan_resume` + the `get_agent` singleton). Previously the configured global default was **absent** from this chain, so a profile with no model set (e.g. the default profile) silently routed every turn to LM Studio regardless of the model chosen in Settings (`ModelDefaultResolutionTest` guards it). **Intentionally strict** (a generic chat fallback would be wrong): specialized model roles (speculative-decoding draft/target pair, ambassador TTS/STT) and the explicit availability probes (`validate()`, cost estimation). **Model roles (ADR-11)**: `models.roles.{fast_utility,deep_reasoning,summarizer}` form an implicit tier over the utility-model settings (single source `agentx_ai/model_roles.py`; membership + effective-chain preview via `GET /api/models/roles`): a member with an empty/"inherit" value follows its role, an explicit value always wins, an unset role is a no-op, and any model setting accepts an explicit `role:<name>` ref — the registry expands `role:` refs defensively (`_fallback_chain` + both `*_with_fallback` entries) so the sentinel never reaches a provider lookup. **Memory stages inherit**: each stage resolves `explicit override (role: expanded) → its model role → consolidation.feature_default_model (bulk) → default chat` (`_resolve_stage_model`); consolidation stage defaults are **`"inherit"`** (v0.21.176 — previously pinned `lmstudio:…`, which silently shadowed a set role so consolidation ignored the family). Existing installs keep their saved per-stage values (ConfigManager no-merge); `POST /api/models/roles/adopt` clears those overrides in one click so they follow the role. **Model-family coverage invariant** (`ModelFamilyCoverageTest`): every general-purpose LLM feature-model setting must resolve through a family — i.e. be a `ROLE_MEMBERS` source (now incl. the ambassador **aide**, `ambassador.aide.model` → `fast_utility`) — OR sit on one of two documented escape-hatch lists in `model_roles.py`: `INHERITS_AGENT_MODEL` (bucket b — intentionally falls through to the calling agent's own model: `preferences.default_model`, `feature_default_model`, `planner.model`, `ambassador.model`, `reasoning_model`, `drafting_model`) or `EXEMPT_SPECIALIZED` (bucket c — not a general-purpose text LLM: embeddings, cross-encoder reranker, `images.default_model`, and the speculative `draft_model`↔`target_model` matched pair). A new `*_model` setting with no bucket fails the guard, so nothing silently bypasses the family system. **Tool-capability gate** (`agent/core.py::_model_supports_tools`): the chat path won't send tool definitions to a model that can't use them (OpenRouter 404s "no endpoints support tool use" otherwise). It checks the resolved model's `get_capabilities().supports_tools`, and — because OpenRouter reports `tools=False` for an *uncached* model — warms the model catalog (`fetch_models`) once and re-checks before stripping tools, defaulting to "tools on" on any probe failure (never strips from a capable model on a cold cache). The model picker surfaces `supports_image` (image *output*, derived from `output_modalities`) beside `supports_vision` (image *input*). **Context-window mis-resolution (v0.21.197):** the same uncached-model blind spot sizes the *context budget* wrong — an OpenRouter `:latest` route (or any model missing from the catalog) resolves to `DEFAULT_CAPABILITIES.context_window` (8192), so `_resolve_model_budget` hands the whole ledger/compaction path an ~8k window on a 256k model and the rolling digest fires almost immediately. Escape hatch: a **per-model context-window override** (`context_limits.models.{id}`, resolved for *all* providers by `get_context_limit_overrides`, `context_window` wins over `caps.context_window`) — now editable in Settings → Model Limits for any provider (a `null` model value calls `ConfigManager.unset` to drop the override). The picker flags `:latest` OpenRouter routes (`PreparedModel.warnLatest`); the compaction knobs (`context.verbatim_budget_ratio`, `context.summary_trigger_ratio`, `context.recent_floor`) are exposed via the `context` section of `/config/update` (allowlisted).

### Agent Identity & Multi-Agent Attribution (Phase 16)

Each profile has a Docker-style `agent_id` (auto-generated, immutable, adj-adj-noun, ~83K combos). `self_channel` = `_self_{agent_id}`; recall searches `[active_channel, _self_{agent_id}, _global]`. Attribution is **agent-specific** (`agent_id` is the durable key, display name is prose only):
- **Agents are first-class entities**: consolidation `_ensure_agent_entities` upserts `Entity(type="Agent")` per agent in `_global`, id `agent:{user_id}:{agent_id}`, canonical key `properties.agent_id`, name + prior names as `aliases`. Facts link via normal `[:ABOUT]`.
- **Name stamping**: assistant turns carry the display name in `Turn.metadata["agent_name"]` (set by writers in `views.py`/`core.py`/`alloy/executor.py`); `episodic.store_turn` stamps `Turn`/`AgentParticipant` nodes so the kit builds a roster (`get_conversation_roster`) without importing `ProfileManager`.
- **Per-agent routing**: `_resolve_subject_channel` honors `subject_agent_id` → `_self_{that_agent}`; assistant self-extraction routes each turn by its own producing `agent_id`.
- **Rename-safety**: `update_profile` propagates a name change to the Agent entity's `aliases`; `dedupe_entities` never merges `Agent` nodes (keyed by `agent_id`).
- **Backfill / debug**: `task memory:backfill-agent-attribution[:apply]` rewrites legacy generic self-facts deterministically; `task memory:debug-attribution -- --scenario <directive|cross-agent|mixed> --agents "..."` drives a scripted multi-agent conversation through real consolidation (non-destructive by default; `--isolate` for a sterile read).
- **Delegation is legible on replay (v0.21.198)**: `load_recent_turns` drops tool_call/tool_result rows, so a `delegate_to` call vanishes on cold rehydration — the executor's stored assistant turn is the only surviving trace. It's framed `[Delegated to {name}] Task: {task[:200]}\nResult: {…}` (`alloy/executor.py`), so the supervisor sees it *delegated* (who + what), not just "a summary was generated." Still attributed to the specialist `agent_id` (consolidates to the specialist's self channel). Durable structured recording stays the agent's call via `update_conversation_state` (INV-8: no executor auto-write).
- **Delegation project-provenance (v0.21.198)**: a specialist's tool context is rebound to the parent conversation's `workspace_id`, but `_build_specialist_messages` also injects `render_project_identity_block` + `render_instructions_block` (`kit/workspaces/retrieval.py`) from `current_context().workspace_id` — so a delegated agent is *told* which project it's in (name, instructions, file-tool guidance), not just silently able to touch it. Guard: `AlloyDelegationMetricsTest.test_specialist_messages_include_project_context`.

### Memory Settings & Eval Pinning (override discipline)

Since the settings overhaul (S1, v0.21.156) the memory kit reads settings **live**: every read
is a `get_settings()` call (TTL-cached, 60s) or goes through an instance `settings` property
that falls back to a live read. There are **zero module-level `settings = get_settings()`
snapshots** — `scripts/check_docs.py` ratchets the count at 0; never add one.

- **Runtime changes apply without a restart** — `save_memory_settings()` busts the TTL cache,
  so a UI save is visible on the very next read (worst case ≤60s for other processes).
- **`MEMORY_SETTINGS_PATH` is absolute** (anchored to the repo root; unit-asserted equal to
  `ConfigManager.CONFIG_PATH.parent / "memory_settings.json"`), with a warn-and-read fallback
  for a legacy CWD-relative file. A corrupt overrides file falls back to defaults (INV-1) but
  is **logged and surfaced** as `settings_file_status` in `GET /api/memory/settings`;
  `POST /api/memory/{settings,recall-settings}` schema-validates the update and rejects the
  whole write with per-key errors (`400 {"error", "errors": {key: msg}}`).
- **Instance override slots** (unit tests): `ExtractionService`, `RecallLayer`, `AgentMemory`,
  and `MemoryAuditLogger` all use the same pattern — `obj._settings = <Settings|mock>`
  overrides that instance; `None` (default) ⇒ live `get_settings()` via the `settings` property.

**Temporary overrides** (tests, eval harnesses, experiments) use ONE mechanism:

```python
from agentx_ai.kit.agent_memory.config import get_settings, pin_memory_settings
override = get_settings().model_copy(update={...})
with pin_memory_settings(override):
    ...  # every get_settings() read sees `override`; the TTL cache can't revert it
```

Checked BEFORE the TTL cache; nesting-safe (an inner pin restores the outer); busts the cache
on exit; never touches disk. Process-global — single-process eval/test use only. Canonical
users: `eval_recall._run_arm` (still pins `retrieval_cache_enabled=False` — the retrieval
cache key does NOT include technique toggles, so stale-config hits are silent),
`eval_consolidation.handle`, `debug_attribution.handle`. `save_memory_settings()` is for REAL,
durable user-facing changes only — never for a temporary override (it writes
`data/memory_settings.json`; a live dev server picks it up immediately).

**Deliberately boot-frozen** (restart or pin before first use — evals already do): the
embedding provider/dispatcher singleton (provider choice + queue settings at first build),
DB connection parameters (first driver/engine/client build), and job intervals
(worker/registry boot).

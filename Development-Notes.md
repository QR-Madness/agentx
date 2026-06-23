# AgentX Development Notes

> Deep subsystem internals, extracted from `CLAUDE.md` so it can stay a light index.
> **Not auto-loaded** — read the relevant section when working that area. Keep in sync with the
> code and update alongside changes (docs-maintenance rule in `CLAUDE.md`). For the full API
> contract use [`OpenApi.yaml`](OpenApi.yaml) + `docs-site/src/content/docs/api/endpoints.md`;
> the tables here are the orientation copy.

## Contents

- [Backend Subsystems](#backend-subsystems) — Prompts, Conversation Context, Ambassador, Logging
- [API Endpoints](#api-endpoints-full-reference) — full reference + the chat-stream SSE contract
- [Agent Memory Internals](#agent-memory-internals) — Interface, Extraction, Procedural, Context Gating, Web Research, Model Resolution, Attribution

---

## Backend Subsystems

### Prompts (`prompts/`)

`PromptManager` singleton composes prompts. The **conversational global system prompt** comes from a durable **layered stack** (`layers.py`: `LayerStore` over `ConfigManager` key `prompts.layers`; `BUILTIN_LAYERS` ship a versioned `default` overlaid by the user's `override` — `effective = override ?? default`; `update_available` diffs a released default bump against `base_version`). `compose_prompt` sources global content from `LayerStore.compose()` (edits persist across restarts) and only attaches a prompt-profile's sections for an explicit non-`is_default` selection (default profile's sections fold into the stack — no double-injection). Legacy `/prompts/global*` are back-compat shims. This stack governs **only** the conversational persona — internal *feature* prompts (reasoning, planner `decompose`, extraction, compression) come from the separate `SystemPromptLoader` (`loader.py`; optional `data/system_prompts.yaml`). **Placeholders**: `placeholders.py::substitute_placeholders` replaces a whitelist (`{agent_name}`/`{date}`/`{time}`) at the end of `compose_system_prompt`; client mirrors it in `lib/promptPlaceholders.ts`.

### Conversation Context (`agent/context.py` + `agent/context_ledger.py`)

**Context Ledger** (`agent/context_ledger.py`, v0.21.90): the chat-stream preamble is built as a list of `LedgerBlock(key, priority, content, min_tokens, max_tokens, shrink_fn, mandatory)` and allocated by `assemble_ledger` within `min(verbatim_ratio·window, window−reserved)`. **Mandatory** blocks (base prompt, active-workflow supervisor) are emitted full first; the rest compete by priority — fit full, else `shrink_fn(content, remaining)` if `≥ min_tokens`, else drop. A synthetic history entry rides at `history_priority` (50), so the verbatim transcript outranks the droppable recall supplement (30) but yields to higher blocks. **Allocation order ≠ emission order**: blocks emit in caller registration order (canonical: base → supervisor/participants → checkpoints → scratchpad → stable_core → recall → summary), then fitted history, then the new turn. Shrink helpers: `shrink_tail`, `shrink_lines_newest_n` (checkpoints, newest-N), `shrink_memory_to_facts` (bundle → `## Known Facts` only). `LedgerResult.allocations` is the per-block token report (Context Inspector seam; logged). **Stable memory core** (Foundation #3): `AgentMemory.get_salient_core()` (cheap non-vector `SemanticMemory.get_salient_facts`/`get_salient_entities`, `ORDER BY salience`, excludes superseded/`temporal_context=past`; gated by `salient_core_*`) is the prio-70 stable block carrying the reflex procedures; `remember(query=message)` is the prio-30 supplement, deduped against the core by id. `assemble_turn_context` is now a thin **wrapper** over `assemble_ledger` (registers each SYSTEM block as mandatory) — byte-identical for the `agent/core.py` + `alloy/executor.py` callers. **Token estimation** (Foundation #6, v0.21.102): all sizing flows through one shared module — `agentx_ai/tokens.py` `estimate_tokens(text)` / `estimate_messages(msgs)` (tiktoken `o200k_base`, chars/4 fallback when unavailable, and a >20K-char fast path so hot loops don't re-tokenize giant tool outputs). The ledger's `estimate_text_tokens`/`_estimate_messages`, `streaming/helpers.estimate_tokens`, the rolling-summary trigger, and transcript rehydration all delegate to it; `shrink_tail` verifies+retrims against it rather than assuming a fixed ratio. The legacy `ContextManager.prepare_context`/`estimate_tokens` and the message-count knobs (`auto_summarize_at`/`max_messages`) were removed; `ContextConfig` keeps only `summary_model`.

Per-turn context (legacy summary): keep the SYSTEM preamble + as much **recent verbatim transcript** as fits `context.verbatim_budget_ratio` (0.7) of the model's real window; drop oldest overflow (covered by the rolling summary). In-memory `SessionManager` is **rehydrated** from durable `conversation_logs` on a cold session (`conversation_history.py::hydrate_session_from_history`) — both the interactive stream (`views.py`) and `Agent.chat()` (Foundation #4b: the **background/queued chat** path now resumes warm too; idempotent, so no double-load). Alloy specialists are deliberately *not* transcript-hydrated — they're task-scoped by contract and already resume warm via shared-channel recall. The rolling summary is **context-window-triggered** (`maybe_update_summary`, token-based) and **persisted** in Redis (`conversation_summary_storage.py`). Checkpoints/scratchpad (`checkpoint_storage.py`) are Redis-keyed per conversation, re-injected each turn; eviction is anchor-preserving (keeps the first), and the `checkpoint` tool has a `replace` mode. After a turn, `views.py::generate_sse` persists via pure builders in `streaming/persistence.py` (user → tool → steer → assistant Turns); a hard Stop (`GeneratorExit`) persists the **partial** turn (`metadata.interrupted`); folded steers persist as `user` turns (`metadata.steered`). **Thinking/CoT is process, not result**: it streams live (shown collapsed in the client) but is **never persisted** to `conversation_logs`. Old turns persisted before this still restore stored `metadata.thinking`.

### File Workspaces & Document RAG (`kit/workspaces/`, Slice 1, v0.21.103)

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

### Agent Shells (`kit/shell/`, v0.21.108)

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

A dedicated agent running *parallel* to a conversation that briefs the user on a turn **without polluting the main transcript** (the load-bearing invariant). `AmbassadorService.brief_turn` resolves the default ambassador (`get_default_ambassador()`; an `AgentProfile` with `kind='ambassador'`, hidden from chat). The profile's `system_prompt` is the personality voice; functional personas come from `AmbassadorConfig.{briefing,qa,draft}_persona` overrides ?? code defaults. Grounds **read-only** via `load_recent_turns`, runs `resolve_with_fallback` + token-streams via `provider.stream` (never raises; cancel settles sidecar to `cancelled`), emits namespaced `ambassador_*` SSE. The persona **speaks TO the reader** (second person, names the agent). The briefing grounds on the turn's substance via client-gathered `artifacts` (tool calls / sources / exhibits, `lib/ambassadorTurn.ts::gatherTurnContext`). Free-form questions (`answer_question`) and spoken questions (`route_voice_command`'s `answer` action) share **one agentic answer core** (`_agentic_answer`): a single streaming `provider.stream` loop that offers a **read-only tool belt** (`agent/ambassador_tools.py`: `summarize_conversation`/`explore_conversation`/`read_conversation`/`list_conversations` — SELECT-only — plus `survey_conversations`, a digest-rich cross-conversation view (each session's own rolling summary via `conversation_summary_storage.get_summary` when present, else the first/last snippet; model-free — the ambassador synthesizes the app-wide summary itself), and `list_agents`, the **agent roster** (`kind=='agent'` profiles only) surfacing each agent's role tags, delegation availability, role blurb, and its model's **live provider capabilities** (input/output modalities + tools/vision/speech/transcription flags, via `registry.resolve_with_fallback` → `provider.get_capabilities`, degrading **per-agent**) — the input for multi-modal routing; `execute_tool` never raises) — a round carrying `tool_calls` runs the tool (emits `ambassador_tool_call`/`_result` SSE) and loops; a content-only round **is** the streamed answer. Bounded by `_MAX_TOOL_ROUNDS` (last round forces tools off); never-raises, degrading to a grounded one-shot. **Tools are always on** (no gate — experimental app). **Grounding is tool-first**: the prompt pre-loads only `_LEAN_GROUNDING_TURNS` recent turns, so the ambassador *reads* the conversation (or surveys others via `list_conversations` → `read_conversation`) for real depth — this is why the tools actually fire. **Conversational memory**: prior settled Q&A is fed back as dialogue turns (`_thread_history`, read-only over `qa:`) so follow-ups have context (text **and** voice — the ambassador's *own conversation*, never mixed into the main transcript). Provider resolution is DRY'd through `_resolve_answerer`; the answer persona (`_build_answer_persona` + `_answer_persona` + `_TOOLS_NOTE`) states its capabilities so "what can you do?" is answered right. Tool reads name **each conversation by its own producing agent** (`conversation_history.load_recent_labeled_turns` / the survey's `agents` column read `metadata->>'agent_name'`), so a cross-conversation survey never mislabels one session's agent with the active one. The client panel header shows the **ambassador profile's own name + avatar** (`AmbassadorPanel`'s `AmbassadorMark` takes the profile `avatar`), since the ambassador is a customizable profile. The panel's **focus is independent of the chat tab** (`focusedConversationId`, snapshotted from the active tab on open, then "stays put"; an `AmbassadorConversationSwitcher` over the open tabs + a "current conversation" jump change it) — and it's told the **active conversation** as ambient context (`active_conversation` {id,title} on ask/voice → `_active_conversation_note`), so it knows where the person is *now* even when focused elsewhere. **Loading a conversation is pure display** — `AmbassadorPanel`'s voice auto-speak fires only for items seen `streaming` *this session* (`locallyStreamedRef`), so switching/reopening never re-synthesizes TTS or speaks history (the earlier bug billed for it). The tool belt's "this conversation" is the **focused** one (`execute_tool(focused_conversation_id=…)`); the answer persona is **multi-agent** (names each agent from its conversation, not one global "active agent"). `logging_kit/async_noise.py::install` (called from the client-abort-prone ambassador async views) downgrades Python 3.14's benign `CancelledError exception in shielded future` (a client aborting an in-flight TTS/SSE request) to debug while delegating all other loop errors. Also does **outbound relay** (`draft_relay_message`: shapes a rough intent into a first-person message the user reviews and relays into the conversation as a real **user** turn via `ConversationContext.relayToConversation` — the ambassador never speaks into the transcript as itself). Output persists ONLY to the Redis **sidecar** — never `conversation_logs`/`conv_summary:`. **Slice 1b** unified briefings + Q&A into one ordered **thread** ("Inquiry") under the `amb_thread:{thread_id}` family (entry-oriented — one record per ambassador turn carrying its optional `question`; ordered by `created_at`; `thread_id` defaults to the conversation id) carrying its own `title`. The pre-1b per-family public API (`create_qa`/`set_qa_answer`/`set_summary`/`get_qa`/`list_briefings`/…) is preserved as **thin projections** over the entry store, and legacy `ambassador:` records still replay (one-place fold in `list_thread`). Tool-call chips are **persisted on the entry** (`set_entry_tool_calls`, called inside `_agentic_answer`) so they survive a reload. `GET/PATCH /api/agent/ambassador/thread/{thread_id}` replays (`thread_payload` → `{thread_id, title, entries}`) / renames (`set_thread_title`); the `{conversation_id}` endpoint is a back-compat shim. A brand-new Inquiry **auto-titles from its first question** (`AmbassadorService._maybe_autotitle` → `ambassador_storage.derive_title`, model-free, text + voice); a `title_auto` flag on the meta guards a manual rename from being overwritten. Client: `AmbassadorContext.threadFor` renders one **Inquiry** stream (briefings as their own turns via `BriefingItem`), with `titleFor`/`renameThread`/`clearThread` driving the switcher rename + a `⋯` menu (brief / rename / clear). Runs ride detached-run infra via `start_chat_run(indexed=False)`. **The ambassador is a parallel operator, not a turn-by-turn briefer** (the UI is deliberately *boringly stable*): `AmbassadorPanel` is **de-coupled from turns** — no Turns strip. The panel is conversation-level — **"Brief this conversation"** (`briefConversation` → an `ask` with a summary intent) + starter chips + free-form ask/relay; the ambassador scopes to conversations through its read-only tools (it never receives per-turn `artifacts`). Per-turn work is **CC**: the chat's `MessageActions` button (`ChatPanel.handleAmbassador`) forwards a turn *into* the Inquiry as an `ask` ("brief me on this turn: …") — like an email into the thread — and opens the panel. The header is **one stable compact command bar in both modes** (voice only adds a subtle accent tint — never a layout morph, so the Voice/Text toggle never relocates and can't strand you); the body **auto-scrolls** (`useStickyScroll` + jump-to-latest); answers carry **copy** + relative timestamps. **Voice and text share one body** — the Inquiry stream *is* the transcript (a spoken question persists as a `qa:` entry), so only the footer differs: the text composer ↔ `components/ambassador/VoiceBar.tsx` (PTT mic + `voiceCommand` → spoken answer/relay-confirm + settings popover). The old full-screen `VoiceSurface` (orb + separate caption log) and `lib/voiceCaptions.ts` were retired. **Voice (TTS):** `AmbassadorService.synthesize` speaks a briefing/answer via the profile's `AmbassadorConfig.{voice_mode,speech_model,voice,speech_speed}` block — `ModelProvider.synthesize_speech` (only `OpenRouterProvider` implements it, via OpenAI-compatible `/audio/speech` → MP3 bytes; base raises `NotImplementedError`); resolved **strictly** (no chat fallback) with precedence override → profile → `config.ambassador.*` → shipped default `openrouter:microsoft/mai-voice-2`, degrading to a typed `SpeechUnavailable` (→ `422`) when unconfigured. The client plays it through a framework-agnostic `lib/audio.ts::SpeechPlayer` singleton (blob-URL cache + playback queue + autoplay-unlock) behind `hooks/useSpeech.ts`; `AmbassadorPanel` adds per-item speaker buttons; a voice-enabled ambassador (`ambassador.voice_mode`) **leads with a Voice tab** ([Voice | Text]) — the immersive `components/ambassador/VoiceSurface.tsx` (Discord-call layout: PTT mic + captions + settings popover). **STT (voice input):** `AmbassadorService.transcribe` → `ModelProvider.transcribe_speech` (OpenRouter `/audio/transcriptions`, base64; default `openrouter:openai/whisper-1`); client `lib/audioRecorder.ts` captures **raw PCM via Web Audio → WAV** (NOT MediaRecorder — webkit2gtk's is broken) behind `hooks/useDictation.ts`. **Voice command routing:** a transcript → `route_voice_command` first **classifies intent** (`_voice_command_persona`, strict JSON `{action,text}`, defensive parse → `answer` fallback); an `answer` is then produced by the shared agentic core (`_answer_to_text` → `_agentic_answer`, so spoken questions get the **same tools + continuity** as typed ones, persisted `qa:`), while a `relay` returns the drafted message the user confirms in a strip and sends via `ConversationContext.relayToConversation` (never auto-sent). VoiceSurface: hold-to-talk default + tap-to-toggle (localStorage `agentx:voice:pttMode`), barge-in (speak stops on record), instant transcript echo, captions both ways, toggle max-duration safety. Per-model **voice dropdowns** via `lib/voiceCatalog.ts` + `components/common/VoicePicker.tsx` (curated set + free-text). Tauri mic perms: CSP `media-src`, macOS `Info.plist`/`entitlements.plist`, Linux webkit2gtk hook `src-tauri/src/lib.rs::enable_webview_microphone` (enables `enable-media-stream` + auto-grants user-media). **Command Deck (Slice 4, `0.21.116`):** the same `AmbassadorPanel` also mounts as a **standalone, app-wide full-screen surface** (`SURFACES.ambassadorDeck` → `stubs.AmbassadorDeckContent`, opened from ⌘K / TopBar) in a conversation-less **deck mode** — pass a `deckThreadId` prop and the panel binds to a single persistent per-user thread (`lib/ambassadorDeck.ts::deckThreadId` → `deck:{user_id|default}`) instead of a conversation. The minted id rides the existing `conversation_id` seam as the thread key, so **no backend change**: ask/voice/thread all accept it, grounding/`_thread_history` degrade to empty (no real conversation), and the conversation-agnostic tools (`survey_conversations`/`list_agents`/`list_conversations`) carry it; the deck never writes `conversation_logs`, so it can't appear in its own survey. In deck mode the conversation-coupled affordances are gated off (`isDeck`): brief-this-conversation, the relay mode + composer (relay needs an open tab — voice/text relay return a friendly note), and the conversation switcher (a plain renamable title stands in); starter chips become `DECK_STARTERS` (survey + roster). **Deferred:** multiple named standalone Inquiries (needs a per-user thread registry + `GET /ambassador/threads`) and deck→conversation relay (needs the deferred `POST /ambassador/relay`). **Deferred** (Todo §16.6): floating CC sticky player, recording persistence/history/GC, streaming answers, cross-agent delegation (relay JSON is target-extensible).

### Logging (`logging_kit/`)

Centralized logging. `configure_logging()` (called from `settings.py` with `LOGGING_CONFIG=None`) installs a root `QueueHandler` → `QueueListener` feeding console (rich color + category badge + run tag via `handler.py`/`highlighters.py`/`categories.py`), an in-memory `RingBufferHandler` (backs `/api/logs`), and a **daily-rotating** gzip `archive.py` file (`DailyArchiveHandler`, midnight/UTC → `agentx-YYYY-MM-DD.log.gz`). When auth is set up, completed days are **sealed** with envelope AES-256-GCM by `archive_crypto.py`: a random DEK (encrypts archives) wrapped by a Scrypt KEK from the login password, stored in `data/logs/keyring.json`. The DEK is unwrapped + cached in process memory on login (sealing is lazy — the hot path never needs the key; days that roll while locked stay redacted-plaintext `.gz` until the next `seal_pending`). Password change re-wraps the DEK (`rewrap_dek`, O(1)); `reencrypt_all` is the deep-rotation path. Retention (`prune_old`, default 30 days) is ours, not `backupCount`. Auth-disabled → plaintext-gzip fallback. Ops: `task logs:keys:status|seal|rotate-keys|rotate-keys:deep` (cmd `rotate_log_keys`). `context.py` stamps a per-turn `run_id` (reuses `streaming/status.py::current_run_id`); `redaction.py` scrubs secrets at capture; `llm_cards.py` renders compact/full LLM request logs; `banner.py` is the startup banner. All behavior is governed by `AGENTX_LOG_*` flags (`flags.py`; decorations on by default, off → historical plain output). The client mirrors categories in `lib/logCategories.ts`.

---

## API Endpoints (full reference)

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

The main streaming chat path. Runs **detached server-side** (survives client disconnect); first event `run_started` carries `run_id`. Events: `run_started, start, chunk, status, steer, tool_call, tool_result, exhibit, done, error, close`.

- **Routing**: optional `target_agent_id` routes to a specific agent by `agent_id` (priority `workflow_id > target_agent_id > agent_profile_id > default`). An inline `@agent-id`/`@name` in the message overrides the selection (suppressed in workflows); the client composer offers `@`-autocomplete. With `alloy.allow_adhoc_delegation`, the agent gets a `delegate_to` tool (depth-limited, no self-delegation; emits `delegation_*` events).
- **Exhibits**: the internal `present_exhibit` tool emits a typed `exhibit` event (declarative Gallery→Exhibit→Element tree) instead of tool cards. Element types: `mermaid`, `choice` (clicking submits as the next user turn), `table` (sortable/scrollable/expand-to-modal), `citation` (active foldable sources vs passive links). A successful `web_search` auto-emits a passive `citation` exhibit (deduped by URL; id `exh_src_<tool_call_id>`; toggle `citations.auto_capture_web_search`).
- **Status events** (`{phase, label, detail?, group?, progress?}`): a coarse per-phase activity feed (`recalling`/`composing`/`thinking`/`running_tool`/`reading`). Rides the run's Redis event bus (not generator yields) via an ambient `emit_status()` (`streaming/status.py`, run resolved from a `ContextVar`), so it replays on re-attach.
- **Steer**: a live `steer` event carries a user message folded into the running turn.

**Plan termination & cancellation.** `PlanExecutor` wraps its subtask loop in `try/finally`: a hard Stop (`GeneratorExit`) mid-subtask resets the in-flight subtask to `pending` and marks the plan **`interrupted`** (resumable) instead of stuck `running`; cooperative cancel marks `cancelled` + `clear_cancel`s the flag. The Stop handler in `views.py` persists the interrupted plan card faithfully. Cancellation is **cooperative** (GeneratorExit lands at a `yield`), made prompt by capping tool wall-clock (`web_search` bounded by `search.timeout`, default 15s). Tool execution stays **synchronous** (off-thread/`asyncio` deadlocked Stop).

---

## Agent Memory Internals

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

Every LLM feature resolves `provider:model` through the registry. `resolve_with_fallback(model, *, preferred_fallback=)` (sync) and `complete_with_fallback(...)` (async, execution-retry) make a feature **never hard-fail the turn**: an unconfigured/unreachable provider falls back to the caller's active agent-profile model (`preferred_fallback`) → global default (`preferences.default_model` / `models.defaults.chat`) → first healthy provider. Kill-switch `models.fallback_enabled`; best-effort health cache (fed by `health_check`). **Foundation #4** extended this from the Ambassador to **every feature site** — the main chat path (`views.py`, surfacing a substitution as a `model_fallback` status notice), reasoning (CoT/ReAct/Reflection/ToT), drafting candidates/pipeline stages, the planner, plan execution, conversation summarization, the prompt enhancer, and Alloy specialists. Sites resolving the agent's own model fall through to the global default; reasoning/drafting sub-models do likewise (they hold no agent-model handle). **Intentionally strict** (a generic chat fallback would be wrong): specialized model roles (speculative-decoding draft/target pair, ambassador TTS/STT) and the explicit availability probes (`validate()`, cost estimation). **Memory stages inherit**: each stage resolves `explicit override → consolidation.feature_default_model (bulk) → default chat` (`_resolve_stage_model`); stage defaults stay `lmstudio:…` so local users stay cheap and cloud-only users auto-fall-back.

### Agent Identity & Multi-Agent Attribution (Phase 16)

Each profile has a Docker-style `agent_id` (auto-generated, immutable, adj-adj-noun, ~83K combos). `self_channel` = `_self_{agent_id}`; recall searches `[active_channel, _self_{agent_id}, _global]`. Attribution is **agent-specific** (`agent_id` is the durable key, display name is prose only):
- **Agents are first-class entities**: consolidation `_ensure_agent_entities` upserts `Entity(type="Agent")` per agent in `_global`, id `agent:{user_id}:{agent_id}`, canonical key `properties.agent_id`, name + prior names as `aliases`. Facts link via normal `[:ABOUT]`.
- **Name stamping**: assistant turns carry the display name in `Turn.metadata["agent_name"]` (set by writers in `views.py`/`core.py`/`alloy/executor.py`); `episodic.store_turn` stamps `Turn`/`AgentParticipant` nodes so the kit builds a roster (`get_conversation_roster`) without importing `ProfileManager`.
- **Per-agent routing**: `_resolve_subject_channel` honors `subject_agent_id` → `_self_{that_agent}`; assistant self-extraction routes each turn by its own producing `agent_id`.
- **Rename-safety**: `update_profile` propagates a name change to the Agent entity's `aliases`; `dedupe_entities` never merges `Agent` nodes (keyed by `agent_id`).
- **Backfill / debug**: `task memory:backfill-agent-attribution[:apply]` rewrites legacy generic self-facts deterministically; `task memory:debug-attribution -- --scenario <directive|cross-agent|mixed> --agents "..."` drives a scripted multi-agent conversation through real consolidation (non-destructive by default; `--isolate` for a sterile read).

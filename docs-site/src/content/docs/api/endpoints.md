# API Endpoints

Base URL: `http://localhost:12319/api/`

All endpoints return JSON. POST endpoints accept `application/json` bodies. All endpoints handle CORS preflight (`OPTIONS`) automatically.

A machine-readable OpenAPI 3.0 spec mirroring this reference lives at `OpenApi.yaml` in the repo root (lint it with `task api:spec:lint`) — browse it interactively in the [API Explorer](/docs/api-explorer).

---

## Core

### Index

```
GET /api/index
```

Returns `{"message": "Hello, AgentX AI!"}`.

### Health Check

```
GET /api/health
GET /api/health?include_memory=true
```

Returns API, translation, and (optionally) memory system health.

**Response:**
```json
{
  "status": "healthy",
  "api": {"status": "healthy"},
  "translation": {
    "status": "healthy",
    "models": {
      "language_detection": "eleldar/language-detection",
      "translation": "facebook/nllb-200-distilled-600M"
    }
  },
  "memory": {
    "neo4j": {"status": "healthy"},
    "postgres": {"status": "healthy"},
    "redis": {"status": "healthy"}
  }
}
```

Translation status is `"not_loaded"` if models haven't been initialized yet. Memory section only appears when `include_memory=true`.

---

## Translation

### Language Detection

```
GET  /api/tools/language-detect-20
POST /api/tools/language-detect-20
```

**Request (POST):**
```json
{"text": "Bonjour, comment allez-vous?"}
```

**Response:**
```json
{
  "original": "Bonjour, comment allez-vous?",
  "detected_language": "fr",
  "confidence": 98.5
}
```

GET accepts `?text=...` query parameter.

### Translation

```
POST /api/tools/translate
```

**Request:**
```json
{
  "text": "Hello, world!",
  "targetLanguage": "fra_Latn"
}
```

Uses NLLB-200 language codes (e.g., `eng_Latn`, `fra_Latn`, `deu_Latn`, `spa_Latn`).

**Response:**
```json
{
  "original": "Hello, world!",
  "translatedText": "Bonjour, monde!",
  "targetLanguage": "fra_Latn"
}
```

### Search-backend health

```
GET /api/tools/search-health
```

Probes the active web-search backend (Tavily/Brave) with a trivial query to confirm it is
configured and reachable. Powers the "Test connection" button in Settings → Web Search.

**Response:**
```json
{
  "ok": true,
  "backend": "tavily",
  "count": 1,
  "error": null
}
```

---

## MCP (Model Context Protocol)

### List Servers

```
GET /api/mcp/servers
```

Returns configured MCP servers and connection status.

**Response:**
```json
{
  "servers": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "status": "connected",
      "tools": ["read_file", "write_file", "list_directory"],
      "tools_count": 3,
      "resources_count": 0
    }
  ]
}
```

### Validate / Server Detail

```
POST /api/mcp/servers/validate
GET  /api/mcp/servers/{name}
```

`validate` checks a server config without saving it (`{"server": {...}}`). The detail
endpoint returns the stored config plus live status for a single server.

### List Tools

```
GET /api/mcp/tools
GET /api/mcp/tools?server=filesystem
```

Returns available tools from connected servers. Filter by server name with `?server=`.

### List Resources

```
GET /api/mcp/resources
GET /api/mcp/resources?server=filesystem
```

Returns available resources from connected servers. Filter by server name with `?server=`.

### Connect

```
POST /api/mcp/connect
```

Connect to one or all configured MCP servers.

**Request (single):**
```json
{"server": "filesystem"}
```

**Response:**
```json
{
  "status": "connected",
  "server": "filesystem",
  "tools_count": 3,
  "resources_count": 0
}
```

**Response (OAuth server needing consent — HTTP 202):**
```json
{
  "status": "auth_required",
  "server": "remote-oauth",
  "authorization_url": "https://provider.example.com/authorize?..."
}
```

Open `authorization_url` in a browser; the connect completes in the background once the
user authorizes (poll `GET /api/mcp/servers` for the transition to `connected`).

**Request (all):**
```json
{"all": true}
```

**Response:**
```json
{
  "results": {
    "filesystem": {"status": "connected", "tools_count": 3},
    "github": {"status": "error", "error": "..."}
  }
}
```

### Disconnect

```
POST /api/mcp/disconnect
```

**Request (single):** `{"server": "filesystem"}`
**Request (all):** `{"all": true}`

### OAuth 2.1 (remote servers)

Remote servers (`sse` / `streamable_http`) can require OAuth 2.1. Add an `auth` block to
the server config:

```json
{"auth": {"type": "oauth", "scope": "mcp:tools", "client_id": "optional", "client_secret": "${VAR}"}}
```

With no `client_id`, AgentX registers itself dynamically (RFC 7591) after discovering the
authorization server via protected-resource metadata (RFC 9728); `client_id`/`client_secret`
are for providers without dynamic registration. Tokens + the registration persist per server
under `data/mcp_oauth/` and refresh automatically.

```
GET  /api/mcp/oauth/callback              # OAuth redirect target (public; state-validated)
POST /api/mcp/servers/{name}/auth/reset   # forget tokens + registration (fresh sign-in)
POST /api/mcp/servers/{name}/auth/cancel  # abort an in-flight sign-in (leaves stored tokens)
```

The callback is the loopback redirect URI (RFC 8252) registered with the authorization
server — override the advertised URL with `AGENTX_OAUTH_REDIRECT_URL` when the API is not
on `http://localhost:12319`. Server payloads from `GET /api/mcp/servers` carry an
`auth_state` object (`authorized` / `pending` / `error`) for OAuth servers. `authorized`
means real tokens are stored — not merely that a registration file exists (the SDK writes
that at dynamic-registration time, before consent), so a cancelled or denied sign-in never
reads as authorized. `auth/cancel` aborts the pending browser consent flow so a late
completion can't retroactively flip the server to signed-in.

---

## Providers

### List Providers

```
GET /api/providers
```

Returns configured model providers and their status.

**Response:**
```json
{
  "providers": [
    {
      "name": "lmstudio",
      "status": "configured",
      "models": ["llama3.2", "deepseek-r1"]
    },
    {
      "name": "anthropic",
      "status": "not_configured",
      "error": "API key not set"
    }
  ]
}
```

### List Models

```
GET /api/providers/models
GET /api/providers/models?provider=openai
```

Returns all available models with capabilities. Filter by provider with `?provider=`.

**Response:**
```json
{
  "models": [
    {
      "name": "claude-3-5-sonnet-latest",
      "provider": "anthropic",
      "context_window": 200000,
      "supports_tools": true,
      "supports_vision": true,
      "supports_streaming": true,
      "cost_per_1k_input": 0.003,
      "cost_per_1k_output": 0.015
    }
  ],
  "count": 5
}
```

### Provider Health

```
GET /api/providers/health
```

Async health check of all configured providers.

**Response:**
```json
{
  "status": "healthy",
  "providers": {
    "lmstudio": {"status": "healthy"},
    "anthropic": {"status": "healthy"},
    "openai": {"status": "unhealthy", "error": "..."}
  }
}
```

`status` is `"healthy"` if all pass, `"degraded"` if any fail.

### Model Roles

```
GET /api/models/roles
```

The three model roles (`fast_utility`, `deep_reasoning`, `summarizer`) with their configured models, plus every member setting's current resolution chain. A member with an empty value follows its role; an explicit value always wins; any model setting may also be set to `role:<name>` explicitly.

**Response:**
```json
{
  "roles": {
    "summarizer": {"label": "Summarizer", "description": "…", "model": ""}
  },
  "members": [
    {"member": "compression", "label": "Tool-output compression",
     "role": "summarizer", "kind": "config", "source": "compression.model",
     "explicit": "", "role_model": "", "effective": "", "following": "fallback"}
  ]
}
```

`following` is `explicit` (member's own value wins), `role` (inheriting the role's model), or `fallback` (neither set — the provider fallback chain decides). Set roles via `POST /api/config/update` with `{"models": {"roles": {"summarizer": "provider:model"}}}` — `""` clears a role; `role:` refs and non-`provider:model` strings are rejected.

---

## Agent

### Run Task

```
POST /api/agent/run
```

Execute a task using the full agent pipeline (planning + reasoning).

**Request:**
```json
{
  "task": "Analyze the sentiment of this text: I love this product!",
  "reasoning_strategy": "chain_of_thought"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | yes | Task description |
| `reasoning_strategy` | string | no | `"auto"`, `"cot"`, `"tot"`, `"react"`, `"reflection"` |

**Response:**
```json
{
  "task_id": "a1b2c3d4",
  "status": "complete",
  "answer": "The sentiment is positive...",
  "plan_steps": 2,
  "reasoning_steps": 3,
  "tools_used": [],
  "models_used": ["llama3.2"],
  "total_tokens": 450,
  "total_time_ms": 1234.5
}
```

### Chat

```
POST /api/agent/chat
```

Conversational interaction with session management.

**Request:**
```json
{
  "message": "What can you help me with?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "claude-3-5-sonnet-latest",
  "profile_id": "default",
  "temperature": 0.7,
  "use_memory": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | yes | — | User message |
| `session_id` | string | no | auto-generated | Session ID for continuity |
| `model` | string | no | from config | Model to use |
| `profile_id` | string | no | `"default"` | Prompt profile ID |
| `temperature` | float | no | `0.7` | Sampling temperature |
| `use_memory` | bool | no | `true` | Enable memory recall/storage |

**Response:**
```json
{
  "task_id": "a1b2c3d4",
  "status": "complete",
  "response": "I can help you with...",
  "answer": "I can help you with...",
  "thinking": "Let me consider what this user needs...",
  "has_thinking": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reasoning_trace": 0,
  "reasoning_steps": 0,
  "tokens_used": 150,
  "total_tokens": 150,
  "total_time_ms": 892.3
}
```

### Chat (Streaming)

```
POST /api/agent/chat/stream
```

Same request body as `/agent/chat`. Optional extras: `workflow_id` (run an Agent Alloy workflow — a "Team" in the UI), `target_agent_id` (route this turn to a specific agent by its Docker-style `agent_id`), `disable_delegation` (Solo mode — suppress ad-hoc delegation for this turn: no `delegate_to` tool, no roster prompt; ignored when `workflow_id` is set), `workspace_id` (attach a document workspace), and `images` (**vision input** — an array of `{workspace_id, doc_id, media_type}` refs to images already uploaded via `POST /agent/chat/images`). A vision-capable model receives the images as image blocks; a non-vision model gets the text only, with a `status` notice. An image-only turn may send an empty `message`. Routing priority: `workflow_id` > `target_agent_id` > `agent_profile_id` > default; an unknown `target_agent_id` yields an `error` event. Returns Server-Sent Events (SSE).

**Response:** `Content-Type: text/event-stream`

SSE events in order:

| Event | Data | When |
|-------|------|------|
| `run_started` | `{"run_id": "..."}` | First event — identifies the detached run for re-attach |
| `start` | `{"task_id": "...", "model": "..."}` | Generation begins |
| `chunk` | `{"content": "token text"}` | Each token |
| `status` | `{"phase": "running_tool", "label": "Running web_search…", "detail"?, "group"?, "progress"?}` | Coarse per-phase activity (`recalling`/`composing`/`thinking`/`running_tool`/`reading`) for a live status line; rides the run bus, so it replays on re-attach |
| `steer` | `{"id": "...", "message": "..."}` | A user steered the running turn (folded in as a fresh user turn at the next safe boundary); echoed so every client shows the steer bubble inline |
| `tool_call` | `{"tool": "name", "arguments": {...}}` | Tool invocation starts |
| `tool_result` | `{"tool": "name", "content": "..."}` | Tool result (truncated to 500 chars) |
| `workspace_attached` | `{"workspace_id": "..."}` | A generated image landed in a workspace; emitted after the `image` exhibit so a conversation with no workspace can durably attach the personal Home store client-side |
| `done` | `{"task_id": "...", "thinking": "...", "has_thinking": bool, "total_time_ms": float, "session_id": "..."}` | Generation complete |
| `error` | `{"error": "message"}` | On failure |
| `close` | `{}` | Run settled; tail ends |

The streaming endpoint supports the same tool-use loop as the non-streaming endpoint (up to 10 rounds). Memory storage happens after the stream completes.

**Detached execution.** The run is driven by a server-side daemon thread that fans SSE events into a Redis stream and persists turns on completion — independent of the HTTP connection. Closing or switching the tab does **not** stop the run; it plays to completion and can be re-attached. This is what prevents the in-flight response (and, for a new chat, the whole conversation) from being lost on disconnect.

**Example with curl:**
```bash
curl -N -X POST http://localhost:12319/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "llama3.2"}'
```

### Re-attach to a Run

```
GET /api/agent/chat/stream/attach?run_id=<id>
```

Replays the buffered SSE events for a detached run from the start, then follows live until completion. A reopened tab uses this to resume an in-flight conversation. Emits a `run_missing` event (instead of replaying) when the run's buffer has expired — the client then restores from conversation history.

### List Detached Runs

```
GET /api/agent/chat/runs
```

Lists the caller's detached chat runs (newest first, capped at 50). Recovery surfaces — the Relay inbox and the conversation selector — use this to find runs whose owning tab was closed and offer to re-attach. Runs are indexed per user, so callers only see their own. Each entry: `{run_id, status, message, session_id, created_at, updated_at}`.

### Cancel a Run

```
POST /api/agent/chat/runs/{run_id}/cancel
```

Cooperatively cancels a detached run; the runner checks the flag at SSE-event boundaries and stops pulling from the provider. Returns `{"run_id": "...", "cancel_requested": bool}`.

### Steer a Run (live steering)

```
POST /api/agent/chat/runs/{run_id}/steer
```

Body: `{"message": "...", "mode"?: "queue"}`. Folds a message into an in-flight turn **without stopping it**: the message is queued and the streaming tool loop drains it at the next safe boundary (after a tool round, or instead of ending) and folds it in as a fresh user turn so the agent course-corrects. Owner-only (it injects content). The message is echoed onto the run's event bus as a `steer` event so every connected client (live + re-attached) shows the steer bubble inline. Returns `{"run_id": "...", "steer_accepted": bool}`. `400` blank message, `403` not the owner, `404` unknown run, `409` run not active.

### Ambassador — Brief a Turn

```
POST /api/agent/ambassador/brief-turn
POST /api/agent/ambassador/ask
POST /api/agent/ambassador/draft
POST /api/agent/ambassador/voice-command
POST /api/agent/ambassador/speak
POST /api/agent/ambassador/transcribe
GET  /api/agent/ambassador/stream?run_id=<id>
POST /api/agent/avatar/generate                  # generate an agent avatar (OpenRouter) → Home workspace
POST /api/agent/chat/images                       # upload an image for vision input (multipart `file`) → Home workspace; returns a ChatImageRef
POST /api/agent/ambassador/relay
POST /api/agent/ambassador/dispatch
GET  /api/agent/ambassador/threads
POST /api/agent/ambassador/threads
GET    /api/agent/ambassador/thread/{thread_id}
PATCH  /api/agent/ambassador/thread/{thread_id}
DELETE /api/agent/ambassador/thread/{thread_id}
GET  /api/agent/ambassador/{conversation_id}
```

The **Ambassador** (Phase 16.6) is a dedicated agent that runs *parallel* to a conversation and briefs you on a single turn — a middleman between the conversation and you for large-context / complex situations — **without polluting** the conversation. It is any agent profile with an `ambassador` section; the global default is `config.ambassador.profile_id` (falls back to the default agent profile).

- **`POST /brief-turn`** — body `{conversation_id, message_id, assistant_text, user_text?, agent_name?, artifacts?}`. Starts a parallel briefing of one turn and returns `{run_id}`. The run is detached and **un-indexed** (`indexed=false`), so it never appears in `/agent/chat/runs`. It reads conversation context read-only and writes only to a Redis **sidecar** (the `ambassador:` key prefix) — never `conversation_logs` or the rolling summary, so nothing it produces re-enters the main agent's context. `agent_name` lets the briefing speak of the agent by name; `artifacts` (`{tools?, sources?, exhibits?}`, gathered + compacted client-side) carries what the agent actually *did* this turn so the briefing grounds on the turn's substance, not just its prose.
- **`POST /ask`** — body `{conversation_id, qa_id, question, agent_name?, artifacts?}`. Ask the ambassador a free-form question about the conversation; returns `{run_id, qa_id}`. Same detached/un-indexed/sidecar machinery as `brief-turn`, persisting under the `qa:` key family so Q&A replays independently of per-turn briefings. The answer is grounded only in the conversation (read-only).
- **`POST /draft`** — body `{conversation_id, intent, agent_name?, artifacts?, fresh?}`. The **outbound relay** (you → agent): the ambassador shapes a rough intent into a clear, first-person message and returns `{draft}` for you to review/edit. The client then relays it into the conversation as a real *user* turn (or steers the running turn) — the ambassador never speaks into the transcript itself, so the no-pollution invariant holds. With `fresh: true` (dispatch) the draft is a **self-contained task** for a worker (`agent_name`) to start cold — no `conversation_id` required. Degrades to the raw intent when no provider is configured.
- **`POST /dispatch`** — body `{agent_id, text}`. The ambassador **write-side** (hand a task to a worker): mints a **brand-new conversation** and runs the chosen worker (resolved by `agent_id`, agents only) **headless** on `text` as its first **user** turn (`enqueue_background_chat`). Returns `{ok, conversation_id, job_id}` so the client opens + watches the new conversation. You authored the task — the ambassador never speaks into the transcript as itself (INV-2). Gated by `config.ambassador.dispatch.enabled` (default on; `422` when off); unknown/non-worker `agent_id` → `400`. v1 is confirm-first (you pick the worker + send).
- **`POST /voice-command`** — body `{conversation_id, transcript, agent_name?, artifacts?}`. **Voice-mode intent routing:** the ambassador interprets a spoken command and returns `{action: "answer"|"relay", text, qa_id?}` — a question it answers (spoken; persisted to the `qa:` sidecar so the Text tab shows it), or an instruction it drafts as a first-person **relay** the user reviews and sends into the conversation as a real user turn (never auto-sent). Never fails the call (degrades to a spoken notice). `text` for a `relay` is shaped via the draft persona; the JSON is forward-compatible with a `target` for future cross-agent delegation.
- **`POST /speak`** — body `{text, agent_profile_id?, voice?, model?}`. **Voice (TTS):** synthesizes spoken audio for a briefing summary or Q&A answer via the ambassador's speech model (OpenRouter `/audio/speech`) and returns the raw **MP3** bytes (`audio/mpeg`). Model/voice precedence: request override → the ambassador profile's `speech_model`/`voice` → `config.ambassador.*` → shipped default (`microsoft/mai-voice-2`); the speech model resolves strictly (no chat fallback). Stateless — the client already holds the text, so nothing is written to the transcript or sidecar. Degrades to a structured `422 {error, code}` (e.g. `voice_unconfigured` when no OpenRouter key is set) rather than failing. Played by the client's `SpeechPlayer` (synthesized once, cached for replay); opted-in ambassadors get an immersive **voice mode** that auto-speaks new briefings.
- **`POST /transcribe`** — body `{audio: <base64>, format?, agent_profile_id?, model?, language?}`. **Voice input (STT):** transcribes a push-to-talk recording via the ambassador's STT model (OpenRouter `/audio/transcriptions`) and returns `{text}`. Model precedence: request override → the ambassador profile's `transcription_model` → `config.ambassador.*` → shipped default (`openai/whisper-1`), strict. The client routes the transcript into the **reviewable input** — it is **never** auto-sent (pre-send confirmation), and a flubbed take can be re-recorded (retake). Degrades to a structured `422 {error, code}` (e.g. `transcription_unconfigured`). Client capture: `lib/audioRecorder.ts` (getUserMedia + MediaRecorder) behind `hooks/useDictation.ts`; hold-Space or the immersive voice-mode record button drives it.
- **`GET /stream?run_id=`** — tail a briefing **or Q&A** run's namespaced SSE: `ambassador_start`, `ambassador_chunk` (one per streamed delta), `ambassador_done` (status `done` | `empty_provider` | `cancelled`), `ambassador_error`; `run_missing` if the buffer expired. Cancel via `POST /api/agent/chat/runs/{run_id}/cancel` (settles the sidecar to `cancelled`, preserving partial text). A missing/unreachable provider degrades gracefully to an `empty_provider` notice rather than failing.
- **`GET /thread/{thread_id}`** — replay the unified ambassador thread (an **"Inquiry"**): `{thread_id, title, entries: [{id, kind: "briefing"|"qa", question, content, status, toolCalls, message_id?, run_id?, created_at, updated_at}]}`. Briefings and Q&A are one ordered conversation (oldest-first), each entry carrying its **persisted tool-call chips** (they now survive a reload), plus the thread's own `title`. `thread_id` defaults to the conversation id. The client renders this as one stream and lets you rename the Inquiry.
- **`PATCH /thread/{thread_id}`** — rename the Inquiry, body `{title}` → `{thread_id, title}`. An empty title clears it (the client falls back to the chat conversation's title).
- **`DELETE /thread/{thread_id}`** — clear the Inquiry (its briefings, answers, and title) → `{thread_id, cleared: true}`.
- **`GET /{conversation_id}`** — **back-compat shim**: replay as the old `{conversation_id, briefings: [...], qa: [...]}` shape (now projected from the unified thread). New clients use `/thread/{thread_id}`.

### Agent Status

```
GET /api/agent/status
```

Returns current agent status (`idle`, `planning`, `reasoning`, `executing`, `complete`, `failed`, `cancelled`).

### Cancel Plan

```
POST /api/agent/plans/cancel
```

Requests cancellation of the agent's active plan execution.

### Plan Status

```
GET /api/agent/plans/{plan_id}/status?session_id=<sid>
```

Reads the Redis-tracked state of a plan (status, completed count, per-subtask
state) so a client can reconcile a persisted "running" plan after a reload.
Returns `{ "found": false }` with HTTP 200 when the state has expired (1h TTL).
The `resumable` flag is `true` when the plan is active with non-terminal work
left and carries a structural snapshot.

### Resume Plan

```
POST /api/agent/plans/{plan_id}/resume
```

Resumes an interrupted plan. Rebuilds it from Redis (`PlanStateStore.load_plan`)
and continues executing only its not-yet-terminal subtasks, streaming SSE the
same way the chat endpoint does (first event `plan_resumed`, then subtask events
→ `plan_complete` → `done`). The run is detached (survives client disconnect)
and the synthesis is persisted as an assistant turn. Single-agent only for now
(alloy plan resumption is a follow-up). Returns `404` when the plan is not
resumable (missing, expired, or already finished).

Body: `{ "session_id": str, "agent_profile_id"?: str, "model"?: str, "temperature"?: number, "use_memory"?: bool }`

---

## Background Chat

Long-running conversations can be queued and polled instead of streamed.

```
POST /api/chat/background
GET  /api/chat/background              # list recent jobs (?limit=, max 50)
GET  /api/chat/background/{job_id}     # fetch one job
DELETE /api/chat/background/{job_id}   # dismiss from the inbox
```

**Request (POST):** `message` is required; the rest mirror `/agent/chat`.
```json
{
  "message": "Summarize today's research notes",
  "session_id": "optional",
  "agent_profile_id": "optional",
  "workflow_id": "optional",
  "model": "optional",
  "use_memory": true
}
```

**Response (POST):** `202 Accepted`
```json
{ "job_id": "a1b2c3d4", "status": "queued" }
```

Poll `GET /api/chat/background/{job_id}` for the job record (status + result once complete);
`404` if unknown.

---

## Tool Outputs

Oversized tool outputs are compressed and stored for later retrieval (see Context Gating).

```
GET    /api/tool-outputs                 # ?pattern=<tool-name-glob> (default *)
GET    /api/tool-outputs/{storage_key}   # ?offset= &limit= &metadata_only=true
DELETE /api/tool-outputs/{storage_key}
```

**Response (list):**
```json
{ "outputs": [ { "storage_key": "...", "tool": "...", "size": 18342, "created_at": "..." } ], "count": 1 }
```

The detail endpoint returns the full content, a paginated slice (`offset`/`limit`), or
metadata only (`metadata_only=true`). `404` if the key is missing or expired.

---

## Workspaces (Projects)

File workspaces & document RAG — surfaced in the client as **Projects**. A workspace is a
named, persistent container of uploaded files with a searchable manifest, plus user-authored
**description** and **instructions** (the instructions ride every chat turn's context), and
**durable conversation membership** (a conversation belongs to at most one project). Bytes
live in a content-addressed blob store; the manifest in Postgres; chunk vectors in pgvector.
Upload validates type/size/quota then ingests in the background (parse → chunk → embed →
auto tag + summary), so a document moves `pending` → `ready` (or `failed`).

```
GET    /api/workspaces                               # list
POST   /api/workspaces                               # { "name": "..." } → 201
GET    /api/workspaces/{workspace_id}                # detail (description, instructions, document_count, used_bytes, allow_shell, shell_backend)
PATCH  /api/workspaces/{workspace_id}                # { name?, description?, instructions?, allow_shell?, shell_backend? }
DELETE /api/workspaces/{workspace_id}                # delete (cascades documents + blobs + memberships + shell container)
GET    /api/workspaces/{workspace_id}/documents      # manifest list (tags/summary/status)
POST   /api/workspaces/{workspace_id}/documents      # multipart field `file` → 201 (status=pending)
POST   /api/workspaces/{workspace_id}/documents/text # { filename, content } → 201 (create md/txt doc; 409 on filename collision)
PUT    /api/workspaces/{workspace_id}/documents/{document_id}/text  # { content, expected_sha256? } → 200 (replace content, re-ingests; 409 on ETag mismatch)
GET    /api/workspaces/{workspace_id}/documents/{document_id}
DELETE /api/workspaces/{workspace_id}/documents/{document_id}
GET    /api/workspaces/{workspace_id}/documents/{document_id}/raw   # serve blob bytes (e.g. generated avatars)
POST   /api/workspaces/{workspace_id}/documents/{document_id}/reingest  # retry ingestion (blob reused) → 202
GET    /api/workspaces/{workspace_id}/conversations                 # the project's conversations (same shape as /api/conversations)
PUT    /api/workspaces/{workspace_id}/conversations/{conversation_id}    # add to project (upsert — moves from any other project)
DELETE /api/workspaces/{workspace_id}/conversations/{conversation_id}    # remove from project
GET    /api/workspaces/{workspace_id}/shell/container          # container backend: status + live stats
POST   /api/workspaces/{workspace_id}/shell/container/{action} # action ∈ start|stop|reset|remove
```

`description` is capped at 500 chars and `instructions` at 8000 (`400` beyond). Membership
`PUT` returns `400` for a non-UUID conversation id or for `ws_home` (the reserved personal
Home space is not a project); `DELETE` returns `404` if the conversation isn't linked to
that workspace. Conversation rows from `/api/conversations` carry a `workspace_id` field
(null when the conversation is in no project).

**Response (manifest list):**
```json
{ "documents": [ {
  "id": "doc_…", "filename": "report.pdf", "content_type": "application/pdf",
  "size_bytes": 174892, "sha256": "…", "tags": ["…"], "summary": "…",
  "status": "ready", "error": null, "created_at": "…", "updated_at": "…"
} ] }
```

Upload errors: `415` unsupported file type, `413` per-file size limit or workspace quota
exceeded. Supported v1 types: PDF + text/markdown/code.

**Text documents (create/update):** the `/documents/text` endpoints accept JSON and are limited
to agent-writable types (`md`/`markdown`/`txt`; filenames may carry one folder level, e.g.
`research/notes.md`). Create returns `409` with `{ code: "conflict", document_id }` when the
filename already exists — update is an explicit, separate act. Update replaces the whole
content and re-ingests (`status` returns to `pending`); pass the document's last-known
`expected_sha256` for optimistic concurrency — a mismatch returns `409` with `current_sha256`.
Identical content is a no-op (no re-ingest). These endpoints back the Projects hub editor and
mirror the agent's `create_document`/`update_document` tools.

**Attaching to a conversation:** pass `workspace_id` in the `/api/agent/chat/stream` request body,
or link the conversation durably via `PUT /api/workspaces/{id}/conversations/{conversation_id}`.
Turn precedence: an explicit request `workspace_id` wins (and self-heals the membership record);
otherwise the server falls back to the conversation's stored membership and emits a
`workspace_attached` SSE event so the client re-learns the binding. The agent then sees a
**project identity block** (always, even for an empty project), the project's **instructions**,
and the file manifest in its context, and can call the project tools (`project_search` — legacy
alias `workspace_search` still executes — `document_query`, `read_document`, plus
`create_document`/`update_document` for durable writes); document hits are auto-cited
(`source_type: "doc"`).

**Agent shells (opt-in, per workspace).** Set `allow_shell: true` on a workspace to expose sandboxed
shell tools (`run_command`, `write_file`/`read_file`/`list_files`) to agents whose conversation is
attached to it. `shell_backend` picks the sandbox:
- `bubblewrap` (default) — a locked-down jail: filesystem limited to a per-conversation working copy of
  the workspace, **no network**, scrubbed env, time-limited. Needs `bubblewrap` on the server.
- `container` — a **persistent per-workspace Docker container** the agent can `pip`/`apt`-install into,
  with **network on** (its own bridge; no access to AgentX's DBs/secrets). Requires `shell.docker.enabled`
  and a reachable Docker daemon (dev: host Docker; prod: the dind sidecar in `docker-compose.shell.yml`).
  Manage it via the `/shell/container` endpoints (status/stats + `start`/`stop`/`reset`/`remove`); the
  container is removed when the workspace is deleted.

```json
// GET /api/workspaces/{id}/shell/container
{ "container": { "state": "running", "image": "python:3.14-slim",
  "memory_usage": "48MiB / 2GiB", "cpu_percent": "0.10%", "install_size": "12MB (virtual 160MB)",
  "last_used_at": 1718900000, "idle_gc_at": 1719504800 } }
```

---

## Agent Profiles

Agent profiles define identity plus per-agent settings (model, temperature, prompt, memory channel).

```
GET    /api/agent/profiles
POST   /api/agent/profiles
GET    /api/agent/profiles/{profile_id}
PATCH  /api/agent/profiles/{profile_id}
DELETE /api/agent/profiles/{profile_id}
POST   /api/agent/profiles/{profile_id}/set-default
POST   /api/agent/profiles/{profile_id}/set-default-ambassador
```

Profiles carry a `kind` (`agent` or `ambassador`). Agents and ambassadors have
**separate defaults**: `set-default` marks the default agent; `set-default-ambassador`
marks the default ambassador (the one briefings use). Ambassador-kind profiles are
hidden from the chat agent selector, delegation, and `@`-mention routing.

**Profile object** (returned by GET/POST):
```json
{
  "id": "default",
  "name": "AgentX",
  "agent_id": "bold-cosmic-falcon",
  "avatar": null,
  "description": "General-purpose assistant",
  "tags": ["research", "fast"],
  "default_model": "lmstudio:llama3.2",
  "temperature": 0.7,
  "prompt_profile_id": "default",
  "system_prompt": null,
  "reasoning_strategy": "auto",
  "enable_memory": true,
  "memory_channel": "_global",
  "enable_tools": true,
  "direct_mode": false,
  "allowed_tools": null,
  "blocked_tools": [],
  "available_for_delegation": false,
  "delegation_hint": null,
  "is_default": true,
  "created_at": "2026-04-01T12:00:00",
  "updated_at": "2026-04-01T12:00:00"
}
```

`POST` requires `name` (an `id` and `agent_id` are generated if omitted) and returns `201`.
`PATCH` accepts a partial object; `agent_id` is immutable.

`available_for_delegation` (opt-in, default `false`) puts the profile on the ad-hoc
delegation roster ("Join the team roster" in the UI); `delegation_hint` is a one-line
specialty shown to teammates deciding whom to delegate to (falls back to `description`;
trimmed, max 200 chars).

---

## Multi-Agent (Agent Alloy)

Supervisor + specialist workflows with delegation over shared memory channels (Phase 16, v1).
**"Agent Teams" is the user-facing name** for this system — routes, payloads, and config keys
keep the `alloy` vocabulary. See the [Agent Teams feature guide](../features/multi-agent.md)
for concepts.

```
GET    /api/alloy/workflows
POST   /api/alloy/workflows
GET    /api/alloy/workflows/{workflow_id}
PATCH  /api/alloy/workflows/{workflow_id}
DELETE /api/alloy/workflows/{workflow_id}
```

**Workflow object** (responses wrap a single workflow as `{"workflow": {...}}`):
```json
{
  "id": "research-team",
  "name": "Research Team",
  "description": "Supervisor delegates lookups to a researcher",
  "supervisor_agent_id": "bold-cosmic-falcon",
  "members": [
    { "agent_id": "bold-cosmic-falcon", "role": "supervisor", "delegation_hint": null },
    { "agent_id": "calm-lunar-otter", "role": "specialist", "delegation_hint": "web research" }
  ],
  "routes": [],
  "shared_channel": "_alloy_research-team",
  "canvas": {},
  "created_at": "2026-04-28T04:18:41",
  "updated_at": "2026-04-29T03:28:38"
}
```

`POST` validates the workflow (kebab-case `id`, exactly one supervisor, known `agent_id`s) and
returns `201`; `400` on validation failure. To run a workflow, pass `workflow_id` to
`POST /api/agent/chat/stream`. `DELETE` returns `{"deleted": true}`.

---

## Conversations

```
GET /api/conversations                          # ?limit= (max 100) &offset= &channel=
GET /api/conversations/{conversation_id}/messages
```

**Response (list):** conversations summarized from `conversation_logs`.
```json
{
  "conversations": [
    {
      "conversation_id": "…",
      "created_at": "…",
      "last_message_at": "…",
      "message_count": 12,
      "channel": "_global",
      "first_user_message": "…",
      "last_message": "…"
    }
  ],
  "total": 1
}
```

**Response (messages):** each message carries `role`, `content`, `timestamp`, `turn_index`, and a `metadata` object. For assistant turns produced under a known agent profile, `metadata.agent_id` (Docker-style id) and `metadata.agent_name` (resolved display name) attribute the message — used to reconstruct multi-agent transcripts on reload. Absent on historical/unattributed turns.

---

## Prompts

### List Profiles

```
GET /api/prompts/profiles
```

**Response:**
```json
{
  "profiles": [
    {
      "id": "default",
      "name": "Default Assistant",
      "description": "General-purpose AI assistant",
      "is_default": true,
      "sections_count": 3,
      "enabled_sections": 2
    }
  ]
}
```

### Profile Detail

```
GET /api/prompts/profiles/{profile_id}
```

Returns the full profile with all sections and the composed prompt preview.

**Response:**
```json
{
  "profile": {
    "id": "default",
    "name": "Default Assistant",
    "description": "General-purpose AI assistant",
    "is_default": true,
    "sections": [
      {
        "id": "identity",
        "name": "Identity",
        "type": "system",
        "content": "You are a helpful AI assistant.",
        "enabled": true,
        "order": 0
      }
    ]
  },
  "composed_prompt": "You are a helpful AI assistant.\n\n..."
}
```

### Get Global Prompt

```
GET /api/prompts/global
```

> **Now a back-compat shim.** The global system prompt is composed from a durable
> **layer stack** (see *Prompt Stack* below). This endpoint returns the composed stack;
> `global/update` persists the posted blob as the reserved `legacy-global` layer.

**Response:**
```json
{
  "global_prompt": {
    "content": "Always be helpful and concise.",
    "enabled": true
  }
}
```

### Update Global Prompt

```
POST /api/prompts/global/update
```

**Request:**
```json
{
  "content": "Always be helpful and concise.",
  "enabled": true
}
```

### Prompt Stack (Layers)

The conversational global system prompt is composed from an ordered stack of editable
**layers**. Built-in layers ship a versioned `default` (the sidecar); the user's edit is
stored separately as an `override` (`effective = override ?? default`). Untouched built-ins
keep receiving release improvements; edited layers are pinned and never silently overwritten.
A released bump to a built-in's `default` surfaces `update_available` (review the diff, then
**acknowledge** to keep your text or **reset** to adopt the new default).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prompts/layers` | GET / POST | List the stack (`{layers, composed}`) or create a custom layer (`{title, content?}`) |
| `/api/prompts/layers/reorder` | POST | Reorder the stack (`{order: [id, …]}`) |
| `/api/prompts/layers/{id}` | PATCH / DELETE | Update (`{content?, title?, enabled?}` — `content` sets the override) or delete a custom layer (built-ins can't be deleted) |
| `/api/prompts/layers/{id}/reset` | POST | Reset a built-in's override back to the shipped default |
| `/api/prompts/layers/{id}/acknowledge` | POST | Mark a bumped built-in default as seen (keep the override, clear the badge) |

**Layer shape:**
```json
{
  "id": "core-principles",
  "title": "Core Principles",
  "kind": "builtin",
  "default": "You are an intelligent AI assistant…",
  "default_version": 1,
  "override": null,
  "base_version": null,
  "effective": "You are an intelligent AI assistant…",
  "enabled": true,
  "order": 0,
  "modified": false,
  "update_available": false
}
```

### List Sections

```
GET /api/prompts/sections
```

Returns all available prompt sections with their content and ordering.

### Compose Preview

```
GET /api/prompts/compose
GET /api/prompts/compose?profile_id=default
```

Returns the fully composed system prompt that would be sent to the model.

**Response:**
```json
{
  "system_prompt": "You are a helpful AI assistant.\n\nAlways be helpful and concise.\n\n...",
  "profile_id": "default"
}
```

### MCP Tools Prompt

```
GET /api/prompts/mcp-tools
```

Returns the auto-generated prompt describing available MCP tools for injection into the system prompt.

**Response:**
```json
{
  "mcp_tools_prompt": "You have access to the following tools:\n\n- read_file: Read a file...",
  "tools_count": 5
}
```

### Prompt Templates

Reusable prompt snippets with tags, plus an LLM-backed prompt enhancer.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prompts/templates` | GET / POST | List or create templates |
| `/api/prompts/templates/tags` | GET | List template tags |
| `/api/prompts/templates/{template_id}` | GET / PUT / DELETE | Get, update, or delete a template |
| `/api/prompts/templates/{template_id}/reset` | POST | Reset a template to its default |
| `/api/prompts/enhance` | POST | Rewrite/enhance a prompt via the LLM |
| `/api/prompts/feature-defaults` | GET | Shipped defaults for the overridable feature prompts (extraction, relevance, planner, enhancement) |

---

## Memory

### Channels

```
GET  /api/memory/channels
POST /api/memory/channels
```

**GET** — List all channels with item counts per type.

**Response:**
```json
{
  "channels": [
    {
      "name": "_global",
      "is_default": true,
      "item_counts": {
        "turns": 42,
        "entities": 15,
        "facts": 28,
        "strategies": 3,
        "goals": 2
      }
    }
  ]
}
```

**POST** — Create a new channel. Name must be alphanumeric with hyphens/underscores.

**Request:** `{"name": "my-project"}`

### Delete Channel

```
DELETE /api/memory/channels/{name}
```

Deletes a channel and all associated data across Neo4j, PostgreSQL, and Redis. Cannot delete `_global`.

**Response:**
```json
{
  "message": "Channel 'my-project' deleted successfully",
  "deleted": {
    "turns": 10, "entities": 5, "facts": 8,
    "strategies": 1, "goals": 0, "conversations": 2,
    "postgres_rows": 15, "redis_keys": 3
  }
}
```

### Delete Conversation

```
DELETE /api/memory/conversations/{conversation_id}
```

Deletes a single conversation and its turns from all databases.

### List Entities

```
GET /api/memory/entities
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |
| `search` | string | — | Text search on entity name |
| `type` | string | — | Filter by entity type |

**Response:**
```json
{
  "entities": [...],
  "total": 45,
  "page": 1,
  "limit": 20,
  "has_next": true
}
```

### Entity Graph

```
GET /api/memory/entities/{entity_id}/graph
GET /api/memory/entities/{entity_id}/graph?depth=3
```

Returns the entity with connected facts and relationships.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `depth` | int | 2 | Traversal depth (max 3) |

Returns 404 if entity not found.

### List Facts

```
GET /api/memory/facts
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |
| `min_confidence` | float | 0.0 | Minimum confidence (0.0–1.0) |
| `search` | string | — | Text search on fact claim |

### List Strategies

```
GET /api/memory/strategies
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |

### List Procedures

```
GET /api/memory/procedures
```

Distilled procedural memory — the "how we work here" deltas the `distill_procedures`
consolidation job mints from corrections/steers and explicit user rules. Each procedure has a
natural-language `trigger`, a replayable `body`, a `rationale`, a `scope` (channel), and a
`strength` (replay/reinforce count).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `_all` | Filter by channel |
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Items per page (max 100) |

### Memory Stats

```
GET /api/memory/stats
```

Returns total counts and per-channel breakdowns.

**Response:**
```json
{
  "totals": {"entities": 45, "facts": 120, "strategies": 8, "turns": 300, "procedure_candidates": 5},
  "by_channel": {
    "_global": {"entities": 30, "facts": 80, "strategies": 5, "turns": 200},
    "my-project": {"entities": 15, "facts": 40, "strategies": 3, "turns": 100}
  }
}
```

### Usage Metrics

```
GET /api/metrics/usage
```

Aggregates the unified `usage_events` spend ledger (chat, multi-agent, ambassador, and
voice TTS/STT) into overall totals plus per-model, per-agent, and per-source breakdowns
and a daily time series. `avg_latency_ms` is read from `conversation_logs`. Degrades
gracefully (zeros + `"unavailable": true`, HTTP 200) when the database is offline.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `days` | int | 14 | Window size in days (clamped 1–90) |

**Response:**
```json
{
  "totals": {
    "turns": 42, "tokens_input": 18500, "tokens_output": 9200,
    "tokens_total": 27700, "cost_total": 0.184, "cost_currency": "USD",
    "avg_latency_ms": 2310.5
  },
  "by_model": [
    {"model": "claude-opus-4-7", "turns": 30, "tokens_input": 15000,
     "tokens_output": 7000, "tokens_total": 22000, "cost_total": 0.16}
  ],
  "by_agent": [
    {"agent_id": "bold-cosmic-falcon", "turns": 18, "tokens_input": 9000,
     "tokens_output": 4500, "tokens_total": 13500, "cost_total": 0.11},
    {"agent_id": "_default", "turns": 4, "tokens_input": 1200,
     "tokens_output": 600, "tokens_total": 1800, "cost_total": 0.012}
  ],
  "by_source": [
    {"source": "chat", "turns": 30, "tokens_input": 15000,
     "tokens_output": 7000, "tokens_total": 22000, "cost_total": 0.16},
    {"source": "ambassador_tts", "turns": 6, "tokens_input": 0,
     "tokens_output": 0, "tokens_total": 0, "cost_total": 0.004}
  ],
  "daily": [
    {"date": "2026-05-28", "turns": 20, "tokens_total": 13000, "cost_total": 0.09}
  ],
  "days": 14
}
```

### Consolidation Settings

```
GET  /api/memory/settings
POST /api/memory/settings
```

**GET** — Returns consolidation settings (extraction, relevance filter, entity linking, quality thresholds) plus default prompts, and `settings_file_status` (`{path, exists, error}` — `error` is non-null when a corrupt overrides file forced the defaults fallback).

**POST** — Update consolidation settings. Accepts partial updates. Values are schema-validated: any invalid value rejects the whole update with `400 {"error", "errors": {key: message}}` (nothing persisted). Saved changes apply live — no API restart needed.

### Recall Settings

```
GET  /api/memory/recall-settings
POST /api/memory/recall-settings
```

**GET** — Returns recall layer settings (hybrid search, entity-centric, query expansion, HyDE, self-query technique toggles and parameters).

**POST** — Update recall settings. Accepts partial updates, with the same schema validation and `400 {"error", "errors"}` reject-whole contract as `/api/memory/settings`.

### Run Consolidation

```
POST /api/memory/consolidate
```

Manually triggers the consolidation pipeline.

**Request (optional):**
```json
{"jobs": ["consolidate", "patterns", "promote"]}
```

If no jobs specified, runs the default set: consolidate, patterns, promote.

### Reset Consolidation

```
POST /api/memory/reset
```

Clears consolidated timestamps from all conversations, allowing reprocessing.

**Request (optional):**
```json
{"delete_memories": true}
```

When `delete_memories` is true, also deletes all entities, facts, and strategies. Useful when extraction logic has changed.

### Export Memory

```
POST /api/memory/export
```

Serializes the user's memory graph (conversations/turns, facts/entities, strategies, goals, tool-invocations) plus the PostgreSQL audit mirror into a single round-trippable envelope keyed by stable node ids. Re-import with `/api/memory/import`.

**Request (optional):**
```json
{"channel": "_all"}
```

`channel` defaults to `"_all"` (every channel). Exports are **text-only** — embeddings are regenerated from text on import, so files are small, deterministic, git-diffable, and portable across embedding models.

**Response:** `{"export": { ...envelope... }}` — `schema_version`, `embedder` (provenance), and per-type node collections.

Scriptable equivalent: `task memory:export -- --channel _global`.

### Import Memory

```
POST /api/memory/import
```

Restores an export idempotently by `MERGE`-ing each node on its stable id. Embeddings are regenerated from each node's canonical text with the importing instance's model (exports are text-only).

**Request:**
```json
{"data": { ...envelope... }, "mode": "merge", "channel": "_global"}
```

- `mode`: `"merge"` (default) upserts and leaves other data untouched; `"replace"` wipes the target channel for the user first, so it ends up matching the file exactly.
- `channel`: overrides the wipe scope for `replace` mode (defaults to the file's channel).

**Response:** `{"imported": {"mode", "channel", "recomputed_embeddings", "imported": {<type>: {"created", "total"}}, ...}}`. Returns `400` for a missing envelope, bad `mode`, or an unsupported (newer) `schema_version`.

Scriptable equivalent: `task memory:import -- --input snapshot.json --mode replace --channel _global`.

### Detail & Streaming Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/entities/{entity_id}` | GET | Single entity detail |
| `/api/memory/facts/{fact_id}` | GET | Single fact detail |
| `/api/memory/facts/{fact_id}/remember` | POST | Boost a fact's salience ("remember this"); `{to?}` clamped to [0,1] |
| `/api/memory/facts/{fact_id}/forget` | POST | Forget a fact — soft-retire (default) or `{hard:true}` to delete |
| `/api/memory/facts/{fact_id}/entities` | POST/DELETE | Link/unlink an entity to a fact (`{entity_id}`, ABOUT edge); returns the fact's updated `{id,name,type}` entity list |
| `/api/memory/facts/{fact_id}/provenance` | GET | Where the fact was learned (origin conversation + turn snippet) |
| `/api/memory/consolidate/stream` | GET | Consolidation progress via SSE |
| `/api/memory/checkpoints` | GET/DELETE | List or clear a conversation's model-authored checkpoints (`?conversation_id=`) |
| `/api/memory/user-history` | POST | Browse the user's past turns + top facts (`{topic?, limit?, channel?}`) |

---

## Jobs

### List Jobs

```
GET /api/jobs
```

Returns all registered consolidation jobs with their status and worker info.

**Response:**
```json
{
  "jobs": [
    {
      "name": "consolidate",
      "status": "idle",
      "enabled": true,
      "last_run": "2026-03-10T14:30:00Z",
      "last_duration_ms": 1500,
      "run_count": 42
    }
  ],
  "worker": {
    "status": "running",
    "uptime_seconds": 3600
  }
}
```

### Job Detail

```
GET /api/jobs/{job_name}
```

Returns job details with recent execution history (last 10 runs).

**Response:**
```json
{
  "job": {...},
  "history": [
    {"timestamp": "...", "duration_ms": 1200, "success": true, "metrics": {...}}
  ]
}
```

### Run Job

```
POST /api/jobs/{job_name}/run
```

Manually triggers a specific job. Returns execution result with metrics.

### Toggle Job

```
POST /api/jobs/{job_name}/toggle
```

Enable or disable a scheduled job.

**Request:** `{"enabled": false}`

### Clear Stuck Jobs

```
POST /api/jobs/clear-stuck
```

Clears jobs stuck in `running` state (e.g., after a crash).

**Response:**
```json
{
  "success": true,
  "cleared_jobs": ["consolidate"],
  "message": "Cleared 1 stuck job(s)"
}
```

---

## Config

### Get Config

```
GET /api/config
```

Returns the current runtime configuration with secrets (API keys) redacted.

### Context Limits

```
GET /api/config/context-limits
```

Returns the resolved context-window limit per configured model — used by the client for token budgeting.

### Settings Manifest

```
GET /api/settings/manifest
```

Canonical machine-readable registry of every user-tunable setting across both stores (memory settings + config): per key — `store`, `type`, `default`, current `value` (secrets redacted), `writable_via` (which endpoint changes it, `null` = server-side only), and model-role linkage (`role_member`/`role`). The substrate for settings tooling and the future settings agent.

### Update Config

```
POST /api/config/update
```

Updates runtime configuration. Persists to `data/config.json` and hot-reloads providers. Sectioned partial update — handled sections: `providers`, `preferences`, `llm_settings`, `context_limits`, `prompt_enhancement`, `planner`, `search`, `alloy`, `ambassador`, `images`, `vision`, and `models.roles`.

**Request:**
```json
{
  "providers": {
    "anthropic": {"api_key": "sk-ant-..."},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "timeout": 300}
  },
  "preferences": {
    "default_model": "claude-3-5-sonnet-latest",
    "enable_memory_by_default": true
  },
  "llm_settings": {
    "default_temperature": 0.7,
    "default_max_tokens": 4096
  }
}
```

All fields are optional — only provided fields are updated.

**Response:**
```json
{
  "status": "ok",
  "message": "Config updated and applied",
  "updated": ["providers.anthropic.api_key", "preferences.default_model"]
}
```

API keys are redacted from `GET /api/config` responses.

---

## Logs

Read-only access to the server's logs for the client **Log panel**. Backed by an
in-memory ring buffer (live) and a compressed on-disk archive (history). Records are
redacted of secrets at capture time. The whole group is gated by
`AGENTX_LOG_API_ENABLED` (returns `404` when off) and is auth-gated by the normal
middleware when `AGENTX_AUTH_ENABLED` is set.

### Recent

```
GET /api/logs
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | string | — | Filter by level (DEBUG/INFO/WARNING/ERROR) |
| `category` | string | — | Filter by category key (provider/memory/stream/mcp/…) |
| `run_id` | string | — | Filter to a single chat run |
| `search` | string | — | Case-insensitive substring match |
| `since` | int | — | Only records with `id` greater than this |
| `limit` | int | 500 | Max records (1–2000) |

**Response:**
```json
{
  "available": true,
  "logs": [
    {
      "id": 1421, "ts": 1749200000.12, "level": "INFO",
      "logger": "agentx_ai.providers.anthropic", "category": "provider",
      "run_id": "chat_run_ab12cd", "conversation_id": null, "agent_id": null,
      "message": "request model=anthropic:claude-opus-4 messages=14"
    }
  ]
}
```

### Live stream (SSE)

```
GET /api/logs/stream
```

Replays the current buffer, then follows live. Emits `log` events (one record each)
plus heartbeat comments. `Content-Type: text/event-stream`. Returns `503` if the
per-process subscriber cap is reached.

### Categories

```
GET /api/logs/categories
```

Returns the category registry (`key`, `label`, `emoji`, `color`) the client uses to
color rows.

### Archive

```
GET /api/logs/archive
GET /api/logs/archive/status
GET /api/logs/archive/{name}
```

`GET /api/logs/archive/status` reports the vault state — `keyring_present`, `unlocked`
(is a key cached in memory, i.e. are sealed segments downloadable right now),
`sealed_segments` / `pending_segments`, `encryption_enabled`, and `retention_days`.

List the daily archive segments (`data/logs/agentx-YYYY-MM-DD.log.gz`) and download one.
Segment names are validated against path traversal.

When authentication is set up, completed days are **sealed** with AES-256-GCM keyed to the
login password (envelope encryption; see the [Logging](#logging) notes). Sealed segments
carry `encrypted: true` in the list and end in `.gz.enc`. Downloading one decrypts it on the
fly to the inner gzip — but only while the vault is **unlocked** (a key is cached from a
recent login); otherwise the download returns `423 Locked` (re-authenticate to unlock). Pass
`?raw=true` to download the encrypted bytes untouched. With auth disabled, archives stay
redacted-plaintext gzip. Manage keys with `task logs:keys:status | logs:seal |
logs:rotate-keys | logs:rotate-keys:deep`.

---

## Authentication

Authentication is **optional** and disabled by default. When `AGENTX_AUTH_ENABLED=true`,
all `/api/*` routes require a valid session (single root user, bcrypt password, Redis-backed
sessions — Phase 17). Run `task auth:setup` to set the root password.

| Endpoint | Method | Description |
|----------|--------|-------------|
Authenticated routes expect the session token in the **`X-Auth-Token`** header. See the
[Authentication guide](../deployment/authentication.md) for the full model.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/status` | GET / POST | Whether auth is enabled and setup is complete |
| `/api/auth/setup` | POST | Set the initial root password |
| `/api/auth/login` | POST | Log in with the root password → session token |
| `/api/auth/logout` | POST | Destroy the current session |
| `/api/auth/session` | GET | Validate the current session |
| `/api/auth/change-password` | POST | Change the root password |

### Status

`GET /api/auth/status`
```json
{ "auth_required": true, "setup_required": false, "auth_bypass_active": false }
```

### Setup

`POST /api/auth/setup` — only allowed while `setup_required` is true.
```json
{ "password": "at-least-8-chars", "confirm_password": "at-least-8-chars" }
```
Returns `{"message": "Root password configured successfully"}`; `403` if already set up.

### Login

`POST /api/auth/login`
```json
{ "username": "root", "password": "…" }
```
**Response:**
```json
{ "token": "<url-safe-token>", "expires_at": "2026-05-28T12:34:56+00:00", "username": "root" }
```
`401` on bad credentials; `403` if setup is required first.

### Session / Logout / Change Password

- `GET /api/auth/session` → `{ "user_id": 1, "username": "root", "session_created": "…", "last_active": "…" }` (`401` if not authenticated).
- `POST /api/auth/logout` → `{ "message": "Logged out successfully" }`.
- `POST /api/auth/change-password` — body `{ "old_password": "…", "new_password": "…" }`; invalidates all *other* sessions; `401` on a wrong old password.

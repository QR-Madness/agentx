# Streaming & Detached Runs

The chat stream is the most involved integration surface, so it gets its own page. It delivers
tokens as they're generated over **Server-Sent Events (SSE)** and — importantly — keeps running
**server-side even if your connection drops**, so a response (or a whole new conversation) is
never lost on disconnect.

## Start a stream

```
POST /api/agent/chat/stream
```

The body is the same as `POST /api/agent/chat` (`message` required; optional `session_id`,
`model`, `profile_id`, `temperature`, `use_memory`). Two extras route the turn:

- `workflow_id` — run a multi-agent [Agent Alloy](../features/multi-agent.md) workflow.
- `target_agent_id` — route the turn to a specific agent by its Docker-style `agent_id`.

Routing priority is `workflow_id` > `target_agent_id` > `agent_profile_id` > default.

## Event sequence

The response is `Content-Type: text/event-stream`. Events arrive in this order:

| Event | Data | When |
|-------|------|------|
| `run_started` | `{ "run_id": "…" }` | First event — identifies the detached run for re-attach |
| `start` | `{ "task_id": "…", "model": "…" }` | Generation begins |
| `chunk` | `{ "content": "token text" }` | Each token |
| `tool_call` | `{ "tool": "…", "arguments": {…} }` | A tool invocation starts |
| `tool_result` | `{ "tool": "…", "content": "…" }` | Tool result (truncated to 500 chars) |
| `done` | `{ "task_id": "…", "thinking": "…", "total_time_ms": …, "session_id": "…" }` | Generation complete |
| `error` | `{ "error": "message" }` | On failure |
| `close` | `{}` | Run settled; the tail ends |

Multi-agent runs additionally emit `delegation_start`, `delegation_chunk`,
`delegation_tool_call`, `delegation_tool_result`, and `delegation_complete` — see the
[Multi-Agent guide](../features/multi-agent.md#execution--streaming).

!!! note "EventSource vs. fetch"
    The browser `EventSource` API only does `GET`, but this endpoint is `POST`. To read it in a
    browser, stream the response body with `fetch` + a `ReadableStream` reader (see
    [Recipes](recipes.md)). `EventSource` *can* be used for the re-attach endpoint below, which
    is a `GET`.

## Detached runs (survive disconnect)

Each run is driven by a server-side daemon thread that fans its SSE events into a Redis stream
and persists the turn on completion — independent of the HTTP connection. Closing the tab does
**not** stop the run; it plays to completion and can be re-attached.

### Re-attach

```
GET /api/agent/chat/stream/attach?run_id=<id>
```

Replays the buffered events from the start, then follows live until completion. Emits a
`run_missing` event (instead of replaying) once the buffer has expired — at which point you
restore from conversation history instead.

### Discover & cancel

```
GET  /api/agent/chat/runs                       # the caller's detached runs (newest first)
POST /api/agent/chat/runs/{run_id}/cancel       # cooperatively cancel a run
```

`runs` powers recovery surfaces (an inbox, a conversation picker) that offer to re-attach runs
whose owning tab was closed. Cancellation is checked at event boundaries.

## Prefer polling?

For long jobs where you don't need live tokens, queue the turn instead and poll:

```
POST /api/chat/background            → { "job_id": "…", "status": "queued" }
GET  /api/chat/background/{job_id}    → status + result once complete
```

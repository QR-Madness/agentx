# Streaming & Detached Runs

The chat stream is the most involved integration surface, so it gets its own page. It delivers
tokens as they're generated over **Server-Sent Events (SSE)** and — importantly — keeps running
**server-side even if your connection drops**, so a response (or a whole new conversation) is
never lost on disconnect.

## Start a stream

```
POST /api/agent/chat/stream
```

See [API Endpoints: Chat (Streaming)](../api/endpoints.md#chat-streaming) for the full request/response reference.

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
| `exhibit` | `{ "schema_version": 1, "id": "…", "title?": "…", "layout": "stack", "elements": [{ "type": "mermaid", "content": "…" }] }` | A typed, agent-authored exhibit (e.g. a Mermaid diagram) |
| `done` | `{ "task_id": "…", "thinking": "…", "total_time_ms": …, "session_id": "…" }` | Generation complete |
| `error` | `{ "error": "message" }` | On failure |
| `close` | `{}` | Run settled; the tail ends |

When the agent calls the internal `present_exhibit` tool, the turn emits an `exhibit`
event — a declarative Gallery→Exhibit→Element tree the client renders from an element
registry — **instead of** a `tool_call`/`tool_result` pair for that call. Every exhibit
carries a `schema_version`; unknown element `type`s degrade to a source-as-code fallback,
so a newer server can add element types without breaking older clients. Re-emitting the
same `id` amends that exhibit in place. Element types:

- `mermaid` — `{ "type": "mermaid", "content": "graph TD; A-->B", "title?": "…" }` — a diagram.
- `choice` — `{ "type": "choice", "prompt?": "Which DB?", "options": ["PostgreSQL", "Neo4j"] }`
  — interactive options. Clicking one submits it as the user's **next message** (no new
  endpoint); the agent's next turn continues from the answer.
- `table` — `{ "type": "table", "columns": ["Model", "Cost"], "rows": [["opus", "0.40"]], "caption?": "…" }`
  — sortable, scrollable, responsive (collapses to cards on mobile), expandable to a modal.
- `citation` — `{ "type": "citation", "sources": [{ "label": "NLLB", "url?": "https://…", "quote?": "…", "kind": "active"|"passive", "source_type?": "web"|"memory"|"doc" }] }`
  — `active` sources fold out with their quote; `passive` sources are archived record-keeping
  links. Default `kind` is `passive`. The card always carries a "Sources" header; a passive-only
  citation (e.g. auto-captured search results) renders its list inline.

**Auto-captured citations.** When the agent calls the internal `web_search` (or `web_research`)
tool, a passive `citation` exhibit (one source per result, deduped by URL, `source_type: "web"`) is
emitted automatically right after that tool's `tool_result` — so web sources surface in the conversation
without the agent having to present them. The exhibit's `id` is `exh_src_<tool_call_id>`, so it
restores in place from history. Toggle with the `citations.auto_capture_web_search` config flag
(default on). The agent is steered to spotlight a key source as `active` (with a quote) rather than
re-listing web results as inline links.

The `stack` layout (vertical) is the only layout today.

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

See [API Endpoints: Re-attach to a Run](../api/endpoints.md#re-attach-to-a-run) for the full reference.

Replays the buffered events from the start, then follows live until completion. Emits a
`run_missing` event (instead of replaying) once the buffer has expired — at which point you
restore from conversation history instead.

### Discover & cancel

```
GET  /api/agent/chat/runs                       # the caller's detached runs (newest first)
POST /api/agent/chat/runs/{run_id}/cancel       # cooperatively cancel a run
```

See [API Endpoints: List Detached Runs](../api/endpoints.md#list-detached-runs) and
[Cancel a Run](../api/endpoints.md#cancel-a-run) for the full reference.

`runs` powers recovery surfaces (an inbox, a conversation picker) that offer to re-attach runs
whose owning tab was closed. Cancellation is checked at event boundaries.

## Prefer polling?

For long jobs where you don't need live tokens, queue the turn instead and poll:

```
POST /api/chat/background            → { "job_id": "…", "status": "queued" }
GET  /api/chat/background/{job_id}    → status + result once complete
```

See [API Endpoints: Background Chat](../api/endpoints.md#background-chat) for the full reference.

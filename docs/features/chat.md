# Chat

The chat system provides multi-turn conversations with session management, memory integration, tool use, and streaming support.

## Chat Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Agent
    participant PM as PromptManager
    participant M as AgentMemory
    participant P as Provider
    participant T as ToolExecutor

    C->>A: chat(message, session_id, profile_id)
    A->>A: SessionManager.get_or_create(session_id)
    A->>M: store_turn(user_turn)
    A->>M: remember(message) → MemoryBundle

    A->>PM: get_system_prompt(profile_id)
    A->>A: Build messages: system + context + memory + user
    A->>A: _get_tools_for_provider() → MCP tools

    loop Tool-use loop (max 10 rounds)
        A->>P: complete(messages, tools)
        alt Model requests tool calls
            A->>T: Execute each tool call
            T-->>A: Tool results
            A->>A: Append results to messages
        else No tool calls
            Note over A: Break loop
        end
    end

    A->>A: parse_output() → extract <think> tags
    A->>A: Session.add_message(assistant)
    A->>M: store_turn(assistant_turn)
    A-->>C: AgentResult
```

## Two Modes

### Simple Mode (default for chat)

Direct provider completion without planning or reasoning. Used by `POST /api/agent/chat` and `POST /api/agent/chat/stream`.

Flow: prompt composition → provider completion → tool-use loop → output parsing → memory storage.

### Full Agent Mode

Full pipeline with task planning and reasoning. Used by `POST /api/agent/run`.

Flow: task decomposition → reasoning strategy selection → execution → memory storage.

## Streaming

`POST /api/agent/chat/stream` returns Server-Sent Events for real-time token streaming.

**SSE events:**

| Event | Data | When |
|-------|------|------|
| `start` | `{task_id, model}` | Stream begins |
| `chunk` | `{content}` | Each token |
| `tool_call` | `{tool, arguments}` | Tool invocation |
| `tool_result` | `{tool, content}` | Tool result |
| `done` | `{task_id, thinking, has_thinking, total_time_ms, session_id}` | Complete |
| `error` | `{error}` | On failure |

The streaming endpoint supports the same tool-use loop as the non-streaming endpoint. Memory storage happens in a background thread after the stream closes.

```bash
curl -N -X POST http://localhost:12319/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "llama3.2"}'
```

## Session Management

`SessionManager` maintains conversation context across messages within a session.

- Sessions are identified by `session_id` (UUID)
- If no `session_id` is provided, a new session is created
- Include the returned `session_id` in subsequent requests for continuity
- Context includes all prior messages in the session

## Memory Integration

When `use_memory` is `true` (default):

1. **Store user turn** — The user message is saved to episodic memory
2. **Recall** — `AgentMemory.remember(query)` retrieves relevant turns, facts, entities, and strategies
3. **Inject** — `ContextManager.inject_memory()` adds the `MemoryBundle` to the message context
4. **Store assistant turn** — The response is saved with model/latency metadata

Memory operations are wrapped in try/except — the system works normally if databases are unavailable.

## Prompt Composition

Each chat request composes a system prompt from:

1. **Global prompt** — Core persona (always applied)
2. **MCP tools prompt** — Auto-generated tool descriptions
3. **Profile sections** — From the selected profile (via `profile_id`)
4. **Memory context** — Injected relevant memories

See [Prompts](prompts.md) for details.

## Tool-Use Loop

When MCP servers are connected, tools are exposed to the model as function-calling tools. The agent runs a tool-use loop:

1. Provider returns a completion with `tool_calls`
2. Agent executes each tool via `ToolExecutor.call_tool_sync()`
3. Tool results are appended as tool messages
4. Provider is called again with the updated messages
5. Repeat until no more tool calls or `max_tool_rounds` (10) is reached

## Output Parsing

The `OutputParser` extracts `<think>` tags from model output:

- Content within `<think>...</think>` is separated into `AgentResult.thinking`
- `has_thinking` is set to `true` when thinking is extracted
- The remaining content becomes `AgentResult.answer`

## Request Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | User message |
| `session_id` | string | auto | Session for continuity |
| `model` | string | from config | Model override |
| `profile_id` | string | `"default"` | Prompt profile |
| `temperature` | float | `0.7` | Sampling temperature |
| `use_memory` | bool | `true` | Enable memory |

## Related

- [API Endpoints: Agent](../api/endpoints.md#agent) — Full endpoint documentation
- [Prompts](prompts.md) — Prompt composition system
- [MCP](mcp.md) — Tool integration
- [Memory](memory.md) — Memory system

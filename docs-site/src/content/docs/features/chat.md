# Chat

Chat is the everyday way you work with an agent — a multi-turn conversation with memory,
tools, and live streaming. You talk, the agent thinks and acts (calling tools when it needs
to), and everything it does streams back as it happens.

## Talking to an agent

Type in the composer and send. The agent you're talking to is shown in the **agent chip** at
the left of the input row — switch agents there, or `@-mention` one to route a single turn to
it. Each turn, the agent recalls relevant memory, composes its prompt, and runs a tool-use
loop until it has an answer.

### The Relay — the conversation's command center

The **Orbit button** <span class="ax-icon ax-icon--orbit" aria-hidden="true"></span> beside the
composer opens the **Relay**: a glass control center for *this* conversation. A status strip reads out the active agent, model, and context usage;
below it, a tile grid holds every per-conversation control:

- **Thinking mode** (the wide tile) — one selection covering the [thinking patterns](reasoning.md)
  *and* Research Mode (they're mutually exclusive; the menu makes that visible).
- Toggles: **Memory** (locks once the conversation starts), **Solo/Team**
  [delegation](multi-agent.md), and **Background** (arm the next send to run detached).
- Openers: **Model**, **Project**, **Conversation state**, **Attach image / file**,
  **Enhance prompt**, and **Auto-title**.
- Below the grid: **Live runs** (recover a detached run) and the **Background runs** inbox.

On desktop the Relay is a popover above the composer, and the most-used controls also stay as
chips in the composer row. **On mobile the chip row disappears** — the input row becomes
*agent avatar · message · Relay · send*, and the Relay opens as a bottom sheet holding
everything. (The command palette stays app-level; the Relay is the conversation-level surface —
the palette's "Open the Relay" jumps straight in.)

A slim handle above the input toggles the **expanded drafting box** (a sticky preference): the
composer becomes a tall canvas where **Enter inserts a new line** and **Ctrl/Cmd+Enter (or the
send button) submits** — long-form drafting, and the cure for the tiny default input on phones.

## Streaming, and picking up where you left off

Streaming responses arrive token-by-token over Server-Sent Events. The run is **detached from
the HTTP connection** — it keeps generating server-side and persists its turns even if you close
the tab or drop your network.

To recover a run you walked away from, two surfaces list every run still in progress that isn't
already owned by an open tab: a **Live runs** section in the Relay inbox, and a **Resume
Running** section atop the conversation selector. Click **Resume** to restore the conversation
and re-attach.

!!! tip "Background sends"
    Arm **Background** in the Relay before you send and the turn runs detached from the start —
    fire off a long research task, close the tab, and pick the result up from the Background runs
    inbox later.

## Sessions & memory

A **session** carries context across the turns of one conversation. Sessions are keyed by a
`session_id` (a UUID); if you don't pass one, a new session is created and returned — include it
on later requests for continuity, and the agent sees all prior messages in the session.

With **Memory** on (the default), each turn also reaches beyond the current session: the agent
stores the exchange to [memory](memory.md) and recalls relevant past turns, facts, entities, and
strategies to fold into its context. Memory is best-effort — if the databases are unavailable,
chat still works, just without recall.

## Under the hood

A turn is a single streaming pipeline. Everything above is what it looks like; here's what it
does — the full [chat-turn sequence diagram](../architecture/system-design.md#the-chat-turn) lives
on the **System Design** page.

### Two modes

- **Simple chat** — the everyday path: direct provider completion with a tool loop, no
  planning. Flow: prompt composition → provider completion → tool-use loop → output parsing →
  memory storage.
- **Full agent** — the complete pipeline with task planning and reasoning. Flow: task
  decomposition → reasoning-strategy selection → execution → memory storage.

### Prompt composition

Each request composes a system prompt from the **global prompt** (core persona, always applied),
the auto-generated **MCP tools prompt**, the selected **profile's sections** (via `profile_id`),
and the injected **memory context**. See [Prompts](prompts.md) for the full layering.

### Tool-use loop

When MCP servers are connected, their tools are exposed to the model as function-calling tools,
and the agent loops:

1. The provider returns a completion with `tool_calls`.
2. The agent executes each tool call.
3. Tool results are appended as tool messages.
4. The provider is called again with the updated messages.
5. Repeat until there are no more tool calls, or `max_tool_rounds` (10) is reached.

### Output parsing

Any `<think>…</think>` content the model emits is split out into the message's **thinking**
(what powers the thinking bubble); the remaining text becomes the visible **answer**.

### Building on the API

Everything above is available programmatically, too. The REST contract — request schema,
streaming (SSE) events, and the re-attach / cancel mechanics for detached runs — lives in the
[API Reference](../api/endpoints.md) and [Streaming & Detached Runs](../integrate/streaming.md).

## Related

- [Agent Profiles](agent-profiles.md) — configure the agent you're chatting with
- [Prompts](prompts.md) — the prompt composition system
- [Connectors & Tools](mcp.md) — tool integration
- [Memory](memory.md) — the memory system

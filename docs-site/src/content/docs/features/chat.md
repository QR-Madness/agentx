# Chat

Chat is the everyday way you work with an agent — a multi-turn conversation with memory,
tools, and live streaming. You talk, the agent thinks and acts (calling tools when it needs
to), and everything it does streams back as it happens.

## Talking to an agent

Type your message in the **message box** and send. The agent you're talking to is shown right
beside it — switch agents there, or `@-mention` another to hand it a single turn. Each turn, the
agent recalls relevant memory, composes its prompt, and works through a tool-use loop until it has
an answer.

### The Relay — the conversation's command center

The **Relay** <span class="ax-icon ax-icon--orbit" aria-hidden="true"></span> — opened from the
button beside the message box — is the control center for *this* conversation. It gathers
everything that shapes the current chat in one place: the [thinking mode](reasoning.md) (the
thinking patterns and Research Mode), **Memory** on or off, **Solo/Team**
[delegation](multi-agent.md), the **model** and **project**, image and file attachments, prompt
enhancement, and the **Background runs** inbox for detached work. Whatever you change here applies
to this conversation only.

!!! note "On mobile"
    To save space, the composer's quick chips (model, thinking, and friends) tuck into the Relay —
    open it from the button beside the message box to reach them.

## Streaming, and picking up where you left off

Responses stream in token by token, and the run is **detached from your connection** — it keeps
generating on the server and saves its turns even if you close the tab or drop your network.

Walked away from one? Any run still going shows up for you to **resume** — in the Relay's runs
inbox and atop the conversation list — so you can jump straight back into it, live.

!!! tip "Background sends"
    Flip on **Background** in the Relay before you send, and the turn runs detached from the start
    — fire off a long task, close the tab, and collect the result from the Relay later.

## Sessions & memory

A **session** carries context across the turns of one conversation — keep chatting and the agent
sees everything said earlier in it.

With **Memory** on (the default), each turn also reaches beyond the current session: the agent
saves the exchange to [memory](memory.md) and recalls relevant past turns, facts, and strategies to
fold into its context. Memory is best-effort — if its databases are unavailable, chat still works,
just without recall.

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

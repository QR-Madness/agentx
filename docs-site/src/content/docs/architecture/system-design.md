# AgX System Design

The flow diagrams behind AgentX, collected in one place. The feature pages stay readable and
link here for the deep view — this page is the map of how the parts actually move.

## The chat turn

Every message runs through one streaming pipeline: bind the session, recall memory, compose the
prompt, run a bounded tool-use loop, then parse the output and write memory back. The
day-to-day surface is on the [Chat](../features/chat.md) page.

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

## Multi-agent delegation

A [team](../features/multi-agent.md) puts a supervisor (the **Lead**) in charge of the
conversation. It hands focused subtasks to specialists (**Members**) through the `delegate_to`
tool, coordinated by the `AlloyExecutor` over a shared memory channel, then synthesizes their
results into one answer.

```mermaid
graph TD
    U[User message] --> S[Supervisor agent]
    S -->|delegate_to| X[AlloyExecutor]
    X --> A[Specialist A]
    X --> B[Specialist B]
    A --> CH[(Shared channel<br/>_alloy_&lt;workflow_id&gt;)]
    B --> CH
    S --> CH
    A -->|result| S
    B -->|result| S
    S -->|synthesized answer| U
```

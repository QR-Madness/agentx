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

## Speculative decoding

[Drafting](../features/reasoning.md#advanced-multi-model-drafting) can pair two models on a
single generation: a fast **draft** model proposes a batch of tokens, and a stronger
**target** model verifies them — accepting or rejecting each batch against a threshold. It's
off by default; the payoff is cheaper tokens whenever the draft and target agree.

```mermaid
sequenceDiagram
    participant D as Draft model (fast)
    participant T as Target model (strong)

    loop Until done or max iterations
        D->>D: Generate N draft tokens
        D->>T: Send draft for verification
        T->>T: Score each token
        T-->>D: Accept / reject (threshold)
    end
```

## Memory recall

Recall is more than nearest-vector lookup. The [Recall Layer](../features/memory.md#how-recall-finds-the-right-memories)
runs several complementary techniques in parallel and fuses the results, then reranks the pool
by relevance, salience, and recency before handing back a `MemoryBundle`.

```mermaid
graph LR
    Q[Query] --> BASE[Base Retrieval<br/>vector similarity]
    Q --> HYB[Hybrid Search<br/>BM25 + vector, RRF fusion]
    Q --> ENT[Entity-Centric<br/>graph traversal from matched entities]
    Q --> QE[Query Expansion<br/>question→statement transforms]
    Q --> HYDE[HyDE<br/>hypothetical document embedding]
    Q --> SQ[Self-Query<br/>LLM filter extraction]

    BASE & HYB & ENT & QE & HYDE & SQ --> MERGE[Merge + Deduplicate]
    MERGE --> RANK[Rerank<br/>salience, temporal, access boosts]
    RANK --> MB[MemoryBundle]
```

## Memory consolidation

Every 15 minutes a background pass distills recent conversations into durable knowledge —
extracting entities and facts, resolving them against what's already known, and detecting
contradictions before storing. The day-to-day view is on the
[Memory](../features/memory.md#consolidation--turning-talk-into-knowledge) page.

```mermaid
graph TD
    T[Recent Turns] --> RF[Relevance Filter<br/>skip trivial messages]
    RF --> EX[Combined Extraction<br/>entities + facts + relationships in one LLM call]
    EX --> EL[Entity Linking<br/>embedding-based entity resolution]
    EL --> CD[Contradiction Detection<br/>compare new facts against existing]
    CD --> |contradicts| RS[Resolution<br/>prefer_new / prefer_old / flag_review]
    CD --> |no conflict| ST[Store<br/>upsert_entity + learn_fact]
    RS --> ST
```

## MCP client architecture

AgentX reaches [connectors](../features/mcp.md) through a `ToolExecutor` and a persistent
`MCPClientManager`, which loads server definitions from `mcp_servers.json` and holds connections
open over three transports — stdio, SSE, and streamable HTTP.

```mermaid
graph LR
    subgraph AgentX
        A[Agent] --> TE[ToolExecutor]
        TE --> MCM[MCPClientManager]
        MCM --> SR[ServerRegistry]
        SR --> |loads| CF[mcp_servers.json]
    end

    subgraph Transports
        MCM --> STDIO[stdio]
        MCM --> SSE[SSE]
        MCM --> HTTP[Streamable HTTP]
    end

    subgraph External["External MCP Servers"]
        STDIO --> FS[Filesystem]
        STDIO --> GH[GitHub]
        SSE --> BS[Web Search]
        HTTP --> PG[PostgreSQL]
        HTTP --> Custom[Custom...]
    end
```

## MCP tool execution

Within a turn, the agent converts connected MCP tools into the provider's function-calling
format, runs the model, and executes each returned tool call through the manager before looping
back for a final answer. The day-to-day view is on the [Connectors & Tools](../features/mcp.md)
page.

```mermaid
sequenceDiagram
    participant A as Agent
    participant TE as ToolExecutor
    participant MCM as MCPClientManager
    participant S as MCP Server

    A->>A: _get_tools_for_provider()
    Note over A: Convert MCP tools to<br/>provider function-calling format

    A->>A: Provider.complete(messages, tools)
    Note over A: Model returns tool_calls

    loop For each tool_call
        A->>TE: find_tool(name) → ToolInfo
        A->>MCM: call_tool_sync(name, args)
        MCM->>S: Execute via MCP protocol
        S-->>MCM: Result
        MCM-->>A: ToolResult
        A->>A: Append tool result to messages
    end

    A->>A: Provider.complete(messages) → final response
```

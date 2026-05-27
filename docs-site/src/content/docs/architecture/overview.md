# Architecture Overview

AgentX is a two-tier system: a Django REST API backend and a Tauri desktop client. This document covers the backend architecture.

## System Architecture

```mermaid
graph TB
    subgraph Views["Views Layer"]
        V[views.py<br/>HTTP dispatch]
    end

    subgraph Agent["Agent Core"]
        AC[Agent]
        TP[TaskPlanner]
        SM[SessionManager]
        CM[ContextManager]
        OP[OutputParser]
    end

    subgraph Reasoning["Reasoning Framework"]
        RO[Orchestrator]
        CoT[Chain-of-Thought]
        ToT[Tree-of-Thought]
        ReAct[ReAct]
        Ref[Reflection]
    end

    subgraph Drafting["Drafting Framework"]
        Spec[Speculative]
        Pipe[Pipeline]
        Cand[Candidate]
    end

    subgraph Providers["Model Providers"]
        PR[ProviderRegistry]
        LMS[LM Studio]
        ANT[Anthropic]
        OAI[OpenAI]
    end

    subgraph MCP["MCP Client"]
        MCM[ClientManager]
        SR[ServerRegistry]
        TE[ToolExecutor]
        subgraph Transports
            STDIO[stdio]
            SSE[SSE]
            HTTP[Streamable HTTP]
        end
    end

    subgraph Prompts["Prompt System"]
        PM[PromptManager]
        Prof[Profiles]
        Sect[Sections]
        Comp[Composer]
    end

    subgraph Memory["Memory System"]
        MI[AgentMemory Interface]
        EP[Episodic]
        SEM[Semantic]
        PROC[Procedural]
        WM[Working]
        EXT[Extraction]
        CON[Consolidation]
        REC[RecallLayer]
    end

    subgraph Data["Data Layer"]
        Neo4j[("Neo4j")]
        PG[("PostgreSQL<br/>+ pgvector")]
        Redis[("Redis")]
    end

    TK[TranslationKit]

    V --> AC
    V --> TK
    V --> MCM
    V --> PR
    V --> PM
    V --> MI

    AC --> RO
    AC --> TE
    AC --> PM
    AC --> CM
    AC --> MI

    RO --> CoT & ToT & ReAct & Ref
    RO --> PR

    AC -.-> Spec & Pipe & Cand
    Spec & Pipe & Cand --> PR

    PR --> LMS & ANT & OAI
    MCM --> SR --> Transports

    MI --> EP & SEM & PROC & WM
    MI --> EXT & CON & REC
    EP & SEM & PROC --> Neo4j
    EP & SEM --> PG
    WM --> Redis
    CON --> EXT
```

## Request Lifecycle

A `POST /api/agent/chat` request follows this path:

```mermaid
sequenceDiagram
    participant C as Client
    participant V as views.py
    participant A as Agent
    participant PM as PromptManager
    participant S as SessionManager
    participant M as AgentMemory
    participant P as Provider
    participant MCP as ToolExecutor

    C->>V: POST /agent/chat {message, model, profile_id, session_id}
    V->>A: Agent(config)
    A->>S: get_or_create(session_id)
    A->>M: store_turn(user_turn)
    A->>M: remember(query)
    M-->>A: MemoryBundle

    A->>PM: get_system_prompt(profile_id)
    PM-->>A: composed system prompt

    A->>A: build messages (system + context + memory + user)
    A->>A: _get_tools_for_provider() → MCP tools

    loop Tool-use loop (max_tool_rounds)
        A->>P: complete(messages, tools)
        P-->>A: CompletionResult
        alt has tool_calls
            A->>MCP: call_tool_sync(name, args)
            MCP-->>A: ToolResult
            A->>A: append tool result to messages
        else no tool_calls
            Note over A: break loop
        end
    end

    A->>A: parse_output() → extract <think> tags
    A->>S: add_message(assistant)
    A->>M: store_turn(assistant_turn)
    A-->>V: AgentResult
    V-->>C: JSON response
```

## Module Index

| Module | Path | Purpose | Init |
|--------|------|---------|------|
| Agent | `agent/core.py` | Orchestrates reasoning, tools, memory, prompts | Per-request |
| TaskPlanner | `agent/planner.py` | Decomposes tasks into subtasks with goal tracking | Per-request |
| SessionManager | `agent/session.py` | Maintains conversation context across messages | Lazy singleton |
| ContextManager | `agent/context.py` | Token budgeting, memory injection, summarization | Per-request |
| OutputParser | `agent/output_parser.py` | Extracts `<think>` tags from model output | Stateless |
| Reasoning | `reasoning/orchestrator.py` | Selects and executes reasoning strategy | Per-request |
| Drafting | `drafting/` | Speculative decoding, pipelines, candidates | Per-request |
| Providers | `providers/registry.py` | Model-to-provider resolution, model registry | Lazy singleton |
| MCP | `mcp/client.py` | External tool server connections and execution | Lazy singleton |
| Prompts | `prompts/manager.py` | System prompt composition from profiles + sections | Lazy singleton |
| Memory | `kit/agent_memory/memory/interface.py` | Unified API for episodic/semantic/procedural/working memory | Lazy |
| RecallLayer | `kit/agent_memory/recall/layer.py` | Multi-strategy retrieval (hybrid, HyDE, entity-centric) | Per-query |
| Extraction | `kit/agent_memory/extraction/service.py` | LLM-based entity/fact extraction | Per-consolidation |
| Consolidation | `kit/agent_memory/consolidation/worker.py` | Background jobs for memory processing | Background thread |
| Translation | `kit/translation.py` | NLLB-200 translation + language detection | Lazy singleton |
| Config | `config.py` | Runtime config persistence to `data/config.json` | Lazy singleton |

## Design Decisions

**Lazy singletons** — Heavy subsystems (TranslationKit, MCP, Providers, Prompts) use `@lazy_singleton` to defer initialization until first use. Health checks can probe without triggering model loads via `get_if_initialized()`.

**Sync Django + async MCP** — Django runs synchronously. MCP client uses `asyncio` internally. The bridge is `MCPClientManager.call_tool_sync()`, which runs async tool calls on a background event loop thread. The streaming chat endpoint (`agent_chat_stream`) is the only async view.

**Per-request Agent** — Each chat/run request creates a fresh `Agent` instance with its own config. Shared state (sessions, providers, MCP connections) lives in singletons. This keeps the Agent stateless and thread-safe.

**Memory is optional** — All memory operations are wrapped in try/except. The system degrades gracefully when databases are unavailable. `enable_memory=False` skips all memory operations.

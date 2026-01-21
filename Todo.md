# AgentX Development Todo

**Project**: AgentX - AI Agent Platform with MCP Client, Drafting Models & Reasoning Framework  
**Status**: Pre-prototype  
**Last Updated**: 2026-01-21

---

## Overview

AgentX is an AI agent platform that:
1. **Consumes MCP tools** from external servers (filesystem, GitHub, databases, custom servers)
2. **Experiments with drafting models** (speculative decoding, multi-model pipelines, candidate generation)
3. **Implements flexible reasoning patterns** (CoT, ToT, ReAct, reflection/self-critique)
4. **Provides memory-augmented AI** with persistent knowledge graphs

### Current State
- ✅ Translation system functional (200+ languages via NLLB-200)
- ✅ Tauri client with React UI operational
- ✅ Docker services configured (Neo4j, PostgreSQL, Redis)
- ⚠️ Agent memory system designed but not connected
- ⚠️ Several dependency and configuration issues
- ❌ MCP client not yet implemented
- ❌ Drafting/reasoning framework not yet implemented
- ❌ Chat/agent functionality not implemented

---

## Phase 1: Critical Fixes ✅

> **Priority**: HIGH  
> **Goal**: Get the codebase into a working baseline state  
> **Status**: COMPLETE

### 1.1 Missing Dependencies
- [x] Add `redis>=5.0.0` to pyproject.toml
- [x] Add `sqlalchemy>=2.0.0` to pyproject.toml
- [x] Add `mcp>=1.0.0` to pyproject.toml (for Phase 3)
- [x] Add `openai>=1.0.0`, `anthropic>=0.20.0`, `httpx>=0.27.0` for model providers
- [x] Add `networkx>=3.0` for reasoning graphs
- [x] Run `uv sync` to regenerate lock file
- [x] Verify all imports resolve without errors

### 1.2 Taskfile Corrections
- [x] Fix `client:build`: change `bunx next build` → `bun run build`
- [x] Fix `client:start`: change `bunx next start` → `bun run preview`  
- [x] Fix `client:dev`: change `bunx tauri run dev` → `bun run tauri dev`
- [x] Remove Next.js references (project uses Vite)
- [ ] Test full `task dev` workflow

### 1.3 OpenAPI Specification
- [x] Update server URL port: `8000` → `12319`
- [x] Document `POST /tools/translate` endpoint with request/response schemas
- [x] Fix language-detect path: `/language-detect` → `/tools/language-detect-20`
- [x] Add proper response schemas for all endpoints
- [x] Add Error component schema

### 1.4 Environment Configuration
- [x] Create `.env.example` with all required variables
- [x] Update `docker-compose.yml` to use environment variables
- [x] Ensure `.env` is in `.gitignore`
- [x] Add redis-commander to optional `tools` profile

### 1.5 Code Quality (Bonus)
- [x] Replace `print()` statements with `logging` in views.py
- [x] Fix `language_detect` to accept POST with text body (backwards compatible with GET)
- [x] Fix test import paths
- [x] Fix test URL paths to match new API structure
- [x] Update CLAUDE.md with correct API endpoints

---

## Phase 2: Wire Up Existing Code ✅

> **Priority**: HIGH  
> **Goal**: Connect the scaffolded code that already exists  
> **Status**: COMPLETE

### 2.1 Language Detection API Fix
- [x] Modify `views.language_detect()` to accept POST (done in Phase 1)
- [x] Remove hardcoded test string (done in Phase 1)
- [x] Add input validation (done in Phase 1)
- [x] Return structured response with language code and confidence

### 2.2 Agent Memory System Connection
- [x] Renamed `agent_memory.py` → `memory_utils.py` to avoid naming conflict with `agent_memory/` package
- [x] Add database connection health check endpoint: `GET /api/health`
- [x] Create lazy-loading `get_agent_memory()` function
- [x] Make PostgreSQL connection lazy (PostgresConnection class)
- [x] Add `check_memory_health()` for connection status

### 2.3 Code Quality Improvements
- [x] Replace all `print()` statements with `logging`
- [x] Add logging configuration to Django settings
- [x] Add `psycopg2-binary` for PostgreSQL connectivity
- [x] Add `sentence-transformers` for local embeddings
- [x] Add health check tests

---

## Phase 3: MCP Client Integration

> **Priority**: HIGH  
> **Goal**: Enable AgentX to consume tools from external MCP servers

### 3.1 MCP Client Infrastructure
- [ ] Create module structure:
  ```
  api/agentx_ai/mcp/
  ├── __init__.py
  ├── client.py           # MCP client manager
  ├── server_registry.py  # Track connected MCP servers
  ├── tool_executor.py    # Execute tools on remote servers
  └── transports/
      ├── __init__.py
      ├── stdio.py        # stdio transport (subprocess)
      ├── sse.py          # SSE transport (HTTP)
      └── websocket.py    # WebSocket transport (future)
  ```
- [ ] Implement `MCPClientManager` class:
  - [ ] `connect(server_config)` → connection_id
  - [ ] `disconnect(connection_id)`
  - [ ] `list_tools(connection_id?)` → available tools
  - [ ] `call_tool(connection_id, tool_name, args)` → result
  - [ ] `list_resources(connection_id?)` → available resources
  - [ ] `read_resource(connection_id, uri)` → content

### 3.2 Server Configuration
- [ ] Create `mcp_servers.json` configuration format:
  ```json
  {
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
        "transport": "stdio"
      },
      "github": {
        "command": "npx", 
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" },
        "transport": "stdio"
      },
      "custom-db": {
        "url": "http://localhost:8080/mcp",
        "transport": "sse"
      }
    }
  }
  ```
- [ ] Add Taskfile command: `task mcp:list-tools` (list all available tools)
- [ ] Add UI for managing MCP server connections

### 3.3 Standard MCP Server Support
- [ ] Test with `@modelcontextprotocol/server-filesystem`
- [ ] Test with `@modelcontextprotocol/server-github`
- [ ] Test with `@modelcontextprotocol/server-postgres`
- [ ] Test with `@modelcontextprotocol/server-brave-search`
- [ ] Document tested/supported servers

### 3.4 Custom MCP Server Development
- [ ] Create template for custom MCP servers
- [ ] Build example: AgentX Memory MCP Server (expose memory as MCP)
- [ ] Build example: Translation MCP Server (expose translation as MCP)

### 3.5 Tool Discovery & Caching
- [ ] Cache tool schemas on connection
- [ ] Refresh tool list on demand
- [ ] Handle server disconnection gracefully
- [ ] Add tool search/filter functionality

---

## Phase 4: Model Provider Abstraction

> **Priority**: HIGH  
> **Goal**: Support multiple LLM backends with unified interface

### 4.1 Provider Interface
- [ ] Create abstract `ModelProvider` interface:
  ```python
  class ModelProvider(ABC):
      @abstractmethod
      async def complete(self, messages, **kwargs) -> CompletionResult
      
      @abstractmethod
      async def stream(self, messages, **kwargs) -> AsyncIterator[StreamChunk]
      
      @abstractmethod
      def get_capabilities(self) -> ModelCapabilities
  ```

### 4.2 Provider Implementations
- [ ] `OpenAIProvider` - GPT-4, GPT-4-turbo, GPT-3.5
- [ ] `AnthropicProvider` - Claude 3 Opus/Sonnet/Haiku
- [ ] `OllamaProvider` - Local models (Llama, Mistral, etc.)
- [ ] `TogetherProvider` - Together.ai API
- [ ] `LocalTransformersProvider` - HuggingFace transformers (existing)

### 4.3 Model Registry
- [ ] Create `models.yaml` configuration:
  ```yaml
  models:
    gpt-4-turbo:
      provider: openai
      context_window: 128000
      supports_tools: true
      supports_vision: true
      cost_per_1k_input: 0.01
      cost_per_1k_output: 0.03
      
    llama-3-70b:
      provider: ollama
      context_window: 8192
      supports_tools: false
      local: true
      
    claude-3-sonnet:
      provider: anthropic
      context_window: 200000
      supports_tools: true
  ```
- [ ] Model selection based on task requirements
- [ ] Cost tracking and budgeting

---

## Phase 5: Drafting Models Framework

> **Priority**: HIGH  
> **Goal**: Implement multi-model generation strategies

### 5.1 Drafting Infrastructure
- [ ] Create `drafting/` module:
  ```
  api/agentx_ai/drafting/
  ├── __init__.py
  ├── strategies.py      # Drafting strategy definitions
  ├── speculative.py     # Speculative decoding
  ├── pipeline.py        # Multi-model pipelines
  ├── candidate.py       # N-best candidate generation
  └── evaluator.py       # Draft quality evaluation
  ```

### 5.2 Speculative Decoding
- [ ] Implement `SpeculativeDecoder`:
  - [ ] Draft model generates N tokens quickly
  - [ ] Target model verifies/corrects in parallel
  - [ ] Configurable draft length and acceptance threshold
- [ ] Support draft/target model pairs:
  - [ ] GPT-3.5 → GPT-4
  - [ ] Llama-7B → Llama-70B
  - [ ] Local small → API large

### 5.3 Multi-Model Pipeline
- [ ] Implement `ModelPipeline`:
  - [ ] Define pipeline stages (analyze → draft → refine → validate)
  - [ ] Route to different models per stage
  - [ ] Pass context between stages
- [ ] Example pipelines:
  - [ ] Fast-triage: Small model classifies, large model handles complex
  - [ ] Specialist: Different models for code/writing/analysis
  - [ ] Ensemble: Multiple models vote on output

### 5.4 Candidate Generation
- [ ] Implement `CandidateGenerator`:
  - [ ] Generate N candidates (same or different models)
  - [ ] Scoring/ranking strategies:
    - [ ] Self-consistency (majority vote)
    - [ ] Verifier model scoring
    - [ ] Heuristic scoring (length, confidence, etc.)
  - [ ] Best-of-N selection

### 5.5 Drafting Configuration
- [ ] Create `drafting_strategies.yaml`:
  ```yaml
  strategies:
    fast_accurate:
      type: speculative
      draft_model: gpt-3.5-turbo
      target_model: gpt-4-turbo
      draft_tokens: 20
      
    code_review:
      type: pipeline
      stages:
        - model: gpt-4-turbo
          role: generate_code
        - model: claude-3-opus
          role: review_and_critique
        - model: gpt-4-turbo
          role: incorporate_feedback
          
    consensus:
      type: candidate
      models: [gpt-4, claude-3-sonnet, llama-3-70b]
      selection: majority_vote
      candidates_per_model: 1
  ```

---

## Phase 6: Reasoning Framework

> **Priority**: HIGH  
> **Goal**: Implement flexible reasoning patterns for complex tasks

### 6.1 Reasoning Infrastructure
- [ ] Create `reasoning/` module:
  ```
  api/agentx_ai/reasoning/
  ├── __init__.py
  ├── base.py            # Abstract reasoning strategy
  ├── chain_of_thought.py
  ├── tree_of_thought.py
  ├── react.py           # ReAct pattern
  ├── reflection.py      # Self-critique
  ├── orchestrator.py    # Combines strategies
  └── traces/            # Reasoning trace storage
  ```

### 6.2 Chain-of-Thought (CoT)
- [ ] Implement `ChainOfThought`:
  - [ ] Zero-shot CoT ("Let's think step by step...")
  - [ ] Few-shot CoT (with examples)
  - [ ] Auto-CoT (automatically generate examples)
- [ ] Step extraction and validation
- [ ] Confidence scoring per step

### 6.3 Tree-of-Thought (ToT)
- [ ] Implement `TreeOfThought`:
  - [ ] BFS exploration (breadth-first)
  - [ ] DFS exploration (depth-first)
  - [ ] Beam search (top-k branches)
- [ ] State evaluation heuristics
- [ ] Pruning strategies
- [ ] Visualization of thought tree

### 6.4 ReAct (Reasoning + Acting)
- [ ] Implement `ReActAgent`:
  - [ ] Thought → Action → Observation loop
  - [ ] Integration with MCP tools
  - [ ] Max iterations / termination conditions
- [ ] Action planning and validation
- [ ] Observation parsing and integration

### 6.5 Reflection & Self-Critique
- [ ] Implement `ReflectiveReasoner`:
  - [ ] Generate initial response
  - [ ] Self-critique: identify weaknesses
  - [ ] Revise based on critique
  - [ ] Configurable reflection depth
- [ ] Implement `DebateReasoner`:
  - [ ] Multiple "personas" debate
  - [ ] Synthesis of perspectives

### 6.6 Reasoning Orchestrator
- [ ] Implement `ReasoningOrchestrator`:
  - [ ] Select reasoning strategy based on task
  - [ ] Combine strategies (e.g., ToT with ReAct actions)
  - [ ] Fallback strategies on failure
- [ ] Reasoning trace logging for debugging
- [ ] Metrics: steps taken, tokens used, time elapsed

---

## Phase 7: Agent Core

> **Priority**: HIGH  
> **Goal**: Unify MCP, drafting, and reasoning into coherent agent

### 7.1 Agent Architecture
- [ ] Create `agent/` module:
  ```
  api/agentx_ai/agent/
  ├── __init__.py
  ├── core.py            # Main Agent class
  ├── planner.py         # Task decomposition
  ├── executor.py        # Action execution
  ├── context.py         # Context management
  └── session.py         # Conversation session
  ```

### 7.2 Agent Core Implementation
- [ ] Implement `Agent` class:
  ```python
  class Agent:
      def __init__(self, config: AgentConfig):
          self.mcp_client = MCPClientManager()
          self.model_provider = get_provider(config.model)
          self.reasoning = ReasoningOrchestrator()
          self.memory = AgentMemory(config.user_id)
          self.drafting = DraftingStrategy(config.drafting)
      
      async def run(self, task: str) -> AgentResult:
          # Plan → Reason → Act → Reflect loop
          ...
  ```

### 7.3 Task Planning
- [ ] Implement `TaskPlanner`:
  - [ ] Decompose complex tasks into subtasks
  - [ ] Identify required tools/resources
  - [ ] Estimate complexity and model requirements
  - [ ] Create execution plan

### 7.4 Context Management
- [ ] Implement `ContextManager`:
  - [ ] Sliding window for long conversations
  - [ ] Summarization of old context
  - [ ] Priority-based context selection
  - [ ] Memory injection (from AgentMemory)

### 7.5 Agent API Endpoints
- [ ] `POST /api/agent/run` - Execute a task
- [ ] `POST /api/agent/chat` - Conversational interaction
- [ ] `GET /api/agent/status/{task_id}` - Check task status
- [ ] `POST /api/agent/cancel/{task_id}` - Cancel running task
- [ ] `GET /api/agent/traces/{task_id}` - Get reasoning trace

---

## Phase 8: Client Updates

> **Priority**: MEDIUM  
> **Goal**: Update UI to support new agent capabilities

### 8.1 Agent Tab (New)
- [ ] Create AgentTab component
- [ ] Task input with natural language
- [ ] Real-time reasoning trace display
- [ ] Tool usage visualization
- [ ] Cancel/pause controls

### 8.2 Dashboard Updates
- [ ] Connected MCP servers status
- [ ] Model provider status (API keys valid, local models loaded)
- [ ] Recent agent runs with outcomes
- [ ] Resource usage (tokens, API costs)

### 8.3 Settings Tab (New)
- [ ] MCP server configuration UI
- [ ] Model provider API key management
- [ ] Drafting strategy selection
- [ ] Reasoning preferences
- [ ] Memory retention settings

### 8.4 Tools Tab Updates
- [ ] MCP tool browser (all available tools from connected servers)
- [ ] Tool testing interface
- [ ] Reasoning trace explorer
- [ ] Memory/knowledge graph browser

---

## Phase 9: Security & Infrastructure

> **Priority**: MEDIUM  
> **Goal**: Secure and stabilize the platform

### 9.1 Tauri Security
- [ ] Configure Content Security Policy in `tauri.conf.json`
- [ ] Review and restrict Tauri capabilities
- [ ] Disable devtools in production builds

### 9.2 API Security
- [ ] Add rate limiting middleware
- [ ] API key authentication for external access
- [ ] Input validation and sanitization
- [ ] CORS configuration for production

### 9.3 Secrets Management
- [ ] Secure storage for API keys (Keyring/OS keychain)
- [ ] Encryption for sensitive config
- [ ] Audit logging for sensitive operations

---

## Phase 10: Testing

> **Priority**: MEDIUM  
> **Goal**: Ensure reliability through automated testing

### 10.1 Backend Tests
- [ ] Fix and unskip `test_language_detect`
- [ ] Add translation tests:
  - [ ] Test multiple language pairs
  - [ ] Test error handling (invalid language codes)
  - [ ] Test long text handling
- [ ] Add memory system tests:
  - [ ] Test fact storage and retrieval
  - [ ] Test goal lifecycle
  - [ ] Test memory relevance ranking
- [ ] Add MCP tool tests:
  - [ ] Test each tool with valid inputs
  - [ ] Test error handling
- [ ] Add reasoning framework tests:
  - [ ] Test CoT step extraction
  - [ ] Test ToT branching and pruning
  - [ ] Test ReAct action loop
- [ ] Add drafting strategy tests:
  - [ ] Test speculative decoding acceptance
  - [ ] Test pipeline stage transitions
  - [ ] Test candidate ranking

### 10.2 Integration Tests
- [ ] Test full translation flow (client → API → model → response)
- [ ] Test memory persistence across restarts
- [ ] Test Docker service dependencies

### 10.3 Client Tests (Optional for Prototype)
- [ ] Add React component tests
- [ ] Add E2E tests with Playwright/Cypress

---

## Phase 11: Documentation

> **Priority**: LOW  
> **Goal**: Comprehensive documentation for users and developers

### 11.1 User Documentation
- [ ] Update README.md with:
  - Quick start guide
  - Feature overview
  - Screenshots
- [ ] Add installation guide for each platform
- [ ] Document MCP setup for AI assistants

### 11.2 Developer Documentation  
- [ ] Update CLAUDE.md with MCP details
- [ ] Add API documentation (auto-generate from OpenAPI)
- [ ] Add architecture diagrams
- [ ] Document contribution guidelines

### 11.3 Inline Documentation
- [ ] Add docstrings to all public functions
- [ ] Add type hints throughout Python code
- [ ] Document complex algorithms/flows

---

## Backlog (Future Enhancements)

> Items to consider after prototype is complete

- [ ] GPU acceleration for translation models
- [ ] Lazy model loading with progress indicator
- [ ] Multiple user support
- [ ] Cloud sync for memories
- [ ] Plugin system for additional tools
- [ ] Voice input/output
- [ ] Mobile companion app
- [ ] Offline mode with cached models

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Critical Fixes | ✅ Complete | 100% |
| Phase 2: Wire Up Code | ✅ Complete | 100% |
| Phase 3: MCP Client | Not Started | 0% |
| Phase 4: Model Providers | Not Started | 0% |
| Phase 5: Drafting Framework | Not Started | 0% |
| Phase 6: Reasoning Framework | Not Started | 0% |
| Phase 7: Agent Core | Not Started | 0% |
| Phase 8: Client Updates | Not Started | 0% |
| Phase 9: Security | Not Started | 0% |
| Phase 10: Testing | Not Started | 0% |
| Phase 11: Documentation | Not Started | 0% |

---

## Notes

### Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-21 | MCP Client (not server) | AgentX consumes tools from external MCP servers |
| 2026-01-21 | Multi-strategy drafting | Support speculative, pipeline, and candidate generation |
| 2026-01-21 | Flexible reasoning | Support CoT, ToT, ReAct, Reflection patterns |
| 2026-01-21 | Keep agent_memory architecture | Well-designed, integrates with reasoning |
| 2026-01-21 | Use Todo.md for tracking | Simple, version controlled |

### Blockers
_None currently_

### Questions to Resolve
- [ ] Which LLM providers to prioritize? (OpenAI / Anthropic / Ollama / Together)
- [ ] Should agent memory require authentication?
- [ ] Target platforms for distribution? (Windows / macOS / Linux)
- [ ] Reasoning trace storage format? (JSON / SQLite / Neo4j)

---

## Dependencies to Add

```toml
# Add to pyproject.toml for new capabilities
dependencies = [
    # ... existing ...
    
    # MCP Client
    "mcp>=1.0.0",
    
    # Missing from current pyproject.toml
    "redis>=5.0.0",
    "sqlalchemy>=2.0.0",
    
    # Model Providers
    "openai>=1.0.0",
    "anthropic>=0.20.0",
    "httpx>=0.27.0",           # For async HTTP (Ollama, etc.)
    "aiohttp>=3.9.0",          # Alternative async HTTP
    
    # Async Support
    "asyncio>=3.4.3",
    
    # Reasoning/Drafting
    "networkx>=3.0",           # For ToT graph structures
    "numpy>=1.26.0",           # For embeddings/scoring
]
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AgentX Desktop App                          │
├─────────────────────────────────────────────────────────────────────┤
│                           Tauri Client                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Dashboard │ │Translate │ │  Agent   │ │  Tools   │ │ Settings │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                         Django API Layer                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        Agent Core                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │   Planner   │  │  Executor   │  │  Context Manager    │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │  MCP Client    │  │   Reasoning    │  │    Drafting        │     │
│  │  ┌──────────┐  │  │  ┌──────────┐  │  │  ┌──────────────┐  │     │
│  │  │FS Server │  │  │  │   CoT    │  │  │  │ Speculative  │  │     │
│  │  │GH Server │  │  │  │   ToT    │  │  │  │ Pipeline     │  │     │
│  │  │DB Server │  │  │  │  ReAct   │  │  │  │ Candidate    │  │     │
│  │  │Custom... │  │  │  │Reflection│  │  │  └──────────────┘  │     │
│  │  └──────────┘  │  │  └──────────┘  │  └────────────────────┘     │
│  └────────────────┘  └────────────────┘                              │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐     │
│  │Model Providers │  │ Agent Memory   │  │  Translation Kit   │     │
│  │ OpenAI        │  │  Episodic      │  │  NLLB-200          │     │
│  │ Anthropic     │  │  Semantic      │  │  Language Detect   │     │
│  │ Ollama        │  │  Procedural    │  └────────────────────┘     │
│  │ Together      │  │  Working       │                              │
│  └────────────────┘  └────────────────┘                              │
├─────────────────────────────────────────────────────────────────────┤
│                        Data Layer                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Neo4j   │  │  PostgreSQL  │  │  Redis   │  │ Local Models  │   │
│  │ (Graph)  │  │  (pgvector)  │  │ (Cache)  │  │  (HuggingFace)│   │
│  └──────────┘  └──────────────┘  └──────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

                    External MCP Servers
┌──────────────────────────────────────────────────────────────────────┐
│  @mcp/filesystem  │  @mcp/github  │  @mcp/postgres  │  Custom MCP   │
└──────────────────────────────────────────────────────────────────────┘
```

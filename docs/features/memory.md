# Memory System

AgentX features a sophisticated cognitive memory system inspired by human memory architecture, enabling the AI to remember past conversations, learn from interactions, and build a knowledge graph over time.

## Overview

The memory system provides four types of memory:

| Memory Type | Purpose | Storage | Retrieval |
|------------|---------|---------|-----------|
| **Working** | Current conversation, active goals | Redis | Direct lookup |
| **Episodic** | Past conversations, events | Neo4j + PostgreSQL | Vector similarity + temporal |
| **Semantic** | Facts, entities, concepts | Neo4j graph | Graph traversal + vector |
| **Procedural** | Successful strategies, tool patterns | Neo4j graph | Task-type matching |

## Design Principles

- **Extensibility**: Easy to add new memory types, stores, or extraction methods
- **Transparency**: All memory operations traceable per session/conversation
- **Auditability**: Full query trace and operation audit trail in PostgreSQL
- **Channel Scoping**: Memory organized into channels for project-level organization

## Memory Channels

Memory is organized into **channels** — traceable scopes that group related memories:

| Channel | Description | Examples |
|---------|-------------|----------|
| `_global` | Default channel, user-wide memory | Preferences, communication style, general facts |
| `<project>` | Project-specific memory containers | `my-rust-project`, `thesis-research`, `work-api` |

**Key behaviors:**
- Retrieval queries both the active channel AND `_global`, merging results
- Channels are traceable scopes, not isolation boundaries
- Cross-channel operations are logged in the audit trail
- Prominent project facts can be promoted to `_global` based on confidence/frequency thresholds

```python
# Using channels
memory = AgentMemory(user_id="user123", channel="my-rust-project")

# Retrieval automatically merges project + global memories
context = memory.remember("What error handling pattern should I use?")
# Returns: project-specific Rust patterns + global user preferences
```

## Key Features

### Episodic Memory
- Stores complete conversation history with turns
- Vector-based semantic search across conversations
- Temporal filtering and recency boosting
- Automatic consolidation to semantic memory

### Semantic Memory
- Entity recognition and tracking (people, organizations, concepts)
- Fact extraction with confidence scores
- Knowledge graph with relationships
- Entity salience scoring and decay

### Procedural Memory
- Records tool usage patterns
- Learns successful strategies for tasks
- Reinforcement learning from outcomes
- Tool performance analytics

### Working Memory
- Redis-based fast access for current session
- Maintains recent conversation context
- Active goal tracking
- Chain-of-thought reasoning steps

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENT RUNTIME                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   LLM API   │  │  Tool Exec  │  │  Planning   │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         │                │                │                          │
│         └────────────────┼────────────────┘                          │
│                          ▼                                           │
│                 ┌─────────────────┐                                  │
│                 │ Memory Interface│                                  │
│                 └────────┬────────┘                                  │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    Neo4j      │  │  PostgreSQL   │  │    Redis      │
│ Graph+Vector  │  │  pgvector     │  │ Working Mem   │
│               │  │  Time-series  │  │ Cache Layer   │
│ - Semantic    │  │               │  │               │
│ - Episodic    │  │ - Raw logs    │  │ - Hot queries │
│ - Procedural  │  │ - Audit trail │  │ - Session     │
│ - Entities    │  │ - Timeline    │  │ - Rate limits │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Graph Database | Neo4j 5.15+ | Knowledge graph, vector search, relationships |
| Relational DB | PostgreSQL 16+ | Logs, audit, time-series, backup vectors |
| Vector Extension | pgvector 0.7+ | ANN search in PostgreSQL |
| Cache | Redis 7+ | Working memory, session state |
| Embeddings | OpenAI / Local | text-embedding-3-small or nomic-embed-text |

## Usage

### Basic Usage

```python
from agentx_ai.kit.agent_memory import AgentMemory, Turn

# Initialize memory for a user
memory = AgentMemory(user_id="user123", conversation_id="conv456")

# Store a conversation turn
turn = Turn(
    conversation_id="conv456",
    index=0,
    role="user",
    content="What's the weather like today?"
)
memory.store_turn(turn)

# Retrieve relevant memories
context = memory.remember("What did we discuss about weather?")
print(context.to_context_string())

# Learn a new fact
memory.learn_fact(
    claim="User prefers concise responses",
    source="inferred",
    confidence=0.8
)

# Track a goal
from agentx_ai.kit.agent_memory import Goal
goal = Goal(
    description="Help user plan vacation",
    priority=4
)
memory.add_goal(goal)

# Record tool usage for learning
memory.record_tool_usage(
    tool_name="web_search",
    tool_input={"query": "weather"},
    tool_output={"results": [...]},
    success=True,
    latency_ms=250
)
```

### Advanced Retrieval

```python
# Retrieve with filters
context = memory.remember(
    query="Python programming discussion",
    top_k=15,
    include_episodic=True,
    include_semantic=True,
    include_procedural=True,
    time_window_hours=72  # Last 3 days only
)

# Find what worked for similar tasks
strategies = memory.what_worked_for("data analysis task")
for strategy in strategies:
    print(f"Strategy: {strategy.description}")
    print(f"Tools: {strategy.tool_sequence}")
    print(f"Success rate: {strategy.success_count / max(1, strategy.success_count + strategy.failure_count)}")

# Get active goals
goals = memory.get_active_goals()
for goal in goals:
    print(f"[P{goal.priority}] {goal.description} - {goal.status}")
```

## Background Processing

The system includes a consolidation worker that runs background jobs:

- **Consolidation** (every 15 min): Extracts entities and facts from recent conversations
- **Pattern Detection** (hourly): Learns successful strategies from conversation outcomes
- **Memory Decay** (daily): Applies time-based decay to salience scores
- **Cleanup** (daily): Archives old conversations and removes low-salience entities

To run the worker:

```bash
python -m agentx_ai.kit.agent_memory.consolidation.worker
```

## Configuration

Configure the memory system via environment variables:

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# PostgreSQL
POSTGRES_URI=postgresql://agent:password@localhost:5432/agent_memory

# Redis
REDIS_URI=redis://localhost:6379

# Embeddings
EMBEDDING_PROVIDER=openai  # or "local"
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# Memory settings
EPISODIC_RETENTION_DAYS=90
FACT_CONFIDENCE_THRESHOLD=0.7
SALIENCE_DECAY_RATE=0.95
MAX_WORKING_MEMORY_ITEMS=50
```

## Audit Logging

All memory operations are logged to a partitioned PostgreSQL table for traceability:

| Log Level | What's Logged |
|-----------|---------------|
| `off` | No audit logging |
| `writes` | Store, update, delete operations (default) |
| `reads` | All reads and writes with query details |
| `verbose` | Full traces including payloads |

**Logged information includes:**
- Operation type, timestamp, user/session/conversation IDs
- Source channel and target channels (for cross-channel operations)
- Query text and result count for retrievals
- Latency per operation
- Promotion tracking (when facts are promoted from project to `_global`)
- Configuration snapshot (active thresholds at time of operation)

```bash
# Configure audit logging
AUDIT_LOG_LEVEL=writes
AUDIT_RETENTION_DAYS=30
```

## Database Setup

Initialize the memory system schemas before first use:

```bash
# Start database services
task db:up

# Initialize all schemas (Neo4j, PostgreSQL, Redis)
task db:init:schemas

# Or verify existing schemas
task db:verify:schemas
```

This creates:
- **Neo4j**: Vector indexes, uniqueness constraints, channel indexes
- **PostgreSQL**: Memory tables with channel columns, partitioned audit log
- **Redis**: Verifies connectivity and documents key patterns

## Status

The memory system implementation is complete and syntax-error free. Current status:

- ✅ Core memory interfaces implemented
- ✅ Episodic, semantic, procedural, and working memory modules
- ✅ Multi-strategy retrieval engine
- ✅ Background consolidation worker
- ✅ Memory decay and cleanup utilities
- ✅ Database schema initialization (`task db:init:schemas`)
- ✅ Channel scoping support in all schemas
- ✅ Partitioned audit log table (daily partitions)
- ⏳ Agent core integration (wiring memory into chat/run flows)
- ⏳ Entity/fact extraction (LLM-based implementation)
- ⏳ Audit logger instrumentation

## Related Documentation

- [Memory System Architecture](../architecture/memory.md) - Detailed technical architecture
- [Database Stack](../architecture/databases.md) - Infrastructure details

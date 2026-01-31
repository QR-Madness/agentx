# Memory System Architecture

Technical architecture documentation for AgentX's cognitive memory system.

## Design Principles

- **Extensibility**: Easy to add new memory types, stores, or extraction methods without backwards-compatibility constraints
- **Transparency**: All memory operations traceable per session/conversation based on logging settings
- **Auditability**: Full query trace and operation audit trail in PostgreSQL
- **Channel Scoping**: Memory organized into channels — `_global` (default, user-wide) and project channels

## System Architecture

The memory system is organized into four layers:

### 1. Interface Layer
- **AgentMemory** - Main unified API
- Provides high-level operations for storing and retrieving memories
- Manages coordination between different memory types
- Accepts `channel` parameter (default `"_global"`) for memory scoping

### 2. Memory Subsystems

#### Episodic Memory (`memory/episodic.py`)
Stores conversation history with full context:

```python
class EpisodicMemory:
    def store_turn(turn: Turn, channel: str = "_global") -> None
    def store_turn_log(turn: Turn, channel: str = "_global") -> None
    def vector_search(query_embedding, top_k, user_id, channels, time_window_hours) -> List[Dict]
    def get_conversation(conversation_id: str) -> List[Turn]
    def get_recent_turns(user_id, channels, hours, limit) -> List[Dict]
```

**Storage:**
- Neo4j: Graph structure with Turn nodes and relationships (with channel property)
- PostgreSQL: Audit log and time-series backup (with channel column)

**Indexing:**
- Vector index on turn embeddings (1536 dimensions)
- BRIN index on timestamps for efficient time-range queries
- B-tree index on conversation_id and channel

#### Semantic Memory (`memory/semantic.py`)
Manages entities, facts, and conceptual knowledge:

```python
class SemanticMemory:
    def upsert_entity(entity: Entity, channel: str = "_global") -> Entity
    def store_fact(fact: Fact, channel: str = "_global") -> None
    def vector_search_facts(query_embedding, top_k, channels, min_confidence) -> List[Dict]
    def vector_search_entities(query_embedding, top_k, channels) -> List[Dict]
    def get_entity_graph(entity_ids, depth) -> Dict
    def create_relationship(source_id, target_id, rel_type, properties) -> None
    def promote_to_global(fact_id: str, reason: str) -> None
```

**Storage:**
- Neo4j: Entity and Fact nodes with typed relationships and channel property
- Supports multi-hop graph traversal

**Key Relationships:**
- `RELATED_TO`, `PART_OF`, `LOCATED_IN`, `WORKS_FOR`, `KNOWS`, `CREATED_BY`, `REFERENCES`
- `DERIVED_FROM` (Fact → Turn)
- `ABOUT` (Fact → Entity)
- `PROMOTED_FROM` (Fact in _global → original Fact in project channel)

#### Procedural Memory (`memory/procedural.py`)
Tracks tool usage and successful strategies:

```python
class ProceduralMemory:
    def record_invocation(conversation_id, turn_id, tool_name, tool_input,
                         tool_output, success, latency_ms, channel) -> None
    def learn_strategy(description, context_pattern, tool_sequence,
                       from_conversation_id, success, channel) -> Strategy
    def find_strategies(task_description, channels, top_k) -> List[Strategy]
    def reinforce_strategy(strategy_id, success) -> None
    def get_tool_stats(task_type, channel) -> List[Dict]
```

**Storage:**
- Neo4j: Strategy and Tool nodes with performance metrics and channel property
- PostgreSQL: Tool invocation audit log with channel column

**Learning Mechanism:**
- Tracks success/failure counts for strategies
- Calculates success rates for recommendation
- Links strategies to task types and tool sequences

#### Working Memory (`memory/working.py`)
Fast, ephemeral storage for current session:

```python
class WorkingMemory:
    # Key pattern: working:{user_id}:{channel}:{conversation_id}:*
    def add_turn(turn) -> None
    def get_recent_turns(limit) -> List[Dict]
    def set(key, value, ttl_seconds) -> None
    def get(key) -> Optional[Any]
    def delete(key) -> None
    def get_context() -> Dict
    def clear_session() -> None
    def set_active_goal(goal_id, goal_description) -> None
    def get_active_goal() -> Optional[Dict]
    def push_thought(thought) -> None
    def get_thoughts(limit) -> List[Dict]
```

**Storage:**
- Redis lists for turns and thoughts (keys include channel segment)
- Redis strings (with TTL) for context values
- Automatic expiration after 1 hour (configurable)

**Key Patterns:**
- `working:{user_id}:{channel}:{conversation_id}:turns` - Recent turns
- `working:{user_id}:{channel}:{conversation_id}:context` - Context window
- `consolidation:job:{job_id}` - Consolidation job tracking
- `consolidation:lock:{conversation_id}` - Consolidation locks
- `session:{session_id}:*` - Session-scoped ephemeral data

### 3. Retrieval Layer

Multi-strategy retrieval engine combines:

```python
class MemoryRetriever:
    def retrieve(query, user_id, channels, top_k, include_episodic,
                include_semantic, include_procedural,
                time_window_hours) -> MemoryBundle
```

**Channel Behavior:**
- Retrieval always queries both the active channel AND `_global`
- Results are merged and deduplicated
- Cross-channel matches are logged in the audit trail

**Strategies:**
1. **Vector Similarity**: Embed query and search each memory type
2. **Graph Traversal**: Expand from matched entities to related nodes
3. **Temporal Filtering**: Boost recent memories, filter by time window
4. **Reranking**: Ensure diversity, limit items per conversation

**Retrieval Weights:**
- Episodic: 0.3
- Semantic (Facts): 0.25
- Semantic (Entities): 0.2
- Procedural: 0.15
- Recency: 0.1

### 4. Background Processing

Consolidation worker runs scheduled jobs:

```python
class ConsolidationWorker:
    jobs = {
        "consolidate": {"func": consolidate_episodic_to_semantic, "interval_minutes": 15},
        "patterns": {"func": detect_patterns, "interval_minutes": 60},
        "decay": {"func": apply_memory_decay, "interval_minutes": 1440},
        "cleanup": {"func": cleanup_old_memories, "interval_minutes": 1440},
        "promote": {"func": promote_prominent_facts, "interval_minutes": 60},
        "audit_partitions": {"func": manage_audit_partitions, "interval_minutes": 1440}
    }
```

**Jobs:**
- **consolidate_episodic_to_semantic**: Extracts entities and facts from conversations
- **detect_patterns**: Learns strategies from successful conversations
- **apply_memory_decay**: Reduces salience/confidence over time
- **cleanup_old_memories**: Archives old conversations, deletes low-salience entities
- **promote_prominent_facts**: Promotes high-confidence project facts to `_global`
- **manage_audit_partitions**: Creates future partitions, drops old ones per retention policy

### 5. Audit Layer

All memory operations are logged to PostgreSQL for traceability:

```python
class MemoryAuditLogger:
    def log_write(operation, memory_type, channel, affected_ids, metadata) -> None
    def log_read(query, channels_searched, result_count, latency_ms) -> None
    def log_promotion(fact_id, source_channel, reason, thresholds) -> None
```

**Log Levels:**
- `off`: No audit logging
- `writes`: Only log mutations (default)
- `reads`: Log reads and writes
- `verbose`: Full query traces with payloads

**Promotion Thresholds (configurable):**
- `confidence >= 0.85`
- `access_count >= 5`
- `conversations >= 2`

Active thresholds are snapshotted in each audit log entry for reproducibility.

## Memory Channels

Channels organize memory into traceable scopes:

| Channel | Description |
|---------|-------------|
| `_global` | Default channel, user-wide memory (preferences, general facts) |
| `<project>` | Project-specific memory containers (e.g., `my-rust-project`) |

**Key behaviors:**
- Channels are **traceable scopes, not isolation boundaries**
- Retrieval queries both active channel + `_global`, merging results
- Cross-channel intersections are logged in audit trail
- Prominent project facts auto-promote to `_global` based on thresholds

## Database Schemas

### Neo4j Graph Schema

#### Node Types

**Conversation**
```cypher
(:Conversation {
    id: string (unique),
    user_id: string,
    channel: string,  -- '_global' or project name
    started_at: datetime,
    title: string,
    consolidated: datetime,
    patterns_extracted: boolean,
    archived: boolean
})
```

**Turn**
```cypher
(:Turn {
    id: string (unique),
    user_id: string,
    channel: string,
    index: integer,
    timestamp: datetime,
    role: string,
    content: text,
    embedding: vector(1536),
    token_count: integer,
    archived: boolean
})
```

**Entity**
```cypher
(:Entity {
    id: string (unique),
    user_id: string,
    channel: string,
    name: string,
    type: string,
    aliases: list<string>,
    description: text,
    embedding: vector(1536),
    salience: float,
    first_seen: datetime,
    last_accessed: datetime,
    access_count: integer,
    properties: map
})
```

**Fact**
```cypher
(:Fact {
    id: string (unique),
    user_id: string,
    channel: string,
    claim: text,
    confidence: float,
    source: string,
    source_turn_id: string,
    embedding: vector(1536),
    created_at: datetime
})
```

**Goal**
```cypher
(:Goal {
    id: string (unique),
    user_id: string,
    channel: string,
    description: text,
    status: string,
    priority: integer,
    created_at: datetime,
    deadline: datetime,
    embedding: vector(1536)
})
```

**Strategy**
```cypher
(:Strategy {
    id: string (unique),
    user_id: string,
    channel: string,
    description: text,
    context_pattern: string,
    tool_sequence: list<string>,
    embedding: vector(1536),
    success_count: integer,
    failure_count: integer,
    created_at: datetime,
    last_used: datetime
})
```

**Tool**
```cypher
(:Tool {
    name: string (unique),
    usage_count: integer,
    success_count: integer,
    avg_latency_ms: float
})
```

#### Relationship Types

- `(:Conversation)-[:HAS_TURN]->(:Turn)`
- `(:Turn)-[:FOLLOWED_BY]->(:Turn)`
- `(:Conversation)-[:MENTIONS]->(:Entity)`
- `(:Fact)-[:DERIVED_FROM]->(:Turn)`
- `(:Fact)-[:ABOUT]->(:Entity)`
- `(:Fact)-[:PROMOTED_FROM]->(:Fact)` — Links global fact to original project fact
- `(:User)-[:HAS_GOAL]->(:Goal)`
- `(:Goal)-[:SUBGOAL_OF]->(:Goal)`
- `(:Goal)-[:BLOCKED_BY]->(:Goal)`
- `(:Strategy)-[:USES_TOOL]->(:Tool)`
- `(:Strategy)-[:SUCCEEDED_IN]->(:Conversation)`
- `(:Conversation)-[:USED_TOOL]->(:ToolInvocation)-[:INVOKED]->(:Tool)`

#### Channel Indexes

```cypher
// Channel indexes for scoped queries
CREATE INDEX turn_channel IF NOT EXISTS FOR (t:Turn) ON (t.channel);
CREATE INDEX entity_channel IF NOT EXISTS FOR (e:Entity) ON (e.channel);
CREATE INDEX fact_channel IF NOT EXISTS FOR (f:Fact) ON (f.channel);
CREATE INDEX strategy_channel IF NOT EXISTS FOR (s:Strategy) ON (s.channel);
CREATE INDEX conversation_channel IF NOT EXISTS FOR (c:Conversation) ON (c.channel);
CREATE INDEX goal_channel IF NOT EXISTS FOR (g:Goal) ON (g.channel);

// Composite indexes for common query patterns
CREATE INDEX conversation_user_channel IF NOT EXISTS FOR (c:Conversation) ON (c.user_id, c.channel);
CREATE INDEX entity_user_channel IF NOT EXISTS FOR (e:Entity) ON (e.user_id, e.channel);
CREATE INDEX fact_user_channel IF NOT EXISTS FOR (f:Fact) ON (f.user_id, f.channel);
```

#### Vector Indexes

```cypher
// Turn embeddings (episodic memory)
CREATE VECTOR INDEX turn_embeddings IF NOT EXISTS
FOR (t:Turn) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

// Entity embeddings (semantic memory)
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

// Fact embeddings
CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
FOR (f:Fact) ON (f.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

// Strategy embeddings (procedural memory)
CREATE VECTOR INDEX strategy_embeddings IF NOT EXISTS
FOR (s:Strategy) ON (s.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
```

### PostgreSQL Schema

#### conversation_logs
```sql
CREATE TABLE conversation_logs (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    token_count INTEGER,
    model VARCHAR(100),
    channel VARCHAR(100) NOT NULL DEFAULT '_global',
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    UNIQUE(conversation_id, turn_index)
);

CREATE INDEX idx_logs_timestamp ON conversation_logs USING BRIN (timestamp);
CREATE INDEX idx_logs_conversation ON conversation_logs (conversation_id);
CREATE INDEX idx_logs_channel ON conversation_logs (channel);
CREATE INDEX idx_logs_embedding ON conversation_logs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

#### memory_timeline
```sql
CREATE TABLE memory_timeline (
    id BIGSERIAL PRIMARY KEY,
    memory_type VARCHAR(50) NOT NULL,
    neo4j_node_id VARCHAR(100),
    event_time TIMESTAMPTZ NOT NULL,
    summary TEXT,
    embedding vector(1536),
    importance_score FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    channel VARCHAR(100) NOT NULL DEFAULT '_global',
    archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_timeline_time ON memory_timeline USING BRIN (event_time);
CREATE INDEX idx_timeline_type ON memory_timeline (memory_type);
CREATE INDEX idx_timeline_importance ON memory_timeline (importance_score DESC);
CREATE INDEX idx_timeline_channel ON memory_timeline (channel);
CREATE INDEX idx_timeline_embedding ON memory_timeline USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

#### tool_invocations
```sql
CREATE TABLE tool_invocations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name VARCHAR(100) NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB,
    success BOOLEAN,
    latency_ms INTEGER,
    error_message TEXT,
    channel VARCHAR(100) NOT NULL DEFAULT '_global'
);

CREATE INDEX idx_tools_conversation ON tool_invocations (conversation_id);
CREATE INDEX idx_tools_name ON tool_invocations (tool_name);
CREATE INDEX idx_tools_timestamp ON tool_invocations USING BRIN (timestamp);
CREATE INDEX idx_tools_channel ON tool_invocations (channel);
```

#### user_profiles
```sql
CREATE TABLE user_profiles (
    user_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    preferences JSONB DEFAULT '{}',
    expertise_areas JSONB DEFAULT '[]',
    communication_style JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);
```

#### memory_audit_log (Partitioned)
```sql
CREATE TABLE memory_audit_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    operation VARCHAR(50) NOT NULL,      -- 'store', 'retrieve', 'update', 'delete', 'promote'
    memory_type VARCHAR(50) NOT NULL,    -- 'episodic', 'semantic', 'procedural', 'working'
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    conversation_id UUID,
    source_channel VARCHAR(100),         -- Channel where operation originated
    target_channels TEXT[],              -- Channels queried/affected
    query_text TEXT,                     -- For retrieval: the query used
    result_count INTEGER,
    latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    -- Promotion tracking
    promoted_from_channel VARCHAR(100),
    promotion_confidence FLOAT,
    promotion_access_count INTEGER,
    promotion_conversation_count INTEGER,
    -- Configuration snapshot
    config_snapshot JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Daily partitions created automatically
-- Retention managed by drop_old_audit_partitions(retention_days)

CREATE INDEX idx_audit_user ON memory_audit_log (user_id, timestamp);
CREATE INDEX idx_audit_session ON memory_audit_log (session_id, timestamp);
CREATE INDEX idx_audit_operation ON memory_audit_log (operation, timestamp);
CREATE INDEX idx_audit_memory_type ON memory_audit_log (memory_type, timestamp);
CREATE INDEX idx_audit_source_channel ON memory_audit_log (source_channel, timestamp);
```

#### schema_version
```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## Data Flow

### 1. Storing a Conversation Turn

```
User Input → AgentMemory.store_turn()
    ├─→ Generate embedding (if missing)
    ├─→ EpisodicMemory.store_turn() → Neo4j (graph structure)
    ├─→ EpisodicMemory.store_turn_log() → PostgreSQL (audit log)
    └─→ WorkingMemory.add_turn() → Redis (session cache)
```

### 2. Memory Retrieval

```
Query → AgentMemory.remember()
    └─→ MemoryRetriever.retrieve()
        ├─→ Generate query embedding
        ├─→ EpisodicMemory.vector_search() → relevant turns
        ├─→ SemanticMemory.vector_search_facts() → relevant facts
        ├─→ SemanticMemory.vector_search_entities() → relevant entities
        │   └─→ SemanticMemory.get_entity_graph() → expand to related
        ├─→ ProceduralMemory.find_strategies() → applicable strategies
        ├─→ Get active goals and user context
        └─→ Rerank and return MemoryBundle
```

### 3. Background Consolidation

```
ConsolidationWorker (every 15 min)
    └─→ consolidate_episodic_to_semantic()
        ├─→ Query unconsolidated conversations
        ├─→ Extract entities → SemanticMemory.upsert_entity()
        ├─→ Extract facts → SemanticMemory.store_fact()
        └─→ Mark conversation as consolidated
```

## Memory Decay

Implements forgetting curve based on:

- **Time since last access**: Exponential decay
- **Access frequency**: Logarithmic boost
- **Recency**: Inverse exponential boost

```python
# Exponential decay formula
decayed_value = initial_value * (decay_rate ** days_since_access)

# Default decay rate: 0.95 (5% daily decay)
# Half-life ≈ 14 days
```

**Decay Targets:**
- Entity salience: Daily decay for entities not accessed in 24h
- Fact confidence: Weekly decay for inferred facts (slower decay)

## Performance Considerations

### Vector Search Optimization

- Use ANN (Approximate Nearest Neighbors) with IVFFlat index
- Over-fetch (2x top_k) to account for filtering
- Cache embeddings for frequently accessed items

### Graph Traversal Limits

- Limit depth to 2-3 hops to prevent slow queries
- Use `LIMIT` clauses in all Cypher queries
- Consider materialized paths for deep hierarchies

### Redis Memory Management

- Use TTLs on all working memory keys
- Configure LRU eviction policy: `maxmemory-policy allkeys-lru`
- Monitor memory usage with `INFO memory`

### PostgreSQL Time-Series Optimization

- BRIN indexes for timestamp columns (highly efficient for time-series)
- Partition large tables by time range
- Use `EXPLAIN ANALYZE` to optimize slow queries

## Scaling Considerations

### Horizontal Scaling

- **Neo4j**: Use Neo4j Enterprise with clustering (read replicas)
- **PostgreSQL**: Use read replicas for analytics queries
- **Redis**: Use Redis Cluster for distributed caching

### Vertical Scaling

- **Neo4j**: Allocate 2-4GB heap, 1GB+ page cache
- **PostgreSQL**: Tune `shared_buffers`, `work_mem`, `effective_cache_size`
- **Redis**: Allocate 512MB-2GB depending on working memory size

### Data Retention

- Archive conversations older than 90 days (configurable)
- Delete entities with salience < 0.1 not accessed in 30 days
- Keep audit logs in PostgreSQL for compliance (longer retention)

## Security Considerations

- Store embeddings in vector format (not reversible to text)
- Encrypt sensitive data in PostgreSQL
- Use connection pooling to prevent exhaustion
- Implement rate limiting on retrieval operations
- Sanitize user input to prevent Cypher injection

## Monitoring

Key metrics to track:

- **Memory size**: Total nodes/relationships in Neo4j
- **Query performance**: P50, P95, P99 latencies for retrieval
- **Consolidation lag**: Time between conversation end and consolidation
- **Cache hit rate**: Redis cache effectiveness
- **Decay rate**: Number of entities/facts decayed per day

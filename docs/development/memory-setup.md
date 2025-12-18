# Memory System Setup Guide

Complete setup instructions for the AgentX memory system.

## Prerequisites

- Docker and Docker Compose (for containerized databases)
- Python 3.11+ with uv or pip
- OpenAI API key (for embeddings) or local model setup

## Quick Start

### 1. Start Database Services

Use the provided Docker Compose configuration:

```bash
# Start all database services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f neo4j postgres redis
```

### 2. Initialize Neo4j Schema

Run the schema initialization script:

```bash
# Connect to Neo4j browser
open http://localhost:7474

# Or use cypher-shell
docker exec -it agent-neo4j cypher-shell -u neo4j -p your_secure_password
```

Execute the following Cypher commands:

```cypher
// ============================================
// CONSTRAINTS AND INDEXES
// ============================================

// Uniqueness constraints
CREATE CONSTRAINT conversation_id IF NOT EXISTS
FOR (c:Conversation) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT fact_id IF NOT EXISTS
FOR (f:Fact) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT goal_id IF NOT EXISTS
FOR (g:Goal) REQUIRE g.id IS UNIQUE;

CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

// Property indexes for fast lookups
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX fact_confidence IF NOT EXISTS FOR (f:Fact) ON (f.confidence);
CREATE INDEX goal_status IF NOT EXISTS FOR (g:Goal) ON (g.status);
CREATE INDEX turn_timestamp IF NOT EXISTS FOR (t:Turn) ON (t.timestamp);

// Full-text search indexes
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases, e.description];

CREATE FULLTEXT INDEX fact_search IF NOT EXISTS
FOR (f:Fact) ON EACH [f.claim];

// ============================================
// VECTOR INDEXES
// ============================================

// Turn embeddings (episodic memory)
CREATE VECTOR INDEX turn_embeddings IF NOT EXISTS
FOR (t:Turn) ON (t.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// Entity embeddings (semantic memory)
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// Fact embeddings
CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
FOR (f:Fact) ON (f.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// Strategy embeddings (procedural memory)
CREATE VECTOR INDEX strategy_embeddings IF NOT EXISTS
FOR (s:Strategy) ON (s.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};
```

### 3. Initialize PostgreSQL Schema

Connect to PostgreSQL and run the initialization script:

```bash
# Connect to PostgreSQL
docker exec -it agent-postgres psql -U agent -d agent_memory

# Or use a SQL file
docker exec -i agent-postgres psql -U agent -d agent_memory < init-scripts/01-init.sql
```

SQL initialization script:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- Conversation logs (append-only time series)
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
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),

    UNIQUE(conversation_id, turn_index)
);

-- BRIN index for time-range queries (very efficient for time-series)
CREATE INDEX idx_logs_timestamp ON conversation_logs USING BRIN (timestamp);
CREATE INDEX idx_logs_conversation ON conversation_logs (conversation_id);
CREATE INDEX idx_logs_embedding ON conversation_logs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Memory timeline (unified temporal index)
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
    archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_timeline_time ON memory_timeline USING BRIN (event_time);
CREATE INDEX idx_timeline_type ON memory_timeline (memory_type);
CREATE INDEX idx_timeline_importance ON memory_timeline (importance_score DESC);
CREATE INDEX idx_timeline_embedding ON memory_timeline USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Tool invocations audit
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
    error_message TEXT
);

CREATE INDEX idx_tools_conversation ON tool_invocations (conversation_id);
CREATE INDEX idx_tools_name ON tool_invocations (tool_name);
CREATE INDEX idx_tools_timestamp ON tool_invocations USING BRIN (timestamp);

-- User preferences and profiles
CREATE TABLE user_profiles (
    user_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    preferences JSONB DEFAULT '{}',
    expertise_areas JSONB DEFAULT '[]',
    communication_style JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_profiles_updated
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# PostgreSQL Configuration
POSTGRES_URI=postgresql://agent:your_secure_password@localhost:5432/agent_memory

# Redis Configuration
REDIS_URI=redis://localhost:6379

# Embedding Provider
EMBEDDING_PROVIDER=openai  # or "local"
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
OPENAI_API_KEY=sk-your-api-key-here

# Local Embedding Model (if using local)
LOCAL_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Memory Settings
EPISODIC_RETENTION_DAYS=90
FACT_CONFIDENCE_THRESHOLD=0.7
SALIENCE_DECAY_RATE=0.95
MAX_WORKING_MEMORY_ITEMS=50

# Retrieval Settings
DEFAULT_TOP_K=10
RERANKING_ENABLED=true
```

### 5. Install Python Dependencies

Add required dependencies to your project:

```bash
# Using uv (recommended)
uv add neo4j redis sqlalchemy psycopg2-binary pgvector pydantic-settings

# For OpenAI embeddings
uv add openai

# For local embeddings
uv add sentence-transformers

# Optional: for entity extraction
uv add spacy
python -m spacy download en_core_web_sm
```

Or add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "neo4j>=5.15.0",
    "redis>=5.0.0",
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pgvector>=0.2.0",
    "pydantic-settings>=2.0.0",
    "openai>=1.0.0",  # For OpenAI embeddings
    "sentence-transformers>=2.2.0",  # For local embeddings
]
```

### 6. Verify Installation

Test the memory system:

```python
from agentx_ai.kit.agent_memory import AgentMemory, Turn
from uuid import uuid4

# Initialize memory
memory = AgentMemory(user_id="test_user", conversation_id=str(uuid4()))

# Store a test turn
turn = Turn(
    conversation_id=memory.conversation_id,
    index=0,
    role="user",
    content="Hello, this is a test message."
)
memory.store_turn(turn)

# Retrieve
context = memory.remember("test message")
print(context.to_context_string())

# Clean up
memory.close()
```

## Running the Consolidation Worker

The background consolidation worker should run as a separate process:

```bash
# Development
python -m agentx_ai.kit.agent_memory.consolidation.worker

# Production (with supervisor/systemd)
# See deployment section below
```

## Docker Compose Configuration

Create `docker-compose.yml` for the memory system databases:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-community
    container_name: agent-neo4j
    ports:
      - "7474:7474"  # Browser
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/your_secure_password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_apoc_export_file_enabled=true
      - NEO4J_apoc_import_file_enabled=true
      - NEO4J_apoc_import_file_use__neo4j__config=true
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=2G
      - NEO4J_server_memory_pagecache_size=1G
    volumes:
      - ./data/neo4j/data:/data
      - ./data/neo4j/logs:/logs
      - ./data/neo4j/plugins:/plugins
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: pgvector/pgvector:pg16
    container_name: agent-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=agent
      - POSTGRES_PASSWORD=your_secure_password
      - POSTGRES_DB=agent_memory
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d agent_memory"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: agent-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - ./data/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Optional: Redis GUI
  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: agent-redis-gui
    ports:
      - "8081:8081"
    environment:
      - REDIS_HOSTS=local:redis:6379
    depends_on:
      - redis
```

## Development Workflow

### Testing Memory Operations

```python
# Test episodic memory
from agentx_ai.kit.agent_memory import AgentMemory, Turn
from uuid import uuid4

memory = AgentMemory(user_id="dev_user", conversation_id=str(uuid4()))

# Add multiple turns
for i, content in enumerate(["Hello", "How are you?", "Tell me about Python"]):
    turn = Turn(
        conversation_id=memory.conversation_id,
        index=i,
        role="user" if i % 2 == 0 else "assistant",
        content=content
    )
    memory.store_turn(turn)

# Test retrieval
context = memory.remember("Python programming", top_k=5)
print(f"Found {len(context.relevant_turns)} relevant turns")

# Test semantic memory
from agentx_ai.kit.agent_memory import Entity

entity = Entity(
    name="Python",
    type="ProgrammingLanguage",
    description="High-level programming language"
)
memory.upsert_entity(entity)

# Test procedural memory
memory.record_tool_usage(
    tool_name="code_interpreter",
    tool_input={"code": "print('hello')"},
    tool_output={"result": "hello"},
    success=True,
    latency_ms=150
)
```

### Monitoring

Check database health:

```bash
# Neo4j stats
docker exec agent-neo4j cypher-shell -u neo4j -p password "CALL dbms.listConfig() YIELD name, value WHERE name STARTS WITH 'dbms.memory' RETURN name, value"

# PostgreSQL stats
docker exec agent-postgres psql -U agent -d agent_memory -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema') ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Redis stats
docker exec agent-redis redis-cli INFO memory
```

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs agent-neo4j

# Test connection
docker exec agent-neo4j cypher-shell -u neo4j -p password "RETURN 'Connected' as status"
```

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs agent-postgres

# Test connection
docker exec agent-postgres pg_isready -U agent -d agent_memory
```

### Redis Connection Issues

```bash
# Check if Redis is running
docker ps | grep redis

# Test connection
docker exec agent-redis redis-cli ping
```

### Vector Index Issues

If vector searches are slow:

```cypher
// Check vector index status
SHOW INDEXES YIELD name, type, state WHERE type = "VECTOR";

// Rebuild vector index if needed
DROP INDEX turn_embeddings;
CREATE VECTOR INDEX turn_embeddings FOR (t:Turn) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
```

## Production Deployment

### Using Supervisor

Create `/etc/supervisor/conf.d/agentx-memory-worker.conf`:

```ini
[program:agentx-memory-worker]
command=/path/to/venv/bin/python -m agentx_ai.kit.agent_memory.consolidation.worker
directory=/path/to/agentx-source
user=agentx
autostart=true
autorestart=true
stderr_logfile=/var/log/agentx/memory-worker.err.log
stdout_logfile=/var/log/agentx/memory-worker.out.log
environment=PATH="/path/to/venv/bin"
```

### Using Systemd

Create `/etc/systemd/system/agentx-memory-worker.service`:

```ini
[Unit]
Description=AgentX Memory Consolidation Worker
After=network.target

[Service]
Type=simple
User=agentx
WorkingDirectory=/path/to/agentx-source
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python -m agentx_ai.kit.agent_memory.consolidation.worker
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable agentx-memory-worker
sudo systemctl start agentx-memory-worker
sudo systemctl status agentx-memory-worker
```

## Next Steps

1. Implement entity extraction (see `extraction/entities.py`)
2. Implement fact extraction (see `extraction/facts.py`)
3. Configure monitoring and alerting
4. Set up backup procedures for databases
5. Tune database parameters for your workload

## Related Documentation

- [Memory System Overview](../features/memory.md)
- [Memory Architecture](../architecture/memory.md)
- [Database Stack](../architecture/databases.md)

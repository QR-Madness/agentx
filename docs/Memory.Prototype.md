# AI Agent Memory Architecture

A reference implementation for building a cognitive memory system for LLM-powered agents using Neo4j (graph + vector), PostgreSQL (relational + vector), and supporting services.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Memory Types](#memory-types)
3. [Technology Stack](#technology-stack)
4. [Docker Compose Setup](#docker-compose-setup)
5. [Database Schemas](#database-schemas)
6. [Python Implementation](#python-implementation)
7. [Memory Operations](#memory-operations)
8. [Background Jobs](#background-jobs)
9. [Kubernetes Deployment](#kubernetes-deployment)
10. [Usage Patterns](#usage-patterns)

---

## Architecture Overview

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
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                 ┌─────────────────┐
                 │  Consolidation  │
                 │    Worker       │
                 │  (Background)   │
                 └─────────────────┘
```

---

## Memory Types

| Type | Purpose | Storage | Retrieval Strategy |
|------|---------|---------|-------------------|
| **Working** | Current conversation, active goals | Redis + in-context | Direct lookup |
| **Episodic** | Past conversations, events | Neo4j graph | Vector similarity + temporal |
| **Semantic** | Facts, entities, concepts | Neo4j graph | Graph traversal + vector |
| **Procedural** | Successful strategies, tool patterns | Neo4j graph | Task-type matching |

---

## Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Graph Database | Neo4j | 5.15+ | Knowledge graph, vector search, relationships |
| Relational DB | PostgreSQL | 16+ | Logs, audit, time-series, backup vectors |
| Vector Extension | pgvector | 0.7+ | ANN search in PostgreSQL |
| Cache | Redis | 7+ | Working memory, session state |
| Queue | Redis Streams | 7+ | Async job processing |
| Embeddings | OpenAI / Local | - | `text-embedding-3-small` or `nomic-embed-text` |
| Python | 3.11+ | - | Runtime |

---

## Docker Compose Setup

```yaml
# docker-compose.yml
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
      # Memory settings - adjust based on your system
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=2G
      - NEO4J_server_memory_pagecache_size=1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_plugins:/plugins
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
      - postgres_data:/var/lib/postgresql/data
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
      - redis_data:/data
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

  # Optional: Background worker for consolidation
  consolidation-worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: agent-consolidation
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=your_secure_password
      - POSTGRES_URI=postgresql://agent:your_secure_password@postgres:5432/agent_memory
      - REDIS_URI=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      neo4j:
        condition: service_healthy
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_plugins:
  postgres_data:
  redis_data:
```

### PostgreSQL Init Script

```sql
-- init-scripts/01-init.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For fuzzy text search

-- Conversation logs (append-only time series)
CREATE TABLE conversation_logs (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    turn_index INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    token_count INTEGER,
    model VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- Adjust dimension as needed
    
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
    memory_type VARCHAR(50) NOT NULL,  -- 'episode', 'fact', 'entity', 'reflection'
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

---

## Database Schemas

### Neo4j Schema (Cypher)

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

// ============================================
// EPISODIC MEMORY SCHEMA
// ============================================

// Example: Create a conversation with turns
// MERGE (c:Conversation {id: $conv_id})
// SET c.started_at = datetime($started_at),
//     c.user_id = $user_id,
//     c.title = $title

// CREATE (t:Turn {
//     id: $turn_id,
//     index: $index,
//     timestamp: datetime(),
//     role: $role,
//     content: $content,
//     embedding: $embedding,
//     token_count: $tokens
// })
// MERGE (c)-[:HAS_TURN]->(t)

// ============================================
// SEMANTIC MEMORY SCHEMA
// ============================================

// Entity types: Person, Organization, Concept, Location, Event, Document, etc.
// 
// Entity properties:
//   - id: UUID
//   - name: string
//   - type: string (label also applied as :Person, :Organization, etc.)
//   - aliases: list of strings
//   - description: string
//   - embedding: vector
//   - salience: float (0-1, importance score)
//   - first_seen: datetime
//   - last_accessed: datetime
//   - access_count: integer
//   - properties: map (flexible key-value)

// Relationship types between entities:
//   - RELATED_TO {type, weight, source}
//   - PART_OF
//   - LOCATED_IN
//   - WORKS_FOR
//   - KNOWS
//   - CREATED_BY
//   - REFERENCES
//   - CAUSED
//   - PRECEDES

// ============================================
// PROCEDURAL MEMORY SCHEMA
// ============================================

// Tools
// CREATE (tool:Tool {
//     name: 'web_search',
//     description: 'Search the web for information',
//     schema: $json_schema,
//     avg_latency_ms: 0,
//     success_rate: 1.0,
//     usage_count: 0
// })

// Strategies (learned patterns)
// CREATE (s:Strategy {
//     id: $strategy_id,
//     description: 'Use code interpreter for data analysis',
//     context_pattern: 'data analysis|csv|spreadsheet',
//     embedding: $embedding,
//     success_count: 0,
//     failure_count: 0,
//     last_used: datetime()
// })

// Link strategy to task types
// MERGE (s)-[:EFFECTIVE_FOR]->(:TaskType {name: 'data_analysis'})
// MERGE (s)-[:USES_TOOL]->(tool:Tool {name: 'code_interpreter'})

// ============================================
// GOAL TRACKING SCHEMA
// ============================================

// Goals
// CREATE (g:Goal {
//     id: $goal_id,
//     description: $description,
//     status: 'active',  // 'active', 'completed', 'abandoned', 'blocked'
//     priority: $priority,  // 1-5
//     created_at: datetime(),
//     deadline: datetime($deadline),
//     embedding: $embedding
// })

// Goal hierarchy
// MATCH (parent:Goal {id: $parent_id}), (child:Goal {id: $child_id})
// MERGE (child)-[:SUBGOAL_OF]->(parent)

// Goal dependencies
// MATCH (g1:Goal {id: $goal_id}), (g2:Goal {id: $blocker_id})
// MERGE (g1)-[:BLOCKED_BY]->(g2)

// ============================================
// USER MODEL SCHEMA
// ============================================

// User preferences and profile
// MERGE (u:User {id: $user_id})
// SET u.name = $name,
//     u.created_at = coalesce(u.created_at, datetime()),
//     u.last_active = datetime()

// MERGE (u)-[:HAS_PREFERENCE]->(:Preference {
//     domain: 'communication',
//     key: 'verbosity',
//     value: 'concise'
// })

// MERGE (u)-[:HAS_EXPERTISE {level: 'expert'}]->(:Topic {name: 'python'})
// MERGE (u)-[:INTERESTED_IN]->(:Topic {name: 'machine_learning'})

// ============================================
// REFLECTION / LEARNING SCHEMA
// ============================================

// Reflections on outcomes
// CREATE (r:Reflection {
//     id: $reflection_id,
//     type: 'failure_analysis',  // or 'success_pattern', 'insight'
//     content: $critique_text,
//     embedding: $embedding,
//     created_at: datetime()
// })
// MATCH (c:Conversation {id: $conv_id})
// MERGE (c)-[:HAS_REFLECTION]->(r)
```

---

## Python Implementation

### Project Structure

```
agent_memory/
├── __init__.py
├── config.py
├── connections.py
├── models.py
├── memory/
│   ├── __init__.py
│   ├── interface.py      # Main API
│   ├── episodic.py       # Conversation memory
│   ├── semantic.py       # Facts and entities
│   ├── procedural.py     # Strategies and tools
│   ├── working.py        # Redis-based working memory
│   └── retrieval.py      # Multi-strategy retrieval
├── extraction/
│   ├── __init__.py
│   ├── entities.py       # NER and entity linking
│   ├── facts.py          # Fact extraction
│   └── relationships.py  # Relationship extraction
├── consolidation/
│   ├── __init__.py
│   ├── worker.py         # Background job runner
│   ├── jobs.py           # Consolidation jobs
│   └── decay.py          # Memory decay functions
├── embeddings.py
└── utils.py
```

### Configuration

```python
# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_secure_password"
    
    # PostgreSQL
    postgres_uri: str = "postgresql://agent:your_secure_password@localhost:5432/agent_memory"
    
    # Redis
    redis_uri: str = "redis://localhost:6379"
    
    # Embeddings
    embedding_provider: str = "openai"  # or "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_api_key: str = ""
    
    # Local embedding model (if using local)
    local_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    
    # Memory settings
    episodic_retention_days: int = 90
    fact_confidence_threshold: float = 0.7
    salience_decay_rate: float = 0.95  # Daily decay multiplier
    max_working_memory_items: int = 50
    
    # Retrieval settings
    default_top_k: int = 10
    reranking_enabled: bool = True
    
    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### Database Connections

```python
# connections.py
from contextlib import contextmanager
from typing import Generator
import redis
from neo4j import GraphDatabase, Driver, Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLSession

from .config import get_settings

settings = get_settings()


# Neo4j
class Neo4jConnection:
    _driver: Driver = None
    
    @classmethod
    def get_driver(cls) -> Driver:
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
        return cls._driver
    
    @classmethod
    @contextmanager
    def session(cls) -> Generator[Session, None, None]:
        driver = cls.get_driver()
        session = driver.session()
        try:
            yield session
        finally:
            session.close()
    
    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None


# PostgreSQL
engine = create_engine(settings.postgres_uri, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_postgres_session() -> Generator[SQLSession, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Redis
class RedisConnection:
    _client: redis.Redis = None
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        if cls._client is None:
            cls._client = redis.from_url(
                settings.redis_uri,
                decode_responses=True
            )
        return cls._client
    
    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
```

### Embeddings

```python
# embeddings.py
from typing import List, Union
import numpy as np
from functools import lru_cache

from .config import get_settings

settings = get_settings()


class EmbeddingProvider:
    """Unified embedding interface supporting multiple providers."""
    
    def __init__(self):
        self.provider = settings.embedding_provider
        self._client = None
        self._model = None
    
    def _init_openai(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)
    
    def _init_local(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            settings.local_embedding_model,
            trust_remote_code=True
        )
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for one or more texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.provider == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        if self._client is None:
            self._init_openai()
        
        response = self._client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            self._init_local()
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """Convenience method for single text."""
        return self.embed(text)[0]


@lru_cache
def get_embedder() -> EmbeddingProvider:
    return EmbeddingProvider()
```

### Data Models

```python
# models.py
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import uuid4


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    index: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    embedding: Optional[List[float]] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str  # 'Person', 'Organization', 'Concept', etc.
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    embedding: Optional[List[float]] = None
    salience: float = 0.5
    properties: Dict[str, Any] = Field(default_factory=dict)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0


class Fact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    claim: str
    confidence: float = 0.8
    source: str  # 'extraction', 'user_stated', 'inferred'
    source_turn_id: Optional[str] = None
    entity_ids: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    status: str = "active"  # 'active', 'completed', 'abandoned', 'blocked'
    priority: int = 3  # 1-5
    parent_goal_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None


class Strategy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    context_pattern: str  # Regex or keywords
    tool_sequence: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None


class MemoryBundle(BaseModel):
    """Aggregated retrieval result for context injection."""
    relevant_turns: List[Dict[str, Any]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    strategies: List[Dict[str, Any]] = Field(default_factory=list)
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    user_context: Dict[str, Any] = Field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """Format memory bundle as context for LLM prompt."""
        sections = []
        
        if self.relevant_turns:
            turns_text = "\n".join(
                f"[{t['timestamp']}] {t['role']}: {t['content'][:200]}..."
                for t in self.relevant_turns[:5]
            )
            sections.append(f"## Relevant Past Conversations\n{turns_text}")
        
        if self.facts:
            facts_text = "\n".join(
                f"- {f['claim']} (confidence: {f['confidence']:.0%})"
                for f in self.facts[:10]
            )
            sections.append(f"## Known Facts\n{facts_text}")
        
        if self.entities:
            entities_text = "\n".join(
                f"- {e['name']} ({e['type']}): {e.get('description', 'N/A')}"
                for e in self.entities[:10]
            )
            sections.append(f"## Relevant Entities\n{entities_text}")
        
        if self.active_goals:
            goals_text = "\n".join(
                f"- [{g['priority']}] {g['description']} ({g['status']})"
                for g in self.active_goals
            )
            sections.append(f"## Active Goals\n{goals_text}")
        
        return "\n\n".join(sections)
```

### Memory Interface (Main API)

```python
# memory/interface.py
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from ..models import Turn, Entity, Fact, Goal, Strategy, MemoryBundle
from ..embeddings import get_embedder
from ..connections import Neo4jConnection, get_postgres_session, RedisConnection
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .procedural import ProceduralMemory
from .working import WorkingMemory
from .retrieval import MemoryRetriever


class AgentMemory:
    """
    Unified interface for agent memory operations.
    
    Usage:
        memory = AgentMemory(user_id="user123")
        
        # Store a conversation turn
        memory.store_turn(turn)
        
        # Retrieve relevant context for a query
        context = memory.remember("What did we discuss about Python?")
        
        # Learn a new fact
        memory.learn_fact("User prefers concise responses", source="inferred")
    """
    
    def __init__(self, user_id: str, conversation_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.embedder = get_embedder()
        
        # Sub-modules
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.working = WorkingMemory(user_id, conversation_id)
        self.retriever = MemoryRetriever(self)
    
    # ==================== STORAGE ====================
    
    def store_turn(self, turn: Turn) -> None:
        """Store a conversation turn in episodic memory."""
        # Generate embedding if not provided
        if turn.embedding is None:
            turn.embedding = self.embedder.embed_single(turn.content)
        
        # Store in Neo4j (graph)
        self.episodic.store_turn(turn)
        
        # Store in PostgreSQL (logs)
        self.episodic.store_turn_log(turn)
        
        # Update working memory
        self.working.add_turn(turn)
    
    def learn_fact(
        self,
        claim: str,
        source: str = "extraction",
        confidence: float = 0.8,
        entity_ids: Optional[List[str]] = None,
        source_turn_id: Optional[str] = None
    ) -> Fact:
        """Add a fact to semantic memory."""
        fact = Fact(
            claim=claim,
            source=source,
            confidence=confidence,
            entity_ids=entity_ids or [],
            source_turn_id=source_turn_id,
            embedding=self.embedder.embed_single(claim)
        )
        self.semantic.store_fact(fact)
        return fact
    
    def upsert_entity(self, entity: Entity) -> Entity:
        """Add or update an entity in semantic memory."""
        if entity.embedding is None:
            text = f"{entity.name}: {entity.description or entity.type}"
            entity.embedding = self.embedder.embed_single(text)
        
        return self.semantic.upsert_entity(entity)
    
    def add_goal(self, goal: Goal) -> Goal:
        """Add a goal to track."""
        if goal.embedding is None:
            goal.embedding = self.embedder.embed_single(goal.description)
        
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (u:User {id: $user_id})
                CREATE (g:Goal {
                    id: $goal_id,
                    description: $description,
                    status: $status,
                    priority: $priority,
                    created_at: datetime(),
                    embedding: $embedding
                })
                MERGE (u)-[:HAS_GOAL]->(g)
            """,
                user_id=self.user_id,
                goal_id=goal.id,
                description=goal.description,
                status=goal.status,
                priority=goal.priority,
                embedding=goal.embedding
            )
        return goal
    
    def record_tool_usage(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int,
        turn_id: Optional[str] = None
    ) -> None:
        """Record tool invocation for procedural learning."""
        self.procedural.record_invocation(
            conversation_id=self.conversation_id,
            turn_id=turn_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            success=success,
            latency_ms=latency_ms
        )
    
    # ==================== RETRIEVAL ====================
    
    def remember(
        self,
        query: str,
        top_k: int = 10,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        time_window_hours: Optional[int] = None
    ) -> MemoryBundle:
        """
        Retrieve relevant memories for the given query.
        
        This is the main retrieval method that combines multiple
        strategies and returns a unified MemoryBundle.
        """
        return self.retriever.retrieve(
            query=query,
            user_id=self.user_id,
            top_k=top_k,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            include_procedural=include_procedural,
            time_window_hours=time_window_hours
        )
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals for the user."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_GOAL]->(g:Goal)
                WHERE g.status = 'active'
                OPTIONAL MATCH (g)-[:SUBGOAL_OF]->(parent:Goal)
                RETURN g, parent
                ORDER BY g.priority DESC
            """, user_id=self.user_id)
            
            goals = []
            for record in result:
                goal_data = dict(record["g"])
                if record["parent"]:
                    goal_data["parent"] = dict(record["parent"])
                goals.append(Goal(**goal_data))
            return goals
    
    def get_user_context(self) -> Dict[str, Any]:
        """Get user profile and preferences."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})
                OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(p:Preference)
                OPTIONAL MATCH (u)-[exp:HAS_EXPERTISE]->(t:Topic)
                OPTIONAL MATCH (u)-[:INTERESTED_IN]->(i:Topic)
                RETURN u,
                       collect(DISTINCT p) AS preferences,
                       collect(DISTINCT {topic: t.name, level: exp.level}) AS expertise,
                       collect(DISTINCT i.name) AS interests
            """, user_id=self.user_id)
            
            record = result.single()
            if not record:
                return {}
            
            return {
                "user": dict(record["u"]) if record["u"] else {},
                "preferences": [dict(p) for p in record["preferences"]],
                "expertise": record["expertise"],
                "interests": record["interests"]
            }
    
    def what_worked_for(self, task_description: str, top_k: int = 5) -> List[Strategy]:
        """Find successful strategies for similar tasks."""
        return self.procedural.find_strategies(task_description, top_k)
    
    # ==================== WORKING MEMORY ====================
    
    def get_working_context(self) -> Dict[str, Any]:
        """Get current working memory state."""
        return self.working.get_context()
    
    def set_working_context(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set a value in working memory."""
        self.working.set(key, value, ttl_seconds)
    
    # ==================== LIFECYCLE ====================
    
    def reflect(self, outcome: Dict[str, Any]) -> None:
        """
        Trigger reflection on conversation outcome.
        Called at end of conversation or after task completion.
        """
        from ..consolidation.jobs import trigger_reflection
        trigger_reflection(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            outcome=outcome
        )
    
    def close(self) -> None:
        """Clean up connections."""
        self.working.clear_session()
```

### Episodic Memory

```python
# memory/episodic.py
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from ..models import Turn
from ..connections import Neo4jConnection, get_postgres_session


class EpisodicMemory:
    """Handles storage and retrieval of conversation history."""
    
    def store_turn(self, turn: Turn) -> None:
        """Store turn in Neo4j graph."""
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (c:Conversation {id: $conv_id})
                ON CREATE SET c.started_at = datetime()
                
                CREATE (t:Turn {
                    id: $turn_id,
                    index: $index,
                    timestamp: datetime($timestamp),
                    role: $role,
                    content: $content,
                    embedding: $embedding,
                    token_count: $token_count
                })
                
                MERGE (c)-[:HAS_TURN]->(t)
                
                // Link to previous turn if exists
                WITH c, t
                OPTIONAL MATCH (c)-[:HAS_TURN]->(prev:Turn)
                WHERE prev.index = $index - 1
                FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (prev)-[:FOLLOWED_BY]->(t)
                )
            """,
                conv_id=turn.conversation_id,
                turn_id=turn.id,
                index=turn.index,
                timestamp=turn.timestamp.isoformat(),
                role=turn.role,
                content=turn.content,
                embedding=turn.embedding,
                token_count=turn.token_count
            )
    
    def store_turn_log(self, turn: Turn) -> None:
        """Store turn in PostgreSQL for audit/backup."""
        with get_postgres_session() as session:
            session.execute("""
                INSERT INTO conversation_logs 
                (conversation_id, turn_index, timestamp, role, content, token_count, embedding)
                VALUES (:conv_id, :index, :timestamp, :role, :content, :tokens, :embedding)
                ON CONFLICT (conversation_id, turn_index) DO UPDATE
                SET content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding
            """, {
                "conv_id": turn.conversation_id,
                "index": turn.index,
                "timestamp": turn.timestamp,
                "role": turn.role,
                "content": turn.content,
                "tokens": turn.token_count,
                "embedding": str(turn.embedding) if turn.embedding else None
            })
    
    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search episodic memory by vector similarity."""
        with Neo4jConnection.session() as session:
            # Build time filter
            time_filter = ""
            if time_window_hours:
                time_filter = f"AND t.timestamp > datetime() - duration('PT{time_window_hours}H')"
            
            user_filter = ""
            if user_id:
                user_filter = "AND c.user_id = $user_id"
            
            result = session.run(f"""
                CALL db.index.vector.queryNodes('turn_embeddings', $k, $embedding)
                YIELD node AS t, score
                MATCH (c:Conversation)-[:HAS_TURN]->(t)
                WHERE true {time_filter} {user_filter}
                RETURN t.id AS id,
                       t.content AS content,
                       t.role AS role,
                       t.timestamp AS timestamp,
                       c.id AS conversation_id,
                       score
                ORDER BY score DESC
            """,
                k=top_k * 2,  # Over-fetch for filtering
                embedding=query_embedding,
                user_id=user_id
            )
            
            return [dict(record) for record in result][:top_k]
    
    def get_conversation(self, conversation_id: str) -> List[Turn]:
        """Get all turns in a conversation."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (c:Conversation {id: $conv_id})-[:HAS_TURN]->(t:Turn)
                RETURN t
                ORDER BY t.index
            """, conv_id=conversation_id)
            
            return [Turn(**dict(record["t"])) for record in result]
    
    def get_recent_turns(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent turns across all conversations."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
                WHERE c.user_id = $user_id
                  AND t.timestamp > datetime() - duration('PT' + $hours + 'H')
                RETURN t.content AS content,
                       t.role AS role,
                       t.timestamp AS timestamp,
                       c.id AS conversation_id
                ORDER BY t.timestamp DESC
                LIMIT $limit
            """,
                user_id=user_id,
                hours=str(hours),
                limit=limit
            )
            
            return [dict(record) for record in result]
```

### Semantic Memory

```python
# memory/semantic.py
from typing import List, Dict, Any, Optional

from ..models import Entity, Fact
from ..connections import Neo4jConnection


class SemanticMemory:
    """Handles entities, facts, and conceptual knowledge."""
    
    def upsert_entity(self, entity: Entity) -> Entity:
        """Create or update an entity."""
        with Neo4jConnection.session() as session:
            # Use MERGE for idempotent upsert
            session.run("""
                MERGE (e:Entity {id: $id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.aliases = $aliases,
                    e.description = $description,
                    e.embedding = $embedding,
                    e.salience = $salience,
                    e.first_seen = datetime(),
                    e.last_accessed = datetime(),
                    e.access_count = 1,
                    e.properties = $properties
                ON MATCH SET
                    e.name = $name,
                    e.description = coalesce($description, e.description),
                    e.embedding = coalesce($embedding, e.embedding),
                    e.aliases = $aliases + [x IN e.aliases WHERE NOT x IN $aliases],
                    e.last_accessed = datetime(),
                    e.access_count = e.access_count + 1
                
                // Add type-specific label
                WITH e
                CALL apoc.create.addLabels(e, [$type]) YIELD node
                RETURN node
            """,
                id=entity.id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                description=entity.description,
                embedding=entity.embedding,
                salience=entity.salience,
                properties=entity.properties
            )
        return entity
    
    def store_fact(self, fact: Fact) -> None:
        """Store a fact and link to entities."""
        with Neo4jConnection.session() as session:
            session.run("""
                CREATE (f:Fact {
                    id: $id,
                    claim: $claim,
                    confidence: $confidence,
                    source: $source,
                    source_turn_id: $source_turn_id,
                    embedding: $embedding,
                    created_at: datetime()
                })
                
                // Link to source turn if provided
                WITH f
                OPTIONAL MATCH (t:Turn {id: $source_turn_id})
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:DERIVED_FROM]->(t)
                )
                
                // Link to entities
                WITH f
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {id: entity_id})
                MERGE (f)-[:ABOUT]->(e)
            """,
                id=fact.id,
                claim=fact.claim,
                confidence=fact.confidence,
                source=fact.source,
                source_turn_id=fact.source_turn_id,
                embedding=fact.embedding,
                entity_ids=fact.entity_ids
            )
    
    def vector_search_facts(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search facts by vector similarity."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('fact_embeddings', $k, $embedding)
                YIELD node AS f, score
                WHERE f.confidence >= $min_confidence
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                RETURN f.id AS id,
                       f.claim AS claim,
                       f.confidence AS confidence,
                       f.source AS source,
                       collect(e.name) AS entities,
                       score
                ORDER BY score DESC
            """,
                k=top_k,
                embedding=query_embedding,
                min_confidence=min_confidence
            )
            
            return [dict(record) for record in result]
    
    def vector_search_entities(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search entities by vector similarity."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
                YIELD node AS e, score
                
                // Update access stats
                SET e.last_accessed = datetime(),
                    e.access_count = e.access_count + 1
                
                RETURN e.id AS id,
                       e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       e.salience AS salience,
                       score
                ORDER BY score DESC
            """,
                k=top_k,
                embedding=query_embedding
            )
            
            return [dict(record) for record in result]
    
    def get_entity_graph(
        self,
        entity_ids: List[str],
        depth: int = 2
    ) -> Dict[str, Any]:
        """Traverse entity relationships to get context."""
        with Neo4jConnection.session() as session:
            result = session.run("""
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {id: eid})
                
                // Get related entities within depth
                OPTIONAL MATCH path = (e)-[r*1..$depth]-(related:Entity)
                
                // Get facts about these entities
                OPTIONAL MATCH (f:Fact)-[:ABOUT]->(e)
                
                RETURN e AS entity,
                       collect(DISTINCT {
                           entity: related,
                           relationship: type(last(relationships(path))),
                           path_length: length(path)
                       }) AS related,
                       collect(DISTINCT f) AS facts
            """,
                entity_ids=entity_ids,
                depth=depth
            )
            
            entities = []
            all_related = []
            all_facts = []
            
            for record in result:
                entities.append(dict(record["entity"]))
                all_related.extend(record["related"])
                all_facts.extend([dict(f) for f in record["facts"]])
            
            return {
                "entities": entities,
                "related": all_related,
                "facts": all_facts
            }
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a relationship between entities."""
        with Neo4jConnection.session() as session:
            props = properties or {}
            session.run(f"""
                MATCH (s:Entity {{id: $source_id}})
                MATCH (t:Entity {{id: $target_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $properties
            """,
                source_id=source_id,
                target_id=target_id,
                properties=props
            )
```

### Procedural Memory

```python
# memory/procedural.py
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from ..models import Strategy
from ..connections import Neo4jConnection, get_postgres_session
from ..embeddings import get_embedder


class ProceduralMemory:
    """Handles tool usage patterns and successful strategies."""
    
    def __init__(self):
        self.embedder = get_embedder()
    
    def record_invocation(
        self,
        conversation_id: str,
        turn_id: Optional[str],
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int
    ) -> None:
        """Record a tool invocation."""
        # PostgreSQL for audit log
        with get_postgres_session() as session:
            session.execute("""
                INSERT INTO tool_invocations
                (conversation_id, turn_index, tool_name, tool_input, tool_output, success, latency_ms)
                VALUES (:conv_id, :turn_idx, :tool, :input, :output, :success, :latency)
            """, {
                "conv_id": conversation_id,
                "turn_idx": 0,  # Would need proper turn index
                "tool": tool_name,
                "input": json.dumps(tool_input),
                "output": json.dumps(tool_output) if tool_output else None,
                "success": success,
                "latency": latency_ms
            })
        
        # Neo4j for graph relationships
        with Neo4jConnection.session() as session:
            session.run("""
                MERGE (tool:Tool {name: $tool_name})
                ON CREATE SET tool.usage_count = 0, tool.success_count = 0
                SET tool.usage_count = tool.usage_count + 1,
                    tool.success_count = tool.success_count + CASE WHEN $success THEN 1 ELSE 0 END,
                    tool.avg_latency_ms = coalesce(
                        (tool.avg_latency_ms * (tool.usage_count - 1) + $latency) / tool.usage_count,
                        $latency
                    )
                
                WITH tool
                MATCH (c:Conversation {id: $conv_id})
                CREATE (inv:ToolInvocation {
                    timestamp: datetime(),
                    tool_name: $tool_name,
                    success: $success,
                    latency_ms: $latency
                })
                MERGE (c)-[:USED_TOOL]->(inv)
                MERGE (inv)-[:INVOKED]->(tool)
            """,
                conv_id=conversation_id,
                tool_name=tool_name,
                success=success,
                latency=latency_ms
            )
    
    def learn_strategy(
        self,
        description: str,
        context_pattern: str,
        tool_sequence: List[str],
        from_conversation_id: Optional[str] = None,
        success: bool = True
    ) -> Strategy:
        """Record a successful (or failed) strategy pattern."""
        embedding = self.embedder.embed_single(description)
        
        strategy = Strategy(
            description=description,
            context_pattern=context_pattern,
            tool_sequence=tool_sequence,
            embedding=embedding,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            last_used=datetime.utcnow()
        )
        
        with Neo4jConnection.session() as session:
            session.run("""
                CREATE (s:Strategy {
                    id: $id,
                    description: $description,
                    context_pattern: $context_pattern,
                    tool_sequence: $tool_sequence,
                    embedding: $embedding,
                    success_count: $success_count,
                    failure_count: $failure_count,
                    last_used: datetime()
                })
                
                // Link to tools
                WITH s
                UNWIND $tool_sequence AS tool_name
                MATCH (t:Tool {name: tool_name})
                MERGE (s)-[:USES_TOOL]->(t)
                
                // Link to conversation if provided
                WITH s
                OPTIONAL MATCH (c:Conversation {id: $conv_id})
                FOREACH (_ IN CASE WHEN c IS NOT NULL AND $success THEN [1] ELSE [] END |
                    MERGE (s)-[:SUCCEEDED_IN]->(c)
                )
                FOREACH (_ IN CASE WHEN c IS NOT NULL AND NOT $success THEN [1] ELSE [] END |
                    MERGE (s)-[:FAILED_IN]->(c)
                )
            """,
                id=strategy.id,
                description=strategy.description,
                context_pattern=strategy.context_pattern,
                tool_sequence=strategy.tool_sequence,
                embedding=strategy.embedding,
                success_count=strategy.success_count,
                failure_count=strategy.failure_count,
                conv_id=from_conversation_id,
                success=success
            )
        
        return strategy
    
    def find_strategies(
        self,
        task_description: str,
        top_k: int = 5
    ) -> List[Strategy]:
        """Find strategies that worked for similar tasks."""
        embedding = self.embedder.embed_single(task_description)
        
        with Neo4jConnection.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('strategy_embeddings', $k, $embedding)
                YIELD node AS s, score
                WHERE s.success_count > 0
                RETURN s.id AS id,
                       s.description AS description,
                       s.context_pattern AS context_pattern,
                       s.tool_sequence AS tool_sequence,
                       s.success_count AS success_count,
                       s.failure_count AS failure_count,
                       s.success_count * 1.0 / (s.success_count + s.failure_count) AS success_rate,
                       score
                ORDER BY success_rate * score DESC
            """,
                k=top_k * 2,
                embedding=embedding
            )
            
            strategies = []
            for record in result:
                strategies.append(Strategy(
                    id=record["id"],
                    description=record["description"],
                    context_pattern=record["context_pattern"],
                    tool_sequence=record["tool_sequence"],
                    success_count=record["success_count"],
                    failure_count=record["failure_count"]
                ))
            
            return strategies[:top_k]
    
    def reinforce_strategy(self, strategy_id: str, success: bool) -> None:
        """Update strategy success/failure counts."""
        with Neo4jConnection.session() as session:
            if success:
                session.run("""
                    MATCH (s:Strategy {id: $id})
                    SET s.success_count = s.success_count + 1,
                        s.last_used = datetime()
                """, id=strategy_id)
            else:
                session.run("""
                    MATCH (s:Strategy {id: $id})
                    SET s.failure_count = s.failure_count + 1,
                        s.last_used = datetime()
                """, id=strategy_id)
    
    def get_tool_stats(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tool usage statistics, optionally filtered by task type."""
        with Neo4jConnection.session() as session:
            if task_type:
                result = session.run("""
                    MATCH (s:Strategy)-[:USES_TOOL]->(t:Tool)
                    WHERE s.context_pattern CONTAINS $task_type
                    WITH t, 
                         sum(s.success_count) AS successes,
                         sum(s.failure_count) AS failures
                    RETURN t.name AS tool,
                           successes,
                           failures,
                           successes * 1.0 / (successes + failures + 0.1) AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY success_rate DESC
                """, task_type=task_type)
            else:
                result = session.run("""
                    MATCH (t:Tool)
                    RETURN t.name AS tool,
                           t.usage_count AS usage_count,
                           t.success_count AS success_count,
                           t.success_count * 1.0 / (t.usage_count + 0.1) AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY usage_count DESC
                """)
            
            return [dict(record) for record in result]
```

### Working Memory (Redis)

```python
# memory/working.py
from typing import Any, Dict, List, Optional
import json
from datetime import datetime

from ..connections import RedisConnection
from ..config import get_settings

settings = get_settings()


class WorkingMemory:
    """
    Fast, ephemeral memory for current session state.
    Uses Redis for sub-millisecond access.
    """
    
    def __init__(self, user_id: str, conversation_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.redis = RedisConnection.get_client()
        
        # Key prefixes
        self.session_key = f"working:{user_id}:{conversation_id or 'global'}"
        self.turns_key = f"{self.session_key}:turns"
        self.context_key = f"{self.session_key}:context"
    
    def add_turn(self, turn: "Turn") -> None:
        """Add a turn to working memory (keeps last N turns)."""
        turn_data = {
            "id": turn.id,
            "index": turn.index,
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat()
        }
        
        # Push to list, trim to max size
        self.redis.lpush(self.turns_key, json.dumps(turn_data))
        self.redis.ltrim(self.turns_key, 0, settings.max_working_memory_items - 1)
        
        # Set TTL (1 hour default)
        self.redis.expire(self.turns_key, 3600)
    
    def get_recent_turns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent turns from working memory."""
        turns = self.redis.lrange(self.turns_key, 0, limit - 1)
        return [json.loads(t) for t in turns]
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        """Set a value in working memory."""
        full_key = f"{self.context_key}:{key}"
        self.redis.setex(full_key, ttl_seconds, json.dumps(value))
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from working memory."""
        full_key = f"{self.context_key}:{key}"
        value = self.redis.get(full_key)
        return json.loads(value) if value else None
    
    def delete(self, key: str) -> None:
        """Delete a value from working memory."""
        full_key = f"{self.context_key}:{key}"
        self.redis.delete(full_key)
    
    def get_context(self) -> Dict[str, Any]:
        """Get all working memory context."""
        # Get all keys matching the context pattern
        pattern = f"{self.context_key}:*"
        keys = self.redis.keys(pattern)
        
        context = {}
        for key in keys:
            short_key = key.replace(f"{self.context_key}:", "")
            value = self.redis.get(key)
            if value:
                context[short_key] = json.loads(value)
        
        # Include recent turns
        context["recent_turns"] = self.get_recent_turns()
        
        return context
    
    def clear_session(self) -> None:
        """Clear all working memory for this session."""
        pattern = f"{self.session_key}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
    
    # Specialized working memory operations
    
    def set_active_goal(self, goal_id: str, goal_description: str) -> None:
        """Set the currently active goal."""
        self.set("active_goal", {
            "id": goal_id,
            "description": goal_description,
            "set_at": datetime.utcnow().isoformat()
        })
    
    def get_active_goal(self) -> Optional[Dict[str, Any]]:
        """Get the currently active goal."""
        return self.get("active_goal")
    
    def push_thought(self, thought: str) -> None:
        """Push a reasoning step (for chain-of-thought tracking)."""
        thoughts_key = f"{self.session_key}:thoughts"
        self.redis.lpush(thoughts_key, json.dumps({
            "thought": thought,
            "timestamp": datetime.utcnow().isoformat()
        }))
        self.redis.ltrim(thoughts_key, 0, 99)  # Keep last 100 thoughts
        self.redis.expire(thoughts_key, 3600)
    
    def get_thoughts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reasoning steps."""
        thoughts_key = f"{self.session_key}:thoughts"
        thoughts = self.redis.lrange(thoughts_key, 0, limit - 1)
        return [json.loads(t) for t in thoughts]
```

### Memory Retrieval (Multi-Strategy)

```python
# memory/retrieval.py
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from ..models import MemoryBundle
from ..embeddings import get_embedder
from ..config import get_settings

if TYPE_CHECKING:
    from .interface import AgentMemory

settings = get_settings()


@dataclass
class RetrievalWeights:
    """Weights for combining different retrieval strategies."""
    episodic: float = 0.3
    semantic_facts: float = 0.25
    semantic_entities: float = 0.2
    procedural: float = 0.15
    recency: float = 0.1


class MemoryRetriever:
    """
    Multi-strategy retrieval engine.
    Combines vector search, graph traversal, and temporal heuristics.
    """
    
    def __init__(self, memory: "AgentMemory"):
        self.memory = memory
        self.embedder = get_embedder()
        self.weights = RetrievalWeights()
    
    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        include_episodic: bool = True,
        include_semantic: bool = True,
        include_procedural: bool = True,
        time_window_hours: Optional[int] = None
    ) -> MemoryBundle:
        """
        Main retrieval method combining multiple strategies.
        """
        query_embedding = self.embedder.embed_single(query)
        
        bundle = MemoryBundle()
        
        # Strategy 1: Episodic memory (past conversations)
        if include_episodic:
            bundle.relevant_turns = self._retrieve_episodic(
                query_embedding, user_id, top_k, time_window_hours
            )
        
        # Strategy 2: Semantic memory (facts and entities)
        if include_semantic:
            facts = self.memory.semantic.vector_search_facts(
                query_embedding, top_k=top_k
            )
            bundle.facts = facts
            
            entities = self.memory.semantic.vector_search_entities(
                query_embedding, top_k=top_k
            )
            bundle.entities = entities
            
            # Graph expansion: get related entities and their facts
            if entities:
                entity_ids = [e["id"] for e in entities[:5]]
                graph_context = self.memory.semantic.get_entity_graph(
                    entity_ids, depth=2
                )
                
                # Merge additional entities and facts
                existing_entity_ids = {e["id"] for e in bundle.entities}
                for related in graph_context.get("related", []):
                    if related.get("entity") and related["entity"].get("id") not in existing_entity_ids:
                        bundle.entities.append(related["entity"])
                
                existing_fact_ids = {f["id"] for f in bundle.facts}
                for fact in graph_context.get("facts", []):
                    if fact.get("id") not in existing_fact_ids:
                        bundle.facts.append(fact)
        
        # Strategy 3: Procedural memory (what worked before)
        if include_procedural:
            strategies = self.memory.procedural.find_strategies(query, top_k=5)
            bundle.strategies = [
                {
                    "description": s.description,
                    "tool_sequence": s.tool_sequence,
                    "success_rate": s.success_count / max(1, s.success_count + s.failure_count)
                }
                for s in strategies
            ]
        
        # Strategy 4: Active goals
        goals = self.memory.get_active_goals()
        bundle.active_goals = [
            {
                "id": g.id,
                "description": g.description,
                "priority": g.priority,
                "status": g.status
            }
            for g in goals
        ]
        
        # Strategy 5: User context
        bundle.user_context = self.memory.get_user_context()
        
        # Rerank if enabled
        if settings.reranking_enabled:
            bundle = self._rerank(bundle, query_embedding)
        
        return bundle
    
    def _retrieve_episodic(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int,
        time_window_hours: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Retrieve from episodic memory with recency boost."""
        # Vector search
        turns = self.memory.episodic.vector_search(
            query_embedding,
            top_k=top_k * 2,
            user_id=user_id,
            time_window_hours=time_window_hours
        )
        
        # Get recent turns regardless of similarity
        recent = self.memory.episodic.get_recent_turns(
            user_id,
            hours=time_window_hours or 24,
            limit=top_k
        )
        
        # Combine and deduplicate
        seen_ids = set()
        combined = []
        
        for turn in turns + recent:
            turn_id = turn.get("id") or f"{turn.get('conversation_id')}:{turn.get('timestamp')}"
            if turn_id not in seen_ids:
                seen_ids.add(turn_id)
                combined.append(turn)
        
        # Sort by combined score (similarity + recency)
        # This is a simplified scoring; production would be more sophisticated
        return combined[:top_k]
    
    def _rerank(
        self,
        bundle: MemoryBundle,
        query_embedding: List[float]
    ) -> MemoryBundle:
        """
        Rerank retrieved items using cross-encoder or other scoring.
        This is a simplified version; production would use a cross-encoder model.
        """
        # For now, just ensure diversity by limiting items from same conversation
        seen_convs = {}
        filtered_turns = []
        
        for turn in bundle.relevant_turns:
            conv_id = turn.get("conversation_id")
            if conv_id:
                seen_convs[conv_id] = seen_convs.get(conv_id, 0) + 1
                if seen_convs[conv_id] <= 3:  # Max 3 turns per conversation
                    filtered_turns.append(turn)
            else:
                filtered_turns.append(turn)
        
        bundle.relevant_turns = filtered_turns
        
        return bundle
```

---

## Background Jobs

### Consolidation Worker

```python
# consolidation/worker.py
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Optional

from ..connections import RedisConnection
from .jobs import (
    consolidate_episodic_to_semantic,
    detect_patterns,
    apply_memory_decay,
    cleanup_old_memories
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsolidationWorker:
    """
    Background worker that runs memory consolidation jobs.
    """
    
    def __init__(self):
        self.redis = RedisConnection.get_client()
        self.running = True
        self.jobs = {
            "consolidate": {
                "func": consolidate_episodic_to_semantic,
                "interval_minutes": 15
            },
            "patterns": {
                "func": detect_patterns,
                "interval_minutes": 60
            },
            "decay": {
                "func": apply_memory_decay,
                "interval_minutes": 60 * 24  # Daily
            },
            "cleanup": {
                "func": cleanup_old_memories,
                "interval_minutes": 60 * 24  # Daily
            }
        }
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self.running = False
    
    def _should_run_job(self, job_name: str, interval_minutes: int) -> bool:
        """Check if job should run based on last run time."""
        last_run_key = f"consolidation:last_run:{job_name}"
        last_run = self.redis.get(last_run_key)
        
        if not last_run:
            return True
        
        last_run_time = datetime.fromisoformat(last_run)
        return datetime.utcnow() - last_run_time > timedelta(minutes=interval_minutes)
    
    def _mark_job_run(self, job_name: str):
        """Mark job as run."""
        last_run_key = f"consolidation:last_run:{job_name}"
        self.redis.set(last_run_key, datetime.utcnow().isoformat())
    
    def run(self):
        """Main worker loop."""
        logger.info("Consolidation worker started")
        
        while self.running:
            for job_name, job_config in self.jobs.items():
                if not self.running:
                    break
                
                if self._should_run_job(job_name, job_config["interval_minutes"]):
                    logger.info(f"Running job: {job_name}")
                    try:
                        job_config["func"]()
                        self._mark_job_run(job_name)
                        logger.info(f"Job completed: {job_name}")
                    except Exception as e:
                        logger.error(f"Job failed: {job_name} - {e}")
            
            # Sleep between checks
            time.sleep(60)
        
        logger.info("Consolidation worker stopped")


if __name__ == "__main__":
    worker = ConsolidationWorker()
    worker.run()
```

### Consolidation Jobs

```python
# consolidation/jobs.py
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from ..connections import Neo4jConnection, get_postgres_session, RedisConnection
from ..embeddings import get_embedder
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def consolidate_episodic_to_semantic():
    """
    Extract entities and facts from recent episodic memory
    and store in semantic memory.
    """
    embedder = get_embedder()
    
    with Neo4jConnection.session() as session:
        # Get recent conversations not yet processed
        result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE NOT EXISTS(c.consolidated)
              OR c.consolidated < datetime() - duration('PT15M')
            WITH c, collect(t) AS turns
            ORDER BY c.started_at DESC
            LIMIT 10
            RETURN c.id AS conversation_id,
                   [t IN turns | {role: t.role, content: t.content}] AS turns
        """)
        
        for record in result:
            conv_id = record["conversation_id"]
            turns = record["turns"]
            
            # Combine turn content for extraction
            full_text = "\n".join(
                f"{t['role']}: {t['content']}" for t in turns
            )
            
            # Entity extraction (simplified - use NER in production)
            entities = extract_entities_from_text(full_text)
            
            # Store entities
            for entity in entities:
                entity_embedding = embedder.embed_single(
                    f"{entity['name']}: {entity.get('description', entity['type'])}"
                )
                
                session.run("""
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET
                        e.id = randomUUID(),
                        e.type = $type,
                        e.embedding = $embedding,
                        e.first_seen = datetime(),
                        e.salience = 0.5
                    ON MATCH SET
                        e.last_accessed = datetime(),
                        e.access_count = coalesce(e.access_count, 0) + 1
                    
                    WITH e
                    MATCH (c:Conversation {id: $conv_id})
                    MERGE (c)-[:MENTIONS]->(e)
                """,
                    name=entity["name"],
                    type=entity["type"],
                    embedding=entity_embedding,
                    conv_id=conv_id
                )
            
            # Fact extraction (simplified)
            facts = extract_facts_from_text(full_text)
            
            for fact in facts:
                fact_embedding = embedder.embed_single(fact["claim"])
                
                session.run("""
                    CREATE (f:Fact {
                        id: randomUUID(),
                        claim: $claim,
                        confidence: $confidence,
                        source: 'extraction',
                        embedding: $embedding,
                        created_at: datetime()
                    })
                    
                    WITH f
                    MATCH (c:Conversation {id: $conv_id})
                    CREATE (f)-[:DERIVED_FROM]->(c)
                """,
                    claim=fact["claim"],
                    confidence=fact.get("confidence", 0.7),
                    embedding=fact_embedding,
                    conv_id=conv_id
                )
            
            # Mark conversation as consolidated
            session.run("""
                MATCH (c:Conversation {id: $conv_id})
                SET c.consolidated = datetime()
            """, conv_id=conv_id)
            
            logger.info(f"Consolidated conversation {conv_id}: {len(entities)} entities, {len(facts)} facts")


def detect_patterns():
    """
    Analyze successful conversations to extract procedural patterns.
    """
    with Neo4jConnection.session() as session:
        # Find conversations with successful outcomes
        result = session.run("""
            MATCH (c:Conversation)-[:RESULTED_IN]->(o:Outcome {success: true})
            WHERE NOT EXISTS(c.patterns_extracted)
            WITH c, o
            MATCH (c)-[:USED_TOOL]->(inv:ToolInvocation)-[:INVOKED]->(t:Tool)
            WITH c, o, collect(DISTINCT t.name) AS tools
            WHERE size(tools) > 0
            RETURN c.id AS conversation_id,
                   tools,
                   o.task_type AS task_type
            LIMIT 20
        """)
        
        embedder = get_embedder()
        
        for record in result:
            conv_id = record["conversation_id"]
            tools = record["tools"]
            task_type = record.get("task_type", "general")
            
            # Create or update strategy
            strategy_desc = f"Use {', '.join(tools)} for {task_type} tasks"
            embedding = embedder.embed_single(strategy_desc)
            
            session.run("""
                MERGE (s:Strategy {context_pattern: $task_type, tool_sequence: $tools})
                ON CREATE SET
                    s.id = randomUUID(),
                    s.description = $description,
                    s.embedding = $embedding,
                    s.success_count = 1,
                    s.failure_count = 0,
                    s.created_at = datetime()
                ON MATCH SET
                    s.success_count = s.success_count + 1,
                    s.last_used = datetime()
                
                WITH s
                MATCH (c:Conversation {id: $conv_id})
                MERGE (s)-[:SUCCEEDED_IN]->(c)
                SET c.patterns_extracted = true
            """,
                task_type=task_type,
                tools=tools,
                description=strategy_desc,
                embedding=embedding,
                conv_id=conv_id
            )
            
            logger.info(f"Extracted pattern from {conv_id}: {tools}")


def apply_memory_decay():
    """
    Apply time-based decay to memory salience scores.
    """
    decay_rate = settings.salience_decay_rate
    
    with Neo4jConnection.session() as session:
        # Decay entity salience
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.last_accessed < datetime() - duration('P1D')
            SET e.salience = e.salience * $decay_rate
            RETURN count(e) AS decayed_count
        """, decay_rate=decay_rate)
        
        count = result.single()["decayed_count"]
        logger.info(f"Decayed {count} entities")
        
        # Decay fact confidence (slower decay)
        result = session.run("""
            MATCH (f:Fact)
            WHERE f.created_at < datetime() - duration('P7D')
              AND f.source = 'inferred'
            SET f.confidence = f.confidence * $decay_rate
            RETURN count(f) AS decayed_count
        """, decay_rate=decay_rate ** 0.5)  # Slower decay for facts
        
        count = result.single()["decayed_count"]
        logger.info(f"Decayed {count} facts")


def cleanup_old_memories():
    """
    Archive or delete old, low-salience memories.
    """
    retention_days = settings.episodic_retention_days
    
    with Neo4jConnection.session() as session:
        # Archive old episodic memories
        result = session.run("""
            MATCH (c:Conversation)-[:HAS_TURN]->(t:Turn)
            WHERE c.started_at < datetime() - duration('P' + $days + 'D')
              AND NOT EXISTS(c.archived)
            SET c.archived = true, t.archived = true
            RETURN count(DISTINCT c) AS archived_count
        """, days=str(retention_days))
        
        count = result.single()["archived_count"]
        logger.info(f"Archived {count} old conversations")
        
        # Delete very low salience entities
        result = session.run("""
            MATCH (e:Entity)
            WHERE e.salience < 0.1
              AND e.last_accessed < datetime() - duration('P30D')
              AND NOT EXISTS((e)<-[:ABOUT]-(:Fact))
            DETACH DELETE e
            RETURN count(e) AS deleted_count
        """)
        
        # Note: The count won't work correctly after DETACH DELETE
        # This is just for illustration
        logger.info("Cleaned up low-salience orphan entities")


def trigger_reflection(
    conversation_id: str,
    user_id: str,
    outcome: Dict[str, Any]
) -> None:
    """
    Trigger reflection job for a conversation.
    Adds to Redis queue for async processing.
    """
    redis = RedisConnection.get_client()
    
    import json
    job_data = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "outcome": outcome,
        "triggered_at": datetime.utcnow().isoformat()
    }
    
    redis.xadd("reflection_jobs", {"data": json.dumps(job_data)})
    logger.info(f"Queued reflection for conversation {conversation_id}")


# Helper functions for extraction (simplified - use proper NLP in production)

def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from text.
    In production, use spaCy, transformers, or LLM-based extraction.
    """
    # Simplified placeholder - implement with actual NER
    # Example using spaCy:
    # import spacy
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # return [{"name": ent.text, "type": ent.label_} for ent in doc.ents]
    
    return []  # Placeholder


def extract_facts_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract factual claims from text.
    In production, use LLM-based extraction.
    """
    # Simplified placeholder - implement with LLM
    return []  # Placeholder
```

---

## Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-memory
---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: agent-memory
type: Opaque
stringData:
  neo4j-password: "your_secure_password"
  postgres-password: "your_secure_password"
  openai-api-key: "sk-..."
---
# k8s/neo4j.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: agent-memory
spec:
  serviceName: neo4j
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.15-community
        ports:
        - containerPort: 7474
          name: http
        - containerPort: 7687
          name: bolt
        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: neo4j-password
        - name: NEO4J_PLUGINS
          value: '["apoc"]'
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j
  namespace: agent-memory
spec:
  ports:
  - port: 7474
    name: http
  - port: 7687
    name: bolt
  selector:
    app: neo4j
---
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: agent-memory
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg16
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          value: agent
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: postgres-password
        - name: POSTGRES_DB
          value: agent_memory
        resources:
          requests:
            memory: "1Gi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: agent-memory
spec:
  ports:
  - port: 5432
  selector:
    app: postgres
---
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: agent-memory
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--appendonly", "yes"]
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: agent-memory
spec:
  ports:
  - port: 6379
  selector:
    app: redis
---
# k8s/consolidation-worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: consolidation-worker
  namespace: agent-memory
spec:
  replicas: 1
  selector:
    matchLabels:
      app: consolidation-worker
  template:
    metadata:
      labels:
        app: consolidation-worker
    spec:
      containers:
      - name: worker
        image: your-registry/agent-memory-worker:latest
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j:7687"
        - name: NEO4J_USER
          value: "neo4j"
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: neo4j-password
        - name: POSTGRES_URI
          value: "postgresql://agent:$(POSTGRES_PASSWORD)@postgres:5432/agent_memory"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: postgres-password
        - name: REDIS_URI
          value: "redis://redis:6379"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

---

## Usage Patterns

### Basic Usage

```python
from agent_memory import AgentMemory

# Initialize memory for a user/conversation
memory = AgentMemory(
    user_id="user123",
    conversation_id="conv456"
)

# Store conversation turns
from agent_memory.models import Turn

turn = Turn(
    conversation_id="conv456",
    index=0,
    role="user",
    content="How do I optimize my Python code for performance?"
)
memory.store_turn(turn)

# Retrieve relevant context for a new query
context = memory.remember(
    "What about using Cython for speed?",
    top_k=10,
    time_window_hours=24
)

# Format for LLM prompt
prompt_context = context.to_context_string()
print(prompt_context)

# Learn facts
memory.learn_fact(
    claim="User is interested in Python performance optimization",
    source="inferred",
    confidence=0.8
)

# Track goals
from agent_memory.models import Goal

goal = Goal(
    description="Help user optimize their Python codebase",
    priority=4
)
memory.add_goal(goal)

# Record tool usage
memory.record_tool_usage(
    tool_name="code_interpreter",
    tool_input={"code": "import timeit..."},
    tool_output={"result": "..."},
    success=True,
    latency_ms=1200
)

# Get procedural recommendations
strategies = memory.what_worked_for("code optimization task")
for s in strategies:
    print(f"Strategy: {s.description}, Success rate: {s.success_count/(s.success_count+s.failure_count):.0%}")
```

### Integration with LLM Agent

```python
from agent_memory import AgentMemory
from openai import OpenAI

client = OpenAI()

class MemoryAugmentedAgent:
    def __init__(self, user_id: str):
        self.memory = AgentMemory(user_id=user_id)
        self.conversation_id = None
    
    def start_conversation(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.memory.conversation_id = conversation_id
    
    def chat(self, user_message: str) -> str:
        # Store user turn
        user_turn = Turn(
            conversation_id=self.conversation_id,
            index=self._get_turn_index(),
            role="user",
            content=user_message
        )
        self.memory.store_turn(user_turn)
        
        # Retrieve relevant context
        context = self.memory.remember(user_message)
        
        # Build augmented prompt
        system_prompt = self._build_system_prompt(context)
        
        # Get LLM response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        assistant_message = response.choices[0].message.content
        
        # Store assistant turn
        assistant_turn = Turn(
            conversation_id=self.conversation_id,
            index=self._get_turn_index(),
            role="assistant",
            content=assistant_message
        )
        self.memory.store_turn(assistant_turn)
        
        return assistant_message
    
    def _build_system_prompt(self, context: MemoryBundle) -> str:
        base_prompt = "You are a helpful AI assistant with memory of past interactions."
        
        memory_section = context.to_context_string()
        
        if memory_section:
            return f"{base_prompt}\n\n# Context from Memory\n{memory_section}"
        return base_prompt
    
    def _get_turn_index(self) -> int:
        # In practice, track this properly
        turns = self.memory.working.get_recent_turns(limit=100)
        return len(turns)
```

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "agent-memory"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "neo4j>=5.15",
    "psycopg2-binary>=2.9",
    "sqlalchemy>=2.0",
    "redis>=5.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "openai>=1.0",
    "numpy>=1.24",
    # Optional: for local embeddings
    # "sentence-transformers>=2.2",
    # Optional: for entity extraction
    # "spacy>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "ruff>=0.1",
]
local = [
    "sentence-transformers>=2.2",
]
nlp = [
    "spacy>=3.7",
]
```

---

## Quick Start

```bash
# 1. Start services
docker-compose up -d

# 2. Wait for services to be healthy
docker-compose ps

# 3. Initialize Neo4j schema (run once)
# Connect to http://localhost:7474 and run the Cypher schema commands

# 4. Install Python package
pip install -e .

# 5. Set environment variables
export OPENAI_API_KEY="sk-..."

# 6. Run example
python -c "
from agent_memory import AgentMemory
memory = AgentMemory(user_id='test')
print('Memory system initialized!')
"
```

---

## Notes

- **Scaling**: For production, consider Neo4j Enterprise for clustering, or use a managed service like Aura
- **Embeddings**: Start with OpenAI for simplicity; switch to local models (nomic-embed, e5) for cost/privacy
- **Entity Extraction**: The placeholder functions should be replaced with proper NER (spaCy) or LLM-based extraction
- **Monitoring**: Add Prometheus metrics and logging aggregation for observability
- **Backup**: Implement regular Neo4j and PostgreSQL backups

---

*Generated for personal use - feel free to adapt and extend*
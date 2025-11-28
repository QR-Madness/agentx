# Database Stack

AgentX uses a hybrid database architecture combining graph, relational, and in-memory databases.

## Overview

```mermaid
graph TB
    API[Django API]
    Neo4j[Neo4j<br/>Graph Database]
    Postgres[PostgreSQL<br/>+ pgvector]
    Redis[Redis<br/>Cache]

    API -->|Relationships| Neo4j
    API -->|Embeddings| Postgres
    API -->|Cache| Redis

    style Neo4j fill:#4581C3
    style Postgres fill:#336791
    style Redis fill:#DC382D
```

## Database Roles

### Neo4j - Graph Database

**Purpose**: Store and analyze relationships between entities

**Use Cases**:

- Entity relationship mapping
- Knowledge graph construction
- Semantic connections
- Graph-based reasoning

**Configuration**:

- Port: 7474 (browser), 7687 (bolt)
- Data: `./data/neo4j/data`
- Logs: `./data/neo4j/logs`
- Plugins: APOC

**Example Query**:

```cypher
// Create entity with relationships
CREATE (p:Person {name: 'Alice'})
CREATE (c:Concept {name: 'Machine Learning'})
CREATE (p)-[:INTERESTED_IN]->(c)
```

### PostgreSQL + pgvector

**Purpose**: Store vector embeddings for semantic search

**Use Cases**:

- Embedding storage
- Similarity search
- Conversation history
- Structured data

**Configuration**:

- Port: 5432
- Database: `agent_memory`
- User: `agent`
- Data: `./data/postgres`
- Extensions: pgvector

**Example Usage**:

```sql
-- Create embeddings table
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector index
CREATE INDEX ON embeddings
USING ivfflat (embedding vector_cosine_ops);

-- Find similar vectors
SELECT content,
       embedding <=> '[0.1, 0.2, ...]' AS distance
FROM embeddings
ORDER BY distance
LIMIT 10;
```

### Redis - Cache Layer

**Purpose**: High-speed caching and temporary data storage

**Use Cases**:

- Translation cache
- Session storage
- Rate limiting
- Real-time data

**Configuration**:

- Port: 6379
- Data: `./data/redis`
- Max Memory: 512MB
- Eviction: allkeys-lru

**Example Usage**:

```python
import redis

r = redis.Redis(host='localhost', port=6379)

# Cache translation
r.setex('translate:en:fr:hello', 3600, 'bonjour')

# Retrieve cached value
cached = r.get('translate:en:fr:hello')
```

## Data Flow

### Translation Request

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Redis
    participant Model
    participant Neo4j

    Client->>API: POST /translate
    API->>Redis: Check cache

    alt Cache Hit
        Redis-->>API: Return cached
    else Cache Miss
        API->>Model: Translate
        Model-->>API: Result
        API->>Redis: Store cache
        API->>Neo4j: Store relationship
    end

    API-->>Client: Translation
```

### Memory Storage

```mermaid
sequenceDiagram
    participant API
    participant Postgres
    participant Neo4j
    participant Redis

    API->>Postgres: Store embedding
    API->>Neo4j: Create relationships
    API->>Redis: Cache metadata

    Note over Postgres,Redis: Coordinated storage
```

## Data Directories

All database data is stored in local bind mounts:

```
data/
├── neo4j/
│   ├── data/          # Graph database files
│   ├── logs/          # Application logs
│   └── plugins/       # APOC and other plugins
├── postgres/          # PostgreSQL data files
└── redis/             # Redis persistence files
```

### Benefits

- **Easy Backups**: Simple directory copy
- **Portability**: Move between environments
- **Debugging**: Direct file access
- **No Orphaned Volumes**: Explicit data location

## Migration from Docker Volumes

If you have existing data in Docker volumes:

```bash
# Migrate all databases
task db:migrate-volumes

# Or individually
task db:migrate-volumes:neo4j
task db:migrate-volumes:postgres
task db:migrate-volumes:redis
```

This copies data from Docker volumes to `./data/` directories.

## Backup Strategies

### PostgreSQL Backup

Automated backup:
```bash
task db:backup:postgres
```

Manual backup:
```bash
docker exec agent-postgres pg_dump -U agent agent_memory > backup.sql
```

Restore:
```bash
task db:restore:postgres BACKUP_FILE=backup.sql
```

### Neo4j Backup

Using Neo4j dump:
```bash
docker exec agent-neo4j neo4j-admin database dump neo4j --to-path=/backups
```

### Redis Backup

Redis automatically persists with AOF:
```bash
# Copy the appendonly.aof file
cp data/redis/appendonlydir/appendonly.aof.1.base.rdb backups/
```

### Full Backup

Copy entire data directory:
```bash
tar -czf agentx-backup-$(date +%Y%m%d).tar.gz data/
```

## Database Maintenance

### PostgreSQL

Vacuum and analyze:
```bash
docker exec agent-postgres psql -U agent -d agent_memory -c "VACUUM ANALYZE;"
```

### Neo4j

Check database size:
```cypher
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store sizes")
YIELD attributes
RETURN attributes.TotalStoreSize.value AS size
```

### Redis

Get memory stats:
```bash
docker exec agent-redis redis-cli INFO memory
```

## Connection Strings

### Neo4j

```
bolt://localhost:7687
```

Python (neo4j driver):
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "your_secure_password")
)
```

### PostgreSQL

```
postgresql://agent:your_secure_password@localhost:5432/agent_memory
```

Python (psycopg2):
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="agent_memory",
    user="agent",
    password="your_secure_password"
)
```

### Redis

```
redis://localhost:6379
```

Python (redis-py):
```python
import redis

r = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)
```

## Performance Considerations

### Neo4j

- Default heap: 512MB initial, 2GB max
- Page cache: 1GB
- Increase for larger graphs

### PostgreSQL

- Shared buffers: 25% of RAM
- Work memory: 4MB per connection
- Maintenance work memory: 64MB

### Redis

- Max memory: 512MB (configurable)
- Eviction policy: allkeys-lru
- Persistence: AOF enabled

## Security

### Network Isolation

Databases are exposed on localhost only. For production:

```yaml
# docker-compose.yml
services:
  postgres:
    ports: []  # Remove port mapping
    networks:
      - internal  # Use internal network
```

### Authentication

Change default passwords in `docker-compose.yml`:

```yaml
environment:
  - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}  # Use env var
```

### Encryption

For production, enable:

- SSL/TLS for PostgreSQL
- Bolt encryption for Neo4j
- Redis AUTH password

## Monitoring

### Health Checks

All services have built-in health checks:

```bash
docker ps
```

Shows health status for each container.

### Database Shells

Access databases directly:

```bash
# PostgreSQL
task db:shell:postgres

# Redis
task db:shell:redis

# Neo4j Browser
open http://localhost:7474
```

## Troubleshooting

### Connection Refused

Check if services are running:
```bash
docker-compose ps
```

### Data Corruption

Restore from backup:
```bash
task db:restore:postgres BACKUP_FILE=backups/latest.sql
```

### Disk Space

Check data directory sizes:
```bash
du -sh data/*
```

### Performance Issues

Check resource usage:
```bash
docker stats
```

## Next Steps

- [Database Migration Guide](../deployment/migration.md) - Data migration strategies
- [API Reference](../api/endpoints.md) - Database interactions via API
- [Development Tasks](../development/tasks.md) - Database management commands

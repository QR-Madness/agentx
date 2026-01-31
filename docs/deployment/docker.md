# Docker Deployment

Deploying AgentX with Docker.

## Quick Start

```bash
# Start all database services
task runners

# Or use docker-compose directly
docker-compose up -d
```

## Services

The project includes three database services:

| Service | Port | Purpose |
|---------|------|---------|
| Neo4j | 7474 (HTTP), 7687 (Bolt) | Graph database for knowledge graphs |
| PostgreSQL | 5432 | Relational storage with pgvector |
| Redis | 6379 | Caching and working memory |

### Neo4j

Graph database for entity relationships and knowledge graphs.

```bash
# Access Neo4j Browser
open http://localhost:7474

# Default credentials (from .env)
# Username: neo4j
# Password: changeme
```

### PostgreSQL with pgvector

Relational database with vector similarity search.

```bash
# Connect via psql
docker exec -it agentx-postgres psql -U agent -d agentx_memory

# Default credentials (from .env)
# Username: agent
# Password: changeme
# Database: agentx_memory
```

### Redis

In-memory cache for working memory and session data.

```bash
# Connect via redis-cli
docker exec -it agentx-redis redis-cli
```

## Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f neo4j

# Restart a service
docker-compose restart postgres

# Remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Data Persistence

Data is stored in local bind mounts under `./data/`:

```
data/
├── neo4j/
│   ├── data/
│   └── logs/
├── postgres/
│   └── data/
└── redis/
    └── data/
```

### Initializing Data Directories

```bash
task db:init
```

### Migrating from Docker Volumes

If you previously used Docker volumes:

```bash
task db:migrate-volumes
```

## Environment Configuration

Configure services via `.env` file (copy from `.env.example`):

```bash
# Database passwords
NEO4J_PASSWORD=your_secure_password
POSTGRES_PASSWORD=your_secure_password

# Neo4j settings
NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
NEO4J_PLUGINS=["apoc"]

# PostgreSQL settings
POSTGRES_USER=agent
POSTGRES_DB=agentx_memory
```

## Production Considerations

### Security Checklist

- [ ] Change all default passwords in `.env`
- [ ] Use strong, unique passwords for each service
- [ ] Restrict network access (don't expose ports publicly)
- [ ] Enable SSL/TLS for database connections
- [ ] Set up firewall rules

### Backups

```bash
# Backup Neo4j
docker exec agentx-neo4j neo4j-admin database dump neo4j --to-path=/backups

# Backup PostgreSQL
docker exec agentx-postgres pg_dump -U agent agentx_memory > backup.sql

# Backup Redis
docker exec agentx-redis redis-cli BGSAVE
```

### Monitoring

Consider adding:

- Prometheus metrics exporters
- Grafana dashboards
- Log aggregation (ELK stack, Loki)
- Health check endpoints

### Resource Limits

Add resource constraints in `docker-compose.yml`:

```yaml
services:
  neo4j:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Troubleshooting

### Port Conflicts

If ports are already in use:

```bash
# Check what's using a port
lsof -i :7474

# Modify ports in docker-compose.yml
ports:
  - "17474:7474"  # Use different host port
```

### Connection Issues

```bash
# Verify containers are running
docker-compose ps

# Check container logs
docker-compose logs neo4j

# Test connectivity
nc -zv localhost 7687
```

### Data Directory Permissions

```bash
# Fix permissions if needed
sudo chown -R $USER:$USER ./data
```

## See Also

- [docker-compose.yml](https://github.com/QR-Madness/agentx/blob/main/docker-compose.yml)
- [Database Migration Guide](migration.md)
- [Memory Setup](../development/memory-setup.md)

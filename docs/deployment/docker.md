# Docker Deployment

Deploying AgentX with Docker.

## Docker Compose

The project uses Docker Compose for database services:

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

## Services

- Neo4j: Graph database
- PostgreSQL: Relational database with pgvector
- Redis: Cache layer

See [docker-compose.yml](../../docker-compose.yml) for configuration.

## Production Considerations

- Change default passwords
- Use environment variables
- Configure backups
- Set up monitoring
- Enable SSL/TLS

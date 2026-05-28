# Production Deployment

For local development the API runs on your host (via `task dev`) against Dockerized
databases. For production, AgentX also runs the **API itself in a container** using the
`production` Docker Compose profile. This page covers the single-host production stack; to run
multiple isolated instances or expose them publicly, see [Clusters](clusters.md).

## The production profile

The `api` service in `docker-compose.yml` is gated behind `profiles: [production]`, so it only
starts when you opt in with `--profile production`. The `prod:*` tasks wrap this:

| Task | Runs | Purpose |
|------|------|---------|
| `prod:build` | `docker compose --profile production build` | Build the API image |
| `prod:up` | `… up -d` | Start the full stack (API + Neo4j + PostgreSQL + Redis) |
| `prod:down` | `… down` | Stop the stack |
| `prod:restart` | `… restart api` | Restart just the API |
| `prod:logs` | `… logs -f api` | Tail API logs |
| `prod:status` | `… ps` | Container status |
| `prod:shell` | `… exec api /bin/bash` | Shell into the API container |
| `prod:auth:setup` | `… exec api … setup_auth` | Set the root password (see [Authentication](authentication.md)) |
| `prod:warmup` | `… exec api … warmup_embeddings --validate` | Pre-load the embedding model |

The API image is built from the repo `Dockerfile` (Python 3.12 + uv, Node for asset build) and
served by uvicorn (ASGI) on port `12319`. It restarts `unless-stopped` and has a healthcheck
with a 60s start period.

## Services & resource limits

| Service | Image | Memory limit / reservation |
|---------|-------|----------------------------|
| `api` | built from `Dockerfile` | 8G / 4G |
| `neo4j` | `neo4j:5.15-community` | 4G / 2G |
| `postgres` | `pgvector/pgvector:pg16` | 2G / 512M |
| `redis` | `redis:7-alpine` | 768M / 256M |

## Required configuration

Set these in your `.env` (or the cluster `.env`) before `prod:up`:

```bash
# Django — REQUIRED, no safe default
DJANGO_SECRET_KEY=<generate a long random key>
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=your.host.or.ip

# Database credentials — change the defaults
NEO4J_PASSWORD=<strong>
POSTGRES_USER=agent
POSTGRES_PASSWORD=<strong>
POSTGRES_DB=agent_memory

# Host ports (override to avoid conflicts)
API_PORT=12319
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687
POSTGRES_PORT=5432
REDIS_PORT=6379

# Where the API container reads config / persists DB data
AGENTX_CONFIG_DIR=./data
AGENTX_DB_DIR=./data
```

Plus any LLM provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) and CORS settings you
need. See [Configuration](../getting-started/configuration.md) for the full list.

!!! danger "Don't ship the defaults"
    `DJANGO_SECRET_KEY` must be set, `DJANGO_DEBUG` must be `false`, and every database
    password must be changed from `changeme` before exposing the stack.

## Bring-up sequence

```bash
task prod:build          # build the API image
task prod:up             # start API + databases
task prod:auth:setup     # set the root password (if AGENTX_AUTH_ENABLED=true)
task prod:warmup         # pre-load the embedding model (avoids a slow first request)
task prod:status         # confirm everything is healthy
```

Verify the API:

```bash
curl http://localhost:12319/api/health?include_memory=true
```

## Next steps

- [Authentication](authentication.md) — turn on login + the version-compatibility checks
- [Clusters](clusters.md) — run multiple instances, add the public gateway, enable GPU

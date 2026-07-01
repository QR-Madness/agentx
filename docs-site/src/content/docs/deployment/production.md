# Production Deployment

For local development the API runs on your host (via `task dev`) against Dockerized
databases. For production, AgentX runs the **API itself in a container** using the
`production` Docker Compose profile.

The deployment unit is a **cluster** — a self-contained instance with its own `.env`, config,
database storage, and ports. Even a single private instance is just a cluster without the public
gateway. This page covers the production profile and required configuration; for the full
workflow (scaffolding, the optional Nginx + Cloudflare gateway, GPU overlay, running multiple
instances) see [Clusters](clusters.md).

!!! tip "Just want to self-host? Use the published image."
    Clusters come in two flavours: **local** clusters build from the repo and are managed by the
    `task cluster:*` workflow below; **isolated** clusters pull the published Docker image, manage
    themselves via the in-image `agentx` CLI, and need no repo or Taskfile. If you're standing up
    your own instance, start at [Self-Hosting (Docker Hub)](self-hosting.md) — it's the shortest
    path. This page documents the local/repo-managed workflow.

## The production profile

The `api` service in `docker-compose.yml` is gated behind `profiles: [production]`, so it only
starts when you opt in with `--profile production`. You never invoke Compose directly — the
`cluster:*` tasks assemble the right `--env-file` and overlays for you (see the
[Clusters lifecycle table](clusters.md#lifecycle)).

The base `docker-compose.yml` pulls the published API image
(`${AGENTX_IMAGE:-qrmadness/agentx-api:latest}`). Local clusters layer
`docker-compose.build.yml` on top to **build from the repo `Dockerfile`** instead (Python +
uv, Node for local MCP tools); the `cluster:*` tasks add this overlay automatically. The API is
served by uvicorn (ASGI) on port `12319`, restarts `unless-stopped`, and self-initializes its
schemas on boot via the container entrypoint (so `cluster:migrate` is belt-and-suspenders, not
required).

## Services & resource limits

| Service | Image | Memory limit / reservation |
|---------|-------|----------------------------|
| `api` | built from `Dockerfile` | 8G / 4G |
| `neo4j` | `neo4j:5.15-community` | 4G / 2G |
| `postgres` | `pgvector/pgvector:pg16` | 2G / 512M |
| `redis` | `redis:7-alpine` | 768M / 256M |

## Required configuration

Set these in the cluster's `.env` (`clusters/<name>/.env`, scaffolded by `cluster:new`) before
`cluster:up`:

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

# Host ports (override to avoid conflicts between clusters)
API_PORT=12319
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687
POSTGRES_PORT=5432
REDIS_PORT=6379
```

`cluster:new` also writes `AGENTX_CONFIG_DIR` and `AGENTX_DB_DIR` (pointing at
`clusters/<name>/config` and `clusters/<name>/db`) so the container reads config from and
persists database data into the cluster directory.

Plus any LLM provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) and CORS settings you
need. See [Configuration](../getting-started/configuration.md) for the full list.

!!! danger "Don't ship the defaults"
    `DJANGO_SECRET_KEY` must be set, `DJANGO_DEBUG` must be `false`, and every database
    password must be changed from `changeme` before exposing the stack.

## Bring-up sequence

A private single-host instance (no public gateway):

```bash
task cluster:new CLUSTER=prod            # scaffold clusters/prod/ + seed config
# edit clusters/prod/.env: DJANGO_SECRET_KEY, DJANGO_DEBUG=false, passwords, provider keys
task cluster:up CLUSTER=prod             # build + start API + databases
task cluster:migrate CLUSTER=prod        # apply Django + memory schema (vector indexes, etc.)
task cluster:auth:setup CLUSTER=prod     # set the root password (if AGENTX_AUTH_ENABLED=true)
task cluster:warmup CLUSTER=prod         # pre-load the embedding model (avoids a slow first request)
task cluster:status CLUSTER=prod         # confirm everything is healthy
```

!!! warning "Don't skip `cluster:migrate`"
    `cluster:migrate` applies **both** the Django/PostgreSQL ORM tables **and** the Neo4j/PG/Redis
    memory schema (`init_memory_schema`). The memory schema creates the vector indexes recall and
    the semantic-duplicate check rely on — without it, consolidation logs
    `fact_embeddings index missing` and silently stops de-duplicating.

Verify the API:

```bash
curl http://localhost:12319/api/health?include_memory=true
```

## Next steps

- [Authentication](authentication.md) — turn on login + the version-compatibility checks
- [Clusters](clusters.md) — run multiple instances, add the public gateway, enable GPU

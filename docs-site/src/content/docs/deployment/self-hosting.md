# Self-Hosting (Docker Hub)

[![Docker Image Version](https://img.shields.io/docker/v/qrmadness/agentx-api?sort=semver&logo=docker&label=qrmadness%2Fagentx-api)](https://hub.docker.com/r/qrmadness/agentx-api)
[![Image Size](https://img.shields.io/docker/image-size/qrmadness/agentx-api/latest?logo=docker)](https://hub.docker.com/r/qrmadness/agentx-api/tags)
[![Docker Pulls](https://img.shields.io/docker/pulls/qrmadness/agentx-api?logo=docker)](https://hub.docker.com/r/qrmadness/agentx-api)

Run AgentX from the **published Docker image** — no source checkout, no Taskfile.
This is the simplest way to stand up your own instance.

> **Image:** [`qrmadness/agentx-api`](https://hub.docker.com/r/qrmadness/agentx-api) on Docker Hub.

AgentX deployments come in two flavours along an **isolation axis**:

| | **Local cluster** | **Isolated cluster** |
|---|---|---|
| For | Development / hacking on AgentX | Self-hosting / production |
| Source | Built from the repo (`Dockerfile`) | Pulls the published image |
| Managed by | `task cluster:*` (in-repo) | The in-image `agentx` CLI |
| Databases | Dedicated stack | Dedicated stack + own network |
| Init | `task cluster:migrate` | Self-initializes on boot |

This page covers the **isolated** path. For local development clusters see
[Clusters & Gateway](clusters.md).

## Get the bundle

Each release ships a small deployment bundle (attached to the GitHub release and
linked from the [Docker Hub repo](https://hub.docker.com/r/qrmadness/agentx-api)):

```
docker-compose.yml        # API (pulled image) + Neo4j + PostgreSQL + Redis
docker-compose.gpu.yml    # optional NVIDIA GPU overlay
.env.example              # configuration template
README.md
```

## Quick start

```bash
cp .env.example .env
# Edit .env: DJANGO_SECRET_KEY, database passwords, DJANGO_ALLOWED_HOSTS,
# and at least one LLM provider API key.

docker compose up -d
```

On first start the API **self-initializes** its database schemas (idempotent) and
seeds default config. It also downloads the embedding + translation models
(~5 GB) into a persistent cache under `./data/hf-cache`, so later starts are fast.

```bash
curl "http://localhost:12319/api/health?include_memory=true"
```

!!! danger "Don't ship the defaults"
    `DJANGO_SECRET_KEY` must be set, `DJANGO_DEBUG` must be `false`, and every
    database password must be changed from `changeme` before exposing the stack.

### Set the root password

With `AGENTX_AUTH_ENABLED=true` (the default), set the password once on first run
— either from the client's first-run setup screen, or:

```bash
docker compose exec api agentx setup-auth
```

## The `agentx` CLI

All day-2 operations run through the in-image CLI (versioned with the image, so
nothing external to keep in sync):

```bash
docker compose exec api agentx help          # list commands
docker compose exec api agentx status        # health + memory status
docker compose exec api agentx version       # running API version
docker compose exec api agentx migrate       # re-apply migrations + memory schema
docker compose exec api agentx setup-auth --force   # reset the root password
docker compose exec api agentx warmup        # pre-load the embedding model
docker compose exec api agentx export --output /app/data/memory-export.json
docker compose exec api agentx import --input /app/data/memory-export.json
```

!!! note "Backups"
    `agentx export`/`import` cover the **memory graph** only. For a full database
    backup, dump the volumes directly:
    ```bash
    docker compose exec postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > backup.sql
    docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/backups
    ```

## GPU acceleration

With the [NVIDIA Container Toolkit](clusters.md#gpu-acceleration-nvidia-overlay)
on the host, layer the GPU overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
curl -s localhost:12319/api/health | jq .compute   # {"device":"cuda",...}
```

## Updating

```bash
# pin AGENTX_IMAGE to the new version in .env (recommended), or pull :latest
docker compose pull
docker compose up -d
```

Schema migrations re-apply automatically on boot (idempotent); config and data
persist under `./data`.

## Going public

To expose the instance over the internet behind an Nginx + Cloudflare Tunnel
gateway with a shared-secret header, see
[Clusters & Gateway → Gateway](clusters.md#gateway-nginx--cloudflare).

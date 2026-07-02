# AgentX API

[![Docs](https://img.shields.io/badge/docs-agentx.thejpnet.net-6366f1)](https://agentx.thejpnet.net/docs)
[![Image Version](https://img.shields.io/docker/v/qrmadness/agentx-api?sort=semver)](https://hub.docker.com/r/qrmadness/agentx-api/tags)
[![Image Size](https://img.shields.io/docker/image-size/qrmadness/agentx-api/latest)](https://hub.docker.com/r/qrmadness/agentx-api/tags)
[![Pulls](https://img.shields.io/docker/pulls/qrmadness/agentx-api)](https://hub.docker.com/r/qrmadness/agentx-api)

The backend API for **AgentX** — an AI agent platform with multi-model
orchestration, MCP tool integration, reasoning frameworks, and a layered memory
system (episodic / semantic / procedural / working).

This image is the server. It is designed to run as a self-contained ("isolated")
deployment alongside its databases (Neo4j + PostgreSQL/pgvector + Redis) via
Docker Compose — no source checkout or Taskfile required.

## Quick start

Grab the deployment bundle (compose + `.env.example`) from the
[latest release](https://github.com/QR-Madness/agentx/releases), then:

```bash
cp .env.example .env
# edit .env: DJANGO_SECRET_KEY, database passwords, DJANGO_ALLOWED_HOSTS,
# and at least one LLM provider API key
docker compose up -d
```

The API **self-initializes** its database schemas on first boot (idempotent) and
seeds default config. First start also downloads the embedding model (~2.3 GB;
translation models download lazily on first use) into a persistent cache, so
later starts are fast.

```bash
curl "http://localhost:12319/api/health?include_memory=true"
```

If auth is enabled (default), set the root password once:

```bash
docker compose exec api agentx setup-auth
```

## Operations — the `agentx` CLI

Day-2 operations run through the in-image CLI (versioned with the image):

```bash
docker compose exec api agentx help        # list commands
docker compose exec api agentx status      # health + memory status
docker compose exec api agentx migrate     # re-apply migrations + memory schema
docker compose exec api agentx warmup      # pre-load the embedding model
docker compose exec api agentx setup-auth --force   # reset the root password
```

## Tags

- `latest` — the most recent release
- `X.Y.Z` — pinned release (recommended for reproducible deploys)

`linux/amd64`. The image bundles PyTorch + Transformers + sentence-transformers,
so it is large (~4.4 GB); the embedding/translation models themselves download at
runtime into the mounted cache volume.

## Configuration

Set via the `.env` file. Key variables: `DJANGO_SECRET_KEY` (required),
`DJANGO_ALLOWED_HOSTS`, `NEO4J_PASSWORD` / `POSTGRES_PASSWORD`,
`AGENTX_AUTH_ENABLED`, `EMBEDDING_PROVIDER` (`local` | `openai`), `AGENTX_DEVICE`
(`auto` | `cpu` | `cuda`), and provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).

GPU acceleration is available via the `docker-compose.gpu.yml` overlay (NVIDIA
Container Toolkit required).

## Links

- 📚 Documentation: https://agentx.thejpnet.net/docs/deployment/self-hosting
- 🐙 Source: https://github.com/QR-Madness/agentx
- 🐛 Issues: https://github.com/QR-Madness/agentx/issues

Licensed under MIT.

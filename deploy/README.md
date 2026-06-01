# AgentX — Self-hosted (isolated) deployment

Run the AgentX API and its databases from the published Docker image — no source
checkout required. This bundle is a self-contained ("isolated") deployment unit:
the API image plus its own Neo4j, PostgreSQL, and Redis.

## What's in this bundle

| File | Purpose |
|------|---------|
| `docker-compose.yml` | The stack: API (pulled image) + Neo4j + PostgreSQL + Redis |
| `docker-compose.gpu.yml` | Optional NVIDIA GPU overlay for the API |
| `.env.example` | Configuration template — copy to `.env` and fill in |

## Quick start

```bash
cp .env.example .env
# Edit .env: set DJANGO_SECRET_KEY, the database passwords, DJANGO_ALLOWED_HOSTS,
# and at least one LLM provider API key.

docker compose up -d
```

On first start the API **self-initializes** its database schemas (idempotent) and
seeds default config. First boot also downloads the embedding + translation
models (~5 GB) into a persistent cache under `./data/hf-cache`, so subsequent
starts are fast.

Check health:

```bash
curl "http://localhost:12319/api/health?include_memory=true"
```

If `AGENTX_AUTH_ENABLED=true` (the default), set the root password once:

```bash
docker compose exec api agentx setup-auth
```

## Day-2 operations — the `agentx` CLI

All management runs through the in-image CLI (versioned with the image):

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

> `export`/`import` cover the **memory graph** only. For a full database backup,
> dump the volumes directly:
> ```bash
> docker compose exec postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > backup.sql
> docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/backups
> ```

## GPU acceleration (optional)

With the NVIDIA Container Toolkit installed on the host:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Verify: `curl -s localhost:12319/api/health | jq .compute` → `{"device":"cuda",...}`.

## Updating

```bash
# pin AGENTX_IMAGE to the new version in .env (recommended), or pull :latest
docker compose pull
docker compose up -d
```

Schema migrations re-apply automatically on boot (idempotent). Your config and
data persist under `./data`.

## Documentation

Full docs: https://agentx.thejpnet.net/docs/deployment

# AgentX — Self-hosted (isolated) deployment

Run the AgentX API and its databases from the published Docker image — no source
checkout required. This bundle is a self-contained ("isolated") deployment unit:
the API image plus its own Neo4j, PostgreSQL, and Redis.

## What's in this bundle

| File | Purpose |
|------|---------|
| `docker-compose.yml` | The stack: API (pulled image) + Neo4j + PostgreSQL + Redis |
| `docker-compose.gpu.yml` | Optional NVIDIA GPU overlay for the API |
| `docker-compose.gateway.yml` | Nginx token gateway (shared secret + rate limiting) in front of the API |
| `docker-compose.gateway.expose.yml` | Publish the gateway on a host port (BYO reverse proxy / TLS) |
| `docker-compose.tunnel.yml` | Cloudflare Tunnel overlay, dashboard/token flavor (no host `cloudflared`) |
| `docker-compose.tunnel.named.yml` | Cloudflare Tunnel overlay, named/credentials-file flavor |
| `gateway/nginx.conf.example` | Gateway config template — copy to `gateway/nginx.conf` |
| `gateway/cloudflared/config.yml.example` | Named-tunnel ingress template |
| `.env.example` | Configuration template — copy to `.env` and fill in |

## Quick start

```bash
cp .env.example .env
# Edit .env: set DJANGO_SECRET_KEY, the database passwords, DJANGO_ALLOWED_HOSTS,
# and at least one LLM provider API key.

docker compose up -d
```

On first start the API **self-initializes** its database schemas (idempotent) and
seeds default config. First boot also downloads the embedding model (~2.3 GB;
translation models download lazily on first use) into a persistent cache under
`./data/hf-cache`, so subsequent starts are fast.

> **Docker Desktop (macOS/Windows):** unpack this bundle somewhere under a
> file-shared path (e.g. your home directory, not `/tmp`) — the stack uses
> bind mounts under `./data`, which Docker Desktop only allows from shared
> locations.

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

## Going public (gateway + tunnel — recommended)

Front the API with the bundled **Nginx token gateway** — every request must
carry a shared-secret `AgentX-Gateway-Token` header and is rate-limited
(~10 req/s per client IP) before it reaches the API — then expose the gateway
via a Cloudflare Tunnel:

```bash
# 1. Gateway config + shared secret:
mkdir -p gateway
cp gateway/nginx.conf.example gateway/nginx.conf
openssl rand -hex 32     # → AGENTX_GATEWAY_TOKEN in .env

# 2. In the Cloudflare Zero Trust dashboard: create a Tunnel, add a Public
#    Hostname with Service = http://nginx:80 (the gateway, NOT the API),
#    and copy the connector token.

# 3. In .env:
#      TUNNEL_TOKEN=<token>                  # secret
#      AGENTX_PUBLIC_HOST=agentx.example.com
#      AGENTX_GATEWAY_TOKEN=<64-char hex>    # secret
#      AGENTX_TRUST_PROXY=true               # gateway overwrites X-Forwarded-For
#      COMPOSE_FILE=docker-compose.yml:docker-compose.gateway.yml:docker-compose.tunnel.yml

# 4. Up (COMPOSE_FILE makes plain `docker compose` include the overlays):
docker compose up -d

# 5. Smoke test:
curl -I https://agentx.example.com/api/health                    # → 401
curl -I -H "AgentX-Gateway-Token: <token>" \
     https://agentx.example.com/api/health                       # → 200
```

Point your AgentX client at the public URL and put the same token in the
server's **Gateway token** field — it sends the header automatically.

Variants:

- **Named tunnel** (credentials file instead of dashboard token): also
  `cp gateway/cloudflared/config.yml.example gateway/cloudflared/config.yml`,
  follow its embedded setup steps, and use
  `docker-compose.tunnel.named.yml` in `COMPOSE_FILE` instead of
  `docker-compose.tunnel.yml`.
- **Your own reverse proxy / TLS** (no Cloudflare): use
  `docker-compose.gateway.expose.yml` instead of a tunnel overlay; the gateway
  listens on `${AGENTX_GATEWAY_BIND:-127.0.0.1}:${AGENTX_GATEWAY_PORT:-8080}`.
- **Bare tunnel (not recommended):** `docker-compose.tunnel.yml` alone with
  Service = `http://api:12319` publishes the API with only app-auth in front —
  no shared secret, no rate limiting. If you do this, keep
  `AGENTX_AUTH_ENABLED=true` and leave `AGENTX_TRUST_PROXY` unset.

> **Edited `gateway/nginx.conf`?** It's bind-mounted as a single file, whose
> inode the container pins at create time — recreate (don't just restart) the
> gateway to pick up changes: `docker compose up -d --force-recreate nginx`.

Full walkthrough + troubleshooting:
https://agentx.thejpnet.net/docs/deployment/self-hosting/#going-public

## Updating

```bash
# pin AGENTX_IMAGE to the new version in .env (recommended), or pull :latest
docker compose pull
docker compose up -d
```

> If you use overlays (gateway/tunnel/GPU), include them on every command or
> set them once via `COMPOSE_FILE=docker-compose.yml:…` in `.env` as above.

Schema migrations re-apply automatically on boot (idempotent). Your config and
data persist under `./data`.

## Documentation

Full docs: https://agentx.thejpnet.net/docs/deployment

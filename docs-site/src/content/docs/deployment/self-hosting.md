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

## Get the desktop client

The Tauri **desktop client** connects to the server you just started — point it at
your server's URL on first run.

- **Download the latest client:** [github.com/QR-Madness/agentx/releases/latest](https://github.com/QR-Madness/agentx/releases/latest)
- **All releases / a specific version:** [github.com/QR-Madness/agentx/releases](https://github.com/QR-Madness/agentx/releases)

Each client release lists the supported **API protocol version** and ships SHA-256
checksums for every installer. Platform coverage today is **Windows**
(`.exe` / `.msi`) and **Linux** (`.deb` / `.AppImage` / `.rpm`); a macOS build is
not yet in the release matrix.

!!! note "Protocol matching"
    The client must speak the same API **protocol version** as your server (reported
    at `/api/health`). If they differ, the client shows a version-mismatch screen —
    grab the client release that matches your server's protocol.

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

The simplest way to reach an isolated instance from the internet is a
**dashboard-managed Cloudflare Tunnel running as a container** — no `cloudflared`
on the host, no inbound ports opened, no DNS to script. The tunnel's
public-hostname → service mapping lives in the Cloudflare Zero Trust dashboard;
the container just dials out with a token.

### 1. Create the tunnel (Cloudflare dashboard)

In **Zero Trust → Networks → Tunnels**, create a tunnel and copy its **token**.
Add a **Public Hostname** for it:

- **Subdomain/Domain:** the hostname you'll use (e.g. `agentx.example.com`)
- **Service:** `http://api:12319` — the API container, reachable by name because
  the tunnel runs on the same Docker network.

Copy the connector token (the long string in the
`cloudflared … run --token <TOKEN>` command the dashboard shows).

### 2. Configure the tunnel overlay

The bundle **ships `docker-compose.tunnel.yml`** — a small overlay that adds a
`cloudflared` container in front of the API. You don't need to write it; just set
its two values in `.env`. (For reference, it's:)

```yaml
services:
  cloudflared:
    image: cloudflare/cloudflared:latest
    container_name: ${AGENTX_CLUSTER_NAME:-agentx}-cloudflared
    command: tunnel --no-autoupdate run
    environment:
      - TUNNEL_TOKEN=${TUNNEL_TOKEN}
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
    profiles:
      - production
```

Then in `.env`:

```bash
TUNNEL_TOKEN=<your-tunnel-token>     # from the dashboard; keep it secret
AGENTX_PUBLIC_HOST=agentx.example.com
```

`AGENTX_PUBLIC_HOST` extends `ALLOWED_HOSTS`, `CORS_ALLOWED_ORIGINS`, and
`CSRF_TRUSTED_ORIGINS` in one shot, so Django accepts requests arriving with that
host from the tunnel.

### 3. Bring it up

The overlay must be included on **every** `docker compose` invocation (up, pull,
restart). Either pass both `-f` flags each time:

```bash
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml up -d
```

…or make it sticky by setting `COMPOSE_FILE` in `.env` so a bare `docker compose
up -d` always layers it:

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.tunnel.yml
```

The `cloudflared` container waits for the API to be healthy, then dials out.

### 4. Verify

```bash
# The connector registered with Cloudflare's edge:
docker compose logs cloudflared | grep -i "Registered tunnel connection"

# Reachable over the public hostname (health is unauthenticated):
curl -sI https://agentx.example.com/api/health                       # → 200
curl -s "https://agentx.example.com/api/health?include_memory=true" | jq .
```

Then point the desktop client at `https://agentx.example.com` and log in
(`root` + the password you set). Authenticated endpoints return `401` until you
log in — that's app auth, not the tunnel.

!!! danger "Keep auth on"
    This tunnel publishes the API **directly** — there is no shared-secret gateway
    in front of it. Leave `AGENTX_AUTH_ENABLED=true` and set a strong root password
    (`docker compose exec api agentx setup-auth`) **before** the hostname goes live.
    Treat `TUNNEL_TOKEN` as a secret — anyone with it can run your tunnel. Rotate it
    from the dashboard if it leaks.

!!! note "Troubleshooting"
    - **`502` / "no such host" / tunnel can't reach the origin** — the dashboard
      **Service** must be `http://api:12319` (plain HTTP, container name `api`), and
      `cloudflared` must be part of *this* compose project so it shares the network.
      If you renamed the API service or run it elsewhere, match the Service to it.
    - **`400 Bad Request` / "Invalid HTTP_HOST"** — the public hostname isn't in
      Django's allowed hosts. Set `AGENTX_PUBLIC_HOST` to the exact hostname and
      recreate the API container (`docker compose up -d` — env is read at boot).
    - **Desktop client connects but the browser/web client is blocked by CORS** —
      the API allows the desktop (Tauri) origins by default. For a web client served
      from another origin, add it to `CORS_ALLOWED_ORIGINS` (comma-separated).
    - **Chat streaming** (SSE) works over the tunnel with no extra configuration.

!!! note "Want a shared-secret gateway too?"
    To additionally front the API with an Nginx gateway that enforces an
    `AgentX-Gateway-Token` header (defense-in-depth, rate limiting, SSE tuning),
    see [Clusters & Gateway → Gateway](clusters.md#gateway-nginx--cloudflare). That
    path points the tunnel's **Service** at `http://nginx:80` instead of the API,
    and the client sends the gateway token from its per-server setting.

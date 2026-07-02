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
seeds default config. It also downloads the embedding model (~2.3 GB; translation
models download lazily on first use) into a persistent cache under
`./data/hf-cache`, so later starts are fast. Expect the first boot to spend a few
minutes *initializing* — the manager GUI below shows that state explicitly.

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

## Guided setup: the manager GUI

The bundle ships a **web deployment manager** — cluster dashboard, live health (including
the first-boot *initializing* state), per-cluster CPU/memory gauges, lifecycle buttons,
log streaming, and safe destroy — so day-2 operation doesn't require memorizing compose
commands. Three steps from an untarred bundle:

```bash
# 1. Enable the manager profile alongside production (once, in .env):
#      COMPOSE_PROFILES=production,manager
docker compose up -d

# 2. Grab the access token (also stored at ./.manager-token):
docker compose logs manager | grep -A1 token

# 3. Open the GUI and paste the token:
#      http://127.0.0.1:12320
```

From there you can watch the first boot progress (the API card shows **initializing**
while models download, then flips to **up**), tail logs per service, and see CPU/memory
per cluster at a glance.

!!! danger "Keep the manager private"
    The manager drives the Docker socket — treat it like root on the host. It publishes
    on `127.0.0.1` only and always requires its token; never route it through the
    tunnel/gateway. From another machine, use SSH port forwarding:
    `ssh -L 12320:127.0.0.1:12320 <host>`.

!!! note "Run its compose commands from the deployment root"
    The manager container mounts the deployment directory at the same absolute path as
    the host (`${PWD}:${PWD}`) so compose bind mounts resolve identically — run
    `docker compose` for this overlay from the bundle directory. Full reference:
    [Deployment Manager](manager.md).

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

The recommended public setup layers **two things** in front of Django:

1. **The Nginx token gateway** (`docker-compose.gateway.yml`, shipped in the
   bundle) — every request must carry a shared-secret `AgentX-Gateway-Token`
   header and is rate-limited (~10 req/s per client IP) *before* it reaches
   the API. The desktop client sends the header automatically from its
   per-server **Gateway token** setting.
2. **A Cloudflare Tunnel running as a container** — no `cloudflared` on the
   host, no inbound ports opened, TLS terminated at Cloudflare's edge.

### 1. Gateway config + shared secret

```bash
mkdir -p gateway
cp gateway/nginx.conf.example gateway/nginx.conf
openssl rand -hex 32        # → AGENTX_GATEWAY_TOKEN below
```

The gateway overlay **fails closed**: compose refuses to start it while
`AGENTX_GATEWAY_TOKEN` is unset or empty.

### 2. Create the tunnel (Cloudflare dashboard)

In **Zero Trust → Networks → Tunnels**, create a tunnel and copy its **token**.
Add a **Public Hostname** for it:

- **Subdomain/Domain:** the hostname you'll use (e.g. `agentx.example.com`)
- **Service:** `http://nginx:80` — the **gateway**, reachable by name because
  the tunnel runs on the same Docker network. (Pointing it at `http://api:12319`
  directly bypasses the gateway — see the bare-tunnel note below.)

### 3. Configure `.env`

```bash
TUNNEL_TOKEN=<your-tunnel-token>          # from the dashboard; keep it secret
AGENTX_PUBLIC_HOST=agentx.example.com
AGENTX_GATEWAY_TOKEN=<64-char hex>        # from step 1; keep it secret
AGENTX_TRUST_PROXY=true                   # the gateway overwrites X-Forwarded-For

# Make the overlays sticky so a bare `docker compose up -d` always layers them:
COMPOSE_FILE=docker-compose.yml:docker-compose.gateway.yml:docker-compose.tunnel.yml
```

`AGENTX_PUBLIC_HOST` extends `ALLOWED_HOSTS`, `CORS_ALLOWED_ORIGINS`, and
`CSRF_TRUSTED_ORIGINS` in one shot, so Django accepts requests arriving with that
host from the tunnel. `AGENTX_TRUST_PROXY` lets the API attribute requests to the
real client IP (nginx sets `X-Forwarded-For` from Cloudflare's
`CF-Connecting-IP`); leave it unset when nothing overwrites that header.

### 4. Bring it up and verify

```bash
docker compose up -d

# The connector registered with Cloudflare's edge:
docker compose logs cloudflared | grep -i "Registered tunnel connection"

# Gateway enforces the token (health is otherwise unauthenticated):
curl -sI https://agentx.example.com/api/health                        # → 401
curl -sI -H "AgentX-Gateway-Token: <token>" \
     https://agentx.example.com/api/health                            # → 200
```

Then point the desktop client at `https://agentx.example.com`, paste the gateway
token into the server's **Gateway token** field, and log in (`root` + the
password you set). Authenticated endpoints return `401` until you log in —
that's app auth, not the gateway.

### Variants

- **Named tunnel** (credentials file, no dashboard-managed routing): also
  `cp gateway/cloudflared/config.yml.example gateway/cloudflared/config.yml`,
  follow the setup steps embedded in that file, and swap
  `docker-compose.tunnel.yml` for `docker-compose.tunnel.named.yml` in
  `COMPOSE_FILE`.
- **Your own reverse proxy / TLS** (no Cloudflare): swap the tunnel overlay for
  `docker-compose.gateway.expose.yml` — the gateway listens on
  `${AGENTX_GATEWAY_BIND:-127.0.0.1}:${AGENTX_GATEWAY_PORT:-8080}` for your
  proxy to front. Rate limiting keys on the TCP peer address in this mode.

!!! danger "Bare tunnel (no gateway) — not recommended"
    `docker-compose.tunnel.yml` alone, with the dashboard Service set to
    `http://api:12319`, publishes the API **directly** — no shared secret and no
    rate limiting; only app auth protects it. If you must run this way: leave
    `AGENTX_AUTH_ENABLED=true` (the default), set a strong root password
    (`docker compose exec api agentx setup-auth`) **before** the hostname goes
    live, and leave `AGENTX_TRUST_PROXY` unset so a spoofed `X-Forwarded-For`
    can't fake client IPs. Treat `TUNNEL_TOKEN` as a secret — anyone with it can
    run your tunnel; rotate it from the dashboard if it leaks.

!!! note "Troubleshooting"
    - **`401` on every request** — the gateway didn't see a valid
      `AgentX-Gateway-Token` header. Check the client's Gateway token field and
      that `.env`'s `AGENTX_GATEWAY_TOKEN` matches; watch rejections with
      `docker compose logs -f nginx`.
    - **`429 Too Many Requests`** — the per-IP rate limit (~10 req/s, burst 20).
      Sustained 429s from one client usually mean a retry loop.
    - **Edited `gateway/nginx.conf`?** It's bind-mounted as a single file whose
      inode the container pins at create time — recreate, don't restart:
      `docker compose up -d --force-recreate nginx`.
    - **`502` / "no such host" / tunnel can't reach the origin** — the dashboard
      **Service** must be `http://nginx:80` (or `http://api:12319` for a bare
      tunnel), and `cloudflared` must be part of *this* compose project so it
      shares the network.
    - **`400 Bad Request` / "Invalid HTTP_HOST"** — the public hostname isn't in
      Django's allowed hosts. Set `AGENTX_PUBLIC_HOST` to the exact hostname and
      recreate the API container (`docker compose up -d` — env is read at boot).
    - **Desktop client connects but the browser/web client is blocked by CORS** —
      the API allows the desktop (Tauri) origins by default. For a web client served
      from another origin, add it to `CORS_ALLOWED_ORIGINS` (comma-separated).
    - **Chat streaming (SSE)** — the gateway proxies streams with buffering off
      and 600s read timeouts; no extra configuration. To verify a long stream
      survives the full path, run a streaming chat request with `curl -N` through
      the public hostname and confirm chunks keep arriving past the two-minute
      mark without truncation.

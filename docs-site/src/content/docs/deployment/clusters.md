# Clusters & Gateway

A **cluster** is a self-contained AgentX instance — its own `.env`, config, database storage,
and ports — that you can run alongside other clusters on a single host. Clusters build on the
[production profile](production.md) and add an optional public **gateway** (Nginx + Cloudflare
tunnel) and an optional **GPU overlay**.

!!! note "Local vs isolated clusters"
    This page covers **local clusters**: repo-native, built from source, and driven by the
    `task cluster:*` workflow — ideal for development and running several instances on your own
    machine. If you instead want a portable, self-managing instance from the published image
    (no repo, no Taskfile), that's an **isolated cluster** — see
    [Self-Hosting (Docker Hub)](self-hosting.md). Local clusters add the
    `docker-compose.build.yml` overlay (to build the API from source) on top of the same base
    compose the isolated bundle ships.

## On-disk layout

Each cluster lives under `clusters/<name>/`:

```
clusters/
├── template/                 # scaffolding copied by `cluster:new`
│   ├── .env.example
│   ├── nginx.conf.example
│   └── cloudflared/config.yml.example
└── <name>/
    ├── .env                  # cluster-specific secrets + ports
    ├── config/               # app config, seeded from api/defaults/
    │   ├── agent_profiles.yaml
    │   ├── prompt_templates.yaml
    │   ├── workflows.yaml
    │   └── memory_settings.json
    ├── db/                   # bind-mounted database storage
    │   ├── neo4j/  postgres/  redis/
    ├── nginx.conf            # OPTIONAL — its presence enables the gateway overlay
    └── cloudflared/          # OPTIONAL — config.yml + user-provided credentials.json
```

`config/` is seeded from `api/defaults/` (agent profiles, prompt templates, workflows, memory
settings). `config.json` is **not** seeded — it's synthesized at runtime by `ConfigManager`.

## Lifecycle

Cluster lifecycle is owned by the **[deployment manager](manager.md)** (`agentx-manager`);
the `task cluster:*` targets are thin wrappers kept for muscle memory. The manager persists
each cluster's overlay spec in `clusters/<name>/.manager-state.json`, runs every cluster
under its own compose project (`agentx-<name>`), and force-recreates services whose
bind-mounted config changed. `task manager:serve` opens the web GUI.

| Task | Purpose |
|------|---------|
| `cluster:new CLUSTER=<name> [GATEWAY=1]` | Scaffold the directory, copy defaults, write `.env` (identity + path vars, generated `DJANGO_SECRET_KEY`, detected `DJANGO_ALLOWED_HOSTS`). `GATEWAY=1` = gateway + named tunnel + generated `AGENTX_GATEWAY_TOKEN` + `AGENTX_TRUST_PROXY=true`. |
| `cluster:seed CLUSTER=<name> [FORCE=1]` | (Re)seed `config/` from `api/defaults/`; skips existing files unless `FORCE=1` |
| `cluster:gateway:enable CLUSTER=<name>` | Add the gateway (+ named-tunnel scaffolding) to an existing cluster |
| `cluster:up CLUSTER=<name> [REBUILD=1] [NVIDIA=1]` | Start — overlays come from the manager spec; waits out first-boot initialization instead of failing |
| `cluster:down` / `cluster:restart` / `cluster:rebuild` | Stop / config-aware restart / rebuild from source |
| `cluster:destroy CLUSTER=<name>` | `down --volumes` + delete `clusters/<name>/` (typed confirmation) |
| `cluster:adopt CLUSTER=<name>` | One-time migration of a pre-manager cluster onto its own compose project (brief downtime) |
| `cluster:status` / `cluster:logs` / `cluster:logs:gateway` / `cluster:logs:tunnel` | Status & logs |
| `manager:usage [CLUSTER=<name>]` | CPU% / memory per cluster (docker stats) |
| `cluster:migrate` | Apply **Django + memory-schema** migrations via the in-image `agentx` CLI |
| `cluster:makemigrations` | Generate new Django migrations inside the container |
| `cluster:auth:setup` / `cluster:warmup` | Set the root password / pre-load the embedding model |
| `cluster:list` | List clusters with kind + overlay tags |
| `cluster:diff CLUSTER=<name>` | Drift report vs `clusters/template/`: `.env` compared by **key only** (values never printed), `nginx.conf` / `cloudflared/config.yml` vs their examples, `config/` filenames vs `api/defaults/`. Run it after pulling template changes to see what your cluster is missing. |

Compose invocation is assembled by the manager — for example `up` on a gateway cluster runs roughly:

```bash
docker compose -p agentx-<name> --env-file clusters/<name>/.env \
  -f docker-compose.yml \
  [-f docker-compose.build.yml]        # spec kind=source: build the API from the repo
  [-f docker-compose.gateway.yml]      # spec gateway=true
  [-f docker-compose.tunnel.named.yml] # spec tunnel=named
  [-f docker-compose.gpu.yml]          # spec gpu=true
  --profile production up -d
```

The gateway overlays find their files via **`AGENTX_GATEWAY_DIR`** in the cluster `.env`
(set to `./clusters/<name>` by `cluster:new`). The same overlays ship in the isolated
deployment bundle, where the variable defaults to `./gateway`.

## Ports

Running several clusters on one host means giving each unique host ports. Override these in the
cluster's `.env`:

```bash
API_PORT=12319
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687
POSTGRES_PORT=5432
REDIS_PORT=6379
```

## Gateway (Nginx + Cloudflare)

When `clusters/<name>/nginx.conf` is present, `cluster:up` layers in
`docker-compose.gateway.yml` — and, when `clusters/<name>/cloudflared/config.yml` also
exists, `docker-compose.tunnel.named.yml`:

- **nginx** (`nginx:1.27-alpine`, from `docker-compose.gateway.yml`) — renders `nginx.conf`
  via envsubst (only `AGENTX_*` vars), then for every request:
  - validates the **`AgentX-Gateway-Token`** header against `AGENTX_GATEWAY_TOKEN` → `401` if
    missing/invalid, and strips the header before proxying to Django. The overlay **fails
    closed**: compose refuses to start it while `AGENTX_GATEWAY_TOKEN` is unset/empty;
  - rate-limits ~10 req/s per client IP (`CF-Connecting-IP`, falling back to the TCP peer
    address when no tunnel fronts it), burst 20 → `429`;
  - proxies to `api:${API_PORT}` with SSE-friendly settings (`proxy_buffering off`, long
    timeouts) and a tokenless `/__gateway_health` probe.
- **cloudflared** (`cloudflare/cloudflared:latest`, from `docker-compose.tunnel.named.yml`) —
  runs a named tunnel that terminates TLS at Cloudflare's edge and forwards to `nginx:80`.
  Only the configured hostname is routable; everything else returns 404.

No tunnel at all? Layer `docker-compose.gateway.expose.yml` instead to publish nginx on
`${AGENTX_GATEWAY_BIND:-127.0.0.1}:${AGENTX_GATEWAY_PORT:-8080}` for your own reverse
proxy / TLS terminator.

### One-time tunnel setup

```bash
cloudflared tunnel login
cloudflared tunnel create agentx-<cluster>
cp ~/.cloudflared/<tunnel-id>.json clusters/<name>/cloudflared/credentials.json
cloudflared tunnel route dns agentx-<cluster> <public-host>
# then set in clusters/<name>/cloudflared/config.yml:
#   tunnel: <tunnel-id>
#   ingress: - hostname: <public-host>  service: http://nginx:80
```

Set in the cluster `.env`:

```bash
AGENTX_PUBLIC_HOST=agentx.example.com
AGENTX_GATEWAY_TOKEN=$(openssl rand -hex 32)
AGENTX_TRUST_PROXY=true   # nginx overwrites X-Forwarded-For with the real client IP
```

Smoke-test once it's up:

```bash
curl -I https://$AGENTX_PUBLIC_HOST/api/health                                   # → 401 (no token)
curl -I -H "AgentX-Gateway-Token: $AGENTX_GATEWAY_TOKEN" \
        https://$AGENTX_PUBLIC_HOST/api/health                                   # → 200
```

The client sends `AgentX-Gateway-Token` from its per-server `gatewayToken` setting; this is
separate from user [authentication](authentication.md).

!!! note "Browser & desktop clients (CORS)"
    The gateway works with the desktop (Tauri) client and browser-based clients: the nginx
    gateway lets the CORS **preflight** (`OPTIONS`) through to Django (a preflight can't carry
    the custom `AgentX-Gateway-Token` header), and `agentx-gateway-token` is an allowed CORS
    request header. Set the client's per-server **gateway token** so the *actual* request
    carries it. The client's origin (`tauri://localhost`, and `http://localhost:1420` in dev)
    is allowed by default; add others via `CORS_ALLOWED_ORIGINS`.

!!! note "No `cloudflared` on the host (dashboard token tunnel)"
    Prefer not to run `cloudflared tunnel login/create/route` on the host? Run the tunnel as a
    **dashboard-managed token connector** instead — create the Tunnel in the Cloudflare Zero
    Trust dashboard, set its Public Hostname **Service** to `http://nginx:80`, and run the
    `cloudflared` container with `tunnel --no-autoupdate run` + `TUNNEL_TOKEN` (clear the
    scaffolded `cloudflared/config.yml` so it doesn't conflict). Same mechanics as the isolated
    bundle's overlay — see [Self-Hosting → Going public](self-hosting.md#going-public).

!!! tip "Editing `nginx.conf` after the cluster is up"
    `nginx.conf` is bind-mounted as a single file, whose inode the container pins at create
    time. After editing it, recreate (don't just restart) nginx so the change is picked up:
    `docker compose … up -d --force-recreate nginx`.

## Sharing access

A **connection link** hands someone everything their client needs to reach a cluster — the
API URL, an optional display name, and the gateway token — encoded in the URL **fragment**
(`https://<web-app>/#connect=…`), which browsers never send to any server. The recipient
opens the link in the deployed web client, confirms on the connect screen, and signs in with
their password (never part of the link). Desktop users can enter the same URL + gateway token
by hand in the sign-in screen's server picker.

The easiest way to build one is the **[manager GUI's Share button](manager.md#the-web-gui)**
on each cluster card. It prefills the server URL from `AGENTX_PUBLIC_HOST`, embeds the
cluster's `AGENTX_GATEWAY_TOKEN`, and warns about the foot-guns before you copy:

- **A localhost URL doesn't travel** — share the tunnel hostname (set `AGENTX_PUBLIC_HOST`).
- **The web app's origin must be CORS-allowed.** `AGENTX_PUBLIC_HOST` whitelists only its
  *own* origin; a client hosted elsewhere (e.g. `https://app.example.com`) must be added to
  `CORS_ALLOWED_ORIGINS` in the cluster `.env`, then applied with **Up** (Restart doesn't
  re-read env). Otherwise every API call fails in the browser's preflight.
- **Treat the link as a secret** — it embeds the gateway token. Share it over a private
  channel, and rotate `AGENTX_GATEWAY_TOKEN` if a link leaks.

## GPU Acceleration (NVIDIA overlay)

Passing `NVIDIA=1` to `cluster:up` layers in `docker-compose.gpu.yml`, which reserves the
host's NVIDIA GPUs for the `api` service:

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**Prerequisites** — the NVIDIA Container Toolkit on the host:

```bash
# recommended to run as superuser if non-sudo doesn't work
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

On Windows, use Docker Desktop's WSL2 backend with the GPU driver installed on the Windows host
(not inside WSL).

No application config changes are needed: the app resolves its compute device from
`AGENTX_DEVICE` (default `auto` → CUDA when available) and moves **both** the embedding model
(BAAI/bge-m3) and the NLLB-200 translation models onto it. Verify after start-up:

```bash
docker compose --env-file clusters/<name>/.env -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile production exec api uv run python -c "import torch; print(torch.cuda.is_available())"

# Or via the API — no exec needed:
curl -s localhost:<API_PORT>/api/health | jq .compute   # {"device":"cuda","cuda_available":true}
```

For local-dev GPU (outside Docker) and the Windows CUDA-torch caveat, see
[GPU Acceleration](../development/gpu.md).

The payoff is a large speedup for local embeddings and translation; GPU VRAM is separate from
the container's 8G memory reservation.

## Full walkthrough

```bash
task cluster:new CLUSTER=prod GATEWAY=1        # scaffold (with gateway)
# edit clusters/prod/.env: DJANGO_SECRET_KEY, DJANGO_DEBUG=false, passwords,
#   AGENTX_PUBLIC_HOST, AGENTX_GATEWAY_TOKEN, provider keys
# set up the Cloudflare tunnel (see above), drop credentials.json
task cluster:up CLUSTER=prod                   # add NVIDIA=1 for GPU
task cluster:migrate CLUSTER=prod              # Django + memory schema (vector indexes — required)
task cluster:auth:setup CLUSTER=prod           # set root password
task cluster:warmup CLUSTER=prod               # pre-load embeddings
task cluster:status CLUSTER=prod               # verify
```

# Clusters & Gateway

A **cluster** is a self-contained AgentX instance — its own `.env`, config, database storage,
and ports — that you can run alongside other clusters on a single host. Clusters build on the
[production profile](production.md) and add an optional public **gateway** (Nginx + Cloudflare
tunnel) and an optional **GPU overlay**.

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

| Task | Purpose |
|------|---------|
| `cluster:new CLUSTER=<name> [GATEWAY=1]` | Scaffold the directory, copy defaults, write `.env` (sets `AGENTX_CLUSTER_NAME`, `AGENTX_CONFIG_DIR`, `AGENTX_DB_DIR`, detected `DJANGO_ALLOWED_HOSTS`). `GATEWAY=1` also drops `nginx.conf` + `cloudflared/config.yml`. |
| `cluster:seed CLUSTER=<name> [FORCE=1]` | (Re)seed `config/` from `api/defaults/`; skips existing files unless `FORCE=1` |
| `cluster:gateway:enable CLUSTER=<name>` | Add the gateway files to an existing cluster |
| `cluster:up CLUSTER=<name> [REBUILD=1] [NVIDIA=1]` | Start the cluster (auto-includes the gateway overlay if `nginx.conf` exists; `--build` if `REBUILD=1`; GPU overlay if `NVIDIA=1`) |
| `cluster:down` / `cluster:restart` / `cluster:rebuild` | Stop / restart / rebuild |
| `cluster:status` / `cluster:logs` / `cluster:logs:gateway` / `cluster:logs:tunnel` | Status & logs |
| `cluster:migrate` / `cluster:makemigrations` | Django migrations inside the container |
| `cluster:auth:setup` / `cluster:warmup` | Set the root password / pre-load the embedding model |
| `cluster:list` | List clusters with tags (e.g. `gateway`, `missing-env`) |

Compose invocation is assembled automatically — for example `cluster:up` runs roughly:

```bash
docker compose --env-file clusters/<name>/.env \
  -f docker-compose.yml \
  [-f docker-compose.cluster.yml]   # when clusters/<name>/nginx.conf exists
  [-f docker-compose.gpu.yml]       # when NVIDIA=1
  --profile production up -d
```

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
`docker-compose.cluster.yml`, which adds two services:

- **nginx** (`nginx:1.27-alpine`) — renders `nginx.conf` via envsubst (only `AGENTX_*` vars),
  then for every request:
  - validates the **`AgentX-Gateway-Token`** header against `AGENTX_GATEWAY_TOKEN` → `401` if
    missing/invalid, and strips the header before proxying to Django;
  - rate-limits ~10 req/s per client IP (from `CF-Connecting-IP`), burst 20 → `429`;
  - proxies to `api:${API_PORT}` with SSE-friendly settings (`proxy_buffering off`, long
    timeouts) and a tokenless `/__gateway_health` probe.
- **cloudflared** (`cloudflare/cloudflared:latest`) — runs a named tunnel that terminates TLS
  at Cloudflare's edge and forwards to `nginx:80`. Only the configured hostname is routable;
  everything else returns 404.

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
```

Smoke-test once it's up:

```bash
curl -I https://$AGENTX_PUBLIC_HOST/api/health                                   # → 401 (no token)
curl -I -H "AgentX-Gateway-Token: $AGENTX_GATEWAY_TOKEN" \
        https://$AGENTX_PUBLIC_HOST/api/health                                   # → 200
```

The client sends `AgentX-Gateway-Token` from its per-server `gatewayToken` setting; this is
separate from user [authentication](authentication.md).

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

No application config changes are needed: the embedding model (BAAI/bge-m3) and the NLLB-200
translation model auto-detect CUDA. Verify after start-up:

```bash
docker compose --env-file clusters/<name>/.env -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile production exec api uv run python -c "import torch; print(torch.cuda.is_available())"
```

The payoff is a large speedup for local embeddings and translation; GPU VRAM is separate from
the container's 8G memory reservation.

## Full walkthrough

```bash
task cluster:new CLUSTER=prod GATEWAY=1        # scaffold (with gateway)
# edit clusters/prod/.env: DJANGO_SECRET_KEY, DJANGO_DEBUG=false, passwords,
#   AGENTX_PUBLIC_HOST, AGENTX_GATEWAY_TOKEN, provider keys
# set up the Cloudflare tunnel (see above), drop credentials.json
task cluster:up CLUSTER=prod                   # add NVIDIA=1 for GPU
task cluster:auth:setup CLUSTER=prod           # set root password
task cluster:warmup CLUSTER=prod               # pre-load embeddings
task cluster:status CLUSTER=prod               # verify
```

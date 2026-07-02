# Deployment Manager

The **deployment manager** (`agentx-manager`) owns cluster lifecycle for both deployment
flavors — local source-built clusters and isolated image deployments — through one tested
core, exposed as a **CLI** and a **web GUI**. You never assemble `docker compose -f …`
chains by hand.

| | |
|---|---|
| CLI | `agentx-manager` (repo: `uv run --project manager agentx-manager`, or the `task cluster:*` wrappers) |
| GUI | `http://127.0.0.1:12320` — `task manager:serve` (repo) or the `manager` compose profile (bundle) |
| Image | `qrmadness/agentx-manager` (~370 MB; python-slim + docker CLI + built GUI) |

## What it does

- **Explicit overlay specs.** Each deployment's shape lives in `.manager-state.json`
  (kind `source|image`; gateway; tunnel `none|token|named`; expose; gpu; shell) — edited via
  `agentx-manager set` or the GUI, not inferred from which files happen to exist. Pre-manager
  clusters are bootstrapped from the old file-presence rules automatically.
- **Per-cluster compose projects.** Everything runs under `-p agentx-<name>`, so clusters can't
  cross-target each other's containers or share a default network. Migrate an existing running
  cluster once with `agentx-manager adopt <name>` (brief downtime).
- **Config-aware restarts.** Tracked single-file bind mounts (`.env`, `nginx.conf`,
  `cloudflared/config.yml`, `credentials.json`) are hashed; when one changed, `restart`/`up`
  force-recreates exactly the affected services — a plain restart would keep serving the old
  file's pinned inode.
- **First-boot patience.** A fresh deployment downloads models for minutes before the API
  healthcheck passes; `up` recognizes the *initializing* state (running + init markers in logs),
  waits it out, and finishes bringing up dependents like the gateway instead of failing with
  "dependency failed to start".
- **Resource visibility.** `agentx-manager usage` / the GUI's per-cluster gauges aggregate
  `docker stats` (CPU %, memory % + bytes) over the cluster's compose project, with per-service
  breakdown.
- **Destroy, safely.** `agentx-manager destroy <name> --confirm <name>` = `down --volumes` +
  data-directory removal; the GUI requires typing the cluster name. Bundle mode never deletes
  the deployment root itself.
- **Day-2 passthrough.** `agentx-manager agentx <name> -- migrate|setup-auth|warmup|status|…`
  execs the in-image ops CLI, so behavior always matches the deployed image.

## CLI reference

```bash
agentx-manager list                          # deployments + spec tags
agentx-manager status [name]                 # phase (down|initializing|degraded|up) + services
agentx-manager usage  [name]                 # CPU% / memory per cluster (docker stats)
agentx-manager new <name> [--gateway] [--tunnel token|named] [--gpu] [--kind source|image]
agentx-manager gateway-enable [name] [--tunnel …]
agentx-manager up|down|restart|rebuild [name]
agentx-manager set [name] --gateway/--no-gateway --tunnel … --gpu/--no-gpu --shell/--no-shell --expose/--no-expose
agentx-manager adopt [name]                  # one-time move onto the agentx-<name> project
agentx-manager destroy <name> --confirm <name> [--keep-data]
agentx-manager logs [name] [service]
agentx-manager agentx [name] -- <ops-cli command>
agentx-manager serve [--host …] [--port …]   # the web GUI / REST server
```

In **bundle mode** (an untarred deploy bundle is exactly one deployment) the name is optional
everywhere. The root is auto-detected by walking up to the nearest `docker-compose.yml`; force
it with `--root` or `AGENTX_MANAGER_ROOT`.

## The web GUI

Cluster cards with live phase badges (**initializing** is first-class — an amber pulse, not a
false alarm), CPU/memory gauges (5s polling), per-service state + usage drill-in, lifecycle
buttons, a live log stream with service filter, typed-confirmation destroy, and (repo mode)
a New Cluster form that shows generated secrets exactly once.

Lifecycle actions run as background jobs — the GUI stays responsive and toasts the result.

## Security posture

!!! danger "The manager is root-equivalent"
    It drives the Docker socket. Accordingly: it binds **127.0.0.1** by default and refuses a
    non-loopback bind without an explicit `AGENTX_MANAGER_BIND`; every `/api` request requires
    the bearer token (printed on first start, stored at `<root>/.manager-token`, mode 0600, or
    supplied via `AGENTX_MANAGER_TOKEN`); and it must **never** be routed through the
    gateway/tunnel — the shipped nginx template has no route to it. For remote access use SSH
    port forwarding: `ssh -L 12320:127.0.0.1:12320 <host>`.

## Running it

**Repo (dev):**

```bash
task manager:serve            # GUI at http://127.0.0.1:12320
task cluster:new CLUSTER=prod GATEWAY=1   # thin wrapper over agentx-manager new
```

**Bundle (isolated):** the bundle ships `docker-compose.manager.yml`; see the
[guided setup](self-hosting.md#guided-setup-the-manager-gui) in the self-hosting guide.

!!! note "Containerized manager: the same-path mount"
    Compose resolves bind-mount paths on the **client** side, so the manager container must see
    the deployment root at the **same absolute path** as the host — that's the
    `${PWD}:${PWD}` mount in `docker-compose.manager.yml`. Always run its compose commands from
    the deployment root; the CLI validates this at startup and fails loudly otherwise.

## Testing

The manager is covered by a unit suite (overlay truth table, scaffold rendering, config-hash
change detection, registry parsing, server auth — all against a recorded fake compose runner)
and a Docker integration suite (`task manager:test:integration`) that stands up the **real
nginx gateway** with a lightweight fake API upstream and asserts the token gate (401/200),
rate-limit 429s, and the edit-config→restart→live behavior end-to-end.

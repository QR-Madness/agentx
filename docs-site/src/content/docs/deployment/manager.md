# Deployment Manager

The **deployment manager** (`agentx-manager`) owns cluster lifecycle for both deployment
flavors — local source-built clusters and isolated image deployments — through one tested
core, exposed as a **CLI** and a **web GUI**. You never assemble `docker compose -f …`
chains by hand.

| | |
|---|---|
| CLI | `agentx-manager` — repo: `uv run --project manager agentx-manager`, or the `task cluster:*` wrappers · bundle: `docker compose exec manager agentx-manager …` |
| GUI | `http://127.0.0.1:12320` — `task manager:serve` (repo) or the `manager` compose profile (bundle) |
| Image | `qrmadness/agentx-manager` (~370 MB; python-slim + docker CLI + built GUI) |

!!! note "Which one am I?"
    **Repo checkout** — you cloned the source and run things with `task`/`uv`. **Bundle** — you
    downloaded `agentx-deploy.tar.gz` and only have Docker; see
    [Self-Hosting](self-hosting.md). Everything below shows the repo-mode command first; if
    you're on a bundle, prefix it with `docker compose exec manager` (see the CLI reference
    below) and run it from the folder your `.env`/`docker-compose.yml` live in.

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
  `docker stats` (CPU %, memory % + bytes, network ↓/↑ totals — plus a loose live ↓/↑ rate in
  the GUI, handy for watching the first-boot model download) over the cluster's compose
  project, with per-service breakdown. Status and usage cover only the app stack's own
  services — the manager's own container (which shares the bundle's compose project by
  design) is excluded, so a stack that isn't running reads **down**, not degraded.
- **Destroy, safely.** `agentx-manager destroy <name> --confirm <name>` = `down --volumes` +
  data-directory removal; the GUI requires typing the cluster name. Bundle mode never deletes
  the deployment root itself.
- **Day-2 passthrough.** `agentx-manager agentx <name> -- migrate|setup-auth|warmup|status|…`
  execs the in-image ops CLI, so behavior always matches the deployed image.

## CLI reference

Shown in **repo mode** form (`agentx-manager <command>`, e.g. via `uv run --project manager
agentx-manager <command>`). **On a bundle**, run the exact same command inside the container
instead — prefix it with `docker compose exec manager`:

```bash
# repo mode                                  # bundle mode: docker compose exec manager …
agentx-manager list                          # deployments + spec tags
agentx-manager status [name]                 # phase (down|initializing|degraded|up) + services
agentx-manager usage  [name]                 # CPU% / memory / network ↓↑ per cluster (docker stats)
agentx-manager new <name> [--gateway] [--tunnel token|named] [--gpu] [--kind source|image]
agentx-manager gateway-enable [name] [--tunnel …]
agentx-manager up|down|restart|rebuild [name]
agentx-manager set [name] --gateway/--no-gateway --tunnel … --gpu/--no-gpu --shell/--no-shell --expose/--no-expose
agentx-manager adopt [name]                  # one-time move onto the agentx-<name> project
agentx-manager destroy <name> --confirm <name> [--keep-data]
agentx-manager logs [name] [service]
agentx-manager agentx [name] -- <ops-cli command>
agentx-manager serve [--host …] [--port …]   # the web GUI / REST server — this is the one
                                              # already running as the bundle's `manager`
                                              # container; you won't run this one yourself
```

For example, on a bundle, checking cluster status is:

```bash
cd agentx-deploy   # wherever your .env / docker-compose.yml live
docker compose exec manager agentx-manager status
```

In **bundle mode** (an untarred deploy bundle is exactly one deployment) the name is optional
everywhere — `agentx-manager status` alone is enough, no cluster name needed. The root is
auto-detected by walking up to the nearest `docker-compose.yml`; force it with `--root` or
`AGENTX_MANAGER_ROOT`. For almost everything day-to-day, the **web GUI at
http://127.0.0.1:12320 is the easier path** — the CLI is mainly useful for scripting or when
you can't open a browser to the host.

## The web GUI

Cluster cards with live phase badges (**initializing** is first-class — an amber pulse, not a
false alarm), CPU/memory gauges (5s polling), per-service state + usage drill-in, lifecycle
buttons, a live log stream with service filter, typed-confirmation destroy, and (repo mode)
a New Cluster form that shows generated secrets exactly once.

**Share** (per cluster) builds a [connection link](clusters.md#sharing-access) without
touching `.env` by hand: server URL (prefilled from `AGENTX_PUBLIC_HOST`, editable), display
name, and the web-app base where your deployed client lives (remembered per browser). The
link embeds the cluster's gateway token — treat it like a secret; the recipient's password is
never included. The modal warns inline about the classic foot-guns: a localhost URL that
won't travel, a missing gateway token, a web-app origin absent from `CORS_ALLOWED_ORIGINS`
(browsers would block every call), and auth being disabled.

**Enable gateway…** appears on cards without a token gateway: pick a tunnel flavor
(none / Cloudflare token / named), and it scaffolds `nginx.conf` + generates
`AGENTX_GATEWAY_TOKEN` (shown once), same as `task cluster:gateway:enable`. Run **Up** to
apply — Restart won't create the gateway service or re-read env.

**Version chips** keep the build→tag→restart loop honest: the header shows the checkout's
`versions.yaml` version (repo mode), each running cluster shows the version its API actually
reports, and an amber `checkout v…` chip appears when the two differ (rebuild, then Up, to
roll the image out). The manager's own version lives in the muted header caption
(`manager v0.x.y`) — it versions independently of the platform.

In bundle mode the GUI drops the docker vocabulary entirely and becomes a plain-language
single-deployment dashboard: a status hero (**Stopped / Starting up… / Running / Needs
attention**) with **Start / Stop / Restart** and an **Open AgentX** link, resource tiles
(processor, memory, network — including a live ↓ rate while the first start downloads AI
models), a components list with friendly names (AgentX Server, Knowledge Graph, Memory
Store, Cache…), connection details, and a typed-confirmation **Reset**. A fresh bundle
reads **Stopped** until you press Start; creating additional clusters stays a
source-checkout (repo mode) feature. Polling pauses automatically in background tabs.

Lifecycle actions run as background jobs — the GUI stays responsive and toasts the result.

## Security posture

!!! danger "The manager is root-equivalent"
    It drives the Docker socket. Accordingly: it binds **127.0.0.1** by default and refuses a
    non-loopback bind without an explicit `AGENTX_MANAGER_BIND`; every `/api` request requires
    the bearer token (printed on first start, stored at `<root>/.manager-token`, mode 0600, or
    supplied via `AGENTX_MANAGER_TOKEN`); and it must **never** be routed through a
    **cluster's** gateway/tunnel — the shipped nginx template has no route to it. For remote
    access use SSH port forwarding: `ssh -L 12320:127.0.0.1:12320 <host>`.

!!! note "Remote access over a dedicated tunnel"
    If SSH isn't convenient (e.g. switching clusters from a phone), a Cloudflare Tunnel can
    front the manager — but treat it as the sensitive service it is:

    - **Keep the loopback bind.** Run a *host-level* `cloudflared` (or a standalone tunnel
      container with `extra_hosts: ["host.docker.internal:host-gateway"]`) whose ingress
      points at `http://127.0.0.1:12320` (`http://host.docker.internal:12320` from a
      container). No `AGENTX_MANAGER_BIND` needed, and no CORS setup either — the GUI is
      served by the manager itself, so everything stays same-origin.
    - **Give the manager its own tunnel**, not a hostname on a cluster's `cloudflared` —
      otherwise downing that cluster takes your remote control plane with it.
    - **Front the hostname with Cloudflare Access** (email OTP / identity provider). The
      bearer token is strong, but a root-equivalent endpoint on the public internet deserves
      a second gate; with Access in place the token becomes your second factor.
    - Side benefit: the tunnel's HTTPS makes the browser treat the GUI as a secure context,
      so Share-modal clipboard copies work remotely.

## Running it

**Repo (dev):**

```bash
task manager:serve            # GUI at http://127.0.0.1:12320
task cluster:new CLUSTER=prod GATEWAY=1   # thin wrapper over agentx-manager new
```

**Bundle (isolated):** the manager is the only thing in the bundle's default
`COMPOSE_PROFILES` — a plain `docker compose up -d` starts *only* it. Use the
dashboard's **Up** button (or `docker compose exec manager agentx-manager up`)
to bring up the actual stack; see
[Self-Hosting, step 3](self-hosting.md#3--open-the-dashboard-and-bring-up-the-stack).

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

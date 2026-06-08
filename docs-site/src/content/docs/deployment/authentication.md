# Authentication

AgentX ships with optional, single-user authentication (Phase 17). It is **disabled by
default** — a fresh install needs no login. Enable it when exposing a server beyond your local
machine.

## Model

- **One root user.** AgentX is single-tenant: one server = one user. There is a single `root`
  account, no registration.
- **Password**: hashed with bcrypt (cost 12) and stored in the PostgreSQL `agentx_auth` table
  (`api/agentx_ai/auth/service.py`, schema in `queries/postgres_builder.sql`).
- **Sessions**: a 32-byte URL-safe token stored in Redis under `agentx:session:{token}`. The
  session carries `user_id`, `username`, `created_at`, `last_active`, and the client IP/UA.
- **TTL**: `AGENTX_SESSION_TTL` (default `86400`, 24h). The TTL is **extended on every
  validated request**, so active sessions don't expire mid-use.

## Enabling auth

```bash
AGENTX_AUTH_ENABLED=true     # gate all /api/* routes
AGENTX_SESSION_TTL=86400     # session lifetime in seconds (default 24h)
```

Then set the root password:

```bash
task auth:setup          # interactive prompt
task auth:setup:force    # reset an existing password (invalidates all sessions)
task auth:check          # report whether setup is required
```

In containerized deployments use the wrapper `cluster:auth:setup CLUSTER=<name>` (see
[Clusters](clusters.md)), which runs `setup_auth` inside the cluster's API container.

## How requests are gated

When auth is enabled, `AgentXAuthMiddleware` (`auth/middleware.py`) requires a valid session
token on every `/api/*` route, sent in the **`X-Auth-Token`** header. Missing or invalid
tokens get `401`.

A few routes stay public so a client can bootstrap: `/api/health`, `/api/version`,
`/api/auth/login`, `/api/auth/status`, and `/api/auth/setup` (only while setup is required).

!!! note "Localhost bypass"
    When `DEBUG` is on and `AGENTX_AUTH_BYPASS_LOCALHOST` is `true` (the default), requests
    from `127.0.0.1` / `::1` skip auth — convenient for local development. Disable it (or run
    with `DJANGO_DEBUG=false`) for any real deployment.

On the client, the token is stored per-server in localStorage (`agentx:server:{id}:authToken`)
and attached to every request by the API layer (`client/src/lib/api/core.ts`). `AuthContext`
drives login/logout/setup and surfaces `authRequired` / `setupRequired` state.

## Endpoints

Full request/response bodies are in the
[API reference → Authentication](../api/endpoints.md#authentication). In brief:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/status` | GET / POST | Is auth enabled? Is setup complete? |
| `/api/auth/setup` | POST | Set the initial root password |
| `/api/auth/login` | POST | Exchange the password for a session token |
| `/api/auth/logout` | POST | Destroy the current session |
| `/api/auth/session` | GET | Validate / inspect the current session |
| `/api/auth/change-password` | POST | Change the password (invalidates other sessions) |

## Encrypted log archives

Setting a password does more than gate the API: it also **encrypts your durable logs at
rest**. The daily log archive (`data/logs/agentx-YYYY-MM-DD.log.gz`) is sealed with
**AES-256-GCM** so on-disk history is unreadable without your password — defense-in-depth on
top of the secret-redaction that already happens when each line is captured.

It uses **envelope encryption**: a random data key (DEK) encrypts the archives, and your
password only derives a key that *wraps* that DEK in `data/logs/keyring.json`. The practical
consequences:

- **The hot path never needs your password.** The current day is written as a redacted
  plaintext gzip; completed days are **sealed** the moment you log in (the DEK is unwrapped and
  cached in server memory). Days that roll while no one is logged in stay redacted-plaintext
  until the next login or a manual `task logs:seal`.
- **Changing your password is instant.** `change-password` just re-wraps the small key — no
  archive is rewritten. Use `task logs:rotate-keys:deep` for a full re-encrypt under a brand-new
  data key (e.g. if you believe the old one leaked).
- **Lost password ⇒ unrecoverable archives**, by design.
- **Auth disabled (the default) ⇒ no keyring**, so archives stay redacted-plaintext gzip exactly
  as before. Encryption activates only once a password exists. (`AGENTX_LOG_ARCHIVE_ENCRYPT=false`
  forces it off even with auth on.)

Downloading a sealed segment from the Log panel decrypts it on the fly; if the server was
restarted and no one has logged back in, the vault is **locked** and the download returns `423`
until you re-authenticate. Retention defaults to 30 days (`AGENTX_LOG_ARCHIVE_RETENTION_DAYS`).

```bash
task logs:keys:status        # keyring present? unlocked? sealed/pending counts
task logs:seal               # seal any pending plaintext days now (prompts for password)
task logs:rotate-keys        # re-wrap the key under a new password (O(1))
task logs:rotate-keys:deep   # deep rotation: new data key + re-encrypt every segment
```

See the [API reference → Logs archive](../api/endpoints.md#archive) for the segment/status
endpoints.

## Version Compatibility

`versions.yaml` is the single source of truth for the API/client versions and the wire
**protocol version**:

```yaml
api:
  version: "0.20.0"
  protocol_version: 1
  min_client_version: "0.20.0"
client:
  version: "0.20.0"
```

`GET /api/version` (public) returns `{version, protocol_version, min_client_version}` — the
same fields are also embedded in `/api/health`. At startup the client (`AuthContext` +
`lib/api/version.ts`) probes `/api/version` and checks two things:

1. **Protocol** — `protocol_version` must match the client's exactly. `protocol_version` is
   bumped only on breaking API changes.
2. **Minimum client** — the client's semver must be `>= min_client_version`.

If either check fails the app shows `VersionMismatchPage` (with a retry) instead of
connecting, so an out-of-date client never talks to an incompatible server.

!!! note "Gateway token"
    Deployments fronted by the cluster gateway also send an `AgentX-Gateway-Token` header,
    validated by Nginx before the request reaches Django. That is separate from user auth —
    see [Clusters → Gateway](clusters.md#gateway-nginx--cloudflare).

# Hosting Handbook

The operator's reference for running your own AgentX: the URLs and endpoints you point at your
deployment, and the third-party app setup a few connectors need. Everything here assumes
**self-hosting** — if you just want to *use* AgentX, the
[Get Started](../getting-started/quickstart.md) guides are enough.

For the complete environment-variable reference, see
[Configuration](../getting-started/configuration.md); this page gathers the pieces that
specifically depend on *where* and *how* you host.

## Custom endpoints & URLs

Self-hosting means a handful of values point at your own host instead of `localhost`. Set them in
the API's `.env` (the [Configuration](../getting-started/configuration.md) reference lists them all):

| Setting | What it points at |
|---------|-------------------|
| `VITE_API_URL` | The API URL the client talks to — e.g. `https://<host>/api` |
| `AGENTX_OAUTH_REDIRECT_URL` | The OAuth callback base for remote connectors — `https://<host>/api/mcp/oauth/callback` |
| `LMSTUDIO_BASE_URL` | Your local-model endpoint, if you run [LM Studio](../features/providers.md) |
| `DJANGO_ALLOWED_HOSTS` | The hostnames the API answers on |

Behind a token **gateway**, the browser can't carry the gateway token through an OAuth redirect, so
the gateway passes exactly `/api/mcp/oauth/callback` through tokenless (the API validates it by
state). See [Clusters & Gateway](clusters.md) for the gateway setup.

## Connector OAuth apps

Most [connectors](../features/mcp.md) that use OAuth register themselves automatically. A few need a
**pre-registered OAuth app in your own cloud project** — there's no dynamic registration — so you
supply a client ID and secret.

### Google Workspace

Google Drive and the other Google Workspace servers (Docs, Gmail, Slides, …) sit behind a
**Coming soon** badge because they're part of Google's
[Workspace Developer Preview Program](https://developers.google.com/workspace/preview): enrollment
needs a Google **Workspace** account and a registered Cloud project — personal accounts can't
enroll, and approval takes days. Until you're enrolled, sign-in *succeeds* but every tool call
fails with `The caller does not have permission`.

Once enrolled, add the server manually and register a pre-registered OAuth app:

1. In the Google Cloud console, create or select a project; enable the **Google Drive API** and the
   **Google Drive MCP API**.
2. On the OAuth consent screen, add the scopes `…/auth/drive.readonly` and `…/auth/drive.file`.
3. Create an OAuth client ID (Web application) and register AgentX's callback as an authorized
   redirect URI: `http://localhost:12319/api/mcp/oauth/callback` (adjust for your host via
   `AGENTX_OAUTH_REDIRECT_URL`).
4. Set `GOOGLE_DRIVE_CLIENT_ID` / `GOOGLE_DRIVE_CLIENT_SECRET` in the API's `.env` — the catalog
   references them as `${VAR}` so secrets never land in `mcp_servers.json` — or paste them into the
   connector dialog.

**One app, many clusters.** A single OAuth client can list every cluster's
`https://<host>/api/mcp/oauth/callback` as an authorized redirect URI, so you reuse the same client
ID and secret across deployments and let each cluster's `AGENTX_OAUTH_REDIRECT_URL` route consent
back to the right host.

## Where to go next

- [Configuration](../getting-started/configuration.md) — the complete environment-variable reference.
- [Self-Hosting](self-hosting.md) — the Docker-image deployment path and the ops dashboard.
- [Clusters & Gateway](clusters.md) — multiple instances behind one token gateway.
- [Authentication](authentication.md) — logins, sessions, and the gateway token.

# Troubleshooting

The problems people most often hit when setting up and using AgentX, and how to fix them. Each
entry links to the page that covers it in more depth. Commands that start with `docker compose`
run in the folder you unpacked the server package into.

## Setting up the server

**Docker Desktop won't start the stack, or refuses a folder.**
Docker Desktop (Windows and macOS) only mounts folders it shares. Unpack the package under your
home directory, not `/tmp`. See [Self-Hosting](../deployment/self-hosting.md).

**The dashboard stays on "Starting up…" for a long time.**
The first start downloads a ~2.3 GB embedding model and sets up the databases, which takes a few
minutes; the dashboard shows the download rate. If it looks stuck, the dashboard's
**Activity Log** shows exactly where it is.

**Containers stop or restart on their own.**
Check that the machine has enough free memory: the stack reserves about 7 GB. On Docker
Desktop, make sure Docker itself is allowed that much. The dashboard's per-service logs show
which service stopped and why.

## Connecting a client

**The desktop app can't reach the server.**
First check that the dashboard shows **Running**. On the same machine, the address is
`http://localhost:12319`, which the app uses by default. From another machine you need an
address it can reach; [Going public](../deployment/self-hosting.md#going-public) sets one up.

**The web client fails on every request.**
The browser blocks requests unless your server allows the web client's address. Add this to
`.env`, then apply it with **Up** in the dashboard (a restart doesn't re-read `.env`):

```bash
CORS_ALLOWED_ORIGINS=https://agx.thejpnet.net
```

The server also needs an HTTPS address; see
[Going public](../deployment/self-hosting.md#going-public).

**Every request returns `401` after you go public.**
The gateway didn't receive a valid gateway token. Check that the client's **Gateway token**
matches `AGENTX_GATEWAY_TOKEN` in `.env`. The Going public section lists the other gateway
errors (`429`, `502`, `400`).

## Signing in

**You forgot the root password.**
Reset it on the server:

```bash
docker compose exec api agentx setup-auth --force
```

This signs out every session. If log archive encryption is on, archives sealed under the old
password can't be opened again; see
[Encrypted log archives](../deployment/authentication.md#encrypted-log-archives).

## Models

**The model picker is empty.**
Add a provider key in **Settings → Providers**. Models are discovered live from each provider,
so the list fills in once a key works. See [Providers](../features/providers.md).

**LM Studio on the same machine isn't found.**
Inside the server's container, `localhost` is the container itself. In
**Settings → Providers**, set LM Studio's address to `http://host.docker.internal:1234/v1`. On
Linux, also let LM Studio serve on your local network.

**The Temperature slider does nothing with a Claude model.**
That's expected. Current Claude models accept only their default sampling and reject a
temperature, so AgentX doesn't send one to the Anthropic provider. Shape tone with the system
prompt or a thinking pattern instead. See [Agent Profiles](../features/agent-profiles.md).

## Finding the logs

- **In the app:** open the command palette (`⌘K` / `Ctrl K`) and choose **Open Logs**.
- **In the dashboard:** each service has its own log view.
- **On the command line:** `docker compose logs -f api` follows the API's output.

Still stuck? [Open an issue](https://github.com/QR-Madness/agentx/issues) with what you tried
and the relevant log lines.

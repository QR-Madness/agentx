# Quick Start

AgentX runs as two parts: a **server** on a machine you control, which holds your agents, their
memory and their tools, and a **client** you chat through, either the desktop app or the web
client in a browser. This page gets both running and ends at your first conversation. If a term
is new to you, [Concepts](concepts.md) explains it in a sentence or two.

## Before you start

- **A machine for the server, with [Docker](https://docs.docker.com/get-started/get-docker/).**
  On Windows and macOS that means Docker Desktop; on Windows it runs through WSL 2.
- **About 8 GB of free memory.** The stack reserves roughly 7 GB, most of it for the API; 16 GB
  is comfortable. On Docker Desktop, make sure its VM is allowed that much.
- **Disk space** for the container images, plus the ~2.3 GB embedding model that the first start
  downloads.
- **A model to talk to:** an API key from Anthropic, OpenAI, OpenRouter or the Vercel AI
  Gateway, or a model you run yourself with [LM Studio](https://lmstudio.ai/). You add it in the
  app once the server is up.

## 1 · Start the server

Download **`agentx-deploy.tar.gz`** from the
[latest release](https://github.com/QR-Madness/agentx/releases/latest), then:

```bash
tar xzf agentx-deploy.tar.gz && cd agentx-deploy
cp .env.example .env      # no edits needed: secrets are generated on the first start
docker compose up -d      # starts the deployment dashboard
```

On Windows, double-click **`start-manager.bat`** in the unpacked folder instead.

Open the dashboard at **http://127.0.0.1:12320**, paste the access token from
`./.manager-token`, and click **Start AgentX**. The first start takes a few minutes while the
model downloads; the dashboard shows the progress and switches to **Running** when the server is
ready. [Self-Hosting](../deployment/self-hosting.md) covers each step in more detail.

## 2 · Connect a client

**The desktop app** is the simplest choice when the server runs on the same machine. Download
the installer for Windows or Linux from the
[latest release](https://github.com/QR-Madness/agentx/releases/latest) and open it. It looks for
a server at `http://localhost:12319`, so on the same machine there is nothing to enter; for a
server elsewhere, enter its address. On the first run, the setup screen asks you to set the root
password.

**The web client** at [agx.thejpnet.net](https://agx.thejpnet.net) needs no install. Use it once
your server has an HTTPS address, which [Going public](../deployment/self-hosting.md#going-public)
sets up. Before you connect, allow the web client in your server's `.env`, or the browser blocks
every request:

```bash
CORS_ALLOWED_ORIGINS=https://agx.thejpnet.net
```

Then apply it with **Up** in the dashboard; a restart doesn't re-read `.env`.

## 3 · Your first conversation

1. Open **Settings → Model Providers** and add your API key. Keys are stored on your server, never
   in the client.
2. Open **Settings → Model Roles** and set the **Global Default Model**. Every agent without a
   model of its own uses it; without one, new conversations fall back to LM Studio and fail if
   you haven't set it up. You can still pick another model for a single conversation from the
   model chip under the message box.
3. Start a chat with **AgentX**, the general assistant. Every install starts with three agents:
   AgentX, **Researcher** and **Deluxe Image Creator**. [Agent Profiles](../features/agent-profiles.md)
   shows how to change them or build your own.

[Your First Session](first-session.md) walks through these steps with screenshots, then makes an
agent of your own and shows memory at work.

!!! tip "Running LM Studio on the same machine?"
    The server runs inside Docker, where `localhost` means the container itself. In
    **Settings → Model Providers**, set LM Studio's address to `http://host.docker.internal:1234/v1`.
    On Linux, also let LM Studio serve on your local network, because the container reaches your
    machine through Docker's network rather than its loopback address.

Something not working? [Troubleshooting](troubleshooting.md) covers the usual fixes.

## Go further

| You want to… | Go to |
|---|---|
| Give an agent its own model, personality and tools | [Agent Profiles](../features/agent-profiles.md) |
| Connect tools and services | [Connectors & Tools](../features/mcp.md) |
| Let agents hand work to each other | [Agent Teams](../features/multi-agent.md) |
| See what AgentX remembers, and why | [Memory](../features/memory.md) |
| Reach your server from anywhere | [Going public](../deployment/self-hosting.md#going-public) |
| Understand sign-in and the gateway token | [Authentication](../deployment/authentication.md) |
| Change environment variables and config files | [Configuration](configuration.md) |
| Run several instances | [Clusters & Gateway](../deployment/clusters.md) |
| Build AgentX from source | [Development Setup](../development/setup.md) |
| Script against the HTTP API | [API Endpoints](../api/endpoints.md) |

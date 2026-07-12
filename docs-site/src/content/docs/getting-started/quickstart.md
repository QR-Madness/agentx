# Quick Start

AgentX is two pieces: a **server** you run (agents, memory, tools) and a
**desktop client** you point at it. Getting up and running is three moves:

1. **Run a server** — pick your path below.
2. **Connect the client** — installer + your server's URL.
3. **Go further** — expose it, tune it, or build on it.

## 1 · Run a server — pick your path

| Path | Best for | Guide |
|---|---|---|
| **Self-host** — Docker image + web dashboard, nothing else on the host | Running AgentX to *use* it | **[Self-Hosting](../deployment/self-hosting.md)** |
| **From source** — hot-reloading full stack (`task dev`) | Hacking on AgentX itself | **[Installation](installation.md)** |
| **Local clusters** — several isolated prod-like instances, one dashboard | Power users & staging | **[Clusters & Gateway](../deployment/clusters.md)** |

The self-host path is three commands, and the bundled
**[deployment manager](../deployment/manager.md)** takes it from there — watch the
first boot progress live, stream logs, start/stop, and see resource usage from a
dashboard instead of a terminal:

```bash
tar xzf agentx-deploy.tar.gz && cd agentx-deploy
cp .env.example .env      # fill 3 values — the file tells you which
docker compose up -d      # then open http://127.0.0.1:12320
```

## 2 · Connect the client

Grab an installer from the
**[latest release](https://github.com/QR-Madness/agentx/releases/latest)**
(Windows / Linux), enter your server's URL on first run, and set the root password
from the built-in setup screen. Done — you're chatting.

## 3 · Go further

| You want to… | Go to |
|---|---|
| Reach your server from the internet (token gateway + tunnel) | [Going public](../deployment/self-hosting.md#going-public) |
| Understand logins, sessions, and the gateway token | [Authentication](../deployment/authentication.md) |
| Tweak environment variables & config files | [Configuration](configuration.md) |
| Drive everything from the ops dashboard / CLI | [Deployment Manager](../deployment/manager.md) |
| Script against the HTTP API | [API Endpoints](../api/endpoints.md) |
| See how it all fits together | [Architecture Overview](../architecture/overview.md) |
| Explore features (chat, memory, MCP tools, translation, prompts) | [Chat](../features/chat.md) · [Memory](../features/memory.md) · [MCP](../features/mcp.md) · [Translation](../features/translation.md) · [Prompts](../features/prompts.md) |

!!! tip "Windows?"
    Developing on Windows has a few platform notes — see
    [Windows Setup](windows.md).

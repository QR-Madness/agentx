# Connectors & Tools

AgentX is an **MCP (Model Context Protocol) client** — it connects to external tool servers that
give agents real-world reach: web search, filesystems, code hosts, docs, design and productivity
apps, payments, and more. The **Connectors & Tools** page is the control center: your connected
servers, a curated connector catalog, live search of the public MCP registry, the discovered
tools, per-agent tool access, and the skill library.

## The connector catalog

The fastest way to give agents real-world reach. Connectors are grouped into three
**intelligence lenses**, so your agents stay well-rounded across the kinds of intelligence they
need:

- **Global Intelligence** — the world beyond your code: web & search (Exa), research & reference
  (arXiv, Wikipedia, Hugging Face).
- **Technical Intelligence** — software & systems: docs & APIs (AWS Knowledge, Context7,
  Microsoft Learn, Cloudflare Docs, DeepWiki) and code & dev (GitHub, Sentry, Vercel, Playwright,
  Filesystem).
- **Workspace & Apps** — your tools & content: productivity (Notion, Linear, Atlassian, Asana,
  monday.com, Zapier), design (Figma, Canva), payments (Stripe, PayPal), storage (Google Drive),
  and local knowledge-graph memory.

Each connector is a tile that shows at a glance how it signs in — from no sign-in, through
OAuth, to an API key — and whether you've added it. Click one to open a single dialog that walks
the whole lifecycle: guided quick-add with only the fields that connector needs (OAuth chains
straight into browser sign-in), live status and **Connect** once it's added, or the reason it's
still gated. Deeper server management — rename, tool access, reset auth, remove — lives in the
**Servers** section above the catalog.

The **search box filters the catalog live**; when what you want isn't there, the same box falls
back to the official **MCP registry** (`registry.modelcontextprotocol.io`) and maps any result
into a prefilled server form (remote endpoints directly; npm / PyPI / OCI packages as `npx` /
`uvx` / `docker run` commands). Registry entries are community-published — review the commands
and URLs before saving.

### Google Workspace connectors

Google Drive and the other Google Workspace servers sit behind a **Coming soon** badge: they're in
Google's Developer Preview and need a self-registered OAuth app in your own cloud project. If
you're self-hosting and enrolled, the
[Hosting Handbook](../deployment/hosting-handbook.md#google-workspace) walks through the setup.

## Adding a server by hand

Beyond the catalog, servers live in `mcp_servers.json` at the project root (copy
`mcp_servers.json.example`). A server names its **transport** and how to reach it:

```json
{
  "filesystem": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
  }
}
```

Values prefixed with `$` in `env` and `headers` resolve from the environment at connect time, so
secrets stay out of the file.

| Transport | Use case | Key fields |
|-----------|----------|------------|
| `stdio` | Local process servers (most common) | `command`, `args`, `env` |
| `sse` | Remote servers over Server-Sent Events | `url`, `headers` |
| `streamable_http` | Remote HTTP servers | `url`, `headers` |

## Per-agent tool access

Every [agent profile](agent-profiles.md) can narrow which tools it sees: an **allow-list**
(`allowed_tools` — only these are exposed to the model) or a **block-list** (`blocked_tools` —
these are hidden). With neither set, the agent gets every tool from its connected servers. This
is how a specialist stays focused and a delegate can't reach beyond its remit.

## OAuth for remote connectors

Remote servers can require **OAuth 2.1**, and AgentX handles the whole dance — discovery,
dynamic client registration (or a pre-registered client for providers like Google), PKCE, and
token refresh — opening your browser for consent on first connect. Tokens persist per server on
the API host.

The server card tells you where a connection stands: signed in, refreshing on the next connect,
or expired and needing a fresh browser sign-in (which also raises the new-conversation nudge).
**Reset auth** forgets stored tokens for a clean start.

## Under the hood

The agent's [tool loop](chat.md#tool-use-loop) reaches connectors through a `ToolExecutor` and a
persistent `MCPClientManager` that keeps connections alive across requests over stdio, SSE, or
streamable HTTP. See the
[MCP client architecture](../architecture/system-design.md#mcp-client-architecture) and the
[tool-execution flow](../architecture/system-design.md#mcp-tool-execution) on the System Design
page. The programmatic surface — listing servers and tools, connecting, and the registry proxy —
is in the [API Reference](../api/endpoints.md#mcp-model-context-protocol).

## Related

- [Chat](chat.md) — how tools run inside a turn
- [Agent Profiles](agent-profiles.md) — per-agent tool access
- [Architecture Overview](../architecture/overview.md) — where MCP sits in the system

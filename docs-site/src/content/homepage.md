# AgentX

> A self-hosted, glassbox Cognitive OS: persistent memory, Agent Teams that collaborate,
> reasoning, and MCP tools — on any model you bring, behind a REST API you run yourself.

AgentX is an open-source (MIT) agent platform you run on your own infrastructure. There is
no SaaS tier and no vendor lock-in: you own the server, the data stores, and the models.

This is the Markdown rendition of the AgentX home page, served through `Accept` content
negotiation. For a complete, machine-readable index of the documentation see
[/llms.txt](/llms.txt); every page below also has a `.md` twin.

## What makes it different

**It thinks.** A mind you can watch work — reasoning, memory that learns, and a context
window that never overflows.

- **Glassbox, not a black box** — the reasoning, every tool call and result, and each
  delegation stream to screen live. Nothing happens in the dark.
  See [System design](/docs/architecture/system-design.md).
- **Memory that actually learns** — four memory types (episodic, semantic, procedural,
  working) with five recall techniques and background consolidation, so agents remember
  and sharpen across conversations. See [Memory](/docs/features/memory.md).
- **Context that never overflows** — oversized tool outputs are compressed and stored for
  retrieval, and Focus-style trajectory compression keeps long tool loops inside the
  window. See [Chat](/docs/features/chat.md).
- **Prompts you compose** — layered, reusable prompt sections over a global base, with an
  LLM-backed enhancer. See [Prompts](/docs/features/prompts.md).

**It works as a team.** Not one overloaded assistant — a roster of specialists that
delegate, coordinate, and keep you in the loop.

- **Agent Teams that collaborate** — a lead hands subtasks to specialist teammates over a
  shared memory channel, with @-mention routing, ad-hoc agent-to-agent delegation, and
  per-agent tool isolation. See [Multi-agent](/docs/features/multi-agent.md).
- **Ambassador — a parallel narrator** — a dedicated agent runs alongside your
  conversation and briefs you on it, by voice or text, without ever entering the
  transcript.
- **Tools in one line** — connect external tool servers over MCP, or turn any Python
  function into an agent tool with a single `@register_tool` decorator.
  See [MCP](/docs/features/mcp.md).

**It's yours.** You own the whole stack — the server, the models, and every platform it
runs on.

- **Self-hosted, bring your own models** — one interface across LM Studio, Anthropic,
  OpenAI, OpenRouter, and Vercel AI Gateway. See [Providers](/docs/features/providers.md).
- **Runs where you do** — one Tauri v2 client across Windows, macOS, and Linux, an Android
  build, and a browser/PWA mode. See [Installation](/docs/getting-started/installation.md).
- **Open source, yours to keep** — MIT-licensed and built in the open at
  [github.com/QR-Madness/agentx](https://github.com/QR-Madness/agentx).

## Subsystems

| Subsystem | What it does |
|-----------|--------------|
| [Agent](/docs/architecture/overview.md) | Per-request orchestrator. Sessions, context budgeting, tool loop, output parsing. |
| [Reasoning](/docs/features/reasoning.md) | Six thinking patterns — native, step-by-step, step-back, reflection, deep reflection, consensus — auto-selected per task, plus a Tree-of-Thought / ReAct kit for offline runs. |
| [Drafting](/docs/features/reasoning.md) | Speculative decoding, multi-stage pipelines, N-best candidates. |
| [MCP Client](/docs/features/mcp.md) | Connect to external tool servers over stdio, SSE, or streamable HTTP. |
| [Providers](/docs/features/providers.md) | One interface across five providers — swap models per request. |
| [Prompts](/docs/features/prompts.md) | Profile-based composition with a global prompt layer; sections compose at runtime. |
| [Memory](/docs/features/memory.md) | Four memory types, five recall techniques, per-agent self-knowledge, background consolidation. |
| [Orchestration](/docs/features/multi-agent.md) | Agent Teams — a lead hands subtasks to specialist teammates over a shared channel. |

## Stack

| | |
|---|---|
| Deploy | Self-hosted · your infra |
| Backend | Django 5.2 (REST API) |
| Client | Tauri v2 · React 19 |
| Memory | Neo4j (entity graphs) · PostgreSQL + pgvector (vectors) · Redis (working memory) |
| License | MIT |

## Start here

- [Quickstart](/docs/getting-started/quickstart.md) — get a server and client running.
- [Installation](/docs/getting-started/installation.md) — desktop, mobile, and web.
- [Configuration](/docs/getting-started/configuration.md) — providers, keys, and settings.
- [Architecture overview](/docs/architecture/overview.md) — how a turn actually executes.
- [Roadmap](/docs/roadmap.md) — what is shipped, shipping, and next.

## For agents

- [/llms.txt](/llms.txt) — index of every documentation page, as Markdown.
- [/openapi.yaml](/openapi.yaml) — machine-readable REST API contract.
- [/.well-known/api-catalog](/.well-known/api-catalog) — RFC 9727 linkset.
- [API endpoints](/docs/api/endpoints.md) — the endpoint reference in prose.

Every HTML page on this site also answers `Accept: text/markdown` with its Markdown twin,
and advertises it via `<link rel="alternate" type="text/markdown">`.

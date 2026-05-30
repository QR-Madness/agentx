# Integration Overview

AgentX is API-first: the desktop client is just one consumer of the same REST API your own
code can call. The backend runs as a Django service (default `http://localhost:12319`) and the
API is the integration surface for everything — chat, memory, agents, tools, and config.

## The basics

| | |
|---|---|
| **Base URL** | `http://localhost:12319/api/` |
| **Format** | JSON in, JSON out. `POST` bodies are `application/json`. |
| **CORS** | Preflight (`OPTIONS`) is handled automatically. |
| **Errors** | Failures return a flat `{ "error": "message" }` body with an HTTP status that reflects the error kind (`400`, `401`, `404`, `5xx`). |

## Authentication

Auth is **optional and off by default** — one server is one user. When the server is started
with `AGENTX_AUTH_ENABLED=true`, every `/api/*` route requires a session token in the
**`X-Auth-Token`** header. Obtain one from `POST /api/auth/login`. See the
[Authentication guide](../deployment/authentication.md) for the full model.

## What you'll integrate against

- **Chat** — `POST /api/agent/chat` for a single JSON response, or
  `POST /api/agent/chat/stream` for token-by-token [streaming](streaming.md).
- **Memory** — recall and write facts, browse user history, inspect entities/facts.
- **Agents & workflows** — agent profiles (`/api/agent/profiles`) and multi-agent
  [Agent Alloy](../features/multi-agent.md) workflows (`/api/alloy/workflows`).
- **Tools & providers** — MCP servers/tools and model providers.

Jump to the [recipes](recipes.md) for copy-paste `curl`, JavaScript, and Python snippets.

## Versioning & compatibility

The API advertises a `protocol_version` (an integer that increments on breaking changes) and a
`min_client_version`. Clients must match the protocol exactly to connect; mismatches are surfaced
to users on a dedicated screen. Read the live values from `GET /api/version`, and see
[Database Migration](../deployment/migration.md) for upgrade/compatibility notes.

## Full reference

The exhaustive endpoint catalog lives in the [API Reference](../api/endpoints.md) (with request/
response examples) and the [API Models](../api/models.md) reference. A machine-readable OpenAPI
3.0 spec mirroring it lives at `OpenApi.yaml` in the repo root.

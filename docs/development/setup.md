# Development Setup

Set up your development environment for contributing to AgentX.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Backend runtime |
| **uv** | latest | Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/)) |
| **Node.js** | 18+ | Client tooling |
| **bun** | latest | Client package manager ([install](https://bun.sh/docs/installation)) |
| **Docker** | 24+ | Database services (Neo4j, PostgreSQL, Redis) |
| **Docker Compose** | v2+ | Multi-container orchestration |
| **Task** | 3+ | Task runner ([install](https://taskfile.dev/installation/)) |
| **Git** | 2.30+ | Version control |

Optional for Tauri desktop builds:

| Tool | Purpose |
|------|---------|
| **Rust** (stable) | Tauri native compilation |
| **System libs** | `libwebkit2gtk-4.1-dev`, `libssl-dev`, etc. — see [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) |

## First-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/agentx-source.git
cd agentx-source

# 2. Run the automated setup (installs deps, creates data dirs, verifies env)
task setup

# 3. Copy and configure environment variables
cp .env.example .env
# Edit .env — at minimum set NEO4J_PASSWORD and POSTGRES_PASSWORD

# 4. Start database services
task db:up

# 5. Initialize database schemas (Neo4j indexes, PostgreSQL tables)
task db:init:schemas

# 6. Start the full development stack
task dev
```

`task setup` runs `task install` (uv sync + bun install), `task db:init` (creates `data/` directories), and `task check` (verifies prerequisites).

## Docker Services

`task db:up` starts three containers defined in `docker-compose.yml`:

| Service | Container | Port | Data Directory |
|---------|-----------|------|----------------|
| Neo4j | `agent-neo4j` | 7474 (HTTP), 7687 (Bolt) | `data/neo4j/` |
| PostgreSQL + pgvector | `agent-postgres` | 5432 | `data/postgres/` |
| Redis | `agent-redis` | 6379 | `data/redis/` |

Data is **bind-mounted** to `./data/` (not Docker volumes). Run `task db:init` to create the directory structure if it doesn't exist.

```bash
task db:status          # Check container health
task db:down            # Stop all services
task db:shell:postgres  # psql shell
task db:shell:redis     # redis-cli
task db:shell:neo4j     # cypher-shell
```

## Environment Variables

Copy `.env.example` to `.env`. See [Configuration](../getting-started/configuration.md) for the full reference.

Key variables for development:

```bash
# Database passwords — must match docker-compose.yml
NEO4J_PASSWORD=changeme
POSTGRES_PASSWORD=changeme

# Default model (used by agent chat)
DEFAULT_MODEL=llama3.2

# Provider API keys (optional — only needed for cloud models)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Embedding provider: "local" (sentence-transformers) or "openai"
EMBEDDING_PROVIDER=local
```

## Running the Dev Stack

```bash
# Full stack: Docker + API + Tauri client (hot reload)
task dev

# API only (assumes Docker is already running)
task dev:api

# Client only in browser mode (no Tauri, port 1420)
task dev:web
```

`task dev` uses `concurrently` to run Docker services, the Django API (port 12319), and the Tauri client in parallel. All three support hot reload.

## Verification

After setup, verify everything is working:

```bash
# 1. Check prerequisites are installed
task check

# 2. Verify database schemas exist
task db:verify:schemas

# 3. Run fast tests (no model loading)
task test:quick

# 4. Hit the health endpoint
curl http://localhost:12319/api/health?include_memory=true
```

A healthy response includes `"status": "ok"` with database connection statuses.

## Common Issues

**Docker containers won't start**
Check if ports 5432, 6379, 7474, or 7687 are already in use. Stop conflicting services or adjust ports in `docker-compose.yml`.

**`task: command not found`**
Install Task: `sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin`

**`uv: command not found`**
Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Schema initialization fails**
Ensure Docker containers are healthy (`task db:status`) before running `task db:init:schemas`. Neo4j takes 10-20 seconds to become ready after container start.

**Translation tests are slow on first run**
The `TranslationKitTest` suite downloads NLLB-200 (~600MB) on first execution. Use `task test:quick` to skip model-dependent tests during development.

**Embedding dimension mismatch errors**
If you switch `EMBEDDING_PROVIDER` between `local` and `openai`, the vector dimensions differ. Reset the memory tables with `task db:init:schemas` or use the memory reset API endpoint.

## Next Steps

- [Contributing Guide](contributing.md) — Branch naming, commit style, PR checklist
- [Task Commands](tasks.md) — Full list of available `task` commands
- [Configuration](../getting-started/configuration.md) — All environment variables and config files
- [Architecture Overview](../architecture/overview.md) — System design and module layout

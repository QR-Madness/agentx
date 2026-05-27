# Installation

Get AgentX up and running on your local machine.

## Prerequisites

Before installing AgentX, ensure you have:

- **Python 3.10+** - For the Django API
- **Node.js 18+** - For the Tauri client
- **Docker & Docker Compose** - For database services
- **uv** - Python package manager ([installation guide](https://github.com/astral-sh/uv))
- **bun** - Fast JavaScript runtime ([installation guide](https://bun.sh))
- **Task** - Task runner ([installation guide](https://taskfile.dev))

## Quick Install

### 1. Clone the Repository

```bash
git clone https://github.com/QR-Madness/agentx.git
cd agentx
```

### 2. Install Dependencies

```bash
task install
```

This command will:

- Install Python dependencies via `uv sync`
- Install client dependencies via `bun install`
- Install concurrently for running multiple services

### 3. Initialize Databases

```bash
task db:init
```

This creates the required data directories for:

- Neo4j graph database
- PostgreSQL with pgvector
- Redis cache

### 4. Start Development Environment

```bash
task dev
```

This will:

1. Start Docker containers (Neo4j, Postgres, Redis)
2. Launch Django API on `http://localhost:12319`
3. Launch Tauri development window

## Verify Installation

### Check API
```bash
curl http://localhost:12319/api/index
```

Expected response:
```json
{
  "status": "ok",
  "message": "AgentX API is running"
}
```

### Check Databases

```bash
# Neo4j Browser
open http://localhost:7474

# Redis Commander (optional GUI)
open http://localhost:8081
```

Default credentials (from `.env.example`):

- **Neo4j**: `neo4j` / `changeme`
- **PostgreSQL**: `agent` / `changeme`

!!! warning "Change default passwords"
    Update passwords in your `.env` file before deploying to production.

## Manual Installation

If you prefer to install components individually:

### Python API

```bash
# Install dependencies
uv sync

# Run migrations
task api:migrate

# Start API server
task api:runserver
```

### Tauri Client

```bash
cd client

# Install dependencies
bun install

# Start development server
bunx tauri dev
```

### Database Services

```bash
# Start all databases
task runners

# Or start individually
docker-compose up -d neo4j
docker-compose up -d postgres
docker-compose up -d redis
```

## Troubleshooting

### Port Conflicts

If ports are already in use, modify `docker-compose.yml`:

- Neo4j: 7474 (browser), 7687 (bolt)
- PostgreSQL: 5432
- Redis: 6379
- Django API: 12319
- Vite dev server: 1420

### Missing Data Directories

If you see errors about missing data directories:

```bash
task pre-launch-check
```

This will show which directories are missing and how to fix them.

### Docker Volume Migration

If you have existing data in Docker volumes:

```bash
task db:migrate-volumes
```

This migrates data from Docker volumes to local bind mounts in `./data/`.

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn basic usage
- [Configuration](configuration.md) - Customize your setup
- [Development Setup](../development/setup.md) - Set up for development

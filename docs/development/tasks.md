# Task Commands

Complete reference for Taskfile automation commands.

## Installation & Setup

### Install All Dependencies
```bash
task install
```

Runs `uv sync` and `bun install` to install all dependencies.

### Initialize Databases
```bash
task db:init
```

Creates data directories for Neo4j, PostgreSQL, and Redis.

## Development

### Start Development Environment
```bash
task dev
```

Starts all services:

- Docker containers (Neo4j, Postgres, Redis)
- Django API on port 12319
- Tauri development window

### Start Services Only
```bash
task runners
```

Starts Docker containers without API or client.

### Stop All Services
```bash
task teardown
```

Stops and removes Docker containers.

### Pre-Launch Check
```bash
task pre-launch-check
```

Validates that all required directories and dependencies exist.

## API Commands

### Run Django Server
```bash
task api:runserver
```

Starts Django development server on `http://127.0.0.1:12319`.

### Django Shell
```bash
task api:shell
```

Opens Django interactive shell.

### Database Migrations

Create migration files:
```bash
task api:makemigrations
```

Apply migrations:
```bash
task api:migrate
```

## Client Commands

### Start Tauri Dev Mode
```bash
task client:dev
```

Launches Tauri window with hot reload.

### Build Client
```bash
task client:build
```

Builds production client bundle.

### Start Production Client
```bash
task client:start
```

Runs production build (requires `client:build` first).

## Database Management

### PostgreSQL

Open PostgreSQL shell:
```bash
task db:shell:postgres
```

Create backup:
```bash
task db:backup:postgres
```

Backups saved to `./backups/postgres_YYYYMMDD_HHMMSS.sql`

Restore from backup:
```bash
task db:restore:postgres BACKUP_FILE=backups/postgres_20231127_120000.sql
```

### Redis

Open Redis CLI:
```bash
task db:shell:redis
```

### Neo4j

Neo4j Browser: [http://localhost:7474](http://localhost:7474)

### Volume Migration

Migrate all databases from Docker volumes to local bind mounts:
```bash
task db:migrate-volumes
```

Migrate individual databases:
```bash
task db:migrate-volumes:neo4j
task db:migrate-volumes:postgres
task db:migrate-volumes:redis
```

These commands copy data from Docker volumes to `./data/` directories.

### Clean All Data

!!! danger "Destructive Operation"
    This permanently deletes all database data!

```bash
task db:clean
```

You will be prompted for confirmation.

## Testing

### Run All Tests
```bash
task test
```

Runs Django test suite.

### Run Specific Test
```bash
uv run python api/manage.py test agentx_ai.TranslationKitTest
```

### Run Single Test Method
```bash
uv run python api/manage.py test agentx_ai.TranslationKitTest.test_translate_to_french
```

## Documentation

### Serve Documentation Locally
```bash
task docs:serve
```

Opens documentation at [http://127.0.0.1:8000](http://127.0.0.1:8000) with live reload.

### Build Documentation
```bash
task docs:build
```

Builds static documentation site to `site/` directory.

### Deploy Documentation
```bash
task docs:deploy
```

Deploys documentation to GitHub Pages (requires git repository setup).

## Utility Commands

### Default Task
```bash
task
```

Runs sanity check and prompts to start development environment.

### List All Tasks
```bash
task --list
```

Shows all available tasks with descriptions.

### Task Help
```bash
task --help
```

Displays Taskfile help information.

## Task Dependencies

Some tasks automatically run prerequisites:

- `task dev` → runs `task runners` first
- `task client:start` → runs `task client:build` first
- `task runners` → runs `task pre-launch-check` first

## Environment-Specific Tasks

### Development
```bash
# Quick iteration cycle
task dev              # Start everything
# Make changes...
task test             # Verify changes
task teardown         # Stop when done
```

### Testing
```bash
# Run specific tests during development
uv run python api/manage.py test agentx_ai --keepdb
```

### Production Build
```bash
task client:build     # Build optimized client
task api:migrate      # Ensure migrations are applied
```

## Custom Task Variables

Some tasks accept variables:

### Postgres Restore
```bash
task db:restore:postgres BACKUP_FILE=path/to/backup.sql
```

## Debugging Tasks

### Verbose Output
```bash
task --verbose dev
```

### Dry Run
```bash
task --dry dev
```

Shows what would be executed without running commands.

## Tips & Tricks

### Run Multiple Commands
```bash
task install && task db:init && task dev
```

### Background Execution
```bash
task runners &  # Start services in background
```

### Watch Mode
```bash
# API auto-reloads on file changes
task api:runserver

# Client has HMR enabled
task client:dev
```

### Quick Database Reset
```bash
task teardown && task db:clean && task db:init && task runners
```

!!! warning
    This deletes all data!

## Next Steps

- [Development Setup](setup.md) - Configure your environment
- [Testing Guide](testing.md) - Write and run tests
- [Contributing](contributing.md) - Contribution workflow

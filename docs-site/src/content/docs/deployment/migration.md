# Database Migration

Guide for backing up and restoring database data. AgentX stores all database data as bind mounts
under `./data/` (or `clusters/<name>/db/` for a cluster) — there are no Docker named volumes to
migrate.

## Backup and Restore

### PostgreSQL

Backup:
```bash
task db:backup
```

Restore:
```bash
task db:restore BACKUP_FILE=backups/postgres_20231127.sql
```

## Data Directories

All data stored in `./data/`:

```
data/
├── neo4j/
├── postgres/
└── redis/
```

See [Database Stack](../architecture/databases.md) for details.

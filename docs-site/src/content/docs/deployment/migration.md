# Database Migration

Guide for migrating database data.

## From Docker Volumes to Bind Mounts

```bash
# Migrate all databases
task db:migrate-volumes

# Or individually
task db:migrate-volumes:neo4j
task db:migrate-volumes:postgres
task db:migrate-volumes:redis
```

## Backup and Restore

### PostgreSQL

Backup:
```bash
task db:backup:postgres
```

Restore:
```bash
task db:restore:postgres BACKUP_FILE=backups/postgres_20231127.sql
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

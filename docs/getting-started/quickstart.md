# Quick Start

Get started with AgentX in 5 minutes.

## Starting AgentX

### Development Mode

```bash
task dev
```

This starts:

- All database services (Neo4j, Postgres, Redis)
- Django API on port 12319
- Tauri desktop application

### API Only

```bash
task api:runserver
```

### Client Only

```bash
cd client
bunx tauri dev
```

## Using the Translation API

### Detect Language

```bash
curl http://localhost:12319/api/language-detect
```

### Translate Text

```bash
curl -X POST http://localhost:12319/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "target_language": "fr"
  }'
```

Response:
```json
{
  "original": "Hello, world!",
  "translated": "Bonjour le monde!",
  "source_language": "en",
  "target_language": "fr",
  "confidence": 0.98
}
```

### Supported Language Codes

Use ISO 639-1 codes for target languages:

- `en` - English
- `fr` - French
- `es` - Spanish
- `de` - German
- `zh` - Chinese
- `ja` - Japanese
- `ar` - Arabic
- And 193+ more...

## Using the Desktop Application

### Tab Navigation

The application has four main tabs:

1. **Dashboard** - Overview and stats
2. **Translation** - Interactive translation interface
3. **Chat** - AI conversation interface
4. **Tools** - Utilities and settings

### Translation Tab

1. Enter text in the source field
2. Select target language
3. Click "Translate"
4. View results with confidence scores

## Database Access

### PostgreSQL Shell

```bash
task db:shell:postgres
```

### Redis CLI

```bash
task db:shell:redis
```

### Neo4j Browser

Open [http://localhost:7474](http://localhost:7474)

Credentials: `neo4j` / `your_secure_password`

## Common Tasks

### Run Tests

```bash
task test
```

### Database Backup

```bash
task db:backup:postgres
```

Backups are saved in `./backups/`

### Clean Database

```bash
task db:clean
```

!!! warning
    This removes all database data. Use with caution!

### Stop Services

```bash
task teardown
```

Or press `Ctrl+C` to stop the dev environment.

## Development Workflow

1. **Start dev environment**: `task dev`
2. **Make changes** to API or client code
3. **Hot reload** happens automatically
   - Vite for frontend changes
   - Django autoreload for API changes
4. **Run tests**: `task test`
5. **Commit changes**: `git commit`
6. **Stop services**: `Ctrl+C` or `task teardown`

## Next Steps

- [Configuration Guide](configuration.md) - Customize settings
- [Architecture Overview](../architecture/overview.md) - Understand the system
- [Development Setup](../development/setup.md) - Advanced development

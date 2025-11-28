# Configuration

Configure AgentX for your environment.

## Environment Variables

Create a `.env` file in the project root (optional):

```bash
# Django Settings
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# Database Connections
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

POSTGRES_URI=postgresql://agent:your_secure_password@localhost:5432/agent_memory
REDIS_URI=redis://localhost:6379

# API Settings
API_PORT=12319
```

## Database Configuration

### Neo4j

Edit `docker-compose.yml` to customize Neo4j settings:

```yaml
services:
  neo4j:
    environment:
      - NEO4J_AUTH=neo4j/your_secure_password
      - NEO4J_server_memory_heap_max__size=2G
      - NEO4J_server_memory_pagecache_size=1G
```

### PostgreSQL

Configure PostgreSQL in `docker-compose.yml`:

```yaml
services:
  postgres:
    environment:
      - POSTGRES_USER=agent
      - POSTGRES_PASSWORD=your_secure_password
      - POSTGRES_DB=agent_memory
```

### Redis

Adjust Redis memory limits:

```yaml
services:
  redis:
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## Django Settings

Located in `api/agentx_api/settings.py`:

### Database Backend

Currently using SQLite for Django ORM:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

To switch to PostgreSQL:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'agent_memory',
        'USER': 'agent',
        'PASSWORD': 'your_secure_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### CORS Settings

Configure allowed origins in `settings.py`:

```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:1420",  # Vite dev server
    "tauri://localhost",       # Tauri window
]
```

## Tauri Configuration

Located in `client/src-tauri/tauri.conf.json`:

### Window Settings

```json
{
  "windows": [
    {
      "title": "AgentX",
      "width": 800,
      "height": 600,
      "resizable": true,
      "fullscreen": false
    }
  ]
}
```

### Development URL

```json
{
  "build": {
    "devUrl": "http://localhost:1420"
  }
}
```

## Translation Models

Configure models in `api/agentx_ai/kit/translation.py`:

### Language Detection Model

```python
DETECTION_MODEL = "eleldar/language-detection"
```

### Translation Model

Currently using M2M100:

```python
TRANSLATION_MODEL = "facebook/m2m100_418M"
```

To switch to NLLB-200 (larger, more accurate):

```python
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
```

!!! note
    Larger models provide better quality but require more memory and slower inference.

## Development Tools

### Task Configuration

Edit `Taskfile.yaml` to customize commands:

```yaml
tasks:
  api:runserver:
    dir: api/
    cmds:
      - uv run python manage.py runserver 127.0.0.1:12319
```

### Vite Configuration

Located in `client/vite.config.ts`:

```typescript
export default defineConfig({
  server: {
    port: 1420,
    strictPort: true,
  },
  // ... other settings
})
```

## Security Considerations

### Production Checklist

- [ ] Change default database passwords
- [ ] Set `DJANGO_DEBUG=False`
- [ ] Configure `DJANGO_ALLOWED_HOSTS`
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS for API
- [ ] Configure firewall rules
- [ ] Set up database backups
- [ ] Review CORS settings

### Password Management

Never commit passwords to version control. Use:

- Environment variables
- Secret management tools (HashiCorp Vault, AWS Secrets Manager)
- `.env` files (gitignored)

## Performance Tuning

### Model Loading

Translation models are loaded at startup. To reduce memory:

```python
# Use smaller models
TRANSLATION_MODEL = "facebook/m2m100_418M"  # ~500MB
# vs
TRANSLATION_MODEL = "facebook/nllb-200-3.3B"  # ~13GB
```

### Database Connections

Adjust connection pools in production:

```python
DATABASES = {
    'default': {
        # ...
        'OPTIONS': {
            'MAX_CONNS': 20,
            'MIN_CONNS': 5,
        }
    }
}
```

### Redis Caching

Configure Redis for optimal performance:

```bash
# In docker-compose.yml
command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## Next Steps

- [Development Setup](../development/setup.md) - Advanced configuration
- [Database Migration](../deployment/migration.md) - Data management
- [Task Commands](../development/tasks.md) - Available automation

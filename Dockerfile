# syntax=docker/dockerfile:1
# AgentX API Server Dockerfile
# Build: docker build -t agentx-api .
# Run: docker run -p 12319:12319 --env-file .env agentx-api

FROM python:3.14-slim

WORKDIR /app

# System dependencies. curl: healthcheck. bubblewrap: the bubblewrap shell sandbox
# (installs setuid so it jails without --privileged). docker.io: provides the `docker`
# CLI so the container shell backend can drive the dind sidecar via DOCKER_HOST — the
# bundled daemon is never started here (the API only ever acts as a Docker *client*).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bubblewrap \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# uv for Python package management — pinned official binary (reproducible, and
# faster/less flaky than piping the install script through curl).
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /uvx /bin/

# Node + npx (needed for local/stdio MCP tool servers) straight from the official
# image — pinned and self-contained, no nvm download/cache cruft in the layer.
COPY --from=node:24.15.0-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:24.15.0-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node -v && npm -v && npx -v

# uv install behavior: compile bytecode for faster cold starts; copy (not hardlink)
# out of the cache mount since it lives on a different filesystem.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependency layer first for caching. The (large) uv download cache rides a
# BuildKit cache mount, so it speeds rebuilds without bloating the image layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy application code.
# Preserve api/ structure so versions.yaml path calculation works.
# api/defaults/ rides along here and is the seed source for the entrypoint.
COPY api/ ./api/
COPY queries/ ./queries/
COPY versions.yaml ./versions.yaml

# Alembic (PostgreSQL schema migrations) — script_location in alembic.ini is
# relative to the ini file itself, so both must land at /app together. Without
# these the entrypoint's `alembic upgrade head` fails on every boot (ADR-9).
COPY alembic.ini ./alembic.ini
COPY alembic/ ./alembic/

# Operations tooling: self-init entrypoint + the `agentx` ops CLI on PATH.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/agentx /usr/local/bin/agentx
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/agentx

# Create data directory for runtime config
RUN mkdir -p ./data

# Environment variables
ENV DJANGO_SETTINGS_MODULE=agentx_api.settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/api
# Disable huggingface_hub's hf-xet download backend: its Rust download threads
# have been observed outliving a finished model download, keeping the process
# alive after init_memory_schema prints success — which blocks the entrypoint
# from ever reaching uvicorn on a fresh first boot (empty HF cache). Plain
# HTTP downloads are marginally slower and entirely boring.
ENV HF_HUB_DISABLE_XET=1

# Expose API port
EXPOSE 12319

# Health check. First boot eagerly loads the embedding + translation models
# (several GB), so give startup a long grace period before marking the container
# unhealthy — compose `depends_on: condition: service_healthy` waits on this.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -f http://localhost:12319/api/health || exit 1

# Self-init entrypoint runs first (seed config + migrate + schema init), then
# exec's the CMD below. Set AGENTX_AUTO_INIT=false to skip auto-init.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Run with uvicorn for production (ASGI)
CMD ["uv", "run", "uvicorn", "agentx_api.asgi:application", "--host", "0.0.0.0", "--port", "12319"]

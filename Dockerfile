# AgentX API Server Dockerfile
# Build: docker build -t agentx-api .
# Run: docker run -p 12319:12319 --env-file .env agentx-api

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install nvm, node + npm (npx needed for local MCP tools) - Reference: https://nodejs.org/en/download
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

# Install specific Node version (MUST BE SPECIFIC to target node directory on the path)
ENV NODE_VERSION=24.15.0
ENV NVM_DIR=/root/.nvm

RUN chmod +x "$HOME/.nvm/nvm.sh"
RUN \. $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

# Add Node's bin to PATH
ENV NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Validate installations
RUN node -v
RUN npm -v
RUN npx -v

# LONG STEP - 5-15 mins - Install Python dependencies (production only)
RUN uv sync --frozen --no-dev

# Copy application code
# Preserve api/ structure so versions.yaml path calculation works
COPY api/ ./api/
COPY queries/ ./queries/
COPY versions.yaml ./versions.yaml

# Create data directory for runtime config
RUN mkdir -p ./data

# Environment variables
ENV DJANGO_SETTINGS_MODULE=agentx_api.settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/api

# Expose API port
EXPOSE 12319

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:12319/api/health || exit 1

# Run with uvicorn for production (ASGI)
CMD ["uv", "run", "uvicorn", "agentx_api.asgi:application", "--host", "0.0.0.0", "--port", "12319"]

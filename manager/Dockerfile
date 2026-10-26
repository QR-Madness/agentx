# syntax=docker/dockerfile:1
# AgentX Manager image — CLI + web GUI over the host's Docker.
#
# Runs from source at /app (no wheel): the server resolves the GUI at
# /app/ui/dist, and PYTHONPATH keeps the layout identical to the repo's
# manager/ directory. The docker CLI + compose plugin are copied from the
# official docker:cli image (static Go binaries).
#
# Dependencies install from the lockfiles only (uv.lock, ui/bun.lock) and are
# never re-resolved at build time, so the lockfiles describe exactly what the
# image ships. To change a dependency, relock — a rebuild alone never picks up
# a newer release.
#
# The container needs, at runtime (see docker-compose.manager.yml):
#   - /var/run/docker.sock mounted
#   - the deployment root mounted at the SAME absolute path as on the host
#     (compose resolves bind mounts client-side)

FROM oven/bun:1-slim AS ui
WORKDIR /build
COPY ui/package.json ui/bun.lock ./
RUN bun install --frozen-lockfile
COPY ui/ .
RUN bun run build

FROM docker:27-cli AS dockercli

# Python runtime dependencies → /app/.venv, with the same pinned uv as the API
# image (root Dockerfile). uv stays in this stage; the final image gets only
# the environment.
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /bin/
# Compile bytecode for faster cold starts; copy (not hardlink) out of the cache
# mount; never download a uv-managed Python — the venv must link to this
# image's interpreter, which the final stage shares.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen: install exactly what uv.lock pins. --no-dev: skip the test group
# (pytest, httpx). --no-install-project: the app runs from source, so only its
# dependencies belong in the environment.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim
LABEL org.opencontainers.image.title="AgentX Manager" \
      org.opencontainers.image.source="https://github.com/QR-Madness/agentx"

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-compose \
     /usr/local/libexec/docker/cli-plugins/docker-compose

WORKDIR /app
COPY --from=deps /app/.venv .venv
COPY agentx_manager/ agentx_manager/
COPY --from=ui /build/dist ui/dist

# The venv comes first on PATH, so the ENTRYPOINT's `python` (and any
# `--entrypoint python` override) runs against the locked environment.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# `docker compose exec manager agentx-manager <cmd>` (deploy/README.md) needs the
# CLI on PATH. The app runs from source rather than as an installed package, so
# this shim stands in for the console script an install would create. It keeps
# the caller's working directory, which root discovery walks up from.
COPY --chmod=0755 <<'EOF' /usr/local/bin/agentx-manager
#!/bin/sh
exec /app/.venv/bin/python -m agentx_manager.cli "$@"
EOF

EXPOSE 12320
ENTRYPOINT ["python", "-m", "agentx_manager.cli"]
CMD ["serve", "--host", "0.0.0.0"]

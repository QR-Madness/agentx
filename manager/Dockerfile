# AgentX Manager image — CLI + web GUI over the host's Docker.
#
# Runs from source at /app (no wheel): the server resolves the GUI at
# /app/ui/dist, and PYTHONPATH keeps the layout identical to the repo's
# manager/ directory. The docker CLI + compose plugin are copied from the
# official docker:cli image (static Go binaries).
#
# The container needs, at runtime (see docker-compose.manager.yml):
#   - /var/run/docker.sock mounted
#   - the deployment root mounted at the SAME absolute path as on the host
#     (compose resolves bind mounts client-side)

FROM oven/bun:1-slim AS ui
WORKDIR /build
COPY ui/package.json ui/bun.lock* ./
RUN bun install --frozen-lockfile || bun install
COPY ui/ .
RUN bun run build

FROM docker:27-cli AS dockercli

FROM python:3.12-slim
LABEL org.opencontainers.image.title="AgentX Manager" \
      org.opencontainers.image.source="https://github.com/QR-Madness/agentx"

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=dockercli /usr/local/libexec/docker/cli-plugins/docker-compose \
     /usr/local/libexec/docker/cli-plugins/docker-compose

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi>=0.115 "uvicorn>=0.30"
COPY agentx_manager/ agentx_manager/
COPY --from=ui /build/dist ui/dist

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 12320
ENTRYPOINT ["python", "-m", "agentx_manager.cli"]
CMD ["serve", "--host", "0.0.0.0"]

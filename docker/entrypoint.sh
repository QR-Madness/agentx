#!/usr/bin/env bash
# AgentX API container entrypoint.
#
# Makes an isolated (Docker Hub) deployment self-sufficient — no repo, no
# Taskfile required:
#   1. Seed the runtime config dir (/app/data) from the baked defaults if empty.
#   2. Auto-initialize database schemas on boot (idempotent), unless disabled.
#   3. Print a one-line hint if auth setup is still pending.
#   4. exec the container command (uvicorn).
#
# Everything here is idempotent and safe to run on every start.
set -euo pipefail

APP_DIR="/app"
# The config dir is always mounted at /app/data inside the container (compose
# maps the host ${AGENTX_CONFIG_DIR} there); the host path is not our concern.
DATA_DIR="/app/data"
DEFAULTS_DIR="/app/api/defaults"

log() { echo "[entrypoint] $*"; }

# 1) Seed config from baked defaults (mirrors `task cluster:seed`: *.yaml + *.json,
#    but never .example/README). Only copies files that are missing, so user
#    edits are preserved across restarts.
seed_config() {
  mkdir -p "$DATA_DIR"
  shopt -s nullglob
  local seeded=0
  for src in "$DEFAULTS_DIR"/*.yaml "$DEFAULTS_DIR"/*.json; do
    local name
    name="$(basename "$src")"
    local dest="$DATA_DIR/$name"
    if [ ! -f "$dest" ]; then
      cp "$src" "$dest"
      log "seeded config: $name"
      seeded=1
    fi
  done
  [ "$seeded" = "0" ] && log "config dir already populated — no seeding needed"
}

# 2) Auto-init DB schemas. depends_on: service_healthy gates DB readiness under
#    compose, but retry a few times so a transient connection blip isn't fatal.
auto_init() {
  if [ "${AGENTX_AUTO_INIT:-true}" != "true" ]; then
    log "AGENTX_AUTO_INIT=${AGENTX_AUTO_INIT:-} — skipping schema auto-init"
    return 0
  fi
  log "running database migrations + memory schema init..."
  local attempt=1 max=5
  while [ "$attempt" -le "$max" ]; do
    if uv run python "$APP_DIR/api/manage.py" migrate --noinput \
       && uv run python "$APP_DIR/api/manage.py" init_memory_schema; then
      log "schema init complete"
      return 0
    fi
    log "schema init attempt $attempt/$max failed — retrying in 5s..."
    attempt=$((attempt + 1))
    sleep 5
  done
  log "ERROR: schema init failed after $max attempts"
  return 1
}

# 3) Auth setup hint (non-fatal).
auth_hint() {
  if [ "${AGENTX_AUTH_ENABLED:-true}" != "true" ]; then
    return 0
  fi
  if uv run python "$APP_DIR/api/manage.py" setup_auth --check 2>/dev/null | grep -qi "setup required"; then
    log "AUTH: no root password set. Run:  docker compose exec api agentx setup-auth"
    log "      (or use the client's first-run setup screen)."
  fi
}

cd "$APP_DIR"
seed_config
auto_init
auth_hint

log "starting API: $*"
exec "$@"

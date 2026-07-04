#!/usr/bin/env bash
# AgentX API container entrypoint.
#
# Makes an isolated (Docker Hub) deployment self-sufficient — no repo, no
# Taskfile required:
#   1. Seed the runtime config dir (/app/data) from the baked defaults if empty.
#   2. `manage.py bootstrap` — ONE process for Django + Alembic migrations,
#      memory-schema init (stamp fast path on warm boots), warmup signal, and
#      the auth hint. Idempotent; disabled via AGENTX_AUTO_INIT=false.
#   3. If bootstrap reports warmup=needed (first boot, local provider): run
#      warmup_embeddings under a post-success watchdog.
#   4. Print a one-line hint if auth setup is still pending.
#   5. exec the container command (uvicorn).
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

# 2) One-process bootstrap: Django + Alembic migrations, memory-schema stamp
#    fast path, embedding-warmup signal, auth hint — all inside a single
#    `manage.py bootstrap` (stdout contract: BOOTSTRAP <phase>=<state> lines).
#    Warm boots take seconds; the model is never loaded here.
BOOT_OUT="$(mktemp)"

run_bootstrap() {
  if [ "${AGENTX_AUTO_INIT:-true}" != "true" ]; then
    log "AGENTX_AUTO_INIT=${AGENTX_AUTO_INIT:-} — skipping schema auto-init"
    return 0
  fi
  log "running bootstrap (migrations + schema init)..."
  local attempt=1 max=5 rc
  while [ "$attempt" -le "$max" ]; do
    uv run python "$APP_DIR/api/manage.py" bootstrap >"$BOOT_OUT" 2>&1 && rc=0 || rc=$?
    cat "$BOOT_OUT"
    if [ "$rc" -eq 0 ]; then
      log "bootstrap complete"
      return 0
    fi
    if [ "$rc" -eq 2 ]; then
      log "ERROR: bootstrap configuration error (see above) — not retrying"
      return 1
    fi
    log "bootstrap attempt $attempt/$max failed — retrying in 5s..."
    attempt=$((attempt + 1))
    sleep 5
  done
  log "ERROR: bootstrap failed after $max attempts"
  return 1
}

# 3) Explicit embedding warmup, only when bootstrap says the local model isn't
#    cached yet (true first boot). The model load/download leaves non-exiting
#    threads that keep the process alive AFTER the success marker — observed
#    with the hf-xet backend AND with it disabled (HF_HUB_DISABLE_XET=1 stays
#    set; plain HTTP is boring and one suspect fewer) — so warmup runs under a
#    post-success watchdog: once the marker appears, give the process a short
#    grace to exit, then reap it and treat the warmup as succeeded.
WARMUP_SUCCESS_MARKER="Model loaded in"
INIT_EXIT_GRACE="${AGENTX_INIT_EXIT_GRACE:-15}"

warmup_watchdog() {
  local out pid rc marker_grace=0
  out="$(mktemp)"
  uv run python "$APP_DIR/api/manage.py" warmup_embeddings >"$out" 2>&1 &
  pid=$!
  # Relay output live so first-boot progress (model download) stays visible.
  tail -n +1 -f "$out" &
  local tail_pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    if grep -q "$WARMUP_SUCCESS_MARKER" "$out"; then
      marker_grace=$((marker_grace + 5))
      if [ "$marker_grace" -ge "$INIT_EXIT_GRACE" ]; then
        log "warmup succeeded but did not exit within ${INIT_EXIT_GRACE}s — reaping lingering process (non-exiting model-download threads)"
        kill -9 "$pid" 2>/dev/null || true
        break
      fi
    fi
  done
  # A reaped-after-success process reads as rc!=0; the marker is the truth.
  wait "$pid" 2>/dev/null && rc=0 || rc=$?
  kill "$tail_pid" 2>/dev/null || true
  wait "$tail_pid" 2>/dev/null || true
  if grep -q "$WARMUP_SUCCESS_MARKER" "$out"; then
    rc=0
  fi
  rm -f "$out"
  return "$rc"
}

maybe_warmup() {
  if grep -q '^BOOTSTRAP warmup=needed' "$BOOT_OUT"; then
    log "downloading + warming embedding model (first boot)..."
    if ! warmup_watchdog; then
      log "WARN: embedding warmup failed — the API will lazy-load the model on first use"
    fi
  fi
}

# 4) Auth setup hint (non-fatal), from the bootstrap contract line.
auth_hint() {
  if [ "${AGENTX_AUTH_ENABLED:-true}" != "true" ]; then
    return 0
  fi
  if grep -q '^BOOTSTRAP auth=setup_required' "$BOOT_OUT"; then
    log "AUTH: no root password set. Run:  docker compose exec api agentx setup-auth"
    log "      (or use the client's first-run setup screen)."
  fi
}

cd "$APP_DIR"
seed_config
run_bootstrap
maybe_warmup
auth_hint
rm -f "$BOOT_OUT"

log "starting API: $*"
exec "$@"

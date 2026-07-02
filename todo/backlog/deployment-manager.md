# Deployment Manager v2 — Gap Closure Plan

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
>
> **Self-contained implementation plan** for closing the functional gaps found in the
> web deployment manager's first release (v0.21.142–143). Written to be executed by an
> agent with no prior session context — read the Warm-up section first.

---

## Warm-up (read before writing code)

1. **What the manager is:** `manager/` is a standalone uv sub-project (`agentx_manager`
   package) that owns cluster lifecycle over Docker Compose — see **ADR-10 in
   [Decisions.md](../../Decisions.md)** and
   [`docs-site/src/content/docs/deployment/manager.md`](../../docs-site/src/content/docs/deployment/manager.md).
   The CLI (`cli.py`), the FastAPI server (`server.py`), and the React GUI
   (`manager/ui/`) are thin layers over the same core modules.
2. **Read these files** (they are small and the whole architecture):
   `manager/agentx_manager/{spec,overlays,registry,state,lifecycle,scaffold,health,compose,server}.py`,
   `manager/ui/src/api.ts`, `manager/ui/src/App.tsx`, `manager/ui/src/components/*`.
3. **Conventions that gate you:**
   - Tests are the validation instrument (no PRs in this repo; merge to `master` after
     gates pass). Unit tests use the recording `FakeRunner`
     (`manager/tests/unit/conftest.py`) and assert on **argv**; integration tests
     (`pytest -m docker`) run the real nginx gateway with a fake API upstream.
   - Gates: `uv run ruff check manager/` · pyright via
     `manager/pyrightconfig.json` (0 errors) · `task manager:test` ·
     `task manager:test:integration` · `cd manager/ui && bun run build` ·
     `task docs:check` · `task deploy:bundle:check`.
   - Any notable change: bump `versions.yaml` (patch) via `task versions:sync` **and**
     update `Release-Notes.md` (marker + body) in the same commit.
   - Commit sign-off: end commit messages with `Assisted-by: <your model name>`.
     No robot-emoji footers, no Co-Authored-By.
   - GUI styling: don't invest in theme — a separate agent will restyle. Keep using the
     existing semantic tokens (`bg-raised`, `text-fg-secondary`, etc. in
     `manager/ui/src/index.css`) and existing component patterns.
4. **Security invariants (do not weaken):** the manager drives the Docker socket. Every
   `/api` route requires the bearer token; loopback bind by default; the manager is
   never routed through the gateway/tunnel. Generated secrets are shown **once** in
   responses and never logged.

---

## Work items (in order — each lands with its own tests)

### 1. Spec editing over the API + GUI (`PATCH /api/clusters/{name}/spec`)

The overlay spec (gateway / tunnel `none|token|named` / expose / gpu / shell) is
persisted in `.manager-state.json` and editable via `agentx-manager set`, but there is
**no server route and no GUI**.

- **Server:** add `PATCH /api/clusters/{name}/spec` accepting a partial dict of the
  five toggles. Reuse the CLI's logic (`cli.py` `set` branch): apply fields to the
  spec, call `spec.__post_init__()` to re-validate combinations (e.g. `expose` requires
  `gateway`; `tunnel=named` requires `gateway`), persist via `save_state` with fresh
  hashes (`tracked_files` + `compute_hashes`, gateway dir from `registry.gateway_dir`).
  Invalid combos → 422 with the `ValueError` message. **Refactor** the shared
  apply-and-persist into a helper (suggest `spec_ops.py` or a function in `state.py`)
  so CLI and server call one implementation.
- **Response:** the updated spec dict + `{"apply_hint": "run up to apply"}` — spec
  edits do NOT auto-apply; the user runs Up/Restart.
- **GUI:** an "Edit" affordance on the cluster card (pencil next to the spec tags →
  small modal with the five toggles mirroring `NewClusterModal`'s controls). After a
  successful PATCH, toast *"spec updated — Up applies it"* and refresh.
- **Tests:** server unit tests — valid patch persists (reload state and assert), invalid
  combo 422, unknown cluster 404, auth 401. Extend the CLI `set` test if the refactor
  moves code.

### 2. Gateway enable from the GUI (surface the existing endpoint)

`POST /api/clusters/{name}/gateway` already exists (scaffolds `gateway/` files,
generates `AGENTX_GATEWAY_TOKEN`, sets `AGENTX_TRUST_PROXY=true`, flips the spec) but
the GUI never calls it, and the generated token — which the desktop client needs in its
per-server **Gateway token** field — is never shown.

- **GUI:** on cards where `spec.gateway === false`, an "Enable gateway…" action opening
  a modal: tunnel choice (`none|token|named`, same wording as `NewClusterModal`),
  explanation line, confirm → POST → show the result like the new-cluster success view
  (generated secrets **once**, copyable `<pre>`, notes list). Add `api.enableGateway`
  to `manager/ui/src/api.ts`.
- **Server:** endpoint exists; add the missing unit test (success shape, invalid tunnel
  422) — currently untested.
- **Follow-through note in the modal:** "Traffic still needs an exposure overlay — see
  Going public" linking to the docs.

### 3. Jobs that survive reload (`active_job` + `GET /api/jobs`)

The busy spinner is client-local; reloading mid-`up` shows an idle card while the job
runs. The server keeps `Jobs.by_id` in memory but exposes only `GET /api/jobs/{id}`.

- **Server:** include `"active_job": {id, action, status} | null` per cluster in
  `GET /api/clusters` (from `Jobs.active` — look up the job, include only if
  `status == "running"`); add `GET /api/jobs` returning recent jobs (cap the dict —
  keep, say, the last 50 in insertion order; prune in `Jobs.start`).
- **GUI:** on load and on every poll, if a cluster has an `active_job`, treat it as
  the busy state and resume `waitForJob(active_job.id)` (guard against double-polling
  the same job id).
- **Tests:** unit — `Jobs.start` prunes; clusters payload carries the running job
  (inject a never-finishing work callable via an `threading.Event`, then release it);
  GUI logic is thin enough to skip JS tests (repo has no JS test rig for manager/ui).

### 4. Windows-compatible root mount (`AGENTX_DEPLOY_ROOT`)

`docker-compose.manager.yml` interpolates `${PWD}` for the same-path mount — POSIX
shells export `PWD`, **PowerShell/cmd do not**, so native-Windows Docker Desktop users
get an empty mount source and a broken manager.

- **Compose:** change `docker-compose.manager.yml` to use
  `${AGENTX_DEPLOY_ROOT:-${PWD}}` in all three places (env `AGENTX_MANAGER_ROOT`,
  volume, `working_dir`).
- **Env template:** add a commented `AGENTX_DEPLOY_ROOT=` to `deploy/.env.example`
  under the plumbing section: "absolute path to this directory — REQUIRED on native
  Windows (PowerShell doesn't export PWD); harmless elsewhere".
- **Docs:** a note in `docs-site/.../deployment/manager.md` (same-path-mount admonition)
  and the self-hosting page's step 3 if it fits naturally; mention WSL as the smoother
  alternative.
- **Tests:** `task deploy:bundle:check` still passes; `docker compose config` with
  `AGENTX_DEPLOY_ROOT` set resolves the mount (add to the integration suite only if
  cheap — a config-level assertion, no containers).

### 5. Adopt awareness (nudge + button)

Pre-manager clusters run under the legacy default compose project; the manager's
`-p agentx-<name>` up would collide with their `container_name`s. `adopt` exists in the
CLI/server (`POST .../adopt`) but the GUI neither offers it nor detects the need.

- **Server:** in `GET /api/clusters`, add `"needs_adopt": bool` — true when containers
  exist whose label `com.docker.compose.project` ≠ `agentx-<name>` but whose names
  match this cluster's `container_name` prefix (`{AGENTX_CLUSTER_NAME}-`). One
  `docker ps -a --format` call per cluster through the runner; keep it cheap and
  fail-soft (on error → false).
- **GUI:** when `needs_adopt`, show an amber banner on the card: "Running under a
  legacy compose project — Adopt migrates it (brief downtime)" + an **Adopt** button
  wired like the other actions.
- **Tests:** unit test the detection against a `FakeRunner` with canned `docker ps`
  output (match and no-match cases).

### 6. Day-2 surface: auth nudge + maintenance actions

- **Server:** add `GET /api/clusters/{name}/setup` → `{auth_pending: bool}` by exec'ing
  `agentx setup-auth --check` through the existing passthrough (`run_agentx` already
  allows `setup-auth`; add a `--check`-only variant — do NOT expose interactive
  setup-auth over HTTP). Add `POST /api/clusters/{name}/ops/{op}` for the safe,
  non-interactive subset `migrate|warmup` — run as background jobs like lifecycle
  actions (reuse `Jobs`).
- **GUI:** when a cluster is `up` and `auth_pending`, show a card hint: "Root password
  not set — use the desktop client's setup screen or `agentx setup-auth`". Add
  Migrate / Warmup to an overflow ("⋯") menu on the card, job-wired.
- **Tests:** passthrough arg-validation (op allowlist → 404 on anything else), auth 401,
  jobs created. FakeRunner argv assertions for the exec command shape.

### 7. (Polish — do last, skip if time-boxed) Stale-config + job output

- `GET /api/clusters` may include `"stale_services": [...]` (expose
  `lifecycle._stale_services` — promote it to a public helper) so the GUI can badge
  "config changed — Restart will recreate nginx".
- Job output streaming: store the child's combined output per job (bounded deque) and
  add `GET /api/jobs/{id}/output`; GUI shows it in the job toast/modal for `rebuild`.
- Logs stream teardown: surface an end-of-stream line ("— stream ended —") instead of
  silently stopping.

---

## Sequencing & landing

- One branch (suggest `feat/manager-v2-gaps`), items land as separate commits in the
  order above (1→6; 7 optional). Merge to `master` directly after all gates pass —
  **no PR, no stacking**.
- After item 2 and 6 (new endpoints): they are manager-internal — **do not** touch
  `OpenApi.yaml` (that's the Django API's spec, per ADR-10 discussion).
- Update docs in the same commits: `deployment/manager.md` (GUI + REST additions),
  self-hosting step 3 if the auth nudge changes the guided flow.
- Finish with: version bump + Release-Notes body (one consolidated bullet: "Manager v2:
  edit overlays, enable the gateway, resumable jobs, adopt detection, Windows-friendly
  root mount"), and mark this file's items `[x]` with the version tag.

## Verification (end-to-end, after all items)

1. All gates from Warm-up §3.
2. Live pass: `task manager:serve` against the repo — create a throwaway cluster from
   the GUI, edit its spec, enable its gateway (see the token once), up, reload the page
   mid-up (spinner resumes), destroy. `task cluster:destroy` any leftovers; report
   anything still running.
3. Bundle pass: `task deploy:bundle`, untar to a scratch dir,
   `AGENTX_DEPLOY_ROOT=$PWD docker compose config` resolves the manager mount.

## Item checklist

- [ ] 1 · Spec editing (PATCH + GUI modal + shared apply helper)
- [ ] 2 · Gateway enable in the GUI (+ endpoint tests)
- [ ] 3 · Jobs survive reload (`active_job`, `GET /api/jobs`, GUI resume)
- [ ] 4 · `AGENTX_DEPLOY_ROOT` Windows-safe mount (+ env/docs)
- [ ] 5 · Adopt detection + button
- [ ] 6 · Auth nudge + migrate/warmup ops
- [ ] 7 · Polish: stale-config badge · job output stream · logs end-of-stream marker

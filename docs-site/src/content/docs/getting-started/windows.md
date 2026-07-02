# Windows Setup

This guide walks through running AgentX on Windows — both the Django API + databases and the Tauri
desktop client. Windows is a first-class target platform alongside Linux.

!!! warning "Use WSL2 for day-to-day development"
    For anyone doing **day-to-day development** (not just GPU/Docker users), run the full dev loop
    inside **WSL2** (Ubuntu) and treat it as a Linux host — see [Development Setup](../development/setup.md).
    Native Windows works, but it is bumpy around **process lifecycle**: the dev supervisor can leave
    orphaned processes holding ports 12319/1420/1421, and `Ctrl-C` teardown is flaky. Under WSL2 the
    Unix process tooling (`task dev:kill`/`dev:reap`, signal handling) behaves correctly. This guide
    still covers **native Windows** below for users who must; WSL2-specific notes are called out
    where they matter.

## Prerequisites

| Tool | Notes |
|------|-------|
| **Python 3.14+** | From [python.org](https://www.python.org/downloads/windows/) or `winget install Python.Python.3.14` (matches `requires-python` in `pyproject.toml` — check there if this drifts) |
| **uv** | `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` |
| **Node.js 18+** | From [nodejs.org](https://nodejs.org/) or `winget install OpenJS.NodeJS.LTS` |
| **bun** | `powershell -c "irm bun.sh/install.ps1 | iex"` |
| **Docker Desktop** | [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) with the **WSL2 backend** enabled |
| **Task** | `winget install Task.Task` (or `scoop install task`) |
| **Git** | `winget install Git.Git` |

For building the **Tauri desktop client** on Windows you also need:

| Tool | Purpose |
|------|---------|
| **Rust (stable)** | `winget install Rustlang.Rustup` then `rustup default stable` |
| **Visual Studio Build Tools** | The "Desktop development with C++" workload (MSVC linker) |
| **WebView2 Runtime** | Pre-installed on Windows 11; otherwise install the [Evergreen runtime](https://developer.microsoft.com/microsoft-edge/webview2/) |

See the [Tauri Windows prerequisites](https://v2.tauri.app/start/prerequisites/) for the
authoritative list.

## First-time setup

Use **PowerShell** (or Windows Terminal) from the repository root:

```powershell
# 1. Clone
git clone https://github.com/yourusername/agentx-source.git
cd agentx-source

# 2. Install deps + create data dirs + verify env
task setup

# 3. Configure environment
Copy-Item .env.example .env
# Edit .env — at minimum set NEO4J_PASSWORD and POSTGRES_PASSWORD

# 4. Start the database services (Docker Desktop must be running)
task db:up

# 5. Run the full stack (API + client)
task dev
```

`task dev:web` runs the client in the browser (port 1420) without the Tauri shell — handy if you
haven't installed the Rust/MSVC toolchain yet.

## GPU on Windows

The default PyPI `torch` wheel on Windows is **CPU-only**. AgentX will run fine, but the embedding and
translation models stay on the CPU regardless of `AGENTX_DEVICE`. To use an NVIDIA GPU you have two
options:

1. **Install the CUDA build of torch** into the project environment (match the CUDA version to your
   driver):

   ```powershell
   uv pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Run the backend under WSL2** (recommended), where the default Linux `torch` wheel already bundles
   the CUDA runtime. Install the NVIDIA driver on the **Windows host** (not inside WSL) and enable GPU
   support in Docker Desktop → Settings → Resources → WSL Integration.

Either way, confirm it worked:

```powershell
curl.exe -s localhost:12319/api/health | python -m json.tool
# look for "compute": { "device": "cuda", "cuda_available": true }
```

See [GPU Acceleration](../development/gpu.md) for the full device-selection reference.

## Windows gotchas

- **Docker Desktop must be running** before `task db:up` / `task dev` — the database services are
  containers.
- **Long paths / line endings:** keep the repo on an NTFS drive and let Git manage line endings
  (`git config core.autocrlf input`). The translation models download ~600 MB on first run.
- **WSL2 file performance:** if you work inside WSL2, keep the repo on the Linux filesystem
  (`~/projects/...`), not `/mnt/c/...`, to avoid slow I/O.
- **`task` not found:** ensure the install location is on your `PATH`, then restart the terminal.

Once running, continue with the [Quick Start](quickstart.md).

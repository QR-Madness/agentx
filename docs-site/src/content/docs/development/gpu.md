# GPU Acceleration

AgentX runs two on-device model workloads: the **local embedding model** (`BAAI/bge-m3` via
`sentence-transformers`) and the **translation models** (NLLB-200 + language detection). Both honor a
single compute-device setting, so enabling a GPU speeds up memory recall/consolidation **and**
translation.

## How device selection works

The device is resolved once at model load from the `AGENTX_DEVICE` environment variable:

| Value | Behavior |
|-------|----------|
| `auto` (default) | CUDA when `torch.cuda.is_available()`, else CPU |
| `cpu` | Force CPU |
| `cuda` | Force GPU; falls back to CPU (with a warning) if unavailable |
| `cuda:0` | A specific GPU index |

The same resolved device is applied to the embedding model and to both translation models — there is
no separate per-model knob.

## Verifying GPU use

After starting the API, check the live device without needing a shell into the process:

```bash
curl -s localhost:12319/api/health | jq .compute
# { "device": "cuda", "cuda_available": true }
```

The startup logs also print the resolved device for each model:

```
INFO device Compute device resolved to 'cuda' (requested=auto, cuda_available=True).
INFO embeddings Local embedding model 'BAAI/bge-m3' loaded on device 'cuda'.
INFO translation TranslationKit models loaded on device 'cuda' (...).
```

Low-level check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Installing CUDA-enabled PyTorch (local dev)

GPU use depends entirely on having a CUDA build of `torch`, and **the project installs the CPU
build by default** — `uv sync`, `uv run` and every `task` that wraps them pull torch from the `cpu`
dependency group, which carries none of the ~5 GB CUDA payload. That is the right default for most
machines; on an NVIDIA machine you have to opt out of it.

**Linux with an NVIDIA GPU:** pick the CUDA line your driver supports. The header of `nvidia-smi`
shows the newest CUDA version it can run.

- **CUDA 12.8**, the same build as the `:latest-gpu` image (driver R570 or newer):

  ```bash
  export UV_NO_SYNC=1                  # keep uv run / task from re-syncing to the CPU build
  uv sync --no-group cpu --group gpu   # torch from the PyTorch cu128 index
  ```

  uv has no environment variable for `--group`, so `UV_NO_SYNC` is what holds this in place: put it
  in your shell profile, and re-run the `uv sync` line yourself whenever `uv.lock` changes.

- **CUDA 13**, PyPI's build of the pinned torch (driver R580 or newer):

  ```bash
  export UV_NO_GROUP=cpu   # every uv sync / uv run now skips the CPU-only torch build
  uv sync                  # PyPI's torch 2.11 bundles CUDA 13.0
  ```

  The variable must stay set: `uv run` re-syncs the environment on every call, and without it the
  next `uv run` (or `task dev`) quietly swaps the CPU build back in.

**Windows:** PyPI's `torch` wheel is CPU-only on Windows, so the switch above does not help. Run the
API under WSL2 instead. See [Windows Setup](../getting-started/windows.md#gpu-on-windows).

The torch **version** lives in one place — `constraint-dependencies` in `pyproject.toml` — and applies
to every build, so the CPU and CUDA environments can never drift apart.

!!! note
    If `cuda_available` is `false` despite having a GPU, you almost always have the CPU build
    installed — check `python -c "import torch; print(torch.__version__)"`: a `+cpu` suffix means
    the shell that last synced didn't have the variable from your chosen path above set.

## Docker / production

For containerized deployments, GPU passthrough is handled by the `docker-compose.gpu.yml` overlay
(applied via `cluster:up CLUSTER=<name> NVIDIA=1`). The overlay also switches the API to the
**`qrmadness/agentx-api:latest-gpu`** image — the CUDA 12.8 build. The default `:latest` image is
CPU-only, so on its own it cannot use a GPU no matter what the overlay reserves. Source clusters that
build the image locally also need `docker-compose.gpu.build.yml`; the deployment manager adds it for
you. See [Clusters & Gateway → GPU Acceleration](../deployment/clusters.md#gpu-acceleration-nvidia-overlay).

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

GPU use depends entirely on having a CUDA build of `torch`.

- **Linux:** the default PyPI `torch` wheel **bundles the CUDA runtime**, so `task setup` /
  `uv sync` gives you a GPU-capable build automatically — just have an NVIDIA driver installed.
- **Windows:** the default PyPI `torch` wheel is **CPU-only**. You must install the CUDA build
  explicitly (or run under WSL2). See [Windows Setup](../getting-started/windows.md#gpu-on-windows).

To install the CUDA build with uv (adjust the CUDA version to your driver):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

!!! note
    If `cuda_available` is `false` despite having a GPU, you almost always have a CPU-only `torch`
    wheel installed. Reinstall from the CUDA index above.

## Docker / production

For containerized deployments, GPU passthrough is handled by the `docker-compose.gpu.yml` overlay
(applied via `cluster:up CLUSTER=<name> NVIDIA=1`). The default Linux `torch` wheel in the
`python:3.14-slim` image is CUDA-capable, so the overlay plus the NVIDIA Container Toolkit on the host
is all that's needed. See [Clusters & Gateway → GPU Acceleration](../deployment/clusters.md#gpu-acceleration-nvidia-overlay).

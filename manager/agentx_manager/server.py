"""Manager web server: REST + SSE logs + static GUI.

Security model: this process drives the Docker socket (root-equivalent), so
- it binds 127.0.0.1 by default; non-loopback requires AGENTX_MANAGER_BIND,
- every /api request needs the bearer token (AGENTX_MANAGER_TOKEN env, or
  auto-generated on first start into <root>/.manager-token, mode 0600),
- it must never be published through the gateway/tunnel; the shipped nginx
  template has no route to it.

Long operations (up can wait out a first boot) run as background jobs; the
GUI polls /api/clusters and /api/jobs/{id}.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, health, lifecycle, registry, scaffold
from .compose import SubprocessRunner
from .overlays import compose_argv
from .spec import VALID_KINDS, VALID_TUNNELS

TOKEN_FILE = ".manager-token"  # noqa: S105 — filename, not a credential


def _load_or_create_token(root: Path) -> str:
    env_token = os.environ.get("AGENTX_MANAGER_TOKEN")
    if env_token:
        return env_token
    path = root / TOKEN_FILE
    if path.is_file():
        return path.read_text().strip()
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n")
    path.chmod(0o600)
    return token


def _cluster_url(env: dict[str, str]) -> str:
    """Best-effort reachable URL: public gateway host if set, else local direct port."""
    host = env.get("AGENTX_PUBLIC_HOST", "").strip()
    if host:
        return f"https://{host}"
    return f"http://localhost:{env.get('API_PORT', '12319').strip() or '12319'}"


def _cluster_ports(env: dict[str, str]) -> dict[str, int]:
    def _port(key: str, default: str) -> int:
        try:
            return int(env.get(key, default).strip() or default)
        except ValueError:
            return int(default)

    return {
        "api": _port("API_PORT", "12319"),
        "neo4j_http": _port("NEO4J_HTTP_PORT", "7474"),
        "neo4j_bolt": _port("NEO4J_BOLT_PORT", "7687"),
        "postgres": _port("POSTGRES_PORT", "5432"),
        "redis": _port("REDIS_PORT", "6379"),
    }


@dataclass
class Job:
    id: str
    cluster: str
    action: str
    status: str = "running"  # running | done | failed
    detail: str = ""


@dataclass
class Jobs:
    lock: threading.Lock = field(default_factory=threading.Lock)
    by_id: dict[str, Job] = field(default_factory=dict)
    active: dict[str, str] = field(default_factory=dict)  # cluster → job id

    def start(self, cluster: str, action: str, work: Callable[[], lifecycle.LifecycleResult]) -> Job:
        with self.lock:
            active_id = self.active.get(cluster)
            if active_id and self.by_id[active_id].status == "running":
                raise HTTPException(409, f"{cluster} already has a running job ({self.by_id[active_id].action})")
            job = Job(id=uuid.uuid4().hex[:12], cluster=cluster, action=action)
            self.by_id[job.id] = job
            self.active[cluster] = job.id

        def runner() -> None:
            try:
                result = work()
                job.status = "done" if result.ok else "failed"
                job.detail = result.detail
            except Exception as exc:  # noqa: BLE001 — job boundary; surfaced to the client
                job.status = "failed"
                job.detail = str(exc)

        threading.Thread(target=runner, daemon=True, name=f"job-{job.id}").start()
        return job


def create_app(root: Path) -> FastAPI:
    root = root.resolve()
    token = _load_or_create_token(root)
    runner = SubprocessRunner()
    jobs = Jobs()
    net_rates = health.NetRateTracker()  # rates need successive samples → per-process state

    app = FastAPI(title="AgentX Manager", version=__version__)

    def require_token(request: Request) -> None:
        supplied = request.headers.get("X-Manager-Token", "")
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            supplied = supplied or auth.removeprefix("Bearer ")
        if not secrets.compare_digest(supplied, token):
            raise HTTPException(401, "manager token required")

    guarded = Depends(require_token)

    def get_cluster(name: str) -> registry.Cluster:
        try:
            return registry.get(root, name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/meta", dependencies=[guarded])
    def meta() -> dict[str, Any]:
        return {
            "version": __version__,
            "root": str(root),
            "mode": registry.detect_mode(root),
        }

    @app.get("/api/clusters", dependencies=[guarded])
    def clusters() -> list[dict[str, Any]]:
        out = []
        for cluster in registry.discover(root):
            cluster_status = health.status(cluster, runner)
            env = cluster.env
            out.append({
                "name": cluster.name,
                "spec": cluster.spec.to_dict(),
                "phase": cluster_status.phase,
                "services": [asdict(s) for s in cluster_status.services],
                "dir": str(cluster.cluster_dir),
                "url": _cluster_url(env),
                "ports": _cluster_ports(env),
            })
        return out

    @app.get("/api/clusters/{name}/usage", dependencies=[guarded])
    def usage(name: str) -> dict[str, Any]:
        return asdict(health.usage(get_cluster(name), runner, rates=net_rates))

    @app.post("/api/clusters", dependencies=[guarded])
    def create(payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name", "")).strip()
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(422, "name must be alphanumeric (dashes/underscores ok)")
        kind = payload.get("kind", "source")
        tunnel = payload.get("tunnel", "none")
        if kind not in VALID_KINDS or tunnel not in VALID_TUNNELS:
            raise HTTPException(422, f"kind ∈ {VALID_KINDS}, tunnel ∈ {VALID_TUNNELS}")
        try:
            result = scaffold.new_cluster(
                root, name, kind=kind,
                gateway=bool(payload.get("gateway")), tunnel=tunnel,
                gpu=bool(payload.get("gpu")),
            )
        except (FileExistsError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"dir": str(result.cluster_dir), "generated": result.generated, "notes": result.notes}

    @app.post("/api/clusters/{name}/gateway", dependencies=[guarded])
    def enable_gateway(name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        tunnel = (payload or {}).get("tunnel", "none")
        if tunnel not in VALID_TUNNELS:
            raise HTTPException(422, f"tunnel ∈ {VALID_TUNNELS}")
        result = scaffold.enable_gateway(get_cluster(name), tunnel=tunnel)
        return {"generated": result.generated, "notes": result.notes}

    @app.post("/api/clusters/{name}/{action}", dependencies=[guarded])
    def act(name: str, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        cluster = get_cluster(name)
        payload = payload or {}
        if action in ("up", "down", "restart", "rebuild", "adopt"):
            func = getattr(lifecycle, action)
            job = jobs.start(name, action, lambda: func(cluster, runner))
        elif action == "destroy":
            if payload.get("confirm") != name:
                raise HTTPException(422, f"destroy requires confirm={name!r}")
            keep = bool(payload.get("keep_data"))
            job = jobs.start(name, action,
                             lambda: lifecycle.destroy(cluster, runner, remove_data=not keep))
        else:
            raise HTTPException(404, f"unknown action {action!r}")
        return {"job": job.id}

    @app.get("/api/jobs/{job_id}", dependencies=[guarded])
    def job_status(job_id: str) -> dict[str, Any]:
        job = jobs.by_id.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return {"id": job.id, "cluster": job.cluster, "action": job.action,
                "status": job.status, "detail": job.detail}

    @app.get("/api/clusters/{name}/logs", dependencies=[guarded])
    def logs(name: str, service: str = "", tail: int = 200) -> StreamingResponse:
        cluster = get_cluster(name)
        argv = compose_argv(
            cluster.spec, cluster.root, cluster.env_file,
            ["logs", "-f", "--tail", str(tail)] + ([service] if service else []),
        )
        proc = runner.stream(argv, cwd=cluster.root)

        def stream():
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    yield f"data: {json.dumps(line.rstrip())}\n\n"
            finally:
                proc.terminate()

        return StreamingResponse(stream(), media_type="text/event-stream")

    ui_dist = Path(__file__).parent.parent / "ui" / "dist"
    if ui_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=ui_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = ui_dist / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(ui_dist / "index.html")

    return app


def serve(root: Path, host: str = "127.0.0.1", port: int = 12320) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "::1", "localhost") and not os.environ.get("AGENTX_MANAGER_BIND"):
        raise SystemExit(
            "refusing non-loopback bind without explicit AGENTX_MANAGER_BIND — "
            "the manager drives the Docker socket; expose it deliberately or not at all"
        )
    token_path = root / TOKEN_FILE
    print(f"AgentX Manager {__version__} — root {root}")
    if not os.environ.get("AGENTX_MANAGER_TOKEN"):
        _load_or_create_token(root)
        print(f"token: {token_path} (send as Authorization: Bearer …)")
    print(f"GUI: http://{host}:{port}/")
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")

"""agentx-manager CLI — argparse, thin over the core modules.

Root resolution: --root flag > AGENTX_MANAGER_ROOT env > walk up from cwd
until a directory with docker-compose.yml is found. In bundle mode (one
deployment) the cluster name may be omitted everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .compose import SubprocessRunner
from . import health, lifecycle, registry, scaffold
from .overlays import compose_argv
from .passthrough import run_agentx
from .spec import VALID_TUNNELS


def resolve_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    env_root = os.environ.get("AGENTX_MANAGER_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = Path.cwd()
    for candidate in (current, *current.parents):
        if (candidate / "docker-compose.yml").is_file():
            return candidate
    sys.exit("error: no docker-compose.yml found here or above; pass --root or set AGENTX_MANAGER_ROOT")


def validate_root(root: Path) -> None:
    """Fail loudly when running containerized without the same-path mount.

    Compose resolves bind-mount paths on the *client* side, so a manager
    container must see the deployment root at the same absolute path as the
    Docker daemon (mount ${PWD}:${PWD}).
    """
    if not root.is_dir():
        sys.exit(f"error: manager root {root} does not exist in this environment")
    if Path("/.dockerenv").exists() and not (root / "docker-compose.yml").is_file():
        sys.exit(
            f"error: {root} has no docker-compose.yml — when containerized, mount the "
            "deployment root at the SAME absolute path as on the host (…-v $PWD:$PWD)"
        )


def resolve_cluster(root: Path, name: str | None) -> registry.Cluster:
    clusters = registry.discover(root)
    if name:
        for cluster in clusters:
            if cluster.name == name:
                return cluster
        sys.exit(f"error: no cluster named {name!r} (have: {', '.join(c.name for c in clusters) or 'none'})")
    if len(clusters) == 1:
        return clusters[0]
    if not clusters:
        sys.exit("error: no deployments found")
    sys.exit(f"error: multiple clusters ({', '.join(c.name for c in clusters)}) — pass a name")


def _print_result(result: lifecycle.LifecycleResult) -> int:
    stream = sys.stdout if result.ok else sys.stderr
    print(("✓ " if result.ok else "✗ ") + result.detail, file=stream)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentx-manager", description=__doc__)
    parser.add_argument("--root", help="deployment root (repo checkout or untarred bundle)")
    parser.add_argument("--version", action="version", version=f"agentx-manager {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list deployments")

    p = sub.add_parser("status", help="phase + per-service state")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("usage", help="CPU%% / memory per cluster (docker stats)")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("new", help="scaffold a cluster (repo mode)")
    p.add_argument("name")
    p.add_argument("--kind", choices=("source", "image"), default="source")
    p.add_argument("--gateway", action="store_true")
    p.add_argument("--tunnel", choices=VALID_TUNNELS, default="none")
    p.add_argument("--gpu", action="store_true")

    p = sub.add_parser("gateway-enable", help="add the token gateway to an existing deployment")
    p.add_argument("name", nargs="?")
    p.add_argument("--tunnel", choices=VALID_TUNNELS, default="none")

    for cmd, doc in (("up", "start"), ("down", "stop"), ("restart", "restart (config-aware)"),
                     ("rebuild", "rebuild from source + restart"), ("adopt", "migrate onto the manager's compose project")):
        p = sub.add_parser(cmd, help=doc)
        p.add_argument("name", nargs="?")

    p = sub.add_parser("destroy", help="down --volumes + delete the cluster's data")
    p.add_argument("name", nargs="?")
    p.add_argument("--keep-data", action="store_true")
    p.add_argument("--confirm", metavar="NAME", help="type the cluster name to confirm")

    p = sub.add_parser("set", help="edit a deployment's overlay toggles")
    p.add_argument("name", nargs="?")
    for flag in ("gateway", "gpu", "shell", "expose"):
        p.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tunnel", choices=VALID_TUNNELS)

    p = sub.add_parser("logs", help="tail logs")
    p.add_argument("name", nargs="?")
    p.add_argument("service", nargs="?")

    p = sub.add_parser("agentx", help="run the in-image ops CLI (migrate, setup-auth, …)")
    p.add_argument("name", nargs="?")
    p.add_argument("args", nargs=argparse.REMAINDER)

    p = sub.add_parser("serve", help="run the web GUI / REST server")
    p.add_argument("--host", default=os.environ.get("AGENTX_MANAGER_BIND", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("AGENTX_MANAGER_PORT", "12320")))

    args = parser.parse_args(argv)
    root = resolve_root(args.root)
    validate_root(root)
    runner = SubprocessRunner()

    if args.command == "list":
        for cluster in registry.discover(root):
            spec = cluster.spec
            tags = [spec.kind]
            if spec.gateway:
                tags.append("gateway")
            if spec.tunnel != "none":
                tags.append(f"tunnel:{spec.tunnel}")
            if spec.gpu:
                tags.append("gpu")
            print(f"{cluster.name}  [{' '.join(tags)}]  {cluster.cluster_dir}")
        return 0

    if args.command == "status":
        cluster = resolve_cluster(root, args.name)
        result = health.status(cluster, runner)
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"{result.name}: {result.phase}")
            for svc in result.services:
                print(f"  {svc.service:<12} {svc.state}{f' ({svc.health})' if svc.health else ''}")
        return 0

    if args.command == "usage":
        cluster = resolve_cluster(root, args.name)
        result = health.usage(cluster, runner)
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"{result.name}: cpu {result.cpu_percent}%  mem {result.mem_used_bytes // 1024**2} MiB"
                  f" ({result.mem_percent}%)"
                  f"  net rx {result.net_rx_bytes // 1024**2} MiB / tx {result.net_tx_bytes // 1024**2} MiB")
            for svc in result.services:
                print(f"  {svc.service:<24} cpu {svc.cpu_percent:>5}%  mem {svc.mem_used_bytes // 1024**2} MiB"
                      f"  net rx {svc.net_rx_bytes // 1024**2} MiB / tx {svc.net_tx_bytes // 1024**2} MiB")
        return 0

    if args.command == "new":
        try:
            result = scaffold.new_cluster(
                root, args.name, kind=args.kind, gateway=args.gateway,
                tunnel=args.tunnel, gpu=args.gpu,
            )
        except (FileExistsError, FileNotFoundError) as exc:
            sys.exit(f"error: {exc}")
        print(f"✓ created {result.cluster_dir}")
        for key, value in result.generated.items():
            print(f"  {key}={value}  (generated — shown once, already written to .env)")
        for note in result.notes:
            print(f"  → {note}")
        return 0

    if args.command == "gateway-enable":
        cluster = resolve_cluster(root, args.name)
        result = scaffold.enable_gateway(cluster, tunnel=args.tunnel)
        print("✓ gateway enabled")
        for key, value in result.generated.items():
            print(f"  {key}={value}  (generated — shown once, already written to .env)")
        for note in result.notes:
            print(f"  → {note}")
        return 0

    if args.command in ("up", "down", "restart", "rebuild", "adopt"):
        cluster = resolve_cluster(root, args.name)
        func = getattr(lifecycle, args.command)
        return _print_result(func(cluster, runner))

    if args.command == "destroy":
        cluster = resolve_cluster(root, args.name)
        if args.confirm != cluster.name:
            sys.exit(f"error: destroy is irreversible — re-run with --confirm {cluster.name}")
        return _print_result(lifecycle.destroy(cluster, runner, remove_data=not args.keep_data))

    if args.command == "set":
        cluster = resolve_cluster(root, args.name)
        spec = cluster.spec
        for flag in ("gateway", "gpu", "shell", "expose"):
            value = getattr(args, flag)
            if value is not None:
                setattr(spec, flag, value)
        if args.tunnel is not None:
            spec.tunnel = args.tunnel
        spec.__post_init__()  # re-validate combinations
        from .state import compute_hashes, save_state, tracked_files
        save_state(cluster.cluster_dir, spec, compute_hashes(
            tracked_files(cluster.cluster_dir, registry.gateway_dir(cluster), cluster.env_file)
        ))
        print(f"✓ spec updated: {spec.to_dict()}")
        print("  → apply with: agentx-manager up" + (f" {cluster.name}" if args.name else ""))
        return 0

    if args.command == "logs":
        cluster = resolve_cluster(root, args.name)
        tail = ["logs", "-f"] + ([args.service] if args.service else [])
        proc = runner.stream(compose_argv(cluster.spec, cluster.root, cluster.env_file, tail), cwd=cluster.root)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            proc.terminate()
        return 0

    if args.command == "agentx":
        cluster = resolve_cluster(root, args.name)
        extra = [a for a in args.args if a != "--"]
        if extra[:1] == ["setup-auth"]:
            # Interactive (password prompt): inherit this terminal's stdio
            # instead of capturing through the runner.
            import subprocess

            from .overlays import compose_argv as _argv
            argv = _argv(cluster.spec, cluster.root, cluster.env_file,
                         ["exec", "api", "agentx", *extra])
            return subprocess.run(argv, cwd=cluster.root).returncode  # noqa: S603 — list argv
        result = run_agentx(cluster, extra, runner)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    if args.command == "serve":
        from .server import serve
        serve(root, host=args.host, port=args.port)
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

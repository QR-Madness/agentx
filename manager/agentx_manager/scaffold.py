"""Scaffolding: create clusters (repo mode) and enable gateways (both modes).

Replaces `task cluster:new`'s sed pipeline with the same on-disk result:
clusters/<name>/{.env,config/,db/} seeded from clusters/template/ and
api/defaults/, with identity + path variables filled in and secrets
generated (never logged — surfaced once in the return value).
"""

from __future__ import annotations

import secrets
import shutil
import socket
from dataclasses import dataclass, field
from pathlib import Path

from .envfile import upsert_env
from .registry import Cluster, gateway_dir
from .spec import ClusterSpec, Kind, Tunnel
from .state import compute_hashes, save_state, tracked_files

DB_SUBDIRS = ("neo4j/data", "neo4j/logs", "neo4j/plugins", "postgres", "redis")


@dataclass
class ScaffoldResult:
    cluster_dir: Path
    generated: dict[str, str] = field(default_factory=dict)  # secret name → value
    notes: list[str] = field(default_factory=list)


def _lan_ip() -> str:
    """Best-effort LAN IP for DJANGO_ALLOWED_HOSTS (mirrors `hostname -I | awk '{print $1}'`)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.254.254.254", 1))  # no traffic sent; just resolves the route
            return sock.getsockname()[0]
    except OSError:
        return ""


def new_cluster(
    root: Path,
    name: str,
    *,
    kind: Kind = "source",
    gateway: bool = False,
    tunnel: Tunnel = "none",
    gpu: bool = False,
) -> ScaffoldResult:
    template = root / "clusters" / "template"
    if not template.is_dir():
        raise FileNotFoundError(f"no clusters/template under {root} — is this a source checkout?")
    cluster_dir = root / "clusters" / name
    if cluster_dir.exists():
        raise FileExistsError(f"cluster already exists: {cluster_dir}")

    spec = ClusterSpec(
        name=name, kind=kind, gateway=gateway,
        tunnel=tunnel if gateway else "none", gpu=gpu,
    )

    for sub in DB_SUBDIRS:
        (cluster_dir / "db" / sub).mkdir(parents=True)
    (cluster_dir / "config").mkdir()
    defaults = root / "api" / "defaults"
    for pattern in ("*.yaml", "*.json"):
        for src in sorted(defaults.glob(pattern)):
            shutil.copy2(src, cluster_dir / "config" / src.name)

    env_file = cluster_dir / ".env"
    shutil.copy2(template / ".env.example", env_file)

    django_secret = secrets.token_urlsafe(48)
    updates = {
        "AGENTX_CLUSTER_NAME": name,
        "AGENTX_CONFIG_DIR": f"./clusters/{name}/config",
        "AGENTX_DB_DIR": f"./clusters/{name}/db",
        "AGENTX_GATEWAY_DIR": f"./clusters/{name}",
        "DJANGO_SECRET_KEY": django_secret,
    }
    lan_ip = _lan_ip()
    if lan_ip:
        updates["DJANGO_ALLOWED_HOSTS"] = lan_ip

    result = ScaffoldResult(cluster_dir=cluster_dir)
    result.generated["DJANGO_SECRET_KEY"] = django_secret

    if gateway:
        gateway_token = secrets.token_hex(32)
        updates["AGENTX_GATEWAY_TOKEN"] = gateway_token
        updates["AGENTX_TRUST_PROXY"] = "true"
        result.generated["AGENTX_GATEWAY_TOKEN"] = gateway_token
        (cluster_dir / "cloudflared").mkdir()
        shutil.copy2(template / "nginx.conf.example", cluster_dir / "nginx.conf")
        if tunnel == "named":
            shutil.copy2(
                template / "cloudflared" / "config.yml.example",
                cluster_dir / "cloudflared" / "config.yml",
            )
            result.notes.append(
                "named tunnel: edit cloudflared/config.yml (tunnel id + hostname) "
                "and drop credentials.json next to it"
            )
        elif tunnel == "token":
            result.notes.append(
                "token tunnel: set TUNNEL_TOKEN in .env and point the dashboard "
                "Public Hostname Service at http://nginx:80"
            )

    upsert_env(env_file, updates)
    result.notes.append("set database passwords + provider API keys in .env before up")

    cluster = Cluster(spec=spec, root=root, cluster_dir=cluster_dir, env_file=env_file)
    save_state(cluster_dir, spec, compute_hashes(
        tracked_files(cluster_dir, gateway_dir(cluster), env_file)
    ))
    return result


def enable_gateway(cluster: Cluster, *, tunnel: Tunnel = "none") -> ScaffoldResult:
    """Add gateway files + secrets to an existing deployment (repo or bundle)."""
    root = cluster.root
    gw_dir = gateway_dir(cluster)
    # Templates: repo mode has clusters/template; bundles ship gateway/*.example.
    template_nginx = root / "clusters" / "template" / "nginx.conf.example"
    template_cf = root / "clusters" / "template" / "cloudflared" / "config.yml.example"
    if not template_nginx.is_file():
        template_nginx = gw_dir / "nginx.conf.example"
        template_cf = gw_dir / "cloudflared" / "config.yml.example"
    if not template_nginx.is_file():
        raise FileNotFoundError("no nginx.conf template found (clusters/template or gateway/)")

    gw_dir.mkdir(parents=True, exist_ok=True)
    (gw_dir / "cloudflared").mkdir(exist_ok=True)
    result = ScaffoldResult(cluster_dir=cluster.cluster_dir)

    nginx_conf = gw_dir / "nginx.conf"
    if not nginx_conf.is_file():
        shutil.copy2(template_nginx, nginx_conf)
    if tunnel == "named" and not (gw_dir / "cloudflared" / "config.yml").is_file():
        shutil.copy2(template_cf, gw_dir / "cloudflared" / "config.yml")
        result.notes.append("edit cloudflared/config.yml + drop credentials.json")

    updates: dict[str, str] = {"AGENTX_TRUST_PROXY": "true"}
    if not cluster.env.get("AGENTX_GATEWAY_TOKEN"):
        token = secrets.token_hex(32)
        updates["AGENTX_GATEWAY_TOKEN"] = token
        result.generated["AGENTX_GATEWAY_TOKEN"] = token
    if not cluster.env.get("AGENTX_GATEWAY_DIR"):
        rel = gw_dir.relative_to(root) if gw_dir.is_relative_to(root) else gw_dir
        updates["AGENTX_GATEWAY_DIR"] = f"./{rel}" if not Path(rel).is_absolute() else str(rel)
    upsert_env(cluster.env_file, updates)

    cluster.spec.gateway = True
    if tunnel != "none":
        cluster.spec.tunnel = tunnel
    save_state(cluster.cluster_dir, cluster.spec, compute_hashes(
        tracked_files(cluster.cluster_dir, gw_dir, cluster.env_file)
    ))
    return result

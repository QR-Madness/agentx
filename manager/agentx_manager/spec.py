"""ClusterSpec — the explicit, persisted description of one deployment.

Replaces `task cluster:up`'s file-presence sniffing: what overlays a cluster
runs with is recorded in `.manager-state.json` (see state.py) and edited
through the CLI/GUI, not inferred each invocation. For pre-manager clusters,
`spec_from_files` bootstraps a spec from the same file-presence rules the
Taskfile used, so adoption is lossless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["source", "image"]
Tunnel = Literal["none", "token", "named"]

VALID_KINDS = ("source", "image")
VALID_TUNNELS = ("none", "token", "named")


@dataclass
class ClusterSpec:
    """What to run and how to expose it. Paths live in the registry, not here."""

    name: str
    kind: Kind = "image"
    gateway: bool = False
    tunnel: Tunnel = "none"
    expose: bool = False
    gpu: bool = False
    shell: bool = False

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {self.kind!r}")
        if self.tunnel not in VALID_TUNNELS:
            raise ValueError(f"tunnel must be one of {VALID_TUNNELS}, got {self.tunnel!r}")
        if self.tunnel != "none" and not self.gateway:
            # The named tunnel's cloudflared depends_on nginx; the token tunnel
            # without a gateway is the discouraged bare-exposure path — allow it
            # only explicitly via gateway=False + tunnel="token".
            if self.tunnel == "named":
                raise ValueError("tunnel='named' requires gateway=True (cloudflared fronts nginx)")
        if self.expose and not self.gateway:
            raise ValueError("expose=True requires gateway=True (it publishes nginx)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClusterSpec:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def spec_from_files(name: str, cluster_dir: Path, kind: Kind = "source") -> ClusterSpec:
    """Bootstrap a spec for a pre-manager cluster from file presence.

    Mirrors the legacy Taskfile rules: nginx.conf → gateway overlay;
    cloudflared/config.yml → named tunnel.
    """
    gateway = (cluster_dir / "nginx.conf").is_file()
    named = gateway and (cluster_dir / "cloudflared" / "config.yml").is_file()
    return ClusterSpec(
        name=name,
        kind=kind,
        gateway=gateway,
        tunnel="named" if named else "none",
    )

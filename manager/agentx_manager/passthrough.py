"""Day-2 ops passthrough to the in-image `agentx` CLI.

The API image ships an ops CLI (docker/agentx: migrate, setup-auth, warmup,
status, export, import). The manager doesn't reimplement those — it execs
them inside the running api container so behavior always matches the
deployed image version.
"""

from __future__ import annotations

from .compose import ComposeRunner, RunResult, SubprocessRunner
from .overlays import compose_argv
from .registry import Cluster

ALLOWED = ("migrate", "setup-auth", "warmup", "status", "version", "export", "import", "help")


def run_agentx(cluster: Cluster, args: list[str], runner: ComposeRunner | None = None,
               *, interactive: bool = False) -> RunResult:
    if not args or args[0] not in ALLOWED:
        raise ValueError(f"unsupported agentx command {args[:1]!r}; allowed: {ALLOWED}")
    runner = runner or SubprocessRunner()
    exec_flags = [] if interactive else ["-T"]
    argv = compose_argv(
        cluster.spec, cluster.root, cluster.env_file,
        ["exec", *exec_flags, "api", "agentx", *args],
    )
    return runner.run(argv, cwd=cluster.root)

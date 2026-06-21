"""Command sandboxes.

``BubblewrapSandbox`` (default) runs a command in a real FS+network jail; the bind set
follows the merged-usr layout shared by Debian-bookworm-slim (prod) and the dev host.
``LocalSubprocessSandbox`` is a bare fallback used ONLY behind ``shell.allow_unsandboxed``.
``get_sandbox`` picks one, probing once that bwrap actually works (the bind set is right).
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .env import minimal_env

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int
    sandbox: str


class Sandbox(Protocol):
    name: str

    def run(self, command: str, *, cwd: Path, timeout: float, allow_network: bool) -> ExecResult: ...


def _exec(argv: list[str], *, cwd: str, env: dict[str, str], timeout: float, sandbox: str) -> ExecResult:
    """Run ``argv`` synchronously, bounded by ``timeout`` (ADR-1: sync + cooperative)."""
    start = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - deliberately runs the agent's command, inside a bubblewrap jail
            argv, cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return ExecResult(
            stdout=proc.stdout or "", stderr=proc.stderr or "",
            exit_code=proc.returncode, timed_out=False,
            duration_ms=int((time.monotonic() - start) * 1000), sandbox=sandbox,
        )
    except subprocess.TimeoutExpired as e:
        partial_out = e.stdout if isinstance(e.stdout, str) else ""
        partial_err = e.stderr if isinstance(e.stderr, str) else ""
        return ExecResult(
            stdout=partial_out,
            stderr=f"{partial_err}\n[killed: exceeded {timeout:.0f}s timeout]",
            exit_code=-1, timed_out=True,
            duration_ms=int((time.monotonic() - start) * 1000), sandbox=sandbox,
        )


class BubblewrapSandbox:
    """Jail: FS limited to ``cwd`` (rw) + a read-only minimal rootfs; network off by default."""

    name = "bubblewrap"

    def _argv(self, command: str, *, cwd: Path, allow_network: bool) -> list[str]:
        env = minimal_env(str(cwd))
        argv = ["bwrap", "--unshare-all"]
        if allow_network:
            argv += ["--share-net"]
        argv += [
            "--die-with-parent",
            "--new-session",            # block TIOCSTI terminal-injection escapes
            "--clearenv",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",  # noqa: S108 - new tmpfs inside the jail, not a host temp path
            # Read-only rootfs (merged-usr: /bin,/lib,… are symlinks into /usr).
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/sbin", "/sbin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--ro-bind-try", "/usr/local", "/usr/local",   # python lives here in slim
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
        ]
        if allow_network:
            argv += ["--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
                     "--ro-bind-try", "/etc/ssl", "/etc/ssl"]
        for k, v in env.items():
            argv += ["--setenv", k, v]
        argv += ["--bind", str(cwd), str(cwd), "--chdir", str(cwd), "--", "sh", "-lc", command]
        return argv

    def run(self, command: str, *, cwd: Path, timeout: float, allow_network: bool) -> ExecResult:
        return _exec(
            self._argv(command, cwd=cwd, allow_network=allow_network),
            cwd=str(cwd), env={}, timeout=timeout, sandbox=self.name,
        )


class LocalSubprocessSandbox:
    """Bare fallback — env-scrub + cwd + timeout, but NO FS/network jail.

    Only reachable behind ``shell.allow_unsandboxed`` when bubblewrap is unavailable.
    """

    name = "subprocess"

    def run(self, command: str, *, cwd: Path, timeout: float, allow_network: bool) -> ExecResult:
        return _exec(
            ["sh", "-lc", command],
            cwd=str(cwd), env=minimal_env(str(cwd)), timeout=timeout, sandbox=self.name,
        )


@functools.cache
def bubblewrap_works() -> bool:
    """Probe once that bwrap is present AND the bind set actually runs a command."""
    if shutil.which("bwrap") is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="bwrap-probe-") as tmp:
            res = BubblewrapSandbox().run(
                "echo ok", cwd=Path(tmp), timeout=10.0, allow_network=False
            )
        ok = res.exit_code == 0 and res.stdout.strip() == "ok"
        if not ok:
            logger.warning("bubblewrap probe failed (rc=%s, err=%r)", res.exit_code, res.stderr[:200])
        return ok
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bubblewrap probe errored: %s", e)
        return False


def get_sandbox(*, allow_unsandboxed: bool) -> Sandbox | None:
    """The active sandbox, or ``None`` when none is safe to use (caller errors)."""
    if bubblewrap_works():
        return BubblewrapSandbox()
    if allow_unsandboxed:
        logger.warning("bubblewrap unavailable — using UNSANDBOXED subprocess (shell.allow_unsandboxed=true)")
        return LocalSubprocessSandbox()
    return None

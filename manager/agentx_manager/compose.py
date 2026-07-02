"""ComposeRunner — the seam between the manager and the docker CLI.

Everything that shells out goes through this protocol so unit tests can
inject a recorder and assert on exact argv. The real runner is a thin
subprocess wrapper around `docker compose` v2 (and plain `docker` for
stats/inspect).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class RunResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class ComposeRunner(Protocol):
    def run(self, argv: list[str], cwd: Path | None = None, timeout: int | None = None) -> RunResult:
        """Run a command, capturing output. Never raises on non-zero exit."""
        ...

    def stream(self, argv: list[str], cwd: Path | None = None) -> subprocess.Popen:
        """Start a command with piped stdout for streaming (logs -f)."""
        ...


class SubprocessRunner:
    def run(self, argv: list[str], cwd: Path | None = None, timeout: int | None = None) -> RunResult:
        try:
            proc = subprocess.run(  # noqa: S603 — argv is list-form (no shell); running docker is this module's purpose
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return RunResult(argv=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        except FileNotFoundError as exc:
            return RunResult(argv=argv, returncode=127, stdout="", stderr=str(exc))
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return RunResult(argv=argv, returncode=124, stdout=stdout, stderr=f"timeout after {timeout}s\n{stderr}")

    def stream(self, argv: list[str], cwd: Path | None = None) -> subprocess.Popen:
        return subprocess.Popen(  # noqa: S603 — argv is list-form (no shell)
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

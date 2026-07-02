"""Shared unit-test fixtures: a fake repo tree and a recording ComposeRunner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agentx_manager.compose import RunResult


class FakeRunner:
    """Records argv; replies from a queue of canned results (default: success)."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.replies: list[RunResult] = []

    def queue(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.replies.append(RunResult(argv=[], returncode=returncode, stdout=stdout, stderr=stderr))

    def run(self, argv: list[str], cwd: Path | None = None, timeout: int | None = None) -> RunResult:
        self.calls.append(list(argv))
        if self.replies:
            reply = self.replies.pop(0)
            return RunResult(argv=list(argv), returncode=reply.returncode,
                             stdout=reply.stdout, stderr=reply.stderr)
        return RunResult(argv=list(argv), returncode=0, stdout="", stderr="")

    def stream(self, argv: list[str], cwd: Path | None = None) -> subprocess.Popen:
        raise NotImplementedError("unit tests must not stream")


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Minimal source-checkout layout the registry/scaffold recognize."""
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "Taskfile.yml").write_text("version: '3'\n")
    template = tmp_path / "clusters" / "template"
    (template / "cloudflared").mkdir(parents=True)
    (template / ".env.example").write_text(
        "AGENTX_CLUSTER_NAME=\n"
        "AGENTX_CONFIG_DIR=./clusters/<name>/config\n"
        "AGENTX_DB_DIR=./clusters/<name>/db\n"
        "AGENTX_GATEWAY_DIR=./clusters/<name>\n"
        "DJANGO_SECRET_KEY=\n"
        "DJANGO_ALLOWED_HOSTS=\n"
        "AGENTX_GATEWAY_TOKEN=\n"
        "AGENTX_TRUST_PROXY=false\n"
    )
    (template / "nginx.conf.example").write_text("# nginx template\n")
    (template / "cloudflared" / "config.yml.example").write_text("# cf template\n")
    defaults = tmp_path / "api" / "defaults"
    defaults.mkdir(parents=True)
    (defaults / "agent_profiles.yaml").write_text("profiles: []\n")
    (defaults / "memory_settings.json").write_text("{}\n")
    return tmp_path


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Untarred deploy-bundle layout (bundle mode: root == deployment)."""
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "gateway" / "cloudflared").mkdir(parents=True)
    (tmp_path / "gateway" / "nginx.conf.example").write_text("# nginx template\n")
    (tmp_path / "gateway" / "cloudflared" / "config.yml.example").write_text("# cf template\n")
    (tmp_path / ".env").write_text("AGENTX_CLUSTER_NAME=myhost\n")
    return tmp_path

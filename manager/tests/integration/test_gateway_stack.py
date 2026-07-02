"""Docker integration tests (pytest -m docker).

Builds a temp *bundle-mode* deployment that uses the REAL gateway overlay and
REAL nginx template from the repo, with a lightweight fake `api` service
(nginx:alpine answering on the API's container port) standing in for the
4.4 GB API image. Validates through the manager:

- lifecycle up/down under the per-cluster compose project
- gateway behavior end-to-end: tokenless health probe, 401/200 token gate,
  rate-limit 429s (expose mode, TCP-peer key)
- the inode fix: edit gateway/nginx.conf → manager restart → new config live
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from agentx_manager import lifecycle, registry
from agentx_manager.spec import ClusterSpec
from agentx_manager.state import save_state

REPO_ROOT = Path(__file__).resolve().parents[3]
GATEWAY_PORT = 18471
TOKEN = "t" * 64

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0  # noqa: S603,S607
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


FAKE_API_COMPOSE = """\
services:
  api:
    image: nginx:1.27-alpine
    container_name: ${AGENTX_CLUSTER_NAME:-agent}-api
    volumes:
      - ./fake-api.conf:/etc/nginx/conf.d/default.conf:ro
    healthcheck:
      test: ["CMD", "wget", "-q", "-O", "/dev/null", "http://127.0.0.1:12319/api/health"]
      interval: 2s
      timeout: 2s
      retries: 10
    profiles:
      - production
"""

FAKE_API_CONF = """\
server {
    listen 12319;
    location /api/health { add_header Content-Type text/plain; return 200 "fake api healthy\\n"; }
    location / { add_header Content-Type text/plain; return 200 "fake api\\n"; }
}
"""


@pytest.fixture(scope="module")
def deployment():
    if not _docker_available():
        pytest.skip("docker daemon not available")
    # Not tmp_path: the docker daemon may refuse bind mounts from /tmp
    # (Docker Desktop file-sharing, rootless daemons). A repo-local dir is
    # always mountable.
    root = Path(__file__).parent / ".int-run"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    (root / "docker-compose.yml").write_text(FAKE_API_COMPOSE)
    (root / "fake-api.conf").write_text(FAKE_API_CONF)
    shutil.copy2(REPO_ROOT / "docker-compose.gateway.yml", root / "docker-compose.gateway.yml")
    shutil.copy2(REPO_ROOT / "docker-compose.gateway.expose.yml", root / "docker-compose.gateway.expose.yml")
    (root / "gateway").mkdir()
    shutil.copy2(REPO_ROOT / "clusters" / "template" / "nginx.conf.example", root / "gateway" / "nginx.conf")
    (root / ".env").write_text(
        "AGENTX_CLUSTER_NAME=mgrtest\n"
        f"AGENTX_GATEWAY_TOKEN={TOKEN}\n"
        "AGENTX_GATEWAY_DIR=./gateway\n"
        "AGENTX_GATEWAY_BIND=127.0.0.1\n"
        f"AGENTX_GATEWAY_PORT={GATEWAY_PORT}\n"
    )
    spec = ClusterSpec(name="mgrtest", kind="image", gateway=True, expose=True)
    save_state(root, spec, {})
    [cluster] = registry.discover(root)
    assert cluster.spec == spec

    yield cluster

    lifecycle.down(cluster, volumes=True)
    shutil.rmtree(root, ignore_errors=True)


def _get(path: str, token: str | None = None, timeout: float = 5.0) -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{GATEWAY_PORT}{path}")
    request.add_header("Host", "localhost")
    if token:
        request.add_header("AgentX-Gateway-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — local test URL
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _get_body(path: str, token: str | None = None) -> str:
    request = urllib.request.Request(f"http://127.0.0.1:{GATEWAY_PORT}{path}")
    request.add_header("Host", "localhost")
    if token:
        request.add_header("AgentX-Gateway-Token", token)
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        return response.read().decode()


def _wait_gateway(deadline_s: int = 90) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            if _get("/__gateway_health") == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError("gateway never became reachable")


def test_up_and_gateway_token_gate(deployment):
    result = lifecycle.up(deployment)
    assert result.ok, result.detail
    _wait_gateway()

    assert _get("/__gateway_health") == 200          # tokenless health probe
    assert _get("/api/health") == 401                # no token
    assert _get("/api/health", token="x" * 64) == 401  # wrong token
    assert _get("/api/health", token=TOKEN) == 200   # valid token


def test_rate_limit_429_on_burst(deployment):
    codes = [_get("/api/health", token=TOKEN) for _ in range(45)]
    assert codes.count(429) > 0, f"no 429 in burst: {codes}"
    assert codes.count(200) >= 20  # burst capacity passes first


def test_config_edit_then_restart_goes_live(deployment):
    """The inode fix: a plain manager restart must pick up an edited nginx.conf."""
    conf = deployment.cluster_dir / "gateway" / "nginx.conf"
    original = conf.read_text()
    edited = original.replace('return 200 "ok\\n";', 'return 200 "ok-edited\\n";')
    assert edited != original, "nginx template health body changed upstream — update this test"
    conf.write_text(edited)

    result = lifecycle.restart(deployment)
    assert result.ok, result.detail
    assert "force-recreated nginx" in result.detail
    _wait_gateway()
    assert _get_body("/__gateway_health") == "ok-edited\n"

    conf.write_text(original)
    lifecycle.restart(deployment)


def test_down_leaves_nothing(deployment):
    result = lifecycle.down(deployment)
    assert result.ok
    ps = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=com.docker.compose.project=agentx-mgrtest"],  # noqa: S603,S607
        capture_output=True, text=True,
    )
    assert ps.stdout.strip() == ""

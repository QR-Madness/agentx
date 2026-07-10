"""Server auth + surface tests over a temp bundle root (no docker: status
calls go through subprocess and fail soft — we assert auth and shapes)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agentx_manager.server import TOKEN_FILE, create_app


@pytest.fixture
def client(bundle: Path, monkeypatch):
    monkeypatch.delenv("AGENTX_MANAGER_TOKEN", raising=False)
    app = create_app(bundle)
    token = (bundle / TOKEN_FILE).read_text().strip()
    return TestClient(app), token


def test_api_requires_token(client):
    http, _ = client
    assert http.get("/api/meta").status_code == 401
    assert http.get("/api/clusters").status_code == 401
    assert http.post("/api/clusters/x/up").status_code == 401


def test_token_grants_access_and_meta(client):
    http, token = client
    response = http.get("/api/meta", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "bundle"
    assert body["version"]


def test_token_file_is_owner_only(bundle: Path, client):
    assert (bundle / TOKEN_FILE).stat().st_mode & 0o777 == 0o600


def test_env_token_overrides_file(bundle: Path, monkeypatch):
    monkeypatch.setenv("AGENTX_MANAGER_TOKEN", "envtoken")
    http = TestClient(create_app(bundle))
    assert http.get("/api/meta", headers={"X-Manager-Token": "envtoken"}).status_code == 200
    assert http.get("/api/meta", headers={"X-Manager-Token": "wrong"}).status_code == 401


def test_destroy_requires_typed_confirmation(client):
    http, token = client
    headers = {"Authorization": f"Bearer {token}"}
    response = http.post("/api/clusters/myhost/destroy", json={}, headers=headers)
    assert response.status_code == 422
    assert "confirm" in response.json()["detail"]


def test_clusters_include_url_and_ports(client):
    http, token = client
    headers = {"Authorization": f"Bearer {token}"}
    body = http.get("/api/clusters", headers=headers).json()
    assert body[0]["url"] == "http://localhost:12319"
    assert body[0]["ports"] == {
        "api": 12319,
        "neo4j_http": 7474,
        "neo4j_bolt": 7687,
        "postgres": 5432,
        "redis": 6379,
    }


def test_unknown_cluster_404(client):
    http, token = client
    headers = {"Authorization": f"Bearer {token}"}
    assert http.get("/api/clusters/nope/usage", headers=headers).status_code == 404


def test_create_cluster_validates_input(client):
    http, token = client
    headers = {"Authorization": f"Bearer {token}"}
    assert http.post("/api/clusters", json={"name": "../evil"}, headers=headers).status_code == 422
    assert http.post("/api/clusters", json={"name": "ok", "kind": "bogus"}, headers=headers).status_code == 422


# ---------------------------------------------------------------------------
# Connection info (share links)


def test_connection_requires_token_and_404s(client):
    http, token = client
    assert http.get("/api/clusters/myhost/connection").status_code == 401
    headers = {"Authorization": f"Bearer {token}"}
    assert http.get("/api/clusters/nope/connection", headers=headers).status_code == 404


def test_connection_bare_cluster_shape(client):
    http, token = client
    response = http.get("/api/clusters/myhost/connection",
                        headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body == {
        "name": "myhost",
        "url": "http://localhost:12319",
        "public_host": "",
        "api_port": 12319,
        "gateway_enabled": False,
        "tunnel": "none",
        "gateway_token": "",
        "cors_origins": [],
        "auth_enabled": True,
    }


def test_connection_reads_env(bundle: Path, client):
    http, token = client
    with (bundle / ".env").open("a") as env:
        env.write(
            "AGENTX_PUBLIC_HOST=agentx.example.com\n"
            "AGENTX_GATEWAY_TOKEN=tok-abc123\n"
            "CORS_ALLOWED_ORIGINS=https://app.example.com, https://beta.example.com\n"
            "AGENTX_AUTH_ENABLED=false\n"
        )
    body = http.get("/api/clusters/myhost/connection",
                    headers={"Authorization": f"Bearer {token}"}).json()
    assert body["url"] == "https://agentx.example.com"
    assert body["public_host"] == "agentx.example.com"
    assert body["gateway_token"] == "tok-abc123"
    assert body["cors_origins"] == ["https://app.example.com", "https://beta.example.com"]
    assert body["auth_enabled"] is False


def test_gateway_token_not_in_clusters_listing(bundle: Path, client):
    http, token = client
    (bundle / ".env").open("a").write("AGENTX_GATEWAY_TOKEN=tok-abc123\n")
    listing = http.get("/api/clusters",
                       headers={"Authorization": f"Bearer {token}"})
    assert "tok-abc123" not in listing.text


# ---------------------------------------------------------------------------
# Gateway enable (endpoint predates these tests)


def test_enable_gateway_generates_token_once(client):
    http, token = client
    headers = {"Authorization": f"Bearer {token}"}
    response = http.post("/api/clusters/myhost/gateway", json={"tunnel": "none"}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["generated"]["AGENTX_GATEWAY_TOKEN"]
    assert isinstance(body["notes"], list)
    # The connection endpoint now reflects the persisted gateway state.
    connection = http.get("/api/clusters/myhost/connection", headers=headers).json()
    assert connection["gateway_enabled"] is True
    assert connection["gateway_token"] == body["generated"]["AGENTX_GATEWAY_TOKEN"]


def test_enable_gateway_rejects_bad_tunnel(client):
    http, token = client
    response = http.post("/api/clusters/myhost/gateway", json={"tunnel": "bogus"},
                         headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Version truth


def test_meta_repo_version_none_in_bundle_mode(client):
    http, token = client
    body = http.get("/api/meta", headers={"Authorization": f"Bearer {token}"}).json()
    assert body["repo_version"] is None


def test_read_repo_version_parses_api_block(tmp_path: Path):
    from agentx_manager.server import read_repo_version

    (tmp_path / "versions.yaml").write_text(
        "# comment\n"
        "api:\n"
        "  # semantic version\n"
        "  version: \"1.2.3\"\n"
        "  protocol_version: 1\n"
        "client:\n"
        "  version: \"9.9.9\"\n"
    )
    assert read_repo_version(tmp_path) == "1.2.3"
    assert read_repo_version(tmp_path / "missing") is None


def test_clusters_api_version_probed_and_cached(bundle: Path, monkeypatch):
    import agentx_manager.server as server_mod

    calls: list[int] = []

    def fake_fetch(port: int, timeout: float = 0.8) -> str:
        calls.append(port)
        return "9.9.9"

    class FakeStatus:
        phase = "up"
        services = []

    monkeypatch.delenv("AGENTX_MANAGER_TOKEN", raising=False)
    monkeypatch.setattr(server_mod, "_fetch_api_version", fake_fetch)
    monkeypatch.setattr(server_mod.health, "status", lambda cluster, runner: FakeStatus())
    from agentx_manager.server import TOKEN_FILE, create_app
    http = TestClient(create_app(bundle))
    token = (bundle / TOKEN_FILE).read_text().strip()
    headers = {"Authorization": f"Bearer {token}"}

    first = http.get("/api/clusters", headers=headers).json()
    second = http.get("/api/clusters", headers=headers).json()
    assert first[0]["api_version"] == "9.9.9"
    assert second[0]["api_version"] == "9.9.9"
    assert calls == [12319]  # cached after the first success


def test_clusters_api_version_failure_not_cached(bundle: Path, monkeypatch):
    import agentx_manager.server as server_mod

    calls: list[int] = []

    def fake_fetch(port: int, timeout: float = 0.8) -> str:
        calls.append(port)
        return ""

    class FakeStatus:
        phase = "up"
        services = []

    monkeypatch.delenv("AGENTX_MANAGER_TOKEN", raising=False)
    monkeypatch.setattr(server_mod, "_fetch_api_version", fake_fetch)
    monkeypatch.setattr(server_mod.health, "status", lambda cluster, runner: FakeStatus())
    from agentx_manager.server import TOKEN_FILE, create_app
    http = TestClient(create_app(bundle))
    token = (bundle / TOKEN_FILE).read_text().strip()
    headers = {"Authorization": f"Bearer {token}"}

    assert http.get("/api/clusters", headers=headers).json()[0]["api_version"] == ""
    assert http.get("/api/clusters", headers=headers).json()[0]["api_version"] == ""
    assert calls == [12319, 12319]  # retried — failures are never cached


def test_clusters_api_version_skipped_when_down(bundle: Path, monkeypatch):
    import agentx_manager.server as server_mod

    def fake_fetch(port: int, timeout: float = 0.8) -> str:
        raise AssertionError("must not probe a down cluster")

    monkeypatch.delenv("AGENTX_MANAGER_TOKEN", raising=False)
    monkeypatch.setattr(server_mod, "_fetch_api_version", fake_fetch)
    from agentx_manager.server import TOKEN_FILE, create_app
    http = TestClient(create_app(bundle))
    token = (bundle / TOKEN_FILE).read_text().strip()
    # No docker in unit tests → status fails soft → phase "down".
    body = http.get("/api/clusters", headers={"Authorization": f"Bearer {token}"}).json()
    assert body[0]["api_version"] == ""

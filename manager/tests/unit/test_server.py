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

"""Scaffold output, registry discovery, env parsing, and state round-trips."""

from pathlib import Path

import pytest

from agentx_manager import registry, scaffold
from agentx_manager.envfile import parse_env, upsert_env
from agentx_manager.spec import ClusterSpec, spec_from_files
from agentx_manager.state import changed_services, compute_hashes, load_state, save_state


def test_new_cluster_layout_and_env(repo: Path):
    result = scaffold.new_cluster(repo, "prod", gateway=True, tunnel="named")
    d = result.cluster_dir
    assert (d / "db" / "neo4j" / "data").is_dir()
    assert (d / "db" / "postgres").is_dir()
    assert (d / "config" / "agent_profiles.yaml").is_file()
    assert (d / "nginx.conf").is_file()
    assert (d / "cloudflared" / "config.yml").is_file()

    env = parse_env(d / ".env")
    assert env["AGENTX_CLUSTER_NAME"] == "prod"
    assert env["AGENTX_CONFIG_DIR"] == "./clusters/prod/config"
    assert env["AGENTX_DB_DIR"] == "./clusters/prod/db"
    assert env["AGENTX_GATEWAY_DIR"] == "./clusters/prod"
    assert env["AGENTX_TRUST_PROXY"] == "true"
    assert len(env["AGENTX_GATEWAY_TOKEN"]) == 64
    assert env["DJANGO_SECRET_KEY"]
    assert result.generated["AGENTX_GATEWAY_TOKEN"] == env["AGENTX_GATEWAY_TOKEN"]


def test_new_cluster_without_gateway_keeps_trust_proxy_off(repo: Path):
    scaffold.new_cluster(repo, "lan")
    env = parse_env(repo / "clusters" / "lan" / ".env")
    assert env["AGENTX_TRUST_PROXY"] == "false"
    assert env["AGENTX_GATEWAY_TOKEN"] == ""
    assert not (repo / "clusters" / "lan" / "nginx.conf").exists()


def test_new_cluster_refuses_duplicates(repo: Path):
    scaffold.new_cluster(repo, "prod")
    with pytest.raises(FileExistsError):
        scaffold.new_cluster(repo, "prod")


def test_registry_discovers_repo_clusters_and_skips_template(repo: Path):
    scaffold.new_cluster(repo, "a")
    scaffold.new_cluster(repo, "b", gateway=True, tunnel="named")
    clusters = registry.discover(repo)
    assert [c.name for c in clusters] == ["a", "b"]
    assert all(c.spec.kind == "source" for c in clusters)
    b = clusters[1]
    assert b.spec.gateway and b.spec.tunnel == "named"


def test_registry_bootstraps_premanager_cluster_from_files(repo: Path):
    # Simulate a cluster made by the old Taskfile: files, no state.
    d = repo / "clusters" / "legacy"
    (d / "cloudflared").mkdir(parents=True)
    (d / ".env").write_text("AGENTX_CLUSTER_NAME=legacy\n")
    (d / "nginx.conf").write_text("# conf\n")
    (d / "cloudflared" / "config.yml").write_text("# cf\n")
    [cluster] = registry.discover(repo)
    assert cluster.spec.gateway is True
    assert cluster.spec.tunnel == "named"
    assert cluster.spec.kind == "source"


def test_registry_bundle_mode(bundle: Path):
    clusters = registry.discover(bundle)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.name == "myhost"
    assert cluster.spec.kind == "image"
    assert cluster.cluster_dir == bundle
    assert registry.detect_mode(bundle) == "bundle"


def test_bundle_gateway_enable_generates_secret_and_files(bundle: Path):
    [cluster] = registry.discover(bundle)
    result = scaffold.enable_gateway(cluster, tunnel="token")
    env = parse_env(bundle / ".env")
    assert (bundle / "gateway" / "nginx.conf").is_file()
    assert len(env["AGENTX_GATEWAY_TOKEN"]) == 64
    assert env["AGENTX_TRUST_PROXY"] == "true"
    assert env["AGENTX_GATEWAY_DIR"] == "./gateway"
    assert result.generated["AGENTX_GATEWAY_TOKEN"] == env["AGENTX_GATEWAY_TOKEN"]
    # Spec persisted; rediscovery sees the gateway without file sniffing.
    [cluster2] = registry.discover(bundle)
    assert cluster2.spec.gateway is True
    assert cluster2.spec.tunnel == "token"


def test_envfile_upsert_preserves_comments_and_appends(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# comment\nA=1\nB=2\n")
    upsert_env(env, {"B": "3", "C": "4"})
    text = env.read_text()
    assert "# comment" in text
    assert parse_env(env) == {"A": "1", "B": "3", "C": "4"}


def test_state_roundtrip_and_change_detection(tmp_path: Path):
    spec = ClusterSpec(name="x", gateway=True)
    f1 = tmp_path / "nginx.conf"
    f1.write_text("v1")
    files = {"nginx:nginx.conf": f1, "env:.env": tmp_path / ".env"}
    old = compute_hashes(files)
    save_state(tmp_path, spec, old)

    loaded = load_state(tmp_path)
    assert loaded is not None and loaded.spec == spec and loaded.hashes == old

    assert changed_services(old, compute_hashes(files)) == set()
    f1.write_text("v2")
    assert changed_services(old, compute_hashes(files)) == {"nginx"}
    (tmp_path / ".env").write_text("K=1")
    assert changed_services(old, compute_hashes(files)) == {"*"}


def test_spec_from_files_variants(tmp_path: Path):
    assert spec_from_files("x", tmp_path).gateway is False
    (tmp_path / "nginx.conf").write_text("#")
    assert spec_from_files("x", tmp_path).gateway is True
    assert spec_from_files("x", tmp_path).tunnel == "none"
    (tmp_path / "cloudflared").mkdir()
    (tmp_path / "cloudflared" / "config.yml").write_text("#")
    assert spec_from_files("x", tmp_path).tunnel == "named"

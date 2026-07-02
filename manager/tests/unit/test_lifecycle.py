"""Lifecycle behavior against the recording runner: argv shapes, config-change
force-recreate, first-boot waiting, destroy safety."""

from pathlib import Path

from agentx_manager import lifecycle, registry, scaffold
from agentx_manager.state import load_state


def _cluster(repo: Path, name: str = "prod", **kwargs):
    scaffold.new_cluster(repo, name, **kwargs)
    return registry.get(repo, name)


def _flat(calls):
    return [" ".join(argv) for argv in calls]


def test_up_uses_project_and_profile(repo, fake_runner):
    cluster = _cluster(repo)
    result = lifecycle.up(cluster, fake_runner)
    assert result.ok
    [call] = fake_runner.calls
    joined = " ".join(call)
    assert "-p agentx-prod" in joined
    assert "--profile production" in joined
    assert joined.endswith("up -d")
    assert f"-f {repo / 'docker-compose.yml'}" in joined
    assert f"-f {repo / 'docker-compose.build.yml'}" in joined  # kind=source


def test_up_force_recreates_changed_service(repo, fake_runner):
    cluster = _cluster(repo, gateway=True)
    lifecycle.up(cluster, fake_runner)  # baseline hashes persisted
    (cluster.cluster_dir / "nginx.conf").write_text("# edited\n")
    result = lifecycle.up(cluster, fake_runner)
    assert result.ok
    assert "force-recreated nginx" in result.detail
    assert _flat(fake_runner.calls)[-1].endswith("up -d --force-recreate nginx")
    # Hashes refreshed: a third up is clean.
    result = lifecycle.up(cluster, fake_runner)
    assert _flat(fake_runner.calls)[-1].endswith("up -d")


def test_env_change_recreates_everything(repo, fake_runner):
    cluster = _cluster(repo)
    lifecycle.up(cluster, fake_runner)
    (cluster.cluster_dir / ".env").write_text(cluster.env_file.read_text() + "NEW=1\n")
    lifecycle.up(cluster, fake_runner)
    assert _flat(fake_runner.calls)[-1].endswith("up -d --force-recreate")


def test_restart_is_config_aware(repo, fake_runner):
    cluster = _cluster(repo, gateway=True)
    lifecycle.up(cluster, fake_runner)
    # No changes → plain restart.
    result = lifecycle.restart(cluster, fake_runner)
    assert result.ok
    assert _flat(fake_runner.calls)[-1].endswith("restart")
    # Edited nginx.conf → restart becomes a force-recreate up (the inode fix).
    (cluster.cluster_dir / "nginx.conf").write_text("# edited\n")
    result = lifecycle.restart(cluster, fake_runner)
    assert "force-recreated nginx" in result.detail
    assert _flat(fake_runner.calls)[-1].endswith("up -d --force-recreate nginx")


def test_up_waits_out_first_boot(repo, fake_runner, monkeypatch):
    cluster = _cluster(repo)
    monkeypatch.setattr(lifecycle, "POLL_INTERVAL", 0)
    # up fails with dependency error; api inspect says running+unhealthy with
    # init logs; then healthy; retry up succeeds.
    fake_runner.queue(returncode=1, stderr="dependency failed to start: container prod-api is unhealthy")
    fake_runner.queue(stdout="running unhealthy")                  # docker inspect
    fake_runner.queue(stdout="Applying migrations... Downloading") # docker logs
    fake_runner.queue(stdout="running healthy")                    # docker inspect
    fake_runner.queue()                                            # retry up
    result = lifecycle.up(cluster, fake_runner)
    assert result.ok
    assert "first-boot" in result.detail


def test_up_does_not_wait_on_real_failure(repo, fake_runner):
    cluster = _cluster(repo)
    fake_runner.queue(returncode=1, stderr="dependency failed to start: container prod-api is unhealthy")
    fake_runner.queue(stdout="exited none")  # docker inspect: crashed
    result = lifecycle.up(cluster, fake_runner)
    assert not result.ok
    assert "exited" in result.detail


def test_destroy_removes_cluster_dir_repo_mode(repo, fake_runner):
    cluster = _cluster(repo)
    result = lifecycle.destroy(cluster, fake_runner)
    assert result.ok
    assert not cluster.cluster_dir.exists()
    assert _flat(fake_runner.calls)[-1].endswith("down --volumes")


def test_destroy_bundle_mode_never_deletes_root(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    (bundle / "data").mkdir()
    (bundle / "data" / "keep.db").write_text("x")
    result = lifecycle.destroy(cluster, fake_runner)
    assert result.ok
    assert bundle.exists() and (bundle / ".env").is_file()  # root untouched
    assert not (bundle / "data").exists()                   # data removed


def test_rebuild_rejected_for_image_deployments(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    result = lifecycle.rebuild(cluster, fake_runner)
    assert not result.ok
    assert "source" in result.detail


def test_adopt_downs_legacy_project_then_ups(repo, fake_runner):
    cluster = _cluster(repo)
    result = lifecycle.adopt(cluster, fake_runner)
    assert result.ok
    first, second = _flat(fake_runner.calls)
    assert "-p" not in first.split() and first.endswith("down")  # legacy default project
    assert "-p agentx-prod" in second and second.endswith("up -d")


def test_up_persists_state(repo, fake_runner):
    cluster = _cluster(repo, gateway=True)
    lifecycle.up(cluster, fake_runner)
    state = load_state(cluster.cluster_dir)
    assert state is not None
    assert state.spec == cluster.spec
    assert any(key.startswith("nginx:") for key in state.hashes)

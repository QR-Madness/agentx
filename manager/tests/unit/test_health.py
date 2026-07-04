"""Status/usage filtering against the recording runner: the bundle's manager
container shares the deployment's compose project by design, so it shows up
in project-scoped queries as an orphan "service" — health must filter it (and
any other orphan) out via the config-derived service set."""

from agentx_manager import health, registry

STACK = "postgres\nredis\nneo4j\napi\n"


def _row(service: str, state: str = "running", health_str: str = "") -> str:
    return f'{{"Service":"{service}","State":"{state}","Health":"{health_str}"}}'


def _flat(calls):
    return [" ".join(argv) for argv in calls]


def test_status_manager_only_project_is_down(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    fake_runner.queue(stdout=_row("manager"))  # compose ps: only the manager orphan
    fake_runner.queue(stdout=STACK)            # config --services
    result = health.status(cluster, fake_runner)
    assert result.phase == "down"
    assert result.services == []
    ps_call, config_call = _flat(fake_runner.calls)
    assert "-p agentx-myhost" in ps_call and ps_call.endswith("ps --format json")
    assert "--profile production" in config_call and config_call.endswith("config --services")


def test_status_filters_manager_from_mixed_output(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    rows = [_row("manager")] + [_row(s, health_str="healthy") for s in ("postgres", "redis", "neo4j", "api")]
    fake_runner.queue(stdout="\n".join(rows))
    fake_runner.queue(stdout=STACK)
    fake_runner.queue(stdout="running healthy")  # docker inspect myhost-api
    result = health.status(cluster, fake_runner)
    assert result.phase == "up"
    assert sorted(s.service for s in result.services) == ["api", "neo4j", "postgres", "redis"]


def test_status_config_failure_keeps_current_behavior(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    fake_runner.queue(stdout=_row("manager"))
    fake_runner.queue(returncode=1, stderr="boom")  # config --services fails → no filtering
    fake_runner.queue(returncode=1)                 # docker inspect: api absent
    result = health.status(cluster, fake_runner)
    assert result.phase == "degraded"
    assert [s.service for s in result.services] == ["manager"]


def test_status_down_makes_no_config_call(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    fake_runner.queue(stdout="")  # compose ps: nothing
    result = health.status(cluster, fake_runner)
    assert result.phase == "down"
    assert len(fake_runner.calls) == 1  # empty ps skips config --services entirely


def test_usage_excludes_manager_container(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    fake_runner.queue(stdout="aaa manager\nbbb api\n")  # docker ps: id + service label
    fake_runner.queue(stdout=STACK)                     # config --services
    fake_runner.queue(stdout='{"Name":"myhost-api","CPUPerc":"5.0%","MemUsage":"100MiB / 1GiB",'
                             '"MemPerc":"9.8%","NetIO":"1.5MB / 300kB"}')
    result = health.usage(cluster, fake_runner)
    listing, _config, stats = _flat(fake_runner.calls)
    assert 'label=com.docker.compose.project=agentx-myhost' in listing
    assert '{{.Label "com.docker.compose.service"}}' in listing
    assert "bbb" in stats and "aaa" not in stats
    assert result.cpu_percent == 5.0
    assert result.net_rx_bytes == 1_500_000 and result.net_tx_bytes == 300_000


def test_usage_manager_only_returns_zero_without_stats(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    fake_runner.queue(stdout="aaa manager\n")
    fake_runner.queue(stdout=STACK)
    result = health.usage(cluster, fake_runner)
    assert len(fake_runner.calls) == 2  # no docker stats call for an empty stack
    assert result.cpu_percent == 0.0 and result.mem_used_bytes == 0


def test_net_rate_tracker_derives_rates_from_deltas():
    clock = iter([100.0, 105.0, 110.0]).__next__
    tracker = health.NetRateTracker(clock=clock)
    assert tracker.rates("c", 1_000_000, 50_000) == (0.0, 0.0)  # first sample: no baseline
    rx, tx = tracker.rates("c", 51_000_000, 550_000)            # +50MB / +500kB over 5s
    assert rx == 10_000_000.0 and tx == 100_000.0
    rx, tx = tracker.rates("c", 1_000, 100)                     # counter reset (restart) → clamp
    assert rx == 0.0 and tx == 0.0


def test_usage_feeds_rate_tracker(bundle, fake_runner):
    [cluster] = registry.discover(bundle)
    tracker = health.NetRateTracker(clock=iter([0.0, 5.0]).__next__)
    stats_row = '{{"Name":"myhost-api","CPUPerc":"1%","MemUsage":"1MiB / 1GiB","MemPerc":"0.1%","NetIO":"{net}"}}'
    for net in ("100MB / 1MB", "150MB / 1MB"):
        fake_runner.queue(stdout="bbb api\n")
        fake_runner.queue(stdout=STACK)
        fake_runner.queue(stdout=stats_row.format(net=net))
    health.usage(cluster, fake_runner, rates=tracker)
    result = health.usage(cluster, fake_runner, rates=tracker)
    assert result.net_rx_rate == 10_000_000.0  # +50MB over 5s
    assert result.net_tx_rate == 0.0

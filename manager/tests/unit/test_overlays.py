"""Overlay-assembly truth table: every toggle combination → exact file list."""

from pathlib import Path

import pytest

from agentx_manager.overlays import (
    BASE, BUILD, GATEWAY, GATEWAY_EXPOSE, GPU, SHELL, TUNNEL_NAMED, TUNNEL_TOKEN,
    compose_argv, compose_files, project_name,
)
from agentx_manager.spec import ClusterSpec


TRUTH_TABLE = [
    # (spec kwargs, expected files)
    ({"kind": "image"}, [BASE]),
    ({"kind": "source"}, [BASE, BUILD]),
    ({"kind": "image", "gateway": True}, [BASE, GATEWAY]),
    ({"kind": "image", "gateway": True, "expose": True}, [BASE, GATEWAY, GATEWAY_EXPOSE]),
    ({"kind": "image", "gateway": True, "tunnel": "token"}, [BASE, GATEWAY, TUNNEL_TOKEN]),
    ({"kind": "image", "gateway": True, "tunnel": "named"}, [BASE, GATEWAY, TUNNEL_NAMED]),
    ({"kind": "image", "tunnel": "token"}, [BASE, TUNNEL_TOKEN]),  # bare tunnel (discouraged, allowed)
    ({"kind": "source", "gateway": True, "tunnel": "named", "gpu": True},
     [BASE, BUILD, GATEWAY, TUNNEL_NAMED, GPU]),
    ({"kind": "source", "shell": True}, [BASE, BUILD, SHELL]),
    ({"kind": "source", "gateway": True, "expose": True, "gpu": True, "shell": True},
     [BASE, BUILD, GATEWAY, GATEWAY_EXPOSE, GPU, SHELL]),
]


@pytest.mark.parametrize("kwargs,expected", TRUTH_TABLE)
def test_compose_files_truth_table(kwargs, expected):
    assert compose_files(ClusterSpec(name="x", **kwargs)) == expected


def test_base_always_first_and_unique():
    for kwargs, _ in TRUTH_TABLE:
        files = compose_files(ClusterSpec(name="x", **kwargs))
        assert files[0] == BASE
        assert len(files) == len(set(files))


def test_invalid_combinations_rejected():
    with pytest.raises(ValueError):
        ClusterSpec(name="x", tunnel="named")  # named tunnel needs the gateway
    with pytest.raises(ValueError):
        ClusterSpec(name="x", expose=True)  # expose publishes nginx
    with pytest.raises(ValueError):
        ClusterSpec(name="x", kind="bogus")  # type: ignore[arg-type] — runtime validation is the subject
    with pytest.raises(ValueError):
        ClusterSpec(name="x", tunnel="bogus")  # type: ignore[arg-type]


def test_project_name_is_per_cluster():
    assert project_name(ClusterSpec(name="prod")) == "agentx-prod"
    assert project_name(ClusterSpec(name="a")) != project_name(ClusterSpec(name="b"))


def test_compose_argv_shape():
    spec = ClusterSpec(name="prod", kind="source", gateway=True, tunnel="named")
    argv = compose_argv(spec, Path("/repo"), Path("/repo/clusters/prod/.env"), ["up", "-d"])
    assert argv[:4] == ["docker", "compose", "-p", "agentx-prod"]
    assert ["--env-file", "/repo/clusters/prod/.env"] == argv[4:6]
    files = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
    assert files == [f"/repo/{f}" for f in (BASE, BUILD, GATEWAY, TUNNEL_NAMED)]
    assert argv[-4:] == ["--profile", "production", "up", "-d"]

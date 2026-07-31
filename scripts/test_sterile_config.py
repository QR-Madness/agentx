"""Run the backend suite against a STERILE config, to catch machine-dependent tests.

A test that reads the developer's live ``data/config.json`` asserts *that box's*
setup rather than the code's behavior. The failure mode is nasty: it passes for
whoever wrote it, and breaks for the next person — or, worse, only under a
particular test order. The full suite is not CI-gated, so nothing else catches it.

This points ``ConfigManager.CONFIG_PATH`` at a nonexistent file **before** the
tests import anything, so config reads fall back to ``DEFAULT_CONFIG``. Any test
that fails only here was depending on ambient local config; fix it by stubbing
the accessor (or passing an explicit ``cfg`` where the API allows one) rather
than by relaxing the assertion.

Two things to know when stubbing:

* ``patch("agentx_ai.config.get_config_manager")`` only reaches modules that
  import the accessor **inside a function** (resolved at call time). Modules with
  a top-level ``from ..config import get_config_manager`` keep their own
  reference — patch those bindings too. ``providers/catalog.py``,
  ``providers/registry.py``, ``providers/egress.py``, ``prompts/layers.py``,
  ``alloy/org_chart.py``, ``agent/ambassador.py``, ``agent/ambassador_tools.py``,
  ``agent/aide_swarm.py`` and ``agent/tool_output_compressor.py`` all bind early.
* Some helpers read keys through their own indirection (e.g. internal_tools'
  ``_backend_has_key``) — stub at that level, not at the config layer.

Read-only: the real ``data/config.json`` is never moved, read or written.

Usage:
    task test:sterile
    uv run python scripts/test_sterile_config.py agentx_ai.tests.SomeTest
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "api"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")

import django  # noqa: E402

django.setup()

import agentx_ai.config as config_module  # noqa: E402

STERILE_PATH = Path("/nonexistent/agentx-sterile/config.json")
config_module.ConfigManager.CONFIG_PATH = STERILE_PATH
# Discard any singleton built during django.setup(), so the sterile path applies.
config_module._config_manager = None

print(f"[sterile] ConfigManager.CONFIG_PATH -> {STERILE_PATH}")
print("[sterile] Failures here mean a test reads live config — stub it, don't relax it.\n")

from django.conf import settings  # noqa: E402
from django.test.utils import get_runner  # noqa: E402

runner = get_runner(settings)(verbosity=1, interactive=False)
labels = sys.argv[1:] or ["agentx_ai.tests"]
sys.exit(bool(runner.run_tests(labels)))

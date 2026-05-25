#!/usr/bin/env python3
"""Pyright baseline guardrail.

Runs pyright over the backend and fails if the error count *rises* above the
recorded baseline in ``api/.pyright-baseline``. This lets the residual
type-checker debt (the ``Optional``/``Any`` findings tracked in
docs/architecture/CLEANUP_ROADMAP.md) only shrink over time, while keeping CI
green at the current level.

- Count rose above baseline  -> exit 1 (fail).
- Count equals baseline      -> exit 0.
- Count dropped below        -> exit 0, and print a reminder to lower the
  baseline so the gain is locked in.

Usage:
    python scripts/check_pyright_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = REPO_ROOT / "api" / ".pyright-baseline"
PYRIGHT_TARGET = "api/agentx_ai/"


def _read_baseline() -> int:
    if not BASELINE_FILE.exists():
        print(f"No baseline file at {BASELINE_FILE}; create it with the current "
              f"pyright error count.", file=sys.stderr)
        sys.exit(2)
    return int(BASELINE_FILE.read_text().strip())


def _run_pyright() -> int:
    proc = subprocess.run(
        ["uv", "run", "pyright", PYRIGHT_TARGET, "--outputjson"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # pyright exits non-zero when errors exist; the JSON is still on stdout.
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("Failed to parse pyright JSON output:", file=sys.stderr)
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    return int(data["summary"]["errorCount"])


def main() -> int:
    baseline = _read_baseline()
    current = _run_pyright()

    if current > baseline:
        print(f"❌ pyright errors rose: {current} > baseline {baseline}. "
              f"Fix the new errors or justify raising the baseline.")
        return 1

    if current < baseline:
        print(f"✅ pyright errors dropped: {current} < baseline {baseline}. "
              f"Lower api/.pyright-baseline to {current} to lock in the gain.")
        return 0

    print(f"✅ pyright errors at baseline: {current}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

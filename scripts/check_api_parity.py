#!/usr/bin/env python3
"""API spec parity gate: Django routes (`urls.py`) vs the hand-maintained `OpenApi.yaml`.

There is no DRF/spectacular auto-generation here, so the spec is written by hand and the
CLAUDE.md rule is "update both." This catches the two drift directions mechanically:

  - PHANTOM (error): a path documented in OpenApi.yaml with no matching route — the spec
    promises an endpoint that doesn't exist.
  - UNDOCUMENTED (warning): a route in urls.py absent from the spec — usually a forgotten
    spec update; sometimes an intentional internal endpoint (add to ALLOW_UNDOCUMENTED).

Static parse only (regex), no Django import. Usage: python3 scripts/check_api_parity.py [--strict]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_URLS = ROOT / "api" / "agentx_ai" / "urls.py"
SPEC = ROOT / "OpenApi.yaml"
STRICT = "--strict" in sys.argv

# Routes that intentionally live in code but not in the public spec.
ALLOW_UNDOCUMENTED: set[str] = set()

PATH_RE = re.compile(r"""(?:re_)?path\(\s*["']([^"']*)["']""")
PARAM_RE = re.compile(r"<(?:[^:>]+:)?([^>]+)>")  # <int:days> / <name> -> {days}/{name}


def code_routes() -> set[str]:
    routes = set()
    for raw in PATH_RE.findall(APP_URLS.read_text(encoding="utf-8")):
        if not raw or raw.startswith("^"):  # skip re_path regex bodies we can't normalize
            continue
        norm = PARAM_RE.sub(r"{\1}", raw).strip("/")
        routes.add("/" + norm)
    return routes


def spec_paths() -> set[str]:
    paths = set()
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^  (/[^:\s]*):\s*$", line)
        if m:
            paths.add(m.group(1).rstrip("/") or "/")
    return paths


def main() -> int:
    code = code_routes()
    spec = spec_paths()
    phantom = sorted(spec - code)
    undocumented = sorted((code - spec) - ALLOW_UNDOCUMENTED)

    print("🔗 API spec parity (urls.py ↔ OpenApi.yaml)")
    for p in phantom:
        print(f"  ❌ phantom: {p} is in OpenApi.yaml but has no route")
    for r in undocumented:
        print(f"  ⚠️  undocumented: {r} is a route with no OpenApi.yaml path")
    if not phantom and not undocumented:
        print(f"  ✓ {len(code)} routes ↔ {len(spec)} documented paths in parity")
    elif not phantom:
        print(f"  ✓ no phantom paths ({len(undocumented)} undocumented warning(s))")

    return 1 if phantom or (STRICT and undocumented) else 0


if __name__ == "__main__":
    sys.exit(main())

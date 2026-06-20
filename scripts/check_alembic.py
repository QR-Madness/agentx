#!/usr/bin/env python3
"""Alembic migration gate (no DB / no git needed — runs anywhere).

Two invariants that protect the memory-Postgres migration history:

1. **Single head** — exactly one revision must be a head (no `down_revision`
   points to it). Forked history (two heads) means `alembic upgrade head` is
   ambiguous; catch it at PR time, not on a deploy.

2. **Frozen baseline** — `alembic/baseline.sql` is the immutable starting-point
   schema adopted at Alembic cutover. Editing it would retroactively change an
   already-applied revision. The gate fails if its sha256 drifts from the
   recorded `alembic/baseline.sql.sha256`. A real schema change is a NEW revision
   (`task db:revision`), never an edit to the baseline.

Exit non-zero on any violation. Pure stdlib + regex over the revision files.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "alembic" / "versions"
BASELINE = ROOT / "alembic" / "baseline.sql"
BASELINE_HASH = ROOT / "alembic" / "baseline.sql.sha256"

_REV = re.compile(r"^revision(?::\s*str)?\s*=\s*['\"]([^'\"]+)['\"]", re.M)
_DOWN = re.compile(r"^down_revision[^=]*=\s*(.+)$", re.M)


def _parse_down(raw: str) -> set[str]:
    """Revisions referenced by a `down_revision` line (handles None / str / tuple)."""
    return set(re.findall(r"['\"]([^'\"]+)['\"]", raw))


def check_single_head() -> list[str]:
    revs: dict[str, set[str]] = {}
    for f in sorted(VERSIONS.glob("*.py")):
        if f.name == "__init__.py":
            continue
        text = f.read_text()
        m = _REV.search(text)
        if not m:
            return [f"{f.name}: no `revision = '...'` found"]
        d = _DOWN.search(text)
        revs[m.group(1)] = _parse_down(d.group(1)) if d else set()

    if not revs:
        return ["no Alembic revisions found under alembic/versions/"]

    referenced = set().union(*revs.values()) if revs else set()
    heads = [r for r in revs if r not in referenced]
    if len(heads) != 1:
        return [f"expected exactly one migration head, found {len(heads)}: {sorted(heads)}"]
    return []


def check_frozen_baseline() -> list[str]:
    if not BASELINE.exists() or not BASELINE_HASH.exists():
        return ["alembic/baseline.sql or its .sha256 is missing"]
    actual = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    expected = BASELINE_HASH.read_text().split()[0].strip()
    if actual != expected:
        return [
            "alembic/baseline.sql changed but is FROZEN — author a new revision "
            "(`task db:revision -- \"msg\"`) instead of editing the baseline.",
        ]
    return []


def main() -> int:
    errors = check_single_head() + check_frozen_baseline()
    print("🧭 Alembic migration gate")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print("  ✓ single head; baseline frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())

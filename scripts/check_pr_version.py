#!/usr/bin/env python3
"""A pull request that claims a version must end ahead of its base branch.

Version and notes travel with the work: a notable change bumps versions.yaml in
its own commit. Two pull requests written in parallel therefore claim the same
number, and when they change it identically git merges the second one without
a conflict: both changes then ship under one version, with every other check
green. This is the check for that.

    check_pr_version.py --base <rev> [--title "<pull request title>"]

Compares `api.version` in the working tree's versions.yaml with the one at
`<rev>` (CI passes the base branch's tip):

  * behind the base: fails. A merge never goes back a version.
  * equal to the base, and the pull request claims a version (its title carries
    `(vX.Y.Z)`): fails. The base already has that number; the change needs the
    next one.
  * equal, and no version is claimed: passes. Not every change is notable.
  * ahead: passes.

No dependencies beyond git and the standard library.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Version = tuple[int, int, int]

# The moving version is the first bare `version:` key under `api:`.
_API_VERSION = re.compile(r'^api:\s*\n(?:.*\n)*?\s*version:\s*"(\d+)\.(\d+)\.(\d+)"', re.M)
_CLAIM = re.compile(r"\(v(\d+)\.(\d+)\.(\d+)\)")


def api_version(text: str) -> Version | None:
    m = _API_VERSION.search(text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def fmt(v: Version) -> str:
    return ".".join(map(str, v))


def verdict(head: Version, base: Version, title: str = "") -> str | None:
    """None when the pull request may merge, otherwise what is wrong with it."""
    if head < base:
        return f"versions.yaml is at {fmt(head)}, behind the base branch's {fmt(base)}."
    if head == base and _CLAIM.search(title):
        nxt = fmt((base[0], base[1], base[2] + 1))
        return (
            f"The title claims a version, but the base branch is already at {fmt(base)}: another "
            f"change took that number. Merge the base branch and run `task versions:resolve`, or "
            f"set versions.yaml to {nxt} and run `task versions:sync` (then the Release-Notes marker)."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base", required=True, help="git revision of the base branch's tip")
    parser.add_argument("--title", default="", help="the pull request's title")
    args = parser.parse_args(argv)

    head = api_version((ROOT / "versions.yaml").read_text(encoding="utf-8"))
    shown = subprocess.run(
        ["git", "show", f"{args.base}:versions.yaml"], capture_output=True, text=True, cwd=ROOT,
    )
    base = api_version(shown.stdout) if shown.returncode == 0 else None
    print("🔢 Version ahead of the base branch")
    if head is None or base is None:
        print(f"  ✗ could not read api.version (here: {head}, at {args.base}: {base})")
        return 1
    problem = verdict(head, base, args.title)
    if problem:
        print(f"  ✗ {problem}")
        return 1
    state = "no version claimed" if head == base else f"{fmt(base)} → {fmt(head)}"
    print(f"  ✓ {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

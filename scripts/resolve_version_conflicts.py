#!/usr/bin/env python3
"""Resolve version-bump merge conflicts (versions.yaml + Release-Notes.md).

The repo's version-travels-with-the-work habit means two parallel PRs always
collide on versions.yaml, the Release-Notes marker, and the derived manifests/
lockfiles. This encodes the only sensible resolution so `task versions:resolve`
fixes it in one shot:

  * versions.yaml — the higher side wins; if the sides are equal (both PRs
    claimed the same patch) the resolved version is that value +1. Applied to
    both api.version and client.version. (The policy is symmetric, so it works
    identically under merge and rebase, where git swaps ours/theirs.)
  * Release-Notes.md — the `<!-- release-version: … -->` marker becomes the
    resolved version; every other conflicted hunk keeps BOTH sides (theirs
    first, exact-duplicate lines dropped), so no release-note bullet is lost.
  * Derived files (client/package.json, tauri.conf.json, Cargo.toml/.lock,
    pyproject.toml, uv.lock, docs-site version stamps) — taken from either
    side, then rewritten wholesale by `task versions:sync`.

The script stages what it resolved and lists anything still conflicted.
"""

from __future__ import annotations

import re
import subprocess
import sys

VERSIONS = "versions.yaml"
NOTES = "Release-Notes.md"
# Everything versions:sync regenerates from versions.yaml — conflict content
# is irrelevant, any side unblocks the merge and sync rewrites it.
DERIVED = [
    "client/package.json",
    "client/src-tauri/tauri.conf.json",
    "client/src-tauri/Cargo.toml",
    "client/src-tauri/Cargo.lock",
    "pyproject.toml",
    "uv.lock",
    "docs-site/src/config/landing.ts",
    "docs-site/src/content/docs/index.md",
]

SEMVER = re.compile(r'(\d+)\.(\d+)\.(\d+)')
MARKER = re.compile(r'release-version:')


def run(*args: str, check: bool = True) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=check).stdout


def conflicted_files() -> set[str]:
    out = run("git", "diff", "--name-only", "--diff-filter=U")
    return {line.strip() for line in out.splitlines() if line.strip()}


def stage_version(path: str, stage: int) -> tuple[int, int, int] | None:
    """First semver found in the file at merge stage 2 (ours) or 3 (theirs)."""
    try:
        content = run("git", "show", f":{stage}:{path}")
    except subprocess.CalledProcessError:
        return None
    m = SEMVER.search(content)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def fmt(v: tuple[int, int, int]) -> str:
    return f"{v[0]}.{v[1]}.{v[2]}"


def resolve_hunks(path: str, keep: str, marker_line: str | None = None) -> str:
    """Resolve conflict markers in `path`. keep = 'ours' | 'both'.
    A hunk containing the release-version marker becomes `marker_line`."""
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("<<<<<<<"):
            out.append(lines[i])
            i += 1
            continue
        ours: list[str] = []
        theirs: list[str] = []
        i += 1
        while not lines[i].startswith("======="):
            ours.append(lines[i])
            i += 1
        i += 1
        while not lines[i].startswith(">>>>>>>"):
            theirs.append(lines[i])
            i += 1
        i += 1
        hunk_all = ours + theirs
        if marker_line and any(MARKER.search(line) for line in hunk_all):
            out.append(marker_line)
        elif keep == "ours":
            out.extend(ours)
        else:  # both — theirs first (mainline history reads first), no dupes
            out.extend(theirs)
            out.extend(line for line in ours if line not in theirs)
    return "".join(out)


def main() -> int:
    conflicts = conflicted_files()
    if not conflicts:
        print("No merge conflicts found — nothing to resolve.")
        return 1
    touched = conflicts & ({VERSIONS, NOTES} | set(DERIVED))
    if not touched:
        print("No version-related files are conflicted; leaving everything alone.")
        return 1

    # --- The resolved version: higher side wins; equal sides bump by one. ---
    ours = stage_version(VERSIONS, 2) or stage_version(VERSIONS, 1)
    theirs = stage_version(VERSIONS, 3) or ours
    if not ours or not theirs:
        print(f"Could not read a version from both sides of {VERSIONS}.")
        return 1
    hi, lo = max(ours, theirs), min(ours, theirs)
    resolved = fmt(hi) if hi > lo else fmt((hi[0], hi[1], hi[2] + 1))
    print(f"Resolved version: {resolved}  (ours {fmt(ours)} / theirs {fmt(theirs)})")

    staged: list[str] = []

    if VERSIONS in conflicts:
        # Ours wholesale (the branch's protocol/min_client intent), with both
        # quoted version fields set to the resolved version.
        ours_full = run("git", "show", f":2:{VERSIONS}")
        # Anchored so `min_client_version:` is never touched — only the bare
        # `version:` keys (api + client) carry the moving version.
        body = re.sub(
            r'(?m)^(\s*)version:\s*"\d+\.\d+\.\d+"',
            rf'\g<1>version: "{resolved}"',
            ours_full,
        )
        open(VERSIONS, "w", encoding="utf-8").write(body)
        print(f"  ✓ {VERSIONS} → {resolved}")
        staged.append(VERSIONS)

    if NOTES in conflicts:
        content = resolve_hunks(
            NOTES, keep="both", marker_line=f"<!-- release-version: {resolved} -->\n"
        )
        content = re.sub(
            r"<!-- release-version: \d+\.\d+\.\d+ -->",
            f"<!-- release-version: {resolved} -->",
            content,
        )
        open(NOTES, "w", encoding="utf-8").write(content)
        print(f"  ✓ {NOTES} — marker {resolved}, both sides' bullets kept")
        staged.append(NOTES)

    for path in DERIVED:
        if path in conflicts:
            subprocess.run(["git", "checkout", "--theirs", "--", path], check=True)
            staged.append(path)
    derived_hit = [p for p in staged if p in DERIVED]
    if derived_hit:
        print(f"  ✓ derived files taken as-is (versions:sync rewrites): {', '.join(derived_hit)}")

    print("Running task versions:sync …")
    subprocess.run(["task", "versions:sync"], check=True)
    # sync may rewrite derived files that weren't conflicted — stage the lot.
    subprocess.run(["git", "add", "--"] + staged + DERIVED, check=False)

    remaining = conflicted_files()
    if remaining:
        print("Still conflicted (yours to resolve): " + ", ".join(sorted(remaining)))
    else:
        print("All conflicts resolved and staged — review, then continue the merge/rebase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

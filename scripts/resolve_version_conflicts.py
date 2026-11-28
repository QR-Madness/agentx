#!/usr/bin/env python3
"""Resolve version-bump merge conflicts (versions.yaml, Release-Notes.md, Todo.md).

The repo's version-travels-with-the-work habit means two parallel PRs always
collide on versions.yaml, the Release-Notes marker, Todo.md's "Current" line and
the derived manifests/lockfiles. This encodes the only sensible resolution so
`task versions:resolve` fixes it in one shot:

  * versions.yaml — the branch ends strictly above the mainline. If the branch
    is already ahead it keeps its version; otherwise (behind, or both claimed
    the same patch) it takes the mainline's version +1. A branch that merely
    took the mainline's version would ship its change under a number another
    change already has. Applied to both api.version and client.version.
  * Which side is the mainline: in a merge it is "theirs" (the default branch
    merged into the PR branch, which is what the click-to-fix workflow and
    `git merge origin/master` do); in a rebase or cherry-pick git swaps the
    sides, so it is "ours". `--mainline ours|theirs` overrides the guess.
  * Release-Notes.md — the `<!-- release-version: … -->` marker becomes the
    resolved version; every other conflicted hunk keeps BOTH sides (theirs
    first, exact-duplicate lines dropped), so no release-note bullet is lost.
  * Todo.md — only the hunk holding the "Current: **x.y.z**" line is resolved,
    to the resolved version; any other conflict there is real content and is
    left for a person.
  * Derived files (client/package.json, tauri.conf.json, Cargo.toml/.lock,
    pyproject.toml, uv.lock, docs-site version stamps) — taken from either
    side, then rewritten wholesale by `task versions:sync`.

The script stages what it resolved and lists anything still conflicted.
Both conflict styles are read: `merge` (two sides) and `diff3`/`zdiff3` (with
the base between them).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

VERSIONS = "versions.yaml"
NOTES = "Release-Notes.md"
TODO = "Todo.md"
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

Version = tuple[int, int, int]

# The moving version lives on the bare `version:` keys (api + client) — NOT
# the first semver in the file (the header's milestone comment is one).
VERSION_KEY = re.compile(r'(?m)^\s*version:\s*"(\d+)\.(\d+)\.(\d+)"')
MARKER = re.compile(r"release-version:")
TODO_CURRENT = re.compile(r"Current: \*\*\d+\.\d+\.\d+\*\*")


def run(*args: str, check: bool = True) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=check).stdout


def conflicted_files() -> set[str]:
    out = run("git", "diff", "--name-only", "--diff-filter=U")
    return {line.strip() for line in out.splitlines() if line.strip()}


def version_at(spec: str) -> Version | None:
    """The `version:` key's semver in versions.yaml at a git object: `:2:` and
    `:3:` are the conflict's two sides, `HEAD:` and `MERGE_HEAD:` their commits."""
    try:
        content = run("git", "show", f"{spec}{VERSIONS}")
    except subprocess.CalledProcessError:
        return None
    m = VERSION_KEY.search(content)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def side_versions() -> tuple[Version | None, Version | None]:
    """(ours, theirs). From the conflict when versions.yaml is in it; otherwise
    from the two commits, which is the case when both sides bumped to the same
    number: git merges that line without a word and only the notes conflict."""
    ours, theirs = version_at(":2:"), version_at(":3:")
    if ours and theirs:
        return ours, theirs
    ours = version_at("HEAD:")
    for other in ("MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD"):
        theirs = version_at(f"{other}:")
        if theirs:
            break
    return ours, theirs


def fmt(v: Version) -> str:
    return f"{v[0]}.{v[1]}.{v[2]}"


def resolved_version(mainline: Version, branch: Version) -> Version:
    """The version a branch carries once the mainline is merged into it: its own
    if that is already ahead, otherwise one patch above the mainline's."""
    if branch > mainline:
        return branch
    return (mainline[0], mainline[1], mainline[2] + 1)


def mainline_side(choice: str = "auto") -> str:
    """'ours' or 'theirs': the side of the conflict that is the default branch."""
    if choice in ("ours", "theirs"):
        return choice
    git_dir = Path(run("git", "rev-parse", "--git-dir").strip())
    # A rebase or cherry-pick replays the branch's commits onto the mainline, so
    # the mainline is what git calls "ours". A merge brings it in as "theirs".
    replaying = any(
        (git_dir / marker).exists()
        for marker in ("rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD")
    )
    return "ours" if replaying else "theirs"


def resolve_hunks(text: str, keep: str, marker_line: str | None = None) -> str:
    """Resolve conflict hunks in `text`.

    keep = 'both': theirs, then ours' lines that theirs lacks.
    keep = 'ours': ours.
    keep = 'todo': the hunk holding Todo.md's "Current:" line resolves to theirs
                   (the caller then sets its version); every other hunk is left
                   conflicted.
    A hunk containing the release-version marker becomes `marker_line`.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("<<<<<<<"):
            out.append(lines[i])
            i += 1
            continue
        start = i
        ours: list[str] = []
        theirs: list[str] = []
        i += 1
        while not lines[i].startswith(("=======", "|||||||")):
            ours.append(lines[i])
            i += 1
        # diff3 / zdiff3 put the common ancestor between the sides: skip it.
        while not lines[i].startswith("======="):
            i += 1
        i += 1
        while not lines[i].startswith(">>>>>>>"):
            theirs.append(lines[i])
            i += 1
        i += 1
        hunk_all = ours + theirs
        if marker_line and any(MARKER.search(line) for line in hunk_all):
            out.append(marker_line)
        elif keep == "todo":
            if all(any(TODO_CURRENT.search(line) for line in side) for side in (ours, theirs)):
                out.extend(theirs)
            else:
                out.extend(lines[start:i])  # real content: not ours to settle
        elif keep == "ours":
            out.extend(ours)
        else:  # both — theirs first (mainline history reads first), no dupes
            out.extend(theirs)
            out.extend(line for line in ours if line not in theirs)
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--mainline", choices=("auto", "ours", "theirs"), default="auto",
        help="which side of the conflict is the default branch (auto: theirs in a merge, ours in a rebase)",
    )
    parser.add_argument(
        "--no-sync", action="store_true",
        help="skip `task versions:sync` (the derived files are then left as taken)",
    )
    args = parser.parse_args(argv)

    conflicts = conflicted_files()
    if not conflicts:
        print("No merge conflicts found — nothing to resolve.")
        return 1
    touched = conflicts & ({VERSIONS, NOTES, TODO} | set(DERIVED))
    if not touched:
        print("No version-related files are conflicted; leaving everything alone.")
        return 1

    # --- The resolved version: the branch ends strictly above the mainline. ---
    ours, theirs = side_versions()
    if not ours or not theirs:
        print(f"Could not read a version from both sides of {VERSIONS}.")
        return 1
    side = mainline_side(args.mainline)
    mainline, branch = (ours, theirs) if side == "ours" else (theirs, ours)
    resolved = fmt(resolved_version(mainline, branch))
    print(
        f"Resolved version: {resolved}  "
        f"(mainline {fmt(mainline)} is {side} / branch {fmt(branch)})"
    )

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
        Path(VERSIONS).write_text(body, encoding="utf-8")
        print(f"  ✓ {VERSIONS} → {resolved}")
        staged.append(VERSIONS)
    else:
        # Both sides bumped to the same number and git merged them silently, or
        # only the other files conflicted: the version still has to move.
        text = Path(VERSIONS).read_text(encoding="utf-8")
        Path(VERSIONS).write_text(
            re.sub(r'(?m)^(\s*)version:\s*"\d+\.\d+\.\d+"', rf'\g<1>version: "{resolved}"', text),
            encoding="utf-8",
        )
        staged.append(VERSIONS)

    if NOTES in conflicts:
        content = resolve_hunks(
            Path(NOTES).read_text(encoding="utf-8"),
            keep="both", marker_line=f"<!-- release-version: {resolved} -->\n",
        )
        staged.append(NOTES)
        print(f"  ✓ {NOTES} — marker {resolved}, both sides' bullets kept")
    else:
        content = Path(NOTES).read_text(encoding="utf-8")
    Path(NOTES).write_text(
        re.sub(
            r"<!-- release-version: \d+\.\d+\.\d+ -->",
            f"<!-- release-version: {resolved} -->",
            content,
        ),
        encoding="utf-8",
    )
    if NOTES not in staged:
        staged.append(NOTES)

    # Todo.md: settle the version line here rather than leave it to versions:sync,
    # which on a branch older than that step would not touch it.
    todo = Path(TODO)
    if todo.exists():
        text = todo.read_text(encoding="utf-8")
        if TODO in conflicts:
            text = resolve_hunks(text, keep="todo")
        todo.write_text(TODO_CURRENT.sub(f"Current: **{resolved}**", text), encoding="utf-8")
        if "<<<<<<<" not in text:
            staged.append(TODO)
            if TODO in conflicts:
                print(f"  ✓ {TODO} — the Current line → {resolved}")

    for path in DERIVED:
        if path in conflicts:
            subprocess.run(["git", "checkout", "--theirs", "--", path], check=True)
            staged.append(path)
    derived_hit = [p for p in staged if p in DERIVED]
    if derived_hit:
        print(f"  ✓ derived files taken as-is (versions:sync rewrites): {', '.join(derived_hit)}")

    if args.no_sync:
        print("Skipping task versions:sync (--no-sync).")
        subprocess.run(["git", "add", "--", *staged], check=False)
    else:
        print("Running task versions:sync …")
        subprocess.run(["task", "versions:sync"], check=True)
        # sync may rewrite derived files that weren't conflicted — stage the lot.
        subprocess.run(["git", "add", "--", *staged, *DERIVED], check=False)

    remaining = conflicted_files()
    if remaining:
        print("Still conflicted (yours to resolve): " + ", ".join(sorted(remaining)))
    else:
        print("All conflicts resolved and staged — review, then continue the merge/rebase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

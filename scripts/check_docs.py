#!/usr/bin/env python3
"""Doc drift gate for the AgentX workspace.

Drift = a fact asserted in prose in N places with its source of truth elsewhere
(see Decisions.md / the CLAUDE.md Documentation Map). This script makes the cheap,
mechanical half of that a *failing check* instead of something a human must notice.

Errors (exit 1 — block release):
  - broken relative markdown links across the doc set (CLAUDE.md, Memory-Roadmap.md,
    Development-Notes.md, Decisions.md, Repo-Questions.md, Todo.md, todo/**, Release-Notes.md)
  - broken `#fragment` anchors — a link whose heading no longer exists in the target file
  - orphan todo/ files not linked from the Todo.md index

Warnings (printed; exit 0 unless --strict):
  - a "current version" / "reconciled as of" assertion that != versions.yaml api.version
  - Release-Notes.md body over the ~2 KB "fits on one screen" target it sets for itself
  - CLAUDE.md over its context-budget ceiling (it is auto-loaded every agent session)

Usage:  python3 scripts/check_docs.py [--strict]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRICT = "--strict" in sys.argv

RN_BODY_LIMIT = 2048    # Release-Notes.md sets its own "~2 KB / one screen" target
CLAUDE_MD_LIMIT = 20000  # the "light index" ceiling; ratchet DOWN as it slims, never up

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*$")
VERSION_ASSERTIONS = [
    re.compile(r"[Cc]urrent[^\n]*?\b(\d+\.\d+\.\d+)\b"),
    re.compile(r"as of[^\n]*?\bv?(\d+\.\d+\.\d+)\b"),
]

errors: list[str] = []
warnings: list[str] = []
_slug_cache: dict[Path, set[str]] = {}


def doc_files() -> list[Path]:
    files = sorted(ROOT.glob("*.md"))
    todo = ROOT / "todo"
    if todo.is_dir():
        files += sorted(todo.rglob("*.md"))
    return files


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def slugify(heading: str) -> str:
    """GitHub heading-anchor rules: lowercase, drop punctuation (keep word/space/hyphen),
    each whitespace → one hyphen (not collapsed)."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


def heading_slugs(path: Path) -> set[str]:
    if path not in _slug_cache:
        slugs: set[str] = set()
        seen: dict[str, int] = {}
        in_fence = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = HEADING_RE.match(line)
            if not m:
                continue
            base = slugify(m.group(1))
            n = seen.get(base, 0)
            slugs.add(base if n == 0 else f"{base}-{n}")  # GitHub dedupes with -1, -2…
            seen[base] = n + 1
        _slug_cache[path] = slugs
    return _slug_cache[path]


def check_links() -> None:
    for f in doc_files():
        for m in LINK_RE.finditer(f.read_text(encoding="utf-8")):
            raw = m.group(1).strip()
            if raw.startswith(("http://", "https://", "mailto:")):
                continue
            core = raw.split()[0]  # drop any (link "title")
            if core.startswith("#"):
                target_file, frag = f, core[1:]
            else:
                path_part, _, frag = core.partition("#")
                if not path_part:
                    continue
                target_file = f.parent / path_part
                if not target_file.exists():
                    errors.append(f"{rel(f)}: broken link → {raw}")
                    continue
            if frag and target_file.is_file() and target_file.suffix == ".md":
                if frag not in heading_slugs(target_file):
                    errors.append(f"{rel(f)}: broken anchor → {raw} (no heading '#{frag}')")


def check_todo_orphans() -> None:
    index = ROOT / "Todo.md"
    todo = ROOT / "todo"
    if not (index.exists() and todo.is_dir()):
        return
    index_text = index.read_text(encoding="utf-8")
    for f in sorted(todo.rglob("*.md")):
        if rel(f) not in index_text:
            errors.append(f"orphan: {rel(f)} is not linked from Todo.md")


def current_version() -> str | None:
    vf = ROOT / "versions.yaml"
    if not vf.exists():
        return None
    m = re.search(r'^api:\s*\n(?:.*\n)*?\s*version:\s*"([^"]+)"', vf.read_text(), re.M)
    return m.group(1) if m else None


def check_versions() -> None:
    cur = current_version()
    if not cur:
        warnings.append("could not read api.version from versions.yaml")
        return
    targets = {"CLAUDE.md": None, "Todo.md": 20, "Memory-Roadmap.md": 20}
    for name, head in targets.items():
        f = ROOT / name
        if not f.exists():
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        if head:
            lines = lines[:head]
        for i, line in enumerate(lines, 1):
            if "[v" in line:  # historical shipped tags like [v0.21.5] are not drift
                continue
            for pat in VERSION_ASSERTIONS:
                m = pat.search(line)
                if m and m.group(1) != cur:
                    warnings.append(
                        f"{name}:{i}: asserts {m.group(1)} != current {cur} — {line.strip()[:80]}"
                    )


def check_release_notes_size() -> None:
    f = ROOT / "Release-Notes.md"
    if not f.exists():
        return
    body = re.sub(r"<!--.*?-->", "", f.read_text(encoding="utf-8"), flags=re.S).strip()
    if len(body) > RN_BODY_LIMIT:
        warnings.append(
            f"Release-Notes.md body is {len(body)} chars (> {RN_BODY_LIMIT} ≈ one screen / ~2 KB target — trim)"
        )


def check_claude_size() -> None:
    f = ROOT / "CLAUDE.md"
    if f.exists():
        n = len(f.read_bytes())
        if n > CLAUDE_MD_LIMIT:
            warnings.append(
                f"CLAUDE.md is {n} bytes (> {CLAUDE_MD_LIMIT} ceiling) — it is auto-loaded every "
                f"session; move detail to Development-Notes.md (then ratchet the ceiling down)"
            )


def main() -> int:
    check_links()
    check_todo_orphans()
    check_versions()
    check_release_notes_size()
    check_claude_size()

    print("📑 Doc drift check")
    for w in warnings:
        print(f"  ⚠️  {w}")
    for e in errors:
        print(f"  ❌ {e}")
    if not errors and not warnings:
        print("  ✓ links, anchors, todo/ index, versions, Release-Notes + CLAUDE.md size all consistent")
    elif not errors:
        print(f"  ✓ no errors ({len(warnings)} warning(s))")

    return 1 if errors or (STRICT and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())

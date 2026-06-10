#!/usr/bin/env python3
"""Doc drift gate for the AgentX workspace.

Drift = a fact asserted in prose in N places with its source of truth elsewhere
(see Decisions.md / the CLAUDE.md Documentation Map). This script makes the cheap,
mechanical half of that a *failing check* instead of something a human must notice.

Errors (exit 1 — block release):
  - broken relative markdown links across the doc set (CLAUDE.md, Memory-Roadmap.md,
    Development-Notes.md, Decisions.md, Todo.md, todo/**, Release-Notes.md)
  - orphan todo/ files not linked from the Todo.md index

Warnings (printed; exit 0 unless --strict):
  - a "current version" / "reconciled as of" assertion that != versions.yaml api.version
  - Release-Notes.md body over the ~2 KB "fits on one screen" target it sets for itself

Usage:  python3 scripts/check_docs.py [--strict]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRICT = "--strict" in sys.argv

RN_BODY_LIMIT = 2048  # Release-Notes.md sets its own "~2 KB / one screen" target

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
VERSION_ASSERTIONS = [
    re.compile(r"[Cc]urrent[^\n]*?\b(\d+\.\d+\.\d+)\b"),
    re.compile(r"as of[^\n]*?\bv?(\d+\.\d+\.\d+)\b"),
]

errors: list[str] = []
warnings: list[str] = []


def doc_files() -> list[Path]:
    files = sorted(ROOT.glob("*.md"))
    todo = ROOT / "todo"
    if todo.is_dir():
        files += sorted(todo.rglob("*.md"))
    return files


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def current_version() -> str | None:
    vf = ROOT / "versions.yaml"
    if not vf.exists():
        return None
    m = re.search(r'^api:\s*\n(?:.*\n)*?\s*version:\s*"([^"]+)"', vf.read_text(), re.M)
    return m.group(1) if m else None


def check_links() -> None:
    for f in doc_files():
        for m in LINK_RE.finditer(f.read_text(encoding="utf-8")):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0].split(" ", 1)[0].strip()
            if not path_part:
                continue
            if not (f.parent / path_part).exists():
                errors.append(f"{rel(f)}: broken link → {target}")


def check_todo_orphans() -> None:
    index = ROOT / "Todo.md"
    todo = ROOT / "todo"
    if not (index.exists() and todo.is_dir()):
        return
    index_text = index.read_text(encoding="utf-8")
    for f in sorted(todo.rglob("*.md")):
        if rel(f) not in index_text:
            errors.append(f"orphan: {rel(f)} is not linked from Todo.md")


def check_versions() -> None:
    cur = current_version()
    if not cur:
        warnings.append("could not read api.version from versions.yaml")
        return
    targets = {
        "CLAUDE.md": None,                 # whole file
        "Todo.md": 20,                     # header region only
        "Memory-Roadmap.md": 20,           # header region only
    }
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
    n = len(body)
    if n > RN_BODY_LIMIT:
        warnings.append(
            f"Release-Notes.md body is {n} chars (> {RN_BODY_LIMIT} ≈ one screen / ~2 KB target — trim)"
        )


def main() -> int:
    check_links()
    check_todo_orphans()
    check_versions()
    check_release_notes_size()

    print("📑 Doc drift check")
    for w in warnings:
        print(f"  ⚠️  {w}")
    for e in errors:
        print(f"  ❌ {e}")
    if not errors and not warnings:
        print("  ✓ links, todo/ index, versions, and Release-Notes size all consistent")
    elif not errors:
        print(f"  ✓ no errors ({len(warnings)} warning(s))")

    return 1 if errors or (STRICT and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())

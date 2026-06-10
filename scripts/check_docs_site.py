#!/usr/bin/env python3
"""Drift gate for the docs-site (Astro/Starlight marketing + docs).

The root drift gate (check_docs.py) covers CLAUDE.md / Todo / roadmap etc. but NOT the
docs-site — which is exactly how roadmap.md and the homepage rotted unnoticed. This adds:

Errors (block):
  - a homepage/landing or content link to `/docs/<page>` with no matching content file
    (the class that let the homepage point at an "Ambassador" page that didn't cover it)
  - a relative markdown link in `content/docs/**` to a missing file

Warnings:
  - `landing.ts` `version` out of sync with versions.yaml api.version (the hardcoded
    "keep in sync" value that had no enforcement)

Usage: python3 scripts/check_docs_site.py [--strict]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs-site"
DOCS = SITE / "src" / "content" / "docs"
STRICT = "--strict" in sys.argv

errors: list[str] = []
warnings: list[str] = []

# only a /docs route at the START of a link value (after " ' or ( ) — never mid-URL
# (e.g. https://bun.sh/docs/installation must not match)
ROUTE_RE = re.compile(r"""(?<=["'(])/docs(?:/[a-z0-9][a-z0-9/-]*)?""")
MDLINK_RE = re.compile(r"\]\(([^)]+)\)")


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def route_exists(route: str) -> bool:
    sub = route[len("/docs"):].strip("/")  # "" for /docs itself
    if not sub:
        return any((DOCS / f"index.{e}").exists() for e in ("md", "mdx"))
    cands = [DOCS / f"{sub}.md", DOCS / f"{sub}.mdx",
             DOCS / sub / "index.md", DOCS / sub / "index.mdx"]
    return any(c.exists() for c in cands)


def check_route_links() -> None:
    # the homepage (landing.ts + landing/*.astro) and any /docs/ link in content
    sources = (
        list((SITE / "src" / "config").glob("landing.ts"))
        + list((SITE / "src" / "components" / "landing").glob("*.astro"))
        + list(DOCS.rglob("*.md")) + list(DOCS.rglob("*.mdx"))
    )
    for f in sorted(sources):
        for route in dict.fromkeys(ROUTE_RE.findall(f.read_text(encoding="utf-8"))):
            if not route_exists(route):
                errors.append(f"{rel(f)}: link to {route} — no matching docs page")


def check_relative_links() -> None:
    for f in sorted(list(DOCS.rglob("*.md")) + list(DOCS.rglob("*.mdx"))):
        for m in MDLINK_RE.finditer(f.read_text(encoding="utf-8")):
            target = m.group(1).split()[0]
            if target.startswith(("http://", "https://", "mailto:", "#", "/")):
                continue  # external, anchor, or absolute (routes handled above)
            path_part = target.split("#", 1)[0]
            if path_part and not (f.parent / path_part).exists():
                errors.append(f"{rel(f)}: broken relative link → {target}")


def check_version() -> None:
    lt = SITE / "src" / "config" / "landing.ts"
    vf = ROOT / "versions.yaml"
    if not (lt.exists() and vf.exists()):
        return
    lm = re.search(r"""export const version = ['"]([^'"]+)['"]""", lt.read_text())
    vm = re.search(r'^api:\s*\n(?:.*\n)*?\s*version:\s*"([^"]+)"', vf.read_text(), re.M)
    if lm and vm and lm.group(1) != vm.group(1):
        warnings.append(
            f"landing.ts version {lm.group(1)} != versions.yaml {vm.group(1)} — bump it (or run a sync)"
        )


def main() -> int:
    check_route_links()
    check_relative_links()
    check_version()

    print("🌐 Docs-site drift check")
    for w in warnings:
        print(f"  ⚠️  {w}")
    for e in errors:
        print(f"  ❌ {e}")
    if not errors and not warnings:
        print("  ✓ homepage + content links resolve; landing version in sync")
    elif not errors:
        print(f"  ✓ no errors ({len(warnings)} warning(s))")

    return 1 if errors or (STRICT and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())

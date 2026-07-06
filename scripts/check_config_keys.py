#!/usr/bin/env python3
"""Config-key cross-check: DEFAULT_CONFIG (config.py) vs `.get("a.b.c")` read sites.

Two drift classes that are otherwise silent:
  - READ-BUT-UNDEFINED (warning): a `.get("root.sub.key")` whose `root` is a real config
    section but whose full path isn't in DEFAULT_CONFIG — almost always a typo, and it
    silently returns the default forever (the exact class behind the _global/_default bug).
  - DEFINED-BUT-UNREAD (warning): a DEFAULT_CONFIG leaf never read literally and not covered
    by a parent-prefix read — a dead knob (feeds Foundation #6's dead-knob sweep). Noisier
    (subtrees are often read by prefix), so advisory only.

The flattened key inventory (`--inventory`) is also the seed for the Settings Manifest.
Stdlib only; ast-parses DEFAULT_CONFIG (no import side-effects). Usage:
    python3 scripts/check_config_keys.py [--strict] [--inventory]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "api" / "agentx_ai"
STRICT = "--strict" in sys.argv

# literal dotted keys read via a CONFIG handle (config/cfg/get_config()) — scoped so prompt-loader
# and plain-dict `.get("a.b")` calls don't masquerade as config reads. f-strings never match.
READ_RE = re.compile(
    r"""(?:config|cfg|get_config\(\))\.get\(\s*["']([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+)["']"""
)


def default_config() -> dict:
    tree = ast.parse((API / "config.py").read_text(encoding="utf-8"))
    # Module-level literal constants (e.g. DEFAULT_IMAGE_MODEL) are referenced
    # by name inside DEFAULT_CONFIG — inline them so literal_eval can parse it.
    constants: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            try:
                ast.literal_eval(node.value)
            except ValueError:
                continue
            constants[node.targets[0].id] = node.value

    class _Inline(ast.NodeTransformer):
        def visit_Name(self, name: ast.Name) -> ast.expr:
            return constants.get(name.id, name)

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in node.targets
        ):
            inlined = ast.fix_missing_locations(_Inline().visit(node.value))
            return ast.literal_eval(inlined)
    raise SystemExit("could not find DEFAULT_CONFIG in config.py")


def flatten(d: dict, prefix: str = "") -> tuple[set[str], set[str]]:
    """Return (all_paths incl. intermediates, leaf_paths)."""
    allp: set[str] = set()
    leaves: set[str] = set()
    for k, v in d.items():
        path = f"{prefix}{k}"
        allp.add(path)
        if isinstance(v, dict) and v:
            sub_all, sub_leaves = flatten(v, path + ".")
            allp |= sub_all
            leaves |= sub_leaves
        else:
            leaves.add(path)
    return allp, leaves


def read_keys() -> set[str]:
    keys: set[str] = set()
    for f in API.rglob("*.py"):
        if "/migrations/" in f.as_posix():
            continue
        for m in READ_RE.finditer(f.read_text(encoding="utf-8")):
            keys.add(m.group(1))
    return keys


def main() -> int:
    cfg = default_config()
    defined, leaves = flatten(cfg)
    roots = {p.split(".", 1)[0] for p in defined}
    reads = read_keys()

    if "--inventory" in sys.argv:
        for p in sorted(leaves):
            print(p)
        return 0

    # READ-BUT-UNDEFINED: read key whose root is a real section but full path is undefined
    undefined = sorted(k for k in reads if k.split(".", 1)[0] in roots and k not in defined)
    # DEFINED-BUT-UNREAD: leaf neither read literally nor covered by a parent-prefix read
    read_prefixes = {p for k in reads for p in _prefixes(k)} | reads
    dead = sorted(leaf for leaf in leaves if leaf not in reads and leaf not in read_prefixes)

    print("⚙️  Config-key cross-check")
    for k in undefined:
        print(f"  ⚠️  read-but-undefined: config.get(\"{k}\") — not in DEFAULT_CONFIG (a typo silently"
              f" returns the default forever; or an undeclared optional → declare it for the Settings Manifest)")
    if dead:
        print(f"  ⚠️  {len(dead)} defined-but-unread leaf knob(s) (dead-knob candidates; prefix-read subtrees excluded):")
        for k in dead:
            print(f"        {k}")
    if not undefined and not dead:
        print(f"  ✓ {len(leaves)} config leaves all reachable; no undefined reads")
    elif not undefined:
        print(f"  ✓ no undefined reads ({len(dead)} dead-knob warning(s))")

    return 1 if STRICT and (undefined or dead) else 0


def _prefixes(key: str) -> set[str]:
    parts = key.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts))}


if __name__ == "__main__":
    sys.exit(main())

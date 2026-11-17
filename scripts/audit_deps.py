#!/usr/bin/env python3
"""Dependency advisory gate: audits what ships, read from the lockfiles.

Python: each uv lock is exported to pinned requirements (the API twice, as the
default CPU set and as the `gpu` group, since both images are published).
Environment markers are dropped and local version tags (`+cpu`, `+cu128`) are
stripped, so torch is checked under its public version, then `pip-audit
--no-deps` checks every pin against OSV (PyPI's own feed lags it: it missed a
Mako advisory OSV had). JS: `bun audit --production` per bun project, so
dev-only tooling doesn't count. Rust: every registry crate in the desktop
shell's Cargo.lock, checked against OSV directly. A RustSec `unmaintained`
notice is not a vulnerability and isn't gated; `unsound` is.

Every advisory found must be fixed or triaged into security/audit-allowlist.toml
with a reason and a review_by date (Decisions.md ADR-20).

Errors (exit 1). The gate fails closed: an audit that didn't run never reads as clean.
  - an advisory that isn't in the allowlist
  - an allowlist entry past its review_by date, or a malformed entry
  - an audit tool or OSV query that failed, or a pin it couldn't check (all need network)

Warnings (printed; exit 0 unless --strict):
  - an allowlisted id the audit no longer reports (prune it)

Usage:  python3 scripts/audit_deps.py [--strict]
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRICT = "--strict" in sys.argv
ALLOWLIST = ROOT / "security" / "audit-allowlist.toml"

# (label, uv project dir, extra `uv export` args)
UV_PROJECTS = [
    ("api", ".", []),
    ("api-gpu", ".", ["--no-group", "cpu", "--group", "gpu"]),
    ("manager", "manager", []),
]
# (label, bun project dir)
BUN_PROJECTS = [("client", "client"), ("manager-ui", "manager/ui"), ("docs-site", "docs-site")]
# (label, Cargo project dir) — the desktop shell's crates ship in every installer.
CARGO_PROJECTS = [("desktop", "client/src-tauri")]
PROJECTS = {label for label, *_ in UV_PROJECTS + BUN_PROJECTS + CARGO_PROJECTS}
ECOSYSTEMS = ("pypi", "npm", "crates")
OSV_API = "https://api.osv.dev/v1"
# RustSec notices that aren't vulnerabilities. `unsound` is deliberately not here.
NOT_GATED = {"unmaintained", "notice"}
GHSA_RE = re.compile(r"GHSA(?:-[0-9a-z]{4}){3}")

errors: list[str] = []
warnings: list[str] = []
# (eco, project, package, version, primary id) -> {"ids", "summary"}. Keyed, because
# pip-audit can return one advisory twice (from two sources, with different aliases).
hits: dict[tuple[str, str, str, str, str], dict] = {}


def record(eco: str, project: str, package: str, version: str, primary: str,
           ids: set[str], summary: str) -> None:
    hit = hits.setdefault((eco, project, package, version, primary), {"ids": set(), "summary": summary})
    hit["ids"] |= ids | {primary}


def norm(name: str) -> str:
    """PEP 503 normalization, so `PyJWT` in a report matches `pyjwt` in the allowlist."""
    return re.sub(r"[-_.]+", "-", name).lower()


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def last_line(p: subprocess.CompletedProcess[str]) -> str:
    return ((p.stderr or p.stdout).strip().splitlines() or ["no output"])[-1][:200]


def uv_pins(label: str, project_dir: str, extra: list[str]) -> dict[str, set[str]]:
    """The lock as {name: {public versions}}, via `uv export`."""
    p = run(["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", "--no-dev",
             "--format", "requirements-txt", *extra], ROOT / project_dir)
    if p.returncode != 0:
        errors.append(f"{label}: `uv export` failed ({last_line(p)}) — run `uv lock` and retry")
        return {}
    pins: dict[str, set[str]] = defaultdict(set)
    for line in p.stdout.splitlines():
        spec = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not spec:
            continue
        name, sep, version = spec.partition("==")
        if not sep:
            errors.append(f"{label}: can't audit `{spec}` (not a == pin) — pin it from an index")
            continue
        pins[norm(name.strip())].add(version.strip().split("+", 1)[0])
    return pins


def audit_python() -> None:
    """One pip-audit pass over the union of every project's pins, attributed back."""
    per_project = {label: uv_pins(label, d, extra) for label, d, extra in UV_PROJECTS}
    union: dict[str, set[str]] = defaultdict(set)
    for pins in per_project.values():
        for name, versions in pins.items():
            union[name] |= versions
    # pip-audit takes one version per name, so a name pinned differently across
    # projects goes into a later round.
    rounds = max((len(v) for v in union.values()), default=0)
    for i in range(rounds):
        reqs = [f"{name}=={sorted(vs)[i]}" for name, vs in sorted(union.items()) if i < len(vs)]
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as fh:
            fh.write("\n".join(reqs) + "\n")
            fh.flush()
            p = run(["uv", "tool", "run", "pip-audit", "--requirement", fh.name, "--no-deps",
                     "--disable-pip", "--vulnerability-service", "osv",
                     "--format", "json", "--progress-spinner", "off"], ROOT)
        try:
            report = json.loads(p.stdout)
        except json.JSONDecodeError:
            errors.append(f"pip-audit didn't run ({last_line(p)}) — it needs network; rerun")
            return
        for dep in report.get("dependencies", []):
            name, version = norm(dep["name"]), dep.get("version", "?")
            owners = [label for label, pins in per_project.items() if version in pins.get(name, ())]
            if "skip_reason" in dep:  # every pin is a public release, so a skip is a failed lookup
                errors.append(f"{', '.join(owners)}: {name} {version} not audited ({dep['skip_reason']})"
                              " — rerun; if it persists, check that version exists on PyPI")
            for v in dep.get("vulns", []):
                fixed = ", ".join(v.get("fix_versions") or []) or "none yet"
                for label in owners:
                    record("pypi", label, name, version, v["id"], set(v.get("aliases", [])),
                           f"fixed in: {fixed}")


def audit_bun(label: str, project_dir: str) -> None:
    p = run(["bun", "audit", "--production", "--json"], ROOT / project_dir)
    try:
        report = json.loads(p.stdout)
    except json.JSONDecodeError:
        report = None
    # Clean is exit 0 with `{}`, findings exit 1 with a non-empty object. Anything
    # else is a failed run (offline, bun exits 1 with an empty stdout), never a clean one.
    if (not isinstance(report, dict) or not all(isinstance(a, list) for a in report.values())
            or (p.returncode != 0) != bool(report)):
        errors.append(f"{label}: `bun audit` didn't run ({last_line(p)}) — it needs network; rerun")
        return
    for package, advisories in report.items():
        for a in advisories:
            m = GHSA_RE.search(a.get("url", ""))
            primary = m.group(0) if m else f"npm-{a.get('id')}"
            record("npm", label, package, "", primary, set(),
                   f"{a.get('severity', '?')}: {a.get('title', '')[:70]}")


def osv(path: str, body: dict | None = None) -> dict:
    """GET (or POST `body` to) the OSV API. Raises on any failure; the caller fails closed."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{OSV_API}/{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def audit_cargo(label: str, project_dir: str) -> None:
    """Every registry crate in the lock against OSV's crates.io advisories.

    One batch query finds the advisory ids; each id's record then supplies its
    aliases, fixed versions and RustSec kind. GitHub and RustSec often publish
    the same issue under both ids, so a hit is keyed by its RustSec id when it
    has one, and the two reports merge into one.
    """
    lock_path = ROOT / project_dir / "Cargo.lock"
    try:
        lock = tomllib.loads(lock_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        errors.append(f"{label}: can't read {lock_path.relative_to(ROOT)} ({e})")
        return
    crates = sorted({(p["name"], p["version"]) for p in lock.get("package", [])
                     if p.get("source", "").startswith("registry+")})
    records: dict[str, dict] = {}
    try:
        batch = osv("querybatch", {"queries": [
            {"package": {"name": n, "ecosystem": "crates.io"}, "version": v} for n, v in crates]})
        for (name, version), result in zip(crates, batch["results"], strict=True):
            if result.get("next_page_token"):
                errors.append(f"{label}: OSV paged its answer for {name} {version}, which the gate"
                              " doesn't read — extend audit_cargo to follow next_page_token")
            for hit in result.get("vulns", []):
                vid = hit["id"]
                if vid not in records:
                    records[vid] = osv(f"vulns/{vid}")
                d = records[vid]
                affected = [a for a in d.get("affected", [])
                            if a.get("package", {}).get("name") == name]
                kinds = {(a.get("database_specific") or {}).get("informational") for a in affected}
                if d.get("withdrawn") or kinds & NOT_GATED:
                    continue
                ids = {vid, *d.get("aliases", [])}
                primary = next((i for i in sorted(ids) if i.startswith("RUSTSEC-")), vid)
                fixed = sorted({ev["fixed"] for a in affected for r in a.get("ranges", [])
                                for ev in r.get("events", []) if "fixed" in ev})
                record("crates", label, name, version, primary, ids,
                       f"fixed in: {', '.join(fixed) or 'none yet'}; {d.get('summary', '')[:60]}")
    except (OSError, ValueError, KeyError) as e:
        errors.append(f"{label}: OSV query failed ({e}) — it needs network; rerun")


def load_allowlist() -> list[dict]:
    if not ALLOWLIST.exists():
        return []
    rel = ALLOWLIST.relative_to(ROOT)
    try:
        data = tomllib.loads(ALLOWLIST.read_text())
    except tomllib.TOMLDecodeError as e:
        errors.append(f"{rel} doesn't parse ({e}) — fix the TOML")
        return []
    entries = []
    for i, raw in enumerate(data.get("advisory", []), 1):
        where = f"{rel} entry #{i} ({raw.get('package', '?')})"
        missing = [k for k in ("ecosystem", "package", "ids", "reason", "review_by") if not raw.get(k)]
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)} — every entry needs all five")
        elif raw["ecosystem"] not in ECOSYSTEMS:
            errors.append(f"{where}: ecosystem must be one of {', '.join(ECOSYSTEMS)}")
        elif not isinstance(raw["review_by"], dt.date):
            errors.append(f"{where}: review_by must be a bare TOML date, e.g. 2027-01-31")
        elif unknown := set(raw.get("projects", [])) - PROJECTS:
            errors.append(f"{where}: unknown project(s) {sorted(unknown)} — use {sorted(PROJECTS)}")
        else:
            if raw["ecosystem"] == "pypi":
                raw["package"] = norm(raw["package"])
            entries.append(raw)
    return entries


def triage(entries: list[dict]) -> int:
    """Match hits to the allowlist; returns how many were allowlisted."""
    seen: dict[int, set[str]] = defaultdict(set)
    unlisted: dict[tuple[str, str, str], dict] = {}
    allowed = 0
    for (eco, project, package, version, primary), h in hits.items():
        match = next((i for i, e in enumerate(entries)
                      if e["ecosystem"] == eco and e["package"] == package
                      and (not e.get("projects") or project in e["projects"])
                      and h["ids"] & set(e["ids"])), None)
        if match is None:
            u = unlisted.setdefault((package, version, primary), {"projects": set(), "summary": h["summary"]})
            u["projects"].add(project)
        else:
            seen[match] |= h["ids"] & set(entries[match]["ids"])
            allowed += 1
    for (package, version, primary), u in sorted(unlisted.items()):
        at = f"{package} {version}".strip()
        errors.append(f"{', '.join(sorted(u['projects']))}: {at} — {primary} ({u['summary']})"
                      " — upgrade past it, or triage it into security/audit-allowlist.toml")
    today = dt.date.today()
    for i, e in enumerate(entries):
        if e["review_by"] < today:
            errors.append(f"allowlist: {e['package']} passed its review_by ({e['review_by']}) — re-triage:"
                          " upgrade it, or set a new date with a fresh reason")
        for stale in sorted(set(e["ids"]) - seen[i]):
            warnings.append(f"allowlist: {e['package']} {stale} is no longer reported — remove it")
    return allowed


def main() -> int:
    entries = load_allowlist()
    audit_python()
    for label, project_dir in BUN_PROJECTS:
        audit_bun(label, project_dir)
    for label, project_dir in CARGO_PROJECTS:
        audit_cargo(label, project_dir)
    allowed = triage(entries)

    print("🔒 Dependency audit")
    for w in warnings:
        print(f"  ⚠️  {w}")
    for e in errors:
        print(f"  ❌ {e}")
    if not errors and not warnings:
        print(f"  ✓ {len(PROJECTS)} projects clean, apart from {allowed} allowlisted advisory hit(s)"
              f" ({ALLOWLIST.relative_to(ROOT)})")
    elif not errors:
        print(f"  ✓ no errors ({len(warnings)} warning(s))")

    return 1 if errors or (STRICT and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())

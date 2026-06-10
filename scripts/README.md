# scripts/ — gate & maintenance scripts

Dependency-free Python (stdlib only) run via `Taskfile.yml`. They enforce the *mechanical* half
of the repo's anti-drift posture (the judgement half lives in [`Decisions.md`](../Decisions.md)).

| Script | Task | Guards |
|--------|------|--------|
| `check_docs.py` | `docs:check` | inter-doc links + `#anchors`, orphan `todo/` files, version drift, Release-Notes + CLAUDE.md size |
| `check_api_parity.py` | `docs:check` | Django routes (`urls.py`) ↔ `OpenApi.yaml` paths |
| `check_pyright_baseline.py` | `check:types:python:baseline` | pyright error count never rises above `api/.pyright-baseline` |

## The gate protocol (write new gates to this spec, not by imitation)

1. **Errors block, warnings inform.** Errors = drift that is *definitely* wrong (a broken link, a
   spec path with no route) → `exit 1`. Warnings = drift that is *probably* wrong or stylistic (a
   size budget, an undocumented route) → printed, `exit 0`.
2. **`--strict` promotes warnings to errors** (`exit 1` on any). Gates run normal in the dev loop,
   `--strict` where a hard wall is wanted.
3. **Baselines only ratchet down.** A known-debt count (pyright errors, a future TODO count) is
   stored and the gate fails only if the number *rises*. Debt is allowed; *growing* debt is not.
4. **One emoji summary line**, then one line per finding.
5. **Every finding states its fix.** This workspace is agent-first — the primary consumer of gate
   output is an agent acting on it in the same session. `orphan: X is not linked from Todo.md` names
   its remedy; `broken anchor → Y.md#z (no heading '#z')` names its remedy. A finding that doesn't
   tell you what to do costs a follow-up turn.

> New gate? Mirror `check_docs.py`'s shape (an `errors`/`warnings` pair, a per-check function, this
> exit-code contract) and wire it into the matching `task` + `release:check`.

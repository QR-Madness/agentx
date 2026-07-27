# Cores — Portable, Extractable Artifacts

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Companions: [open-platform.md](open-platform.md) (outward contract, egress — this track absorbed
> its portability follow-ups), [Memory-Roadmap.md](../../Memory-Roadmap.md) (testing cores pair
> with the eval harnesses), [foundation.md](foundation.md).

> **Concept ratified 2026-07-27.** Names locked: **Core** (the artifact family — Memory Core,
> Agent Core, …; deliberately *not* project-prefixed), **Construct** (the multi-core bundle),
> **Extract** (the export-with-verified-deletion operation). "Vault" is the *working* name for the
> extract retention area — final name open.

---

### The unifying claim

**Channels are the partitioning primitive, and the machinery is already channel-scoped.** The
portability kit (`kit/agent_memory/portability/` — `[v0.21.22–24]`) exports every node family
(conversations, turns, entities, facts, goals, strategies, tool invocations + the PG audit mirror)
through one channel filter (`exporter.py`), and the deletion half exists as replace-mode import's
scoped `DETACH DELETE` with `include_global=False` — no `_global` spill (`importer.py::_wipe`).
Because ecosystem conventions route everything through channels (`_self_{agent_id}` per agent,
`_project_{ws_id}` per project, workflow channels for teams), **every core type's memory slice is
well-defined for free** — one exporter serves them all. Envelopes are text-only (embeddings
regenerate on import), schema-versioned, idempotent on stable ids: already interchange-shaped.

So this track *generalizes* the shipped system; it does not rework it.

### Taxonomy

| Core | Contents | Import semantics | Distance from today |
|------|----------|------------------|---------------------|
| **Memory Core** | One or more channels of the memory graph + PG mirror | **merge** or **replace** (both shipped) | Shipped, minus the gaps below |
| **Agent Core** | Profile + avatar asset + `_self_` Memory Core + skill *references* | **instantiate** — never merged into an existing agent | New envelope over existing stores |
| **Skill Core** | Skill pack(s) from `data/skills.yaml` | **additive** | Trivial — already portable YAML |
| **Project Core** | Blobs + manifest + description/instructions + `_project_` Memory Core + conversation membership | **instantiate** | New; chunks/embeddings re-derive on ingest (the text-only philosophy applied to files) |
| **Team Core** | Workflow + member Agent Cores by reference (live) / by value (packaged) | **instantiate** | New envelope over `data/workflows.yaml` |
| **Blueprint** *(future)* | Settings presets | **apply** | Gated on the Settings Manifest keystone ([foundation.md](foundation.md)) |
| **Construct** *(future)* | A manifest of cores | per-member semantics | The composition layer |

**Rules (ratified):**

1. **Import semantics are a property of the core type, declared in the envelope** — memory merges,
   agents/projects/teams instantiate, skills add, blueprints apply. No half-applied imports.
2. **Compose by reference inside a live system, by value inside a Construct.** Packaging resolves
   references into embedded copies (OCI model: manifest + content-addressed layers). The workspace
   blob store is already sha256 content-addressed — layer digests and integrity verification come
   nearly free.
3. **Secrets can never enter an envelope** — redaction is schema-enforced (excluded by field type,
   never by remembering to strip).
4. **Imported cores are untrusted input** — instructions/skills/procedures are prompt-injection
   vectors and a marketplace is a supply chain. Everything a core brings in carries
   `origin: imported` provenance so quarantine and bulk-eviction stay possible.
5. **Attribution survives transit** — `agent_id` + channel scoping ride the envelope (the v0.20
   migratability rule already demands the schema-versioned wrapper).

### Extract — export with verified deletion

The hard requirement: serialize a core out, then remove that data from the live system. The
primitives exist (export ✓, scoped wipe ✓); the work is the **safe composition**:

**export → verify → wipe → receipt**, in that order, non-negotiable. The wipe never runs until the
written artifact re-parses cleanly and its node counts reconcile against the live graph.

- **Vault (working name):** Extract writes into a local retention area; the data leaves the *live*
  system immediately but the artifact survives a grace window before any hard-delete convenience.
  Irreversible operations get an undo window.
- **Quiesce:** the channel under extraction is write-frozen for the duration (a consolidation sweep
  landing facts mid-extract is silent loss). The worker's idempotent-marking discipline makes this
  tractable, but it must be explicit.
- **Cross-store sweep:** Neo4j + PG wipe together today; Redis sidecars do not. Extract reuses the
  conversation-DELETE sidecar-clearing sweep and the contents-vs-residue ruling below.

**Proposed contents-vs-residue ruling** (graduates to a Decisions.md ADR when Slice 1 lands):

| Sidecar / store | Ruling | Rationale |
|-----------------|--------|-----------|
| Conversation state + rolling summaries | **Contents** | Part of what the conversation *is*; cheap text |
| Ambassador Inquiry threads | **Residue** (cleared) | Operator-private sidecar; never pollutes, never travels |
| Working memory / run state / job queues | **Residue** | Transient by definition |
| Usage ledger rows | **Stays** (neither) | Billing history is the operator's, not the core's |
| Tool-output store (Redis) | **Residue** | Debug surface, reproducible |

### Testing cores

`portability/cluster.py` already does cluster-wide snapshot/wipe/restore for the eval harnesses —
this slice *promotes* that plumbing to first-class artifacts: named, versioned corpus cores as eval
fixtures, and **synthetic Memory Cores at 10×/100× scale** for the long-term memory-growth testing
[Memory-Roadmap.md](../../Memory-Roadmap.md) wants. Cheapest slice, highest immediate payoff.

### Construct + marketplace posture

Sellable, fully-formed agents (profile + expertise + projects) are the revenue story. The artifact
side is nearly free once rules 2–5 hold (manifest + digests + license metadata + signature); what it
actually gates on is **Phase 19 infrastructure** (hosted registry, identity, payments) — design for
it now, build toward it later. Interop the same way: external formats (Claude Memory, …) are
**adapters out of one canonical format** — a serializer each, never a redesign (the ACP-bridge
argument, `[v0.21.237]`).

### Known gaps (verified 2026-07-27)

- [ ] **Procedures are not in the export schema** — zero references in `portability/`
      (grep-verified). Procedural memory silently doesn't travel; a Memory Core bug regardless of
      this track. → Slice 0.
- [ ] **Avatar cross-store dependency** — profile avatars are `media:{ws}/{doc}` references into
      `ws_home`; an Agent Core must embed the avatar bytes or it imports broken. → Slice 2.
- [ ] **Multi-channel selection** — the exporter takes one channel or all; cores need a set.
      → Slice 0.

### Slices

1. **Slice 0 — envelope completeness:** procedures into the export schema; multi-channel selection.
2. **Slice 1 — Extract:** the verify-before-wipe pipeline + vault + quiesce over Memory Cores;
   promote eval snapshots to named Testing Cores; contents-vs-residue ADR lands.
3. **Slice 2 — Agent + Skill Cores:** envelopes over profiles (+ avatar bytes + `_self_` slice) and
   skills.
4. **Slice 3 — Project Cores:** blobs + manifest + re-ingest on import.
5. **Slice 4+ — Construct manifest** + verify-on-import hardening; **Blueprints** after the
   Settings Manifest.

### Hardening substrate (absorbed from [open-platform.md](open-platform.md), 2026-07-27)

- [ ] **Import dry-run / preview-diff** — show the graph delta before committing (mirrors
      `POST /api/memory/consolidate/preview`) so a bad hand-edit can't silently nuke the graph.
- [ ] **Verify-on-import (default on, opt-out)** — route imported facts through the three-layer
      `check_contradictions` pipeline rather than trusting JSON verbatim; opt-out for trusted/raw
      restores. *The seed of rule 4.*
- [ ] **Memory-as-VCS (diffable, hand-editable snapshots)** — the text-only export + idempotent
      MERGE-on-id import already gives the loop: export → commit/hand-edit → import re-applies.
      Remaining polish: canonical/sorted-key JSON (+ optional NDJSON); a `memory:snapshot`/
      `memory:restore` task pair; per-node content hash to recompute only changed embeddings; a
      "memory log" of snapshots.
- [ ] **Import conflict policy** — skip / overwrite / rename for cross-instance id collisions
      (merge currently overwrites). *Required, not optional, once cores move between machines.*
- [ ] **Recompute PG audit-mirror embeddings on import** — import leaves
      `conversation_logs.embedding` NULL (recall uses recomputed Neo4j `turn_embeddings`); fill
      from content if anything starts querying the PG vector column.
- [ ] **Per-user export** — export/import is single-user (`DEFAULT_USER_ID`) until auth lands;
      scope by authenticated user when multi-user ships.

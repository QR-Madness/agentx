
# AgentX Development To-do

**Project**: AgentX - AI Agent Platform
**Status**: Prototype
**Last Updated**: 2026-06-10

**NOTE**: UI must be highly responsive between PC and mobile devices.

**Versioning**: `versions.yaml` is the single source of truth (run `task versions:sync` after
editing it). Completed work is tagged inline with the version it shipped in, e.g. `[v0.20.1]`.
Bump the version when a unit of work completes — patch for additive/back-compat features, and
bump `protocol_version` only on breaking API changes. Current: **0.21.93** (protocol 1).

> For completed phases and project history, see [roadmap.md](docs-site/src/content/docs/roadmap.md)

> **This file is an index.** The detail lives under [`todo/`](todo/) — one file per active phase
> and per backlog theme (see the map below). The **memory-system** backlog files pair with
> [`Memory-Roadmap.md`](Memory-Roadmap.md), the hardening/experimental roadmap reconciled against
> these items. Per `CLAUDE.md`, update the relevant `todo/` file (and version + release notes)
> alongside the work.

---

## Progress Tracker

| Phase | Status | Completion |
|-------|--------|------------|
| Phases 1-11, 13, 14, 17 | **Complete** | See [roadmap.md](docs-site/src/content/docs/roadmap.md) |
| Phase 12: Documentation | Partial | ~60% |
| Phase 15: Plan Execution | **Core Complete** | Core shipped; parallelism/resumption deferred |
| Phase 16: Multi-Agent Conversations | **In Progress** | ~72% (16.0–16.5 + 16.6 ambassador foundation & TTS voice shipped; 16.7 Ambassador v2 rework planned; Factory UI deferred) |
| Phase 18: UX + Memory Tuning | **In Progress** | ~98% (18.9 done; eval procedural cases + run persistence done; memory import/export shipped `[v0.21.22]` → eval snapshot/restore now unblocked) |

---

## Document Map

### Phases — [`todo/phases/`](todo/phases/)

| File | Contents |
|------|----------|
| [completed.md](todo/phases/completed.md) | Archived/complete phase stubs — **11** (deferred items), **12** (docs), **13** (UX overhaul), **14** (context gating), **17** (server management) |
| [phase-15-plan-execution.md](todo/phases/phase-15-plan-execution.md) | Plan Execution — core complete; deferred follow-ups (parallel subtasks, Alloy resume, mid-tool cancel) |
| [phase-16-multi-agent.md](todo/phases/phase-16-multi-agent.md) | Multi-Agent + Ambassador — 16.x deferred, 16.6 foundation, **16.7 Ambassador v2** (the live planning) |
| [phase-18-ux-memory.md](todo/phases/phase-18-ux-memory.md) | UX Improvements & Memory Tuning — prompt stack, ambassador-as-kind, profile control-center |

### Backlog — [`todo/backlog/`](todo/backlog/)

| File | Contents |
|------|----------|
| [foundation.md](todo/backlog/foundation.md) | The real next-session priority order |
| [workspaces.md](todo/backlog/workspaces.md) | ⭐ File Workspaces & Document RAG |
| [memory-recall.md](todo/backlog/memory-recall.md) | ⭐ Active Memory Recall (3 tiers) · pairs with `Memory-Roadmap.md` |
| [procedural.md](todo/backlog/procedural.md) | ⭐ Procedural Memory (encode → replay → activate) · pairs with `Memory-Roadmap.md` |
| [retrieval-extraction.md](todo/backlog/retrieval-extraction.md) | Retrieval Quality · MCP Tools · Extraction Improvements · pairs with `Memory-Roadmap.md` |
| [conversation-context.md](todo/backlog/conversation-context.md) | Conversation Context & Checkpoints · Memory Area UX |
| [chat-ux.md](todo/backlog/chat-ux.md) | Chat UX & Tool-Call Rendering · Backend Observability · Live Steering |
| [engineering-hardening.md](todo/backlog/engineering-hardening.md) | Grounded tech-debt / consistency items |
| [misc.md](todo/backlog/misc.md) | Uncategorized backlog (defaults, scheduler, sharing, themes, packaging…) |
| [genome-advisor.md](todo/backlog/genome-advisor.md) | Agent Genome & Settings Advisor — the meta-layer |
| [open-platform.md](todo/backlog/open-platform.md) | De-walling the garden — import/export, MCP server, egress |
| [exhibits.md](todo/backlog/exhibits.md) | Rich agent-authored content (declarative protocol) |
| [translation-web-search.md](todo/backlog/translation-web-search.md) | Translation backend · Web Search & Delegation |
| [logging.md](todo/backlog/logging.md) | Logging & Observability Overhaul (`logging_kit`) |

### Other

| File | Contents |
|------|----------|
| [known-future-issues.md](todo/known-future-issues.md) | Architectural concerns at scale + current Blockers |

> **Companion roadmap:** [`Memory-Roadmap.md`](Memory-Roadmap.md) — the memory-system hardening &
> experimental roadmap, reconciled against the ⭐ backlog tracks above.

---

> *Backlog items are things to consider after the prototype is complete.*

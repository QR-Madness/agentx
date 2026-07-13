# Cognitive OS — the North-Star Map

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

> **Nature of this file: abstractive and cross-referencing.** This file owns the *vision* — the
> pillars that make AgX a cognitive OS rather than a chat app with agents. Concrete work items
> live in the referenced backlog/phase files (same ownership rule as `todo/research/`: the
> roadmap/backlog own the work items). Checkboxes here are capture-level stubs, promoted into the
> concrete files when a pillar unparks.

---

### The OS mapping

Take the OS metaphor seriously and the gaps light up:

| OS concept | In AgX | Status |
|------------|--------|--------|
| Kernel | The memory system (episodic/semantic/procedural/working, extraction, recall) | **Shipped — the moat** |
| Processes | Agents → Teams → [Agentic Organizations](agentic-organizations.md) | Live track |
| **Scheduler** | **Prospective memory** — commitments, routines, an agenda | **Missing** |
| **Interrupts** | **Perception** — watchers, webhooks, ambient events | **Missing** |
| **Filesystem** | **The document layer — Docushark** | **Missing** |
| Self-model | Metacognition — track records, genome, self-cost | Partial ([genome-advisor.md](genome-advisor.md)) |
| Permissions | Trust & memory hygiene — provenance, belief revision | Partial ([Memory-Roadmap.md](../../Memory-Roadmap.md)) |
| Economy | Budgets & self-cost analysis | Partial (usage ledger) |
| **Government** | **Governance — the Agency (detection) + the Regulator (enforcement)** | **Vision (Pillar 8)** |
| Shell | Chat, command palette, the ambassador | Live |

**Naming:** the governed whole is **the Commonwealth** (user-facing candidate — literally a
governed community of organizations for the common good; internals keep `agent core`). **The
Agency** is reserved for the ambassador's intelligence apparatus (Pillar 8), not the space.

### Pillar 1 — Scheduler & Prospective Memory

> The system has four memory types; the missing fifth is **prospective memory** — remembering to
> *do* things in the *future*. Today AgX is a brilliant mind that only exists while you're talking
> to it. A cognitive OS has a temporal life.

- Commitments as durable objects: "check on this Thursday" becomes a tracked obligation, not a hope.
- Routines & cadence: a morning briefing the ambassador compiles before you arrive; weekly
  reflection/consolidation reviews; a research team that sweeps sources nightly. Organizations
  don't just have structure — they have *rhythm* (standups, reviews, deliverable dates).
- [ ] **Agenda surface** — "what the org owes you and when"; the single feature that most changes
      what AgX *is*.
- Hooks: memory-kit goals (`add_goal`/`get_active_goals`), ambassador dispatch, per-team cadence
  ([agentic-organizations.md](agentic-organizations.md)), and Phase 19 — an always-on cloud
  deployment is what makes schedules real ([phase-19-cloud-operation.md](../phases/phase-19-cloud-operation.md)).

### Pillar 2 — Perception & Interrupts

> Today every thought starts with the user typing. An OS has ambient awareness — events flow in,
> get triaged, and wake the right process.

- Watchers: webhooks, files, mail, RSS, calendars, cluster events.
- Triage is the ambassador's INV-2-shaped job — read, interpret, never pollute — then dispatch
  into the org through the chain of command
  ([agentic-organizations.md](agentic-organizations.md)).
- Turns the org from a call center that waits for the phone to ring into a staff that notices things.
- Cross-refs: [open-platform.md](open-platform.md) (ingress/egress is the de-walling track).

### Pillar 3 — Process Continuity (running agents & continuation)

> An OS runs processes you can detach from and re-attach to. Running agents, background runs, and
> stream continuation must be rock-solid before anything above them can be trusted.

- Durable running agents; resume/reattach everywhere; background runs as first-class processes; a
  "Mission Control" process-monitor surface (everything alive in the system: runs, delegations,
  jobs, consolidations, watchers).
- **Foundation: the streaming-engine stabilization pass in [chat-ux.md](chat-ux.md)** (Streaming
  Engine Stability & Golden Tests) + the Live Steering "run-aware conversation opens" item there +
  persistent delegation threads ([agentic-organizations.md](agentic-organizations.md) Slice 3).

### Pillar 4 — Filesystem for Thought (Docushark)

> **AgX is the mind, Docushark is the paper.** Memory is sediment — facts, entities, turns,
> machine-shaped. But cognition that matters ends in *artifacts*: briefs, dossiers, decision
> records, diagrams, living state-of-the-world documents. Every memory system eventually faces
> "where does synthesized knowledge live so a human can read, correct, and own it?" —
> [Docushark](https://docushark.app) (same author; a rich document surface with prose pages,
> canvas pages, diagrams, references, DOI resolution — all MCP-commandable) is that substrate.
> **Highlighted integration** — we own both ends of this story.

- Research reports born as Docushark documents with real reference management, not markdown blobs
  ([research.md](research.md)).
- The ambassador's briefing as a *living* document it maintains rather than regenerates.
- The delegation handbook / "Dossier" (phase-16) as a human-editable document; org charters.
- Agents that *maintain* documents over weeks — staleness checks, diagram updates — rather than
  emit and forget.
- Exhibits are the ephemeral rich output; Docushark is the durable one ([exhibits.md](exhibits.md)).

### Pillar 5 — Self-Model & Metacognition

> An OS monitors its own processes. The org gives us the vocabulary: track records and
> performance reviews.

- Track records mined from delegation outcomes — every delegation is training signal for "who's
  actually good at what" (the phase-16 social/delegation graph + Dossier, taken seriously).
- **Self-cost analysis** — agents understand their own cost/latency/capability and reason about
  *why* to delegate: delegate when a peer is cheaper, better, or parallelizable; abstain when
  delegation overhead outweighs the gain. Feeds chain-of-command routing
  ([agentic-organizations.md](agentic-organizations.md)).
- The genome advisor grows into performance review: agents proposing amendments to their own
  prompts/skills from trajectory history, with the user approving like a CEO
  ([genome-advisor.md](genome-advisor.md)).
- The **manager tier** is where this lands first — staffing directors whose expertise *is* the
  roster and model capabilities (tier semantics in
  [agentic-organizations.md](agentic-organizations.md)).
- The org canvas eventually shows *competence*, not just structure.

### Pillar 6 — Trust & Memory Hygiene

> As memory compounds, "what does it believe and why" becomes the product. Boring-sounding, and
> exactly what makes people trust a cognitive OS with their lives.

- [ ] **Memory inbox** — a review queue of what the system learned this week; approve, correct,
      discard.
- Provenance on every claim — traceable back to the conversation or tool call that produced it.
- Contradiction surfacing / belief revision; deliberate forgetting as UX, not just decay.
- Cross-refs: [Memory-Roadmap.md](../../Memory-Roadmap.md), [memory-recall.md](memory-recall.md),
  [retrieval-extraction.md](retrieval-extraction.md).

### Pillar 7 — Attention & Compute Economy

> Organizations have payroll. Token budgets as envelopes, allocated down the hierarchy, degrading
> gracefully instead of failing.

- Budget envelopes per team/project; the manager tier allocates.
- Graceful degradation: drop to local models, shrink recall — never just stop.
- Substrate: the usage ledger; pairs with Pillar 5's self-cost analysis; Phase 19's cost model
  ([phase-19-cloud-operation.md](../phases/phase-19-cloud-operation.md)).

### Pillar 8 — Governance (the Commonwealth, the Agency, the Regulator)

> A governed community of organizations — **the Commonwealth** — with **you as the president**.
> Orgs can run unchecked, but how can you police without an agency? The ambassador is the glue to
> YOU only: your aide for dealing with the whole system once it gets busy and your agents aren't
> always free to answer. These organizations should operate like real teams on research and
> deliver powerful projects without you running a single prompt — governance is what makes that
> safe.

- **The Agency (detection wing)** — the ambassador as your chief of intelligence; its aides as
  operatives. Investigations are **read-only by construction** — the aide invariant (read + call
  a model, write nothing) and INV-2 are literally the constitution's first articles; the
  governance layer inherits its trust model from invariants already locked.
- **The Regulator (enforcement wing)** — the enforcement arm of the genome control plane
  ([genome-advisor.md](genome-advisor.md) family model: propose → validate against the manifest →
  apply within the **autonomy envelope**, else escalate); naming candidate for the @Settings
  Advisor's enforcement duties. Powers: tune, quarantine (pull from rosters), disable a
  misbehaving agent — which requires a first-class **agent quarantine/disable flag, greenfield
  today** (nothing in the system can disable an agent).
- [ ] **Missions** — investigative + enforcement operations that land **mission briefings** in
      the ambassador thread (the `amb_thread:` model). Flagship scenario — **contamination
      tracing**: three conversations consumed a bad document → aides trace exposure through
      memory provenance (epidemiology's contact-tracing, run on knowledge; prerequisite:
      Pillar 6 provenance) → the Regulator quarantines the document and flags/purges derived
      memories → a briefing lands. You never lift a finger.
- [-] **Inspector General** (parked) — the refined "spies" idea: roster-invisible auditors with a
      **private side-channel reporting line** (sidecar, never the transcript) reviewing
      delegation outcomes for divergence from plan. An oversight office, not deception mechanics.
- **Separation of powers** — detection (the Agency) never enforces; enforcement (the Regulator)
  acts only within the autonomy envelope; destructive actions require **presidential
  ratification** (user veto). The envelope plus a readable governance policy is the constitution.
- **Ecosystem health** — "is work ACTUALLY getting done": goal-completion rates, stuck
  delegations, cost anomalies, contradiction rates — the vitals the ambassador watches (the
  Mission Control surface, Pillar 3).
- Cross-refs: [genome-advisor.md](genome-advisor.md) (control plane + family model),
  [agentic-organizations.md](agentic-organizations.md) (the orgs being governed),
  [Memory-Roadmap.md](../../Memory-Roadmap.md) / [memory-recall.md](memory-recall.md)
  (provenance), phase-16 §16.7 (ambassador-as-top-level).

### Cross-cutting principle — lean into OSS

> The cognitive OS should be open the way operating systems are open.

- **Open substrate** — local-first models, self-hosted stores: already the DNA.
- **Open protocols** — MCP in both directions: client today, **AgX as an MCP server** (your org
  and memory callable from other tools) per [open-platform.md](open-platform.md); A2A interop with
  external agents.
- **Open formats** — agent genome + memory export/import as a portable *being* (an agent's
  identity and memories travel as one artifact); the connection-links precedent.
- **Community surface** — skills and connectors as shareable packs; candidate standalone OSS
  libraries someday (the memory kit, the logging kit).
- [open-platform.md](open-platform.md) stays the concrete de-walling track; this pillar is its
  north star.

### Cross-reference map

| Pillar | Concrete home(s) |
|--------|------------------|
| Scheduler & prospective memory | *(unparked — new track when started)*; memory-kit goals; [phase-19-cloud-operation.md](../phases/phase-19-cloud-operation.md) |
| Perception & interrupts | [open-platform.md](open-platform.md); [agentic-organizations.md](agentic-organizations.md) (dispatch path) |
| Process continuity | [chat-ux.md](chat-ux.md) (streaming stability + live steering); [agentic-organizations.md](agentic-organizations.md) (Slice 3/5) |
| Document layer (Docushark) | [exhibits.md](exhibits.md); [research.md](research.md); phase-16 Dossier |
| Self-model & metacognition | [genome-advisor.md](genome-advisor.md); phase-16 social graph + Dossier |
| Trust & memory hygiene | [Memory-Roadmap.md](../../Memory-Roadmap.md); [memory-recall.md](memory-recall.md); [retrieval-extraction.md](retrieval-extraction.md) |
| Attention & compute economy | usage ledger; [phase-19-cloud-operation.md](../phases/phase-19-cloud-operation.md) |
| Governance | [genome-advisor.md](genome-advisor.md) (control plane + autonomy envelope); [agentic-organizations.md](agentic-organizations.md); Pillar 6 (provenance) |
| OSS | [open-platform.md](open-platform.md) |

### Open questions

1. **Which pillar unparks first** — recommendation recorded: prospective memory + the agenda
   surface (it makes the org *alive*), then the Docushark document layer (durable, ownable work —
   and a story only we can tell); perception rides naturally on either.
2. **Where prospective memory lives** — inside the memory kit (a fifth store beside
   episodic/semantic/procedural/working) vs a new scheduler module that *consumes* memory.
3. **Watcher security model** — what may an event source trigger, and with whose budget?
4. **Docushark connector depth** — generic MCP connector vs first-class integration (dedicated
   tools, document-aware prompts, briefing documents as a managed artifact type).
5. **How self-cost data feeds routing** without prompt bloat — ledger-derived hints in the
   delegation roster vs a learned dossier field.
6. **Inspector General mechanics when unparked** — how the private reporting line works (sidecar
   family, retention, who reads it) without becoming deception machinery.
7. **Commonwealth naming rollout** — when user-facing copy adopts "the Commonwealth" (docs first,
   UI later?).
8. **Executive deliberation mechanics** — internal multi-model panels judging reports; how their
   curated memory stays "perfect" (verified-only ingestion?). Pairs with the org doc's executive
   open question.

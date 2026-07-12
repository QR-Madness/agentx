# Memory

Memory is what makes an AgentX agent feel less like a stateless chatbot and more like a
colleague who remembers. Across conversations it keeps what matters — what you discussed, the
facts it learned about you and your work, the strategies that worked — and folds the relevant
pieces back into context on every turn. It's on by default and works quietly in the
background; you rarely have to think about it.

## The four kinds of memory

AgentX models memory in four layers, loosely mirroring human memory:

| Type | Holds | Feels like |
|------|-------|------------|
| **Working** | The current conversation and active goals | Short-term attention |
| **Episodic** | Past conversations and events | "I remember when we…" |
| **Semantic** | Facts, entities, and how they relate | Knowledge you've built up |
| **Procedural** | Strategies and tool patterns that worked | A learned skill |

You don't pick between them — a single turn draws on all four. Working memory tracks the live
session, episodic recall surfaces relevant past turns, semantic memory supplies facts and
entities, and procedural memory suggests approaches that succeeded on similar tasks before.

## Turning it on and off

Memory is enabled by default. The **Memory** toggle in the [Relay](chat.md) controls it per
conversation and **locks once the conversation starts**, so a chat can't switch memory models
mid-stream. With memory off, the agent still works — it just won't store the exchange or recall
anything beyond the current session. Recall is always **best-effort**: if the memory databases
are unavailable, chat keeps working, quietly, without it.

## Channels & Projects

Memory is organized into **channels** — scopes that group related memories so a work project
and a personal thread don't bleed into each other:

- **`_global`** — user-wide memory: your preferences, your style, durable facts about you.
- **A project channel** — everything tied to one [Project](chat.md): its own facts, entities,
  and history, kept together.

Channels are **traceable scopes, not walls.** Every recall queries the active channel *and*
`_global` and merges the results, so a project agent still knows your global preferences.
Standout project facts can be **promoted** to `_global` once they've proven durable — by
confidence and by how often they recur — and cross-channel reads are recorded in the audit
trail.

## What it remembers, automatically

You never file anything by hand. As you talk, AgentX:

- **stores each turn** to episodic memory;
- **extracts facts and entities** — people, organizations, concepts — each with a confidence
  score, and links them into a knowledge graph;
- **learns strategies** — which tool sequences worked for which kinds of task;
- **tracks goals** you set, from active through completed, abandoned, or blocked.

### How recall finds the right memories

Recall is more than nearest-vector lookup. The **Recall Layer** runs several complementary
techniques in parallel and fuses the results, so a query lands on relevant memory even when the
wording is nothing alike:

| Technique | What it adds |
|-----------|--------------|
| **Hybrid search** | Blends keyword (BM25) and vector similarity via Reciprocal Rank Fusion |
| **Entity-centric** | Matches entities in the query, then walks the graph for related facts |
| **Query expansion** | Rewrites a question as a statement for a cleaner vector match |
| **HyDE** | Drafts a hypothetical answer, then finds real memories like it |
| **Self-query** | Pulls structured filters (time, entity type, channel) out of plain language |

Each technique is independently toggleable, and a cross-encoder rerank stage orders the final
pool by relevance, salience, and recency. See the
[recall flow](../architecture/system-design.md#memory-recall) on the System Design page.

### Consolidation — turning talk into knowledge

Every 15 minutes a background pass reads recent conversations and distills them into durable
knowledge: it filters out trivial chatter, extracts entities and facts in one pass, resolves
each entity against what it already knows, and checks new facts against existing ones —
**detecting contradictions** and either superseding the old fact, keeping it, or flagging the
clash for review. Facts carry a **confidence** (from explicit down to uncertain), a **temporal
tag** (current / past / future, which biases retrieval), and **reinforcement signals** (how
often they're accessed) that feed reranking. When you correct the agent — *"no, I meant…"* — it
supersedes the stale fact instead of piling on a contradiction. See the
[consolidation flow](../architecture/system-design.md#memory-consolidation).

Other background jobs keep memory healthy: **pattern detection** (hourly) learns strategies from
outcomes, **decay** (daily) lets unused memories fade, and **cleanup** (daily) archives stale
conversations.

## Browsing & moving your memory

The **Memory drawer** is the window into everything the agent has stored — browse entities and
their subgraphs, facts filtered by confidence, learned strategies, and per-channel counts.

Memory is also **portable.** Export the whole graph to a single JSON envelope and re-import it
elsewhere — for backups, moving between instances, or hand-editing and pushing changes back:

- **Stable IDs** on every node mean import *merges* rather than duplicates — re-importing the
  same file twice is a no-op.
- **Text-only.** Exports never carry embedding vectors; they're regenerated on import with the
  target instance's model, so files stay small, git-diffable, and portable across embedding
  models.
- **Merge or replace.** `merge` upserts and leaves the rest alone; `replace` first wipes the
  target channel so it ends up matching the file exactly.

Use the drawer's Export / Import buttons, or the CLI (`task memory:export` / `task memory:import`
— see [Task Commands](../development/tasks.md#export--import-memory)).

## Settings

**Settings → Memory** governs the system without touching code:

- **Consolidation** — cadence and thresholds for what gets extracted and promoted.
- **Recall** — which retrieval techniques are active (the table above).
- **Conversation context** — the per-turn budget for how much recalled memory folds in.
- **Audit level** — how much of each operation is logged: `off`, `writes` (the default),
  `reads`, or `verbose` full traces — all to a partitioned PostgreSQL table for traceability.

## Under the hood

Memory spans three stores — **Neo4j** (the knowledge graph + vector search), **PostgreSQL +
pgvector** (episodic logs, the audit trail, backup vectors), and **Redis** (working memory and
session state). The design leans on four principles: it's **extensible** (new memory types or
extractors slot in), **transparent** and **auditable** (every operation traceable per
conversation), and **channel-scoped**.

For the deep view:

- [Memory System Architecture](../architecture/memory.md) — schemas, data flow, decay math,
  performance, and scaling.
- [Memory Capabilities](../architecture/memory-capabilities.md) — the full capability matrix.
- [Database Stack](../architecture/databases.md) — the storage infrastructure.
- [Memory Setup](../development/memory-setup.md) — environment variables and first-run schema
  initialization.

The same operations — storing turns, recalling, learning facts, tracking goals — are also
available over REST; see the [API Reference](../api/endpoints.md) and
[Memory models](../api/models.md#memory-models).

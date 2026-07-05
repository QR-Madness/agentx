# Phase 19 — Cloud Operation ("Cloud Mainframe")

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

## Phase 19: Cloud Operation (Planned)

> Run AgentX clusters in the cloud (Fly.io / AWS / one big box) so clients anywhere can
> connect, with a **memory-bundle ingest** flow: a user exports their memory locally,
> uploads the bundle to a freshly provisioned cloud cluster, and picks up where they
> left off. Grounded in the 2026-07-04 scaling assessment (cost projection + bottlenecks);
> chip away slice by slice — each slice is independently shippable.

### What we already have (the head start)

- `POST /api/memory/export` / `import` (`kit/agent_memory/portability/`) — text-only JSON
  envelope (conversations/turns, facts, entities, strategies, goals, tool invocations +
  PG audit mirror), stable-id MERGE import, **embeddings regenerated on import** (records
  `embedder_info`), per-user + optional channel scope. The right foundation for ingest.
- Cluster provisioning is scripted (manager, ADR-10; per-cluster compose projects), the
  gateway ships shared-secret auth + rate limiting + cloudflared tunnel overlays, boot is
  a fast single-process bootstrap (`manage.py bootstrap`), and the client already does
  multi-server (`ServerContext`, per-server settings) with version/protocol guards.
- Auth is multi-user-capable plumbing (PG `agentx_auth` + Redis sessions) and the memory
  kit filters `user_id` on **every** query — but the app resolves `user_id="default"`
  everywhere in practice. That gap is the fork between per-tenant and shared tenancy.

**Not in the envelope today** (portability gaps): workspace blobs + document chunks (the
RAG three-store), agent profiles (`agent_profiles.yaml`), `config.json`, prompt layers,
conversation meta.

### Deployment shapes (in order of effort)

1. **One big box, many clusters** (~zero code): a large host (Hetzner CPX41-class
   ~€28/mo or EC2 t3.xlarge ~$120/mo) running the existing manager with N isolated
   clusters, each behind its own tunnel/gateway port. ~3–5 active tenants per 16 GB
   (Neo4j is the RAM constraint). The friends-and-family alpha shape.
2. **Per-tenant Fly apps** (moderate): one app per tenant (api/neo4j/postgres/redis
   machines on volumes, private networking, Fly proxy TLS). `autostop/autostart` on the
   API machine is the cost lever — **requires lazy translation loading** (19.2.1).
3. **Shared multi-tenant stack** (most code, the real cost win): one API fleet + shared
   PG/Redis/Neo4j keyed by `user_id`. Marginal tenant ≈ storage + embeddings (pennies).
   Requires 19.4.

### The Neo4j problem

- Community edition idles at ~1.5–2 GB JVM (compose: heap 2G + pagecache 1G, limit 4G) —
  dominates per-tenant RAM everywhere.
- Community = **one database per instance, no clustering** (multi-database is Enterprise).
  Per-tenant DBs on one shared instance are impossible — BUT the schema is already
  app-level multi-tenant (`user_id` on every node/query via `CypherFilterBuilder`), so one
  shared Community instance with app-enforced isolation works today. Trust-the-app
  isolation: fine for a managed offering we run, not for adversarial tenants.
- Managed: AuraDB Professional ≈ $65/mo **per instance** (fine as one shared instance,
  ruinous per tenant). Aura Free (200k nodes) genuinely fits a single user's graph —
  legit per-tenant hack; pauses after ~3 idle days.
- Escape hatch (research slice): graph usage is modest (typed nodes, ABOUT/SUBGOAL_OF,
  salience ordering — no deep traversals) → Postgres + Apache AGE or plain relational
  adjacency could absorb it and delete a stateful service. See 19.5.

### Ingest flow (bundle → fresh cluster)

```
client                    control plane                   tenant cluster
  │ upload bundle ────────► S3/R2 presigned URL
  │                         │ provision (Fly API / manager / Terraform)
  │                         │ boot: manage.py bootstrap (seconds)
  │                         │ POST /api/memory/import ◄─ pull bundle
  │                         │   └─ async job: MERGE → re-embed all text
  │ poll /api/jobs/{id} ◄───┘
  │ connect (multi-server entry: https://{tenant}.… + token)
```

Re-embedding is the only expensive ingest step — trivial via API embeddings
(~100k items × ~100 tok ≈ 10M tok ≈ **$0.20** at text-embedding-3-small rates); hours on
shared-CPU local BGE-M3. Cloud profile should default `EMBEDDING_PROVIDER=openai` (the
envelope's `embedder_info` makes the switch safe by design).

### Cost projection (approximate, per always-on tenant cluster)

| Component | Fly.io ~/mo | AWS Fargate ~/mo | Notes |
|---|---|---|---|
| API (2 vCPU shared / 4 GB) | ~$24 | ~$72 | Fly autostop cuts 50–70% once translation is lazy |
| Neo4j (2 GB + 10 GB vol) | ~$13 | ~$40 | or Aura Free ($0, pauses) / shared Aura Pro ~$65 total |
| Postgres (1 GB + vol) | ~$8 | ~$30 (RDS t4g.micro ~$15) | Supabase/Neon free tiers viable |
| Redis (256 MB) | ~$3 | Upstash paygo ≈ $0–2 | sessions/runs/sidecars |
| TLS/ingress | $0 (Fly proxy) | ALB ~$18 + LCU | ALB idle timeout must be ≥600s for SSE |
| **Total / tenant** | **~$45–55** (≈$25–35 w/ autostop) | **~$120–160** | |

Shape 1: ~$30–120/mo total for a handful of tenants. Shape 3: fixed ~$100–200/mo floor,
marginal tenant <$1/mo + their LLM spend. **LLM/API usage dwarfs hosting in every shape**
— metering per-user spend (existing `usage_ledger` + `/metrics/usage`) is the control lever.

### Bottlenecks (ranked)

1. **Translation eager-load** — `TranslationKit` loads NLLB-200 (~600 MB dl, +2 GB RSS) at
   init; kills cold starts + fattens every API machine. Biggest single fix (19.2.1).
2. **Neo4j RAM floor + Community licensing** — decides the tenancy model (above / 19.5).
3. **SSE through cloud proxies** — Fly proxy fine; AWS ALB default 60s idle timeout kills
   quiet streams (rounds can go minutes between events). Raise timeout + rely on detached
   runs/re-attach; a periodic SSE heartbeat comment would make it bulletproof.
4. **Local embeddings on shared CPU** — cloud profile defaults to API embeddings.
5. **Agent Shells** — `container` backend needs dind (EC2/Fly-machines only, never
   Fargate); bubblewrap needs privileges PaaS won't grant. Cloud clusters ship
   `allow_shell` off (already the default); treat shells as an EC2-only feature.
6. **File-based state** (`config.json`, `agent_profiles.yaml`, prompt layers, `data/`
   bind mounts) — pins a tenant to one node/volume. Fine for shapes 1–2; the main
   migration item for shape 3 (→ PG).
7. **Single background worker + single-process bootstrap** — fine per tenant; the shared
   stack eventually wants the `chat_jobs` worker split from the web process.

---

### 19.1 Shape 1 — one big box, manager-packed (config, not code)

- [ ] Provision a large host; run the manager + N isolated clusters, one tunnel each.
- [ ] Document the per-tenant resource curve observed (RAM/CPU/disk per active tenant)
      to calibrate 19.3/19.4 sizing.
- [ ] Backup/restore runbook: volume snapshots + `memory/export` as the logical backup.

### 19.2 Cloud-profile prerequisites (needed by every later shape)

- [ ] **19.2.1 Lazy/optional translation** — deployment profile flag
      (`TRANSLATION_ENABLED=false` or lazy-load on first use); keeps API machines small
      and makes autostop viable.
- [ ] **19.2.2 Async memory import** — `/api/memory/import` becomes a `/api/jobs/*` job
      (it's synchronous today): progress reporting, size limits, and validation of the
      envelope's `embedder_info` against the target cluster's embedding config
      (dimension mismatch = known pgvector failure mode).
- [ ] **19.2.3 Envelope v2** — fold in agent profiles, prompt layers, config,
      conversation meta; optional sidecar tarball for workspace blobs + document
      re-ingest. (Profiles/config are trivial YAML/JSON; blobs are the bulky part.)
- [ ] **19.2.4 SSE heartbeat** — periodic comment frame on `/agent/chat/stream` so idle
      proxies (ALB) can't kill a quiet round.

### 19.3 Shape 2 — per-tenant Fly apps

- [ ] Fly app template (api/neo4j/pg/redis machines + volumes + private networking);
      provisioning driver (Fly Machines API) — could live in the manager as a new
      cluster backend.
- [ ] API machine autostop/autostart; measure cold-start with lazy translation.
- [ ] Per-tenant Neo4j decision: tiny self-run machine vs Aura Free (pauses) — pick per
      tenant tier.
- [ ] Bundle-ingest control-plane flow (presigned upload → provision → import job →
      ready signal → client multi-server entry).

### 19.4 Shape 3 — shared multi-tenant stack

- [ ] Session → `user_id` threading everywhere `"default"` is assumed (chat stream,
      workspaces Home store, jobs, metrics).
- [ ] File-state → PG migration: per-user agent profiles, prompt layers, config.
- [ ] Per-user rate + spend limits over `usage_ledger`.
- [ ] Split the background worker from the web process; scale independently.

### 19.5 Neo4j strategy (decide only when 19.4 is real)

- [ ] Default: one shared Community instance, app-level `user_id` isolation (works today).
- [ ] Research slice: migrate the graph to Postgres (Apache AGE or relational adjacency)
      — deletes the RAM floor, the licensing constraint, and a stateful service. Audit
      actual Cypher surface first (`kit/agent_memory/`) to size the port.
- [ ] Non-option to remember: Aura per tenant never pencils out (~$65/mo/instance).

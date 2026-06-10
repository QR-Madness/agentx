# Logging & Observability Overhaul

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Logging & Observability Overhaul (`logging_kit`)
- [x] Central color/category/run-tag console via QueueHandler→QueueListener; `AGENTX_LOG_*` flags (decorations on by default); plain/json modes; secret redaction; third-party noise tamed
- [x] ASCII startup banner + status table (`AGENTX_LOG_BANNER`)
- [x] Compact LLM request cards (`AGENTX_LLM_LOG_LEVEL` off|summary|full; legacy `DEBUG_LOG_LLM_REQUESTS`→full)
- [x] In-memory ring buffer + `/api/logs`, `/api/logs/stream` (SSE), `/api/logs/categories`; auth-gated via middleware + `AGENTX_LOG_API_ENABLED`
- [x] Compressed log archive: rotating gzip file handler + `/api/logs/archive*` endpoints + client browse/download
- [ ] Optional structured JSONL sidecar for the archive (machine-queryable history)
- [ ] Stamp `conversation_id`/`agent_id` ContextVars at the chat entry (run_id already wired) for richer per-turn correlation
- [ ] Dedupe/rate-limit consecutive identical lines (`… (×N)`); client run-tag → transcript cross-link


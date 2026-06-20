# Foundation — Next-Session Priority Order

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### ▶ FOUNDATION — real next-session priority order (barring the genome/advisor/evolution meta-layer)

> The "fancy" meta-layer (Agent Genome, Settings Advisor, evolution) is captured below but gated on
> foundation. Do these first; they're user-facing, correctness, or reliability — not strategy.

1. ~~**Chat legibility slice**~~ — **shipped.** (a) ~~compact collapsible tool-call rendering~~ — slim
   one-row card (tool-type icon + name + key arg/query + folded `· N results · 1.2s` meta; status =
   colored left border; footer/output relocated into the expanded view), `ToolExecutionBlock` so every
   inheritor (tool_call/tool_result bubbles, delegation nested tools, checkpoints) gets it. (b)
   ~~web-search query inline~~ — `web_search`/`web_research` show the quoted `query` + result count on
   the collapsed row. (c) ~~per-phase SSE `status` events~~ — coarse feed on the run-bus
   `emit_status()` seam (see the Observability cluster; deep sub-phases now a drop-in).
2. ~~**Live steering — message interruption / queue**~~ — **v1 shipped (queue mode).** A steer
   (`POST /agent/chat/runs/{id}/steer`) is pushed to a per-run Redis queue (`chat_run.push_steer`),
   drained by `streaming_tool_loop` (`streaming/steering.py`, run resolved via the `current_run_id`
   contextvar) at the tool-result boundary **and** at the would-end (continues the turn instead of
   ending), folded in as a fresh user turn. Echoed as a `steer` bus event so all clients show the
   bubble inline; client composer stays live (Stop **+** Steer). **Follow-ups:** hard interrupt (abort
   the in-flight provider stream) + persisting the steer as a real `conversation_logs` turn. See
   cluster below.
3. ~~**Stable memory core**~~ — **shipped** as the Context Ledger (Memory-Roadmap §1.8). The transient
   `remember(query=message)` block is now a low-priority, droppable *supplement*; a stable high-salience
   core (`AgentMemory.get_salient_core` → cheap non-vector `ORDER BY salience`, maintained-not-searched
   like the reflex core) is injected every turn. `assemble_turn_context` evolved into `assemble_ledger`
   (`agent/context_ledger.py`): every preamble contributor registers `(priority, min/max_tokens,
   shrink_fn, mandatory)` and one allocator decides what fits — so a growing sidecar can no longer
   silently shrink the transcript (recall now yields to history). Per-block allocation report = the
   Context Inspector's data seam (logged). Settings: `salient_core_*`. **Follow-ups:** migrate the
   `agent/core.py` + `alloy/executor.py` context builders to native ledger blocks (they keep the
   byte-identical `assemble_turn_context` wrapper for now); surface the allocation report as a real
   Context Inspector view.
4. **Finish the reliability guarantees** — (a) ~~extend the Slice-5 model fallback to the remaining
   feature sites~~ **shipped `[v0.21.92]`**: every feature site now resolves via
   `resolve_with_fallback`/`complete_with_fallback` (chat path + reasoning/drafting/`planner`/
   `plan_executor`/`alloy`/summarization/prompt-enhance); specialized roles (speculative draft/target,
   ambassador TTS/STT) and availability probes (`validate()`, cost-est) stay strict by design; the
   chat path surfaces a swap as a `model_fallback` status notice. (b) **warm-resume the queued/multi-agent
   paths** — **shipped `[v0.21.93]`**: the **background-chat** path now hydrates inside `Agent.chat()`
   (mirrors the interactive stream), so a queued job picking up an existing conversation resumes warm.
   The **Alloy specialist** path was investigated and **intentionally left as-is**: specialists are
   task-scoped by contract (the system prompt explicitly tells them they *don't* see the user's
   conversation) and already resume warm through deliberate shared-channel recall (`remember(query=task)`
   surfaces relevant prior workflow/specialist turns); raw-transcript hydration would break that scoping
   and duplicate the existing mechanism. **Foundation #4 complete.**
5. **Cost + gaps** — (a) ~~**cost tracking**~~ + (b) ~~**configure the global default model** (UI gap)~~
   **shipped `[v0.21.92–98]`**: a unified content-free **usage ledger** (`usage_events`) every spend
   site writes to — chat, Alloy, the Ambassador, and **voice TTS/STT** (per-char / per-minute audio
   pricing) — surfaced in `/metrics/usage` with a **by-source** dashboard breakdown; plus a Settings
   control for the global default model. The **per-turn search credit budget** (Tavily spend) +
   **search-spend metering** shipped `[v0.21.100]`: `web_search`/`web_research` now log a
   `source="search"` usage row (estimated credits) and a per-turn window
   (`search.per_turn_limit`, default 8; `agent/search_budget.py`, opened in `streaming_tool_loop`)
   short-circuits a runaway tool loop — background callers stay unbounded. The **persisted tool
   outputs** debugging surface shipped `[v0.21.101]`: a command-palette "Tool Outputs" drawer
   (`components/toolkit/ToolOutputsPanel.tsx`) lists/filters/reads/prunes the Redis-backed store over
   the existing `/api/tool-outputs` API. **Foundation #5 complete.**
6. ~~**Tech-debt sweep**~~ — **shipped (v0.21.102).** All token estimators now flow through one
   `tiktoken`-backed module (`api/agentx_ai/tokens.py`: `estimate_tokens`/`estimate_messages`, chars/4
   fallback + a >20K-char fast path); the ledger's `shrink_tail` verifies against it instead of assuming a
   fixed ratio. Dead knobs removed: `auto_summarize_at`, `max_messages`, the superseded
   `ContextManager.prepare_context`/`estimate_tokens`, and the stale `ContextConfig` fields (only
   `summary_model` remains).

> ⭐ **Major missing capability — File Workspaces & Document RAG** (see section below). Slots near the
> top once the chat-legibility slice lands: today agents can search learned *memory* + the *web* but
> can't be handed a file/codebase/folder. Bigger build than 1–5 but capability-defining, and mostly
> *reuse* — schedule it as its own slice.
>
> The **Settings Manifest** (keystone) is the bridge: foundational *and* the precondition for the
> meta-layer — reasonable once the above land, since it directly cleans up settings + validation.


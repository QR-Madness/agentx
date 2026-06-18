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
4. **Finish the reliability guarantees** — extend the Slice-5 model fallback to the remaining feature
   sites (reasoning/drafting/`planner`/`alloy`, still raw `get_provider_for_model`); **hydrate the
   Alloy + background-chat paths** (Slice-6 follow-up) so multi-agent/queued chats also resume warm.
5. **Cost + gaps** — **per-turn search credit budget** (Tavily spend), **configure the global default
   model** (UI gap), and the **full persisted tool outputs** debugging surface (heavier backend).
6. **Tech-debt sweep** — consolidate the 4 token estimators (→ `tiktoken`), retire dead context knobs
   (`auto_summarize_at`/`max_messages`/stale `ContextConfig`/superseded `prepare_context`).

> ⭐ **Major missing capability — File Workspaces & Document RAG** (see section below). Slots near the
> top once the chat-legibility slice lands: today agents can search learned *memory* + the *web* but
> can't be handed a file/codebase/folder. Bigger build than 1–5 but capability-defining, and mostly
> *reuse* — schedule it as its own slice.
>
> The **Settings Manifest** (keystone) is the bridge: foundational *and* the precondition for the
> meta-layer — reasonable once the above land, since it directly cleans up settings + validation.


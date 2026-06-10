# Chat UX, Observability & Live Steering

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Chat UX & Tool-Call Rendering (density + observability)

> Tool calls — and everything that inherits the tool-call block (checkpoints, exhibit fallbacks,
> web-search cards) — dominate the transcript and hide what matters. Make the chat readable.

- [x] **Compact tool-call rendering** — shipped. `ToolExecutionBlock` is now a slim one-row card by
      default (tool-type lead icon + name + key arg/query + folded `· N results · 1.2s` meta; status =
      colored left border; Call ID / args JSON / result footer + View Output relocated into the
      expanded view). Every inheritor (tool_call/tool_result bubbles, delegation nested tools,
      checkpoints, `recall_user_history`) gets it.
- [x] **Web-search shows its query inline** — shipped. `web_search`/`web_research` surface the quoted
      `query` + parsed result count on the collapsed row (`ToolExecutionBlock` `primaryPreview` +
      `resultCount`). The auto-captured Sources/citation exhibit stays a separate row (linking them is
      a follow-up).
- [ ] **Full tool-call outputs (persisted)** — only a small slice of a tool result is shown today,
      but full outputs are valuable for debugging agent thinking. Persist complete outputs
      (PostgreSQL or similar) and let the UI expand to the whole thing (lazy-loaded), beyond the
      streamed/truncated preview.

### Backend Observability — live operation status over SSE

- [x] **Per-phase status events (coarse)** — shipped. A typed `status` event
      (`{phase, label, detail?, group?, progress?}`) gives the chat a live activity line
      (`recalling` → `composing` → `thinking` → `running_tool` → `reading`) instead of a silent
      "thinking". **Key realization:** the chat path doesn't go through `Agent.run` — `generate_sse`
      inlines the phases and the live client *tails the run's Redis event bus* (`chat_run`), not the
      generator. So all status routes through one ambient `emit_status()` (`streaming/status.py`, run
      resolved from a `ContextVar` set in `chat_run._drive_run`) that appends straight to the bus —
      replays on re-attach for free, throttled/coalesced. Emit points: `views.py::generate_sse`
      (recalling/composing) + `streaming/tool_loop.py` (thinking/running_tool/reading, shared with plan
      exec). Client: `onStatus` → `streamReducer` `activity` → the `ChatPanel` spinner line.
- [ ] **Per-phase status events (deep sub-phases)** — the deferred fine grain: `embedding` /
      `reranking` inside `remember()`/`RecallLayer`, `reasoning_step N` inside the reasoner. Now a
      **drop-in** `emit_status("embedding", …)` — phases are reserved in `STATUS_PHASES` and the
      `{detail, group, progress}` contract fields + client `activity` shape already carry them. Only
      real work: the **embedding daemon thread** can't see the `ContextVar`, so its queued job must
      carry `run_id` → `emit_status(…, run_id=job.run_id)` (the explicit arg already exists). Rides the
      **same tool-loop boundary as Live Steering** (below); build the boundary once.
- [ ] **Context Inspector ("what's in the model's head this turn")** *(my idea, from the Slice-6
      context work)* — now that `assemble_turn_context` builds one well-defined message list, expose it:
      a per-turn debug view showing exactly what was sent to the model (system preamble blocks:
      checkpoints / scratchpad / summary / memory; the verbatim transcript that fit; the new turn) with
      **per-block token counts** and the budget breakdown (verbatim vs reserved vs window). Pairs with
      the per-tab context bar + the "full tool outputs" item — the single best lens for debugging agent
      behavior. Cheap to surface (the assembler already has all of it); gate behind a dev/inspect toggle.

### Live Steering — message interruption & queue (steer a running agent)

> Today a turn is fire-and-forget: once it starts you can only let it finish or **hard-cancel** it
> (`/runs/{run_id}/cancel`). You can't say "wait, also check X" or "stop — you're off track, do Y
> instead" without throwing away the whole run. Steering mid-run is essential for long/agentic turns,
> and it's the most-forgotten gap. Foundation #2 — this is the design cluster for it.

- [x] **Inject-into-running-turn** — shipped. `POST /api/agent/chat/runs/{run_id}/steer` (`{message,
      mode}`, owner-only) pushes to a per-run **steer queue** (`chat_run.push_steer` →
      `chat_run:{run_id}:queue`) **and** echoes a `steer` bus event so all clients render the bubble.
- [x] **Drain at safe boundaries** — shipped. `streaming_tool_loop` drains (`streaming/steering.py`,
      run resolved via the `current_run_id` contextvar) at the **tool-result boundary** *and* at the
      **would-end** (folds the answer-so-far + steer, then `continue`s instead of ending), so the agent
      re-plans mid-trajectory or keeps going after a steer.
- [ ] **Two modes** — only **queue** (fold at the next safe boundary) shipped; the `mode` field is
      carried but **interrupt** (abort the in-flight provider stream / tool wait + re-prompt) is a
      follow-up.
- [x] **Client** — shipped. Composer stays **live during streaming** (`ChatPanel` shows Stop **+**
      Steer; Enter routes to `stream.steer`); the `steer` event appends a `steered` user bubble via
      `useChatStream.onSteer` (flush-then-append, dedupe by id) so live + re-attached clients match.
- [x] **Persist the steer as a real turn** — shipped. Folded steers are captured on
      `ToolLoopResult.steers` and persisted as `user` turns (`metadata.steered` + `steer_round`/
      `after_tools`/`phase` — a procedural-memory signal); restored on reload (`mapServerMessages`
      `steered`). Turn-shaping extracted to pure builders in `streaming/persistence.py`.
- [x] **Hard-stop persists the partial turn** — shipped. A Stop (`GeneratorExit` in `generate_sse`)
      saves progress up to the stop (user + completed tools + steers + partial assistant text,
      `metadata.interrupted`) via the same `_persist_turns` orchestrator; the assistant bubble restores
      with a "stopped" tag. Detach/tab-close still persists normally (run plays on). **Follow-ups:**
      procedural *consumption* (consolidation mining `metadata.steered`), plan-execution-path steer +
      partial capture, background chat-queue jobs, richer `tool_call_id` linkage.
- [ ] **Shares plumbing** — the same boundary still wants to power **Blocking tool-call approval** +
      the in-run **Exhibit `choice`** round-trip (see Future Enhancements). The drain boundary +
      `current_run_id` contextvar are now in place to build on.


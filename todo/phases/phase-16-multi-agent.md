# Phase 16 — Multi-Agent Conversations & Ambassador

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

## Phase 16: Multi-Agent Conversations

> **Priority**: MEDIUM
> **Goal**: Enable multiple agents to collaborate in shared conversation threads
> **Depends on**: Plan execution (Phase 15)

> **Agent Alloy** = the multi-agent system. **Factory** = the visual editor (frontend, not yet built).
> Control flow: supervisor agent owns the conversation; specialists are invoked via a `delegate_to` tool. Opt-in per chat request via `workflow_id`.

### Shipped (16.0–16.5) — moved to [roadmap.md](../../docs-site/src/content/docs/roadmap.md)

- **16.0 Agent Alloy v1 backend** (2026-04-27): `alloy/` package, workflow model + `WorkflowManager` YAML CRUD, `delegate_to` tool, `AlloyExecutor` (shared `_alloy_<id>` channel, child goals, `delegation_*` SSE streaming, depth-limited re-delegation), supervisor framing prompt, `/api/alloy/workflows` CRUD, `alloy.*` config.
- **Parallel / fan-out delegation** + **trace/replay UI** (`[v0.20.1]`): per-delegation tokens/duration/cost/pricing-snapshot persisted; client `AlloyRunTraceModal` groups fan-out into runs.
- **16.1 Message attribution**: `agent_id` on `Turn` + `conversation_logs`; persisted across streaming/non-streaming/background and restored to display names.
- **16.2 Explicit routing** (`[v0.21.2]`): `target_agent_id`, `Session.participants` hydration, multi-agent awareness prompt.
- **16.3 Per-agent tool isolation**: `allowed_tools`/`blocked_tools` enforced in `_get_tools_for_provider`.
- **16.4 Ad-hoc agent-to-agent delegation** (`[v0.21.3]`): workflow-less `AlloyExecutor` mode gated by `alloy.allow_adhoc_delegation`, depth-limited, no self-delegation.
- **16.4c Delegation multimodality** (`[v0.21.172]`): specialist `exhibit`/`workspace_attached` events pass through top-level (were silently dropped → hallucinated `ws_home` URLs + 404s); wires ride `delegation_complete.exhibits` and persist as synthetic `present_exhibit` turns (reload parity); supervisor tool result carries an "already displayed" note; specialist-scoped `InternalToolContext` (attribution fix); specialist profile tool gating applied; empty-final guard covers post-delegation turns. **Follow-ups flagged:** ~~wire-or-remove the dead `alloy.specialist_inherits_supervisor_tools` knob~~ — **removed `[v0.21.226]`** (it was a live Settings toggle that never gated anything; specialist tools come solely from the profile's `allowed_tools`/`blocked_tools`); main-loop `generate_image` exhibit reload parity (synthetic `present_exhibit` turn in `_execute_and_emit_tools`).
- **16.4b Agent Teams — first-class conversational delegation** (`[v0.21.170]`): the missing system-prompt half — `delegation_roster` LedgerBlock (soft nudge + teammates/specialties, same source as the tool enum via `list_adhoc_delegation_targets`), per-profile `delegation_hint`, `available_for_delegation` flipped **opt-in** (default off) with the global gate flipped **on**, per-conversation `disable_delegation` (Solo/Team chip + palette), profile-editor "Team membership" card, and the user-facing **Agent Teams** rename (strings only; internals keep `alloy`, Workspaces→Projects precedent).
- **16.5 @-mention routing** (`[v0.21.4]`/`[v0.21.5]`): `agent/mentions.py` parsing, `AgentParticipant` Neo4j nodes + backfill migration, client `@`-autocomplete composer.
- **Multi-agent attribution**: attribution is now per-agent, not a singleton "agent". Agents are first-class `Entity(type="Agent")` (canonical `properties.agent_id`, name as prose, prior names as aliases); facts attributed to a specific agent (`subject_agent` name → resolved `subject_agent_id`) route to that agent's `_self_` channel — so a directive aimed at Mobius lands in Mobius's memory, not Atlas's. Roster-aware extraction prompts + per-turn responder resolution for "you"; assistant self-extraction routes each turn by its own producing `agent_id`. Display names stamped onto `Turn`/`AgentParticipant` at write-time (`get_conversation_roster`); rename-safety via Agent-entity aliases; `dedupe_entities` skips Agent nodes; deterministic legacy backfill (`task memory:backfill-agent-attribution`).

### 16.x Deferred / Next

- [-] Factory canvas frontend (Tauri client) — backend exposes everything needed → realized at
      **org scope** by the Organization Manager canvas
      ([backlog/agentic-organizations.md](../backlog/agentic-organizations.md)); `Workflow.canvas`
      stays reserved for a future per-team editor
- [-] Declarative route execution (`on_complete`, `on_match:<predicate>`, `on_failure`) — schema accepted but ignored in v1
- [ ] Loop nodes (specialist iterates over a list)
- [ ] Human-in-the-loop checkpoint nodes
- [~] Async delegation (specialist runs in background; supervisor continues other work) —
      **within-turn half shipped `[v0.21.220]`** as `delegate_start` work orders (dispatch
      receipt + auto-folded report at the steering boundaries; would-end barrier; see
      [backlog/agentic-organizations.md](../backlog/agentic-organizations.md)). Cross-turn
      continuation (a work order outliving its turn) still open.
- [ ] "User as supervisor" mode (no LLM supervisor — user manually invokes specialists from the chat UI)
- [ ] Tool-output sharing across agents (specialist's raw tool outputs visible to supervisor, not just final text)
- [ ] Per-workflow tool isolation (specialist inherits a *subset* of supervisor tools, not all)
- [ ] Trace UI follow-up: persist per-tool timing (executor currently stores one rollup turn per delegation → restored runs show delegation-level metrics only); fold specialist tokens into the supervisor done-event cost rollup
- [ ] Attribution follow-up: backfill historical NULL `agent_id` rows
- [ ] **Attribution quality in compound messages** — the `debug_attribution` harness shows
      that on a *mixed* user turn ("I prefer metric… also Mobius, cite sources… Jeff, be
      concise"), a small extraction model (gpt-4o-mini) left the per-agent directives in the
      active channel instead of homing them to each agent's `_self_`, and dropped some facts.
      Clean single directives route correctly. Follow-up: tune the `combined_with_relevance`
      prompt (or default to a stronger extraction model) so multi-directive turns split +
      attribute reliably; add a golden-output regression once stabilized.
- [ ] **Full-roster DI provider** — resolve user-named agents that aren't conversation
      participants (today they demote to `third_party`); inject the full profile roster into
      consolidation without coupling the kit to `ProfileManager`.
- [ ] **Agent social/delegation graph** — mine cross-agent facts ("Atlas is faster at SQL
      than Mobius") into a graph that informs Agent Alloy routing ("who's good at what").
- [ ] **Delegation handbook (the "Dossier")** — a **global, cross-agent registry of who to
      delegate to**: one entry per agent profile with its role/specialties/strengths (seeded from
      the profile's system prompt + capability blurb, hand-editable, optionally enriched by the
      social/delegation graph above). Global to **all** agents — any agent (and the top-level
      ambassador) reads it to pick the right delegate, so routing isn't re-derived per turn. This is
      the **curated/explicit** counterpart to the *mined* social graph (graph informs it; handbook is
      the authoritative, editable source). Backs the ambassador's `list_agents`/roster awareness
      (§16.7) and ad-hoc/Alloy delegation routing with one shared lookup. Stored once (not per
      conversation); surfaced read-only to agents as a compact "who's who," editable in the UI.
- [ ] **Per-agent identity seeding** — on profile create, seed the agent's `_self_` channel
      with an identity fact/entity ("I am Mobius, id …") for stronger self-recall.
- [ ] **Debug-harness extensions** — record/replay real conversations into scenarios;
      assertion-based regression suite (golden attribution outcomes) runnable in CI when a
      provider is configured. ~~Extract the shared cluster snapshot/wipe/restore util~~ —
      **done** as `portability/cluster.py` (with the `eval_recall` harness, Memory-Roadmap
      §2.7); `eval_consolidation` delegates to it.
- [ ] **Memory capability registry** — a code-side `@capability(...)`/registry that
      `architecture/memory-capabilities.md` is generated from or validated against, so the
      manifest can't silently drift from code (the deferred half of the drift decision).
- [-] **Manager agents (agent → team delegation)** — today `delegate_to` only targets individual agents (strict A2A). A **manager agent** could delegate a task to an entire **Team** (workflow), which then runs its own Lead→Members loop and returns a synthesized result — agent → team, not just agent → agent. Hooks: a team-targeted delegation descriptor (enum over workflow ids alongside the existing agent enum in `alloy/delegation_tool.py`), `AlloyExecutor` nesting with depth/parallel accounting across the team boundary, and cost/attribution rollup for the nested team run. Tiering note: an **executive** agent would sit a step higher again (delegating across multiple managers/teams) — capture later. **→ superseded/refined into [backlog/agentic-organizations.md](../backlog/agentic-organizations.md)** (chain of command, org manager panel, persistent delegation threads; team-as-target is recorded there as an open question).

### 16.6 Ambassador Agent (dual-presentation layer) — foundation shipped `[v0.21.32]`

> **Concept**: A customizable "ambassador" agent that runs *parallel* to a
> conversation as the **middleman of information** between the conversation and the
> user — a dedicated interpreter for large-context / complex situations that reads
> the conversation on demand **without polluting** the main transcript or the
> agent's context. Not a thin voice feature; a relay. An ambassador is a normal
> (reliable) agent profile **plus an `ambassador` section** — customized like any
> other profile.

**Shipped (foundation):**
- [x] **Per-turn briefing core**: a CC button on each assistant reply CCs the
      ambassador to brief *that turn*; a right-side **Ambassador** panel (subscribed
      to the active conversation) streams + persists the briefing.
- [x] **Parallel + non-polluting**: dedicated endpoints (`/api/agent/ambassador/*`)
      on the detached-run infra (reconnect/replay/cancel) writing to a Redis
      **sidecar** under the `ambassador:` prefix — never `conversation_logs` /
      `conv_summary:`; `start_chat_run(indexed=False)` keeps the run out of the
      conversation-recovery list. Reads conversation context `SELECT`-only.
- [x] **Profile section**: `AgentProfile.ambassador` (`AmbassadorConfig`:
      enabled/briefing_prompt/verbosity + null speech seam); global default picked
      in **Settings → Ambassador** (`ambassador.profile_id`).
- [x] **Bulletproofing**: graceful empty-provider/error degradation
      (`resolve_with_fallback`, never raises); idempotent re-CC; reload/tab-switch
      replay from the sidecar. Tests: storage round-trip, pollution regression,
      recovery-isolation, graceful degradation.
- [x] **Token-streaming + verbosity budget + cancel-settle** `[v0.21.33]`: the
      briefing now streams via `provider.stream` (per-delta `ambassador_chunk` +
      sidecar `append_chunk`); output budget scales with verbosity
      (`_VERBOSITY_TOKENS`, capped by `ambassador.max_tokens`). **Fix:** a cancel
      (GeneratorExit from `gen.aclose()`) now settles the sidecar to `cancelled`
      (preserving partial text) instead of leaving it stuck on `streaming` — no
      more perpetual "briefing…" spinner after reload. Panel refreshed with
      per-turn status chips + streaming cursor. Tests: streaming round-trip,
      cancel-settle.
- [x] **Human voice + turn-substance grounding** `[v0.21.34]`: rewrote the persona
      so the briefing speaks **to you** (second person, names the agent) instead of
      narrating "the user asked… the assistant replied." And the briefing now sees
      **what the agent actually did** — the client gathers the turn's tool calls,
      cited sources, and table/diagram exhibits (`lib/ambassadorTurn.ts::gatherTurnContext`,
      compact + capped) and posts them as `artifacts`; `_render_artifacts` weaves them
      into the prompt so it interprets the turn, not just the prose. `agent_name` +
      `artifacts` added to `/ambassador/brief-turn`. Tests: artifact-grounded prompt.
      **Name-resolution fix:** the briefed agent's name is now resolved
      (`resolveTurnAgentName`: stamped `agentName` → producing profile by `profileId`
      → conversation profile/`getAgentName()`) at both CC entry points, so restored
      turns (which lack a stamped `agentName`) still get named instead of degrading to
      "your agent". Tests: `client/src/lib/ambassadorTurn.test.ts`.
      **Thinking-truncation fix:** ambassadors think freely, so the token cap now
      budgets `_THINKING_HEADROOM` (reasoning) + a per-verbosity answer allowance,
      instead of a tight cap a thinking model (e.g. Gemini) would spend reasoning —
      which truncated the visible briefing mid-sentence. Visible length is governed
      by a firm prompt LENGTH LIMIT, not the cap; `ambassador.max_tokens` is now an
      optional hard ceiling (unset by default). `finish_reason=length` is logged.
      Tests: budget headroom + ceiling.

- [x] **Outbound relay (you → agent)** `[v0.21.36]`: relay a message into the
      conversation from the ambassador panel — a real **user turn** (or a **steer**
      into the running turn), so the ambassador stays a non-participant (the invariant
      holds: it never speaks into the transcript as itself; the user is the author).
      Client seam: `ConversationContext.{registerRelay,relayToConversation}` — ChatPanel
      registers its tab's send/steer handler; the panel got an Ask/Relay mode toggle.
      The ambassador's value-add is **drafting**: `POST /ambassador/draft` →
      `AmbassadorService.draft_relay_message` shapes a rough intent into a ready-to-send
      first-person message (ghostwriter, not speaker; degrades to the raw intent with no
      provider) which you review/edit before sending ("Refine"). Tests: draft degrade +
      provider completion. Deferred: dictation (speech → intent) feeds this same draft seam.

**Deferred (seams in place):**
> **Superseded by §16.7 (Ambassador v2):** the briefing/Q&A flow below is being
> reframed into a conversational, tool-using ambassador with its own thread — the
> "no auto-brief on open", empty-conversation, and cross-conversation gaps are tracked
> there. The items here remain accurate for the *current* one-shot implementation.
- [ ] **Activation toggle per-conversation** (today: global default + the active
      tab's context).
- [ ] **Dictation (speech → relay)**: capture continuous dictation; on stop, feed the
      captured speech as the *intent* into the existing `/ambassador/draft` → review/edit
      → relay seam (never auto-sends). File inputs remain available (reuse the input path).
- [x] **Spoken briefing (inbound) — TTS** `[v0.21.63]`: the ambassador speaks its
      briefings + Q&A aloud. `ModelProvider.synthesize_speech` (implemented by
      `OpenRouterProvider` via OpenAI-compatible `/audio/speech` → MP3; base raises);
      `AmbassadorService.synthesize` resolves the profile's `ambassador.{speech_model,
      voice,speech_speed}` block **strictly** (no chat fallback; precedence override →
      profile → `config.ambassador.*` → shipped default `microsoft/mai-voice-2`),
      degrading to a typed `SpeechUnavailable` → `422`. New `POST /ambassador/speak`
      returns raw `audio/mpeg`. Client: `lib/audio.ts::SpeechPlayer` (blob cache +
      queue + autoplay-unlock) behind `hooks/useSpeech.ts`; `AmbassadorPanel` per-item
      speaker buttons + an opt-in immersive **voice mode** (auto-speaks new briefings,
      `prefers-reduced-motion`-aware orb, Esc-to-exit). `voice_mode`/`speech_model`/`voice`
      surfaced in the ambassador profile editor's **Voice** card (speech-capable model
      picker via `ModelPickerModal requireCapability="speech"`). Tests: OpenRouter
      synth (mock httpx) + `supports_speech` cap + base-raise; `AmbassadorService.synthesize`
      precedence/degradation; `SpeechPlayer` cache/state vitest.
- [x] **Two-way voice (STT, the user-speaks half)** `[v0.21.64]`: hold-to-talk captures mic
      (Web Audio → WAV; webkit2gtk's MediaRecorder is broken), transcribes via OpenRouter
      `/audio/transcriptions`, and routes the transcript through the ambassador's intent inference
      (`/ambassador/voice-command`) → spoken answer or a reviewable relay (never auto-sent). Voice
      surface rework + intent routing landed in `[v0.21.65]` (see §16.6 vision below).
- [x] **Free-form Q&A** `[v0.21.35]`: ask the ambassador anything about the
      conversation from the panel (`POST /ambassador/ask` → `AmbassadorService.answer_question`,
      a Q&A persona/prompt over the shared `_stream_and_settle` streaming core). Persists
      under the disjoint `qa:` sidecar family (replays via `/ambassador/{conversation_id}`
      → `{briefings, qa}`); client-stable `qa_id`; grounded on a wider transcript window
      + latest-turn artifacts. Panel gained a pinned ask input + a Q&A thread. Tests:
      qa storage round-trip/isolation, answer streaming, qa prompt grounding.

#### 16.6 Voice Mode — UX vision (north star for the post-STT UI rework)

> **The bar:** Microsoft Personal Copilot's voice mode is the closest thing to a
> *perfect* immersive voice experience (OpenAI's was close too) — but both have the
> **same fatal gap: no stable push-to-talk**, and they lack **retake** and **pre-send
> confirmation**. The Ambassador's voice mode should feel like a **Discord voice call**:
> minimal, immersive, calm — and fix exactly those gaps. The current panel will need a
> **major UI rework** to reach this; the STT pass below ships the plumbing behind a
> *placeholder record button*, not this surface.

**In-panel rework shipped** `[v0.21.65]` — `components/ambassador/VoiceSurface.tsx` + the
`[Voice | Text]` tabs in `AmbassadorPanel`; backend `route_voice_command`. The items below marked
`[x]` landed in that slice; the floating CC player + recording lifecycle stay deferred.

- [x] **Immersive panel = three things only** `[v0.21.65]`: `VoiceSurface` is a **push-to-talk
      icon** (hero), **captions**, and a **settings** popover — nothing else. Voice **leads** for a
      voice-enabled ambassador (opens on the Voice tab; Text is the 2nd tab).
- [x] **Stable push-to-talk** `[v0.21.65]`: press/hold/release via mic button + Space (ignored
      while typing); pointerup/leave end a hold; **hold default + tap-toggle** setting
      (localStorage `agentx:voice:pttMode`); toggle-mode **max-duration auto-stop** + stop on
      unmount; distinct idle/listening/transcribing/thinking/speaking states. **Barge-in**: talking
      cuts off the ambassador.
- [x] **Retake confirmation** `[v0.21.65]`: a `relay` draft offers **retake** (discard + re-record)
      before sending.
- [x] **Pre-send confirmation** `[v0.21.65]`: an instruction routes to a **relay draft** shown for
      review/edit; you send (or discard). Never auto-sent. (Questions are answered spoken; a
      first-class **"relay that instead"** override recovers a misroute.)
- [x] **Captioning** `[v0.21.65]`: captions for **both** sides — your transcript + the ambassador's
      spoken line; toggleable; never audio-only.
- [x] **Voice settings popover** `[v0.21.65]`: PTT hold/toggle + captions on/off (localStorage).
      Voice/STT model + per-model **voice dropdown** live in the profile Voice card
      (`lib/voiceCatalog.ts` + `components/common/VoicePicker.tsx`).
- [x] **Voice command intent routing** `[v0.21.65]`: the ambassador **infers intent** —
      `route_voice_command` returns `{action: answer|relay, text}`; answers persist `qa:`, relays go
      through the confirm strip → `relayToConversation`. Forward-compatible `target` for future
      cross-agent delegation.
- [ ] **Headless floating CC sticky player**: CC'ing an ambassador *from a message* spawns a
      small **floating, draggable sticky button** (not the full panel) — **pause/play**, a
      small **close** button, and **keyboard PTT capture when focus isn't in an input**. A
      mini now-playing pill that rides above the conversation.
- [~] **Text mode = a second tab** `[v0.21.65]`: the typed panel is now the **Text** tab (turns,
      Q&A replay with speaker buttons, ask/relay). Playback/history/replay are there; **archival +
      recording housekeeping** still pending (see recording lifecycle below).
- [ ] **Recording lifecycle (manual, user-owned)**: recordings are **not** auto-GC'd — users
      own their own lifecycle. Surface **how much storage** recordings use and give one-tap
      **clear** options (all / by age / by conversation); **recommend clearing at a size
      threshold** (we don't mind heavy usage, we just want it visible). Auto-GC is **deferred**
      — decide the store first (IndexedDB client-side vs server sidecar) and what's even kept
      (user audio? synthesized TTS? captions/transcripts only?).
- [x] **Tauri/runtime mic permission** `[v0.21.64]`: macOS Info.plist
      `NSMicrophoneUsageDescription` + `com.apple.security.device.audio-input` entitlement,
      CSP `media-src`, **and** a Linux webkit2gtk setup hook
      (`src-tauri/src/lib.rs::enable_webview_microphone`) that flips on `enable-media-stream`
      (otherwise `navigator.mediaDevices` is absent in the packaged webview) and auto-grants the
      user-media `permission-request`. Requires an **app rebuild** to take effect (Rust change).
      Windows WebView2 prompts by default. (Verify on each packaged target.)

### 16.7 Ambassador v2 — conversational, tool-using, parallel relay — **shipped** (roadmap items open)

> Slices 0–4 + the stability/invariants close-out are live. What remains open below is
> the **orchestration roadmap** (capability modelling, ad-hoc/autonomous delegation,
> switcher over server history, CC-semantics refinement) — follow-on work, not v2 itself.

> **Why a rework.** 16.6 shipped the ambassador as a *stateless per-turn briefer*
> hard-bound to the active conversation tab: `conversationId = activeTab.sessionId`,
> every brief/ask/voice-command is a **one-shot completion** over a read-only
> transcript snapshot. That model has three problems the user hit:
> 1. **It briefs the immediate turn on open** instead of waiting for you to ask what
>    you want — but the ambassador runs *parallel to* the conversation, not *off* its
>    latest turn. (Surfaces: the prominent "Brief the latest turn" CTA + the voice
>    tab's auto-speak of the freshest item make "open ⇒ briefed" the default feel.)
> 2. **It breaks on an empty conversation** — with no turns, the grounding context is
>    empty and the briefing/voice path has nothing to operate on.
> 3. **It can only see one conversation, one snapshot.** There's no way to ask "what
>    have my agents discovered across my long-running sessions?" — the north-star use
>    case (come back to the cluster, ask the ambassador for an application-wide
>    summary of everything in flight).
>
> **The reframe.** The ambassador becomes a **real conversational agent with its own
> persistent thread**, running parallel to (not derived from) your work, that
> **observes the conversation world through a curated read-only tool belt** and acts
> only when you ask. You talk *to it*; it decides what to look at. The load-bearing
> **no-pollution invariant is unchanged** — it still writes nothing into
> `conversation_logs`/`conv_summary:`, and its tools are SELECT-only.
>
> **The bigger picture — the ambassador is the app's top-level agent.** It sits *above*
> the work, not beside one conversation. The agent profiles (the "settings agents" —
> the workers you configure) **serve it**: it surveys what they're doing, briefs you,
> and (future) dispatches work to them. The human has **dual entry** — talk to the
> ambassador (the front door / orchestrator) *or* drop straight into a worker
> conversation through the UI; both are first-class. This is the through-line that ties
> the slices together: 16.7 builds the ambassador *up* from a per-conversation briefer
> to the orchestration layer. Read-side (survey/brief) lands here; the write-side
> (dispatch/delegate to workers) reuses the relay/`target` seam from 16.6 and grows
> into real cross-agent delegation. The no-pollution invariant still holds at every
> step — orchestration is reads + relays, never the ambassador ghost-writing a worker's
> transcript as itself.

**Design pillars**

- **Wait-for-ask, never auto-brief.** Opening the panel (text or voice) lands on a
  calm idle state — no briefing fires until you ask. "Brief the latest turn" stays as
  an *explicit* affordance; the voice tab no longer auto-speaks on open. The
  ambassador is parallel, not a reaction to your last turn.
- **The ambassador owns its own conversation.** A durable, multi-turn ambassador
  **thread** (roles `user`/`ambassador`, with tool-call records) — persisted in the
  `ambassador:` sidecar, never `conversation_logs`. Two scopes: a thread *about* a
  main conversation (default), and a free-standing **command-deck** thread that isn't
  bound to one tab (for cross-conversation queries). Replaces the disjoint
  per-turn-briefing + flat Q&A records with one coherent thread.
- **A curated, read-only ambassador tool belt** (distinct from the main agent's
  MCP/internal tools — an ambassador-private registry; nothing it calls can write a
  transcript). The agentic loop replaces the one-shot completions.
- **Voice confirms the inferred tool/intent.** "Can you summarize this?" →
  `summarize_conversation`; "explore more on this" → `explore_turn` — two different
  tool calls the ambassador names and **confirms by voice** before running, then
  speaks the result. Generalizes `route_voice_command` from `{answer|relay}` to
  `{answer|relay|tool}`.
- **Stability is a feature.** Empty conversation, no active tab, missing provider,
  tool failure — all degrade gracefully (the never-raise invariant now wraps the tool
  loop, not just a single completion).

**Slice 0 — stop auto-briefing + empty-conversation safety (fast, isolated)** — shipped
- [x] **No brief on open.** Decoupled open from brief: the per-message CC button
      (`ChatPanel.handleAmbassador`) now *only opens* the panel — it no longer fires
      `ccTurn`. Briefing is an explicit action inside the panel ("Brief the latest turn"
      / per-turn) or an ask. The command-palette "Open Ambassador" was already idle; the
      voice tab already opens idle (existing `seedSpoken()` prevents replay). Button
      tooltip updated to "Open the Ambassador (parallel…)".
- [x] **Empty-conversation safety.** `_build_qa_prompt` / `_build_voice_command_prompt`
      now tell the model plainly when the transcript is empty (instead of a context-free
      prompt it could hallucinate against), so brief/ask/voice degrade to a grounded
      "nothing here yet" instead of erroring. Tests: `test_qa_prompt_handles_empty_conversation`,
      `test_answer_question_degrades_on_empty_conversation`.
- [ ] **Decouple the panel from a required `activeTab`** where the global mode will
      live (the empty-state copy already exists; make the ask/voice paths null-safe).
      → folded into Slice 1 (the ambassador thread isn't bound to a saved `sessionId`).

**Slice U — coexisting panel: slide-in, non-modal (the "stop hogging the screen" rework)**

> **The problem (touching reality).** Today the ambassador is a **modal drawer**:
> `lib/surfaces.ts::ambassador` opens `type:'drawer', size:'xxl'` (`min(1000px,100vw)`)
> inside `DrawerPanel` → `.drawer-backdrop` (`position:fixed; inset:0;
> rgba(0,0,0,.6) + blur(4px); z-index:1000`, `aria-modal`, click-to-close). So even
> though the panel is on the right, the **whole conversation is dimmed, blurred, and
> click-locked** behind it. You can't watch the agent work *and* talk to the
> ambassador — which is the entire point of a parallel relay. It must **slide in
> beside** the conversation, both panels live.
>
> **Mechanisms considered:**
> - **(A) Docked layout region (push/shrink) — recommended.** Make the ambassador a
>   first-class column in `AgentXPage` (already `ConversationSidebar | ChatPanel`): add
>   a right-side `AmbassadorDock` that, when open, *shrinks ChatPanel via flex* rather
>   than overlaying it. No backdrop, no blur, both interactive. **Reuse the
>   `ConversationSidebar` pattern** (collapsible + resizable, `agentx:conv-sidebar-*`)
>   for `agentx:ambassador-dock-{open,width}`. Default ~420–480px (not 1000 — a
>   coexisting dock leaves the conversation room), resizable 340–680. *Concern:* the
>   open-state must move out of the modal system into layout state; the entry points
>   (CC button, palette, topbar) toggle the dock via a shared store (keep `surfaces.ts`
>   as the single descriptor so they don't drift). *Concern:* narrow screens can't
>   push — fall back to the current full-screen sheet under ~900px (`useIsMobile`).
> - **(B) Non-modal floating drawer (overlay, no backdrop).** Smallest change: add a
>   `modal:false` variant to `DrawerPanel`/`ModalContext` that drops `.drawer-backdrop`
>   (no dim/blur, click-through) and `aria-modal`. The panel still floats over the
>   right edge but the conversation stays visible/interactive. *Concern:* it *covers*
>   the right slice of the conversation instead of giving it its own space; pair with a
>   narrower default width. Good interim step toward (A).
> - **(C) Split-pane with draggable divider.** (A) plus a VS-Code-style secondary-
>   sidebar drag handle — most powerful, most work. The resize mechanism already exists
>   (ConversationSidebar), so (A) naturally grows into (C).
>
> **Recommendation:** ship **(A)** — a docked, resizable, collapsible, non-modal right
> panel in `AgentXPage`, mobile-falling-back to the sheet. It's the truest "see both
> panels" and reuses an in-repo pattern (sidebar resize/collapse). (B) is the cheap
> interim if we want it *today*; (C) is (A) + a drag handle once it's docked.

- [x] **Mechanism: (A) docked push/shrink**, built *with* a resize handle (close to
      (C) — reuses the `ConversationSidebar` resize/collapse pattern). `AmbassadorDock`
      is a right column in `AgentXPage`'s flex row (after `.agentx-chat`), so opening it
      shrinks the conversation; `AmbassadorDock.css` left-edge resizer.
- [x] **Dock state + persistence**: `AmbassadorDockContext` (`agentx:ambassador-dock-{open,width}`),
      mounted under `ModalProvider`. One entry point — `useOpenAmbassador()` — used by both
      the per-message CC button (`ChatPanel`) and the palette ("Open Ambassador" now also
      navigates to chat first), so they can't drift.
- [x] **Non-modal**: no backdrop/blur, the conversation stays fully interactive; explicit
      close button (reuses `.shell-close-btn`), not Esc (the dock is persistent).
- [x] **Responsive**: docks at `≥880px`; below that (reactive media query) callers fall
      back to the existing full-screen sheet (`DrawerPanel` is already full-screen ≤600px).
      Width clamps 340–680 (default 440 — leaves the conversation room).
- [x] **Payoff check** — verified live in the browser preview (16.7 close-out): chat
      streams left while the ambassador answers/speaks right (no backdrop, both
      interactive); resize drag works with the 340/680 clamps + persistence; the
      ≥880px↔sheet crossover unmounts/remounts the dock (open-state survives); mobile
      is the exact-viewport sheet with an inline close.

**Slice 1 — ambassador thread (its own conversation)**

> **1a shipped (`0.21.75`): conversational memory.** The ambassador now *has its own
> conversation* — `answer_question` feeds prior settled Q&A back as real
> user/assistant dialogue turns (`AmbassadorService._thread_history`, read-only over
> the `qa:` sidecar, oldest-first, capped, in-flight turn excluded), so a follow-up
> ("what about the second one?") has context. Zero client churn — the existing Q&A
> store *is* the thread for now. Test: `test_thread_history_gives_qa_continuity`.
> **1b shipped (`0.21.82`): the `amb_thread:` thread model.** Briefings + Q&A are now
> one ordered **thread** ("Inquiry") under `amb_thread:{thread_id}` (entry-oriented — one
> record per ambassador turn carrying its optional `question`; ordered by `created_at`;
> `thread_id` defaults to the conversation id) carrying its own `title`. Decision:
> **entry-oriented, not message-split** (preserves in-place streaming + briefing
> idempotency). The pre-1b per-family public API is preserved as **thin projections** over
> the entry store, so `ambassador.py` + the existing tests barely changed; legacy
> `ambassador:` records still replay (one-place fold in `list_thread`). Tool-call chips are
> **persisted on the entry** (`set_entry_tool_calls`, inside `_agentic_answer`) so they
> survive a reload — closing the Slice-2 follow-up. The client renders one **Inquiry**
> stream (`AmbassadorContext.threadFor`; briefings as their own turns via `BriefingItem`,
> the per-turn CC trigger staying in the Turns strip) with inline rename in the switcher
> (`titleFor`/`renameThread`, empty → chat title). New endpoint
> `GET/PATCH /api/agent/ambassador/thread/{thread_id}`; the `{conversation_id}` endpoint is
> a back-compat shim. Tests: `AmbassadorStorageTest` (unify+order, tool-call persistence,
> title, legacy fold).

- [x] **Sidecar thread model** (`0.21.82`): one `amb_thread:` entry family + thread meta
      (`title`); the briefing/Q&A public API preserved as projections; legacy records fold
      in via `list_thread`. (Entry-oriented rather than a `role`/`content` message list.)
- [x] **Thread scoping** (`0.21.82`): `thread_id` is a real param, defaulting to the main
      `conversation_id`. Standalone command-deck thread ids (not bound to a tab) are
      Slice 4's to mint — the seam is in place.
- [x] **`GET/PATCH /api/agent/ambassador/thread/{thread_id}`** (`0.21.82`) replays the
      thread (`{thread_id, title, entries}`) + renames it; the `{conversation_id}` endpoint
      is now a back-compat shim.
- [x] **Client thread state** (`0.21.82`): `AmbassadorContext.threadFor` renders one ordered
      Inquiry stream (briefings + Q&A + persisted tool chips); `refresh` replays the thread.
      *(The two internal maps are kept as the source of truth with `threadFor` as a merge
      selector + projected `briefingsFor`/`qaFor` views — lower-risk than a physical state
      collapse, same UX.)*
- [x] **Voice continuity follow-up** — shipped (16.7 close-out). The spoken **router**
      now reads prior settled Q&A as dialogue turns (`_thread_history` in the classify
      messages, same seam as the answer path), so a follow-up like "relay that to the
      agent" classifies in context. (The answer core already had continuity via
      `_seed_answer_messages`.) Test: `test_router_hears_prior_qa_continuity`.

**Slice 2 — the read-only tool belt + agentic loop** — shipped + consolidated (`0.21.76`–`78`)

> **Ungated + unified (`0.21.78`).** The tools never fired because (a) **voice had no
> tools** and (b) the answer path **pre-stuffed the whole transcript** beside the tools,
> so the model never needed them. Fixed by a consolidation onto **one streaming agentic
> core** (`_agentic_answer`): tools always on (no gate), **tool-first grounding**
> (`_LEAN_GROUNDING_TURNS` only — the model reads via tools for depth), and **voice
> answers routed through the same core** (`route_voice_command` classifies →
> `_answer_to_text`), so spoken questions get the same tools + continuity. Also DRY'd
> provider resolution (`_resolve_answerer`), collapsed the qa/tools personas into one
> capability-stating `_answer_persona` (+ `_TOOLS_NOTE`), and reuse `_stream_and_settle`
> for the degrade. Tests rewritten for the streaming core (tool fires, voice drives a
> tool, provider-rejects-tools → grounded fallback).

- [x] **Ambassador tool registry** (`agent/ambassador_tools.py`): SELECT-only,
      separate from `mcp/internal_tools`; `execute_tool` dispatch never raises.
      - `summarize_conversation(conversation_id?)` — "summarize this".
      - `explore_conversation(topic?, conversation_id?)` — "explore more on this".
      - `read_conversation(conversation_id)` — a specific session (after a survey).
      - `list_conversations(limit?)` — the cross-conversation survey primitive
        (`conversation_history.list_recent_conversations`, new read-only enumerator).
      Backed by `load_recent_turns` (read-only). Test: `test_ambassador_tools_are_read_only_and_degrade`.
      *Deferred:* `read_conversation_results` (exhibits/sources) — **shipped `0.21.235`** in the
      v3 close-out (no bibliography plumbing needed — the persisted `present_exhibit` turns are
      the record); `explore_turn` by message-id — needs turn-id plumbing.
- [x] **Agentic turn loop** (`AmbassadorService._answer_with_tools`): bounded
      (`ambassador.max_tool_rounds`, default 4) provider tool-calling loop; emits
      `ambassador_tool_call`/`ambassador_tool_result` SSE; **never-raise** wraps the loop
      and **degrades to a grounded one-shot** on any failure (e.g. a provider that
      rejects `tools`). Client SSE pump ignores the new events (default no-op), so it's
      forward-safe. Tests: `test_answer_with_tools_executes_then_answers`,
      `test_answer_with_tools_falls_back_when_provider_rejects_tools`.
- [x] **Tools persona** (`_build_tools_persona`): the Q&A persona + a note that it has
      read-only tools and should fetch what it needs, then answer in its own voice
      (no markdown, names the agent, never reads tool output back).
- [x] **Voice answers fold through the same loop** (`0.21.78`) — spoken questions drive
      tools + continuity via the shared core.
- [x] **Client tool-call chips** (`0.21.80`) — `streamAmbassador` now surfaces
      `ambassador_tool_call`/`_result`; `AmbassadorContext` captures them onto the
      qa/briefing record (`toolCalls`); `AmbassadorPanel` renders live `ToolChips`
      (spinner → check) in Q&A + briefings (`lib/ambassadorTools.ts::toolChipLabel`).
      *Live-only* (not persisted to the sidecar — gone on reload).
- [x] **Persist tool calls to the sidecar so chips survive reload** (`0.21.82`, with Slice 1b
      thread model — `set_entry_tool_calls` on the entry).
- [~] **Follow-ups:** `QaItem`/`BriefingItem` marks now carry the **ambassador profile's
      avatar** (16.7 close-out — no more generic mark once customized). Voice-path tool
      activity: persisted chips already replay into the Inquiry after settle
      (`onAnswerPersisted` → thread refresh); **live** chips during a voice answer stay
      deferred — they need the voice endpoint to stream (SSE) instead of answering
      server-side.

**Slice 1d — parallel operator (de-couple the panel from turns)** — shipped (`0.21.83`)

> The ambassador is **an operator in its own right**, not a turn-by-turn briefer (UI kept
> *boringly stable*). The panel is **de-coupled from turns** — the Turns strip is gone. It's
> conversation-level: **"Brief this conversation"** (`briefConversation` → an `ask`) + starter
> chips + free-form ask/relay; it scopes to conversations via its read-only tools (no per-turn
> `artifacts` passed). **Per-turn = CC**: the chat's `MessageActions` button forwards a turn
> *into* the Inquiry as an `ask` ("brief me on this turn: …") — like an email into the thread —
> and opens the panel (`ChatPanel.handleAmbassador`). Header redesigned (compact command bar /
> accent-gradient **voice hero**); body **auto-scrolls** (`hooks/useStickyScroll` + jump pill);
> a **⋯ menu** (brief / rename / clear → `DELETE /thread/{id}` via `clearThread`); answers carry
> **copy** + relative timestamps. `AmbassadorConversationSwitcher` has an `inline` variant.
> **Voice mirrored (`0.21.84`, Slice 1e):** voice + text now share **one body** — the Inquiry
> stream *is* the transcript (a spoken question persists as a `qa:` entry), so only the footer
> differs (text composer ↔ `VoiceBar` = PTT mic + `voiceCommand` + relay-confirm + settings). The
> full-screen `VoiceSurface` (orb + caption log) and `lib/voiceCaptions.*` were deleted; the panel
> renders the body unconditionally and swaps the footer on `voiceActive`. **Remaining:** refine CC
> semantics (a CC'd turn as a first-class *message* that retains prior state, vs. today's framed ask).

**Slice 1c — conversations overhaul + active-conversation context** — shipped (`0.21.81`)

> The ambassador was welded to the chat tab (`conversationId = activeTab.sessionId`).
> Fixed: **independent focus** (`focusedConversationId`, "stays put", switched via
> `AmbassadorConversationSwitcher` + "current conversation" jump); **loading is pure
> display** (`locallyStreamedRef` — voice auto-speak only fires for items streamed *this
> session*, so switching/reopening never re-synthesizes TTS or speaks history — the
> cost+speech bug); **active-conversation context** (`active_conversation` {id,title} on
> ask/voice → `_active_conversation_note`, so it knows where the person is *now* even when
> focused elsewhere); **dropped the single "active agent"** (`execute_tool` param renamed
> `focused_conversation_id`; multi-agent answer persona names each agent from its own
> conversation). Relay targets the focused conversation when open; per-turn briefing stays
> gated to the active tab (needs in-memory messages).
- [ ] **Switcher over server history too** (today: open tabs only) — the full command deck.
- [~] **Nameable ambassador conversations — "Inquiries"** — **manual rename shipped (`0.21.82`)**:
      each thread carries its own `title` (`amb_thread:{id}:meta`); the switcher has inline rename
      (`titleFor`/`renameThread`), empty → chat title. **Name: an _Inquiry_** (UI noun, plural
      "Inquiries") — leans into its read-only, investigative role ("what have my agents
      discovered?") and dodges the `survey`/`search`/`research` verbs already used in code.
      **Remaining:** auto-titling (*"Inquiry · {chat title}"* mirror / *"Inquiry — {date}"* or a
      first-question summary for a **standalone** thread) is only worth it once threads can be
      standalone (a "weekly review" inquiry, a cross-conversation survey, a command-deck session)
      with no chat title to borrow — that lands with Slice 4's standalone `thread_id` minting.

**Command deck + ad-hoc delegation — roadmap (foundation laid in 1c)**
- [x] **Agent roster awareness** — a read-only `list_agents` tool — shipped (`0.21.114`).
      Lists `kind=='agent'` profiles (ambassadors excluded) with each agent's role tags,
      delegation availability, a role blurb (`description` → first paragraph of
      `system_prompt`), and — load-bearing for **multi-modal routing** — its model's **live
      provider capabilities**: input/output modalities + tools/vision/speech/transcription
      flags, resolved via `registry.resolve_with_fallback` → `provider.get_capabilities`
      (degrades **per agent**; never-raise). The `is_default` agent is flagged `primary`.
      Auto-advertised through the existing answer/voice agentic loop — no API/SSE change.
      (`agent/ambassador_tools.py`; tests `AmbassadorServiceTest.test_list_agents_*`.)
      Future: fold in the global **Delegation Handbook ("Dossier")** (§16 roadmap) once it lands.
- [ ] **Capability/strength modelling + recognize the primary agent from history.**
- [ ] **Ad-hoc delegation** — the ambassador (top-level agent) dispatches work to the right
      agent, reusing the relay `target` seam. The active-conversation context + per-conversation
      agent names from 1c are the inputs.
- [ ] **Swarm paradigm — aides gather, the ambassador stays high-level.** Today the tool belt
      reads **full transcripts into the ambassador's own context** (each read ~`_READ_TOKEN_BUDGET`,
      bounded by `_MAX_TOOL_ROUNDS`) — fine for one conversation, but a cross-conversation survey
      ("what have my agents discovered?") bloats context fast and gets expensive. Instead, the
      ambassador should **delegate to a swarm of cheap aide models**: each aide reads/condenses ONE
      conversation (or shard) read-only and returns a **high-level digest**, so the ambassador only
      ingests condensed summaries, never raw transcripts. Keeps its context lean + parallelizes the
      survey. Shape: `summarize_conversation`/`read_conversation` (and `survey_conversations`) become
      **fan-out to aide jobs** (their own model tier, e.g. `consolidation.feature_default_model`),
      results merged for the ambassador. Reuses the read-only tool belt as the aides' capability;
      ties into ad-hoc delegation (aides are the read-side, worker agents the write-side). Bound
      fan-out width + per-aide budget; never-raise per aide (one bad read doesn't sink the survey).

**Slice 3 — voice relay that works + read-only tools auto-run** — shipped (`0.21.113`)

> **Reframed from "voice confirms the tool call."** Decision: **confirm writes only**.
> All current ambassador tools are SELECT-only, so spoken read-only intents
> ("summarize this", "explore that") **auto-run** through the existing voice `answer`
> path (`_answer_to_text` → `_agentic_answer`, `with_tools=True`) and speak the result —
> no confirm strip needed. The only **write** is relay, which already confirms. So the
> `{action:'tool'}` generalization + a tool-confirm strip is **deferred** until there are
> real write/dispatch actions (e.g. cross-agent delegation) to confirm.

- [x] **Voice relay actually delivers** (`0.21.113`). Root cause: the relay seam
      (`ConversationContext.relayToConversation`) only has a registered handler for the
      **active tab** (`ChatPanel` registers `relayMessage` per `activeTab`), but the panel
      relayed to its **focused** conversation → silent `false` whenever focus ≠ active, and
      `VoiceBar.sendRelay` swallowed the failure (dead button). Fixed: one pure helper
      (`lib/ambassadorRelay.ts::relayToActiveConversation`) targets the **active**
      conversation (the one with a live handler) and returns `{ok, note}`; `AmbassadorPanel`
      (voice + text) and `VoiceBar` both report where it landed — on failure the draft is
      kept + an error shown. Persona hardened so clear imperatives route to `relay` and the
      ambassador never answers that it "can't talk to the agent". Tests:
      `ambassadorRelay.test.ts`; backend `AmbassadorVoiceCommandTest` (stale `usage` mock
      fixed). The "↦ relay instead" override remains the recovery for a misroute.
- [-] *(deferred)* **`route_voice_command` → `{action: answer|relay|tool, ...}`** + a
      voice tool-confirm strip ("Summarize this conversation?") with barge-in/retake +
      action-naming captions — re-opens once there's a **write/dispatch** action to confirm.
- [-] *(deferred)* **Relay to *any* conversation** (not just the active tab): a server-side
      `POST /api/agent/ambassador/relay` reusing `enqueue_background_chat(session_id=…)`
      (warm-hydrates, appends a real user turn, runs the agent headless, persists) + a
      conversation→profile resolver. Invariant holds (relay is a USER turn via the chat path).

**Slice 4 — command deck: the ambassador as orchestrator (the north-star)**
- [x] **Standalone, top-level ambassador surface** — shipped (`0.21.116`) as the **Command
      Deck**: a full-screen, app-wide surface (`SURFACES.ambassadorDeck`, ⌘K → "Open Command
      Deck") that mounts `AmbassadorPanel` in a conversation-less **deck mode**
      (`deckThreadId` prop) against a single persistent per-user thread (`deck:{user}`,
      `lib/ambassadorDeck.ts`). Reuses the `conversation_id` seam as the thread key →
      **zero backend changes**; the deck thread never touches `conversation_logs`, so it
      can't pollute its own survey. Conversation-bound affordances (brief-this-conversation,
      relay, the conversation switcher) drop away; the cross-conversation tools (survey /
      roster) + free-form ask + voice carry it. Worker conversations stay reachable directly
      (dual entry). Follow-ons:
  - [x] **Multiple named standalone Inquiries** + a switcher over them — shipped. **Backend
        (`0.21.119`):** per-user registry (Redis ZSET `amb_user:{user}:threads`,
        `register/unregister/list_user_threads`, self-healing) + `GET/POST /agent/ambassador/threads`
        (list / mint `inq:{user}:{uuid}`; `deck:{user}` pinned) + the ambassador's `rename_inquiry`
        tool (the belt's lone, self-scoped write — its own Inquiry title only). **Client
        (`0.21.120`):** `AmbassadorInquirySwitcher` in deck mode (create / switch; rename + clear via
        the `⋯` menu, syncing the registry) over `AmbassadorContext.{inquiries,listInquiries,
        createInquiry}`; pure `lib/ambassadorDeck.ts::orderInquiries` (home pinned, then recency).
  - [x] **Auto-titling** an Inquiry from its first question — shipped (`0.21.117`), model-free.
        `AmbassadorService._maybe_autotitle` titles an empty thread from its first question
        (text + voice) via `ambassador_storage.derive_title`; a `title_auto` flag on
        `set_thread_title` guards a manual rename from being clobbered.
  - [x] **Relay from the deck** to a chosen conversation — shipped. **Backend (`0.21.121`):**
        `POST /agent/ambassador/relay {conversation_id, text}` runs the target conversation's
        agent headless via `enqueue_background_chat` (real user turn; profile resolved from the
        conversation's last stamped turn via `latest_agent_name`/`get_profile_by_name`, else
        default). **Client (`0.21.122`):** `lib/ambassadorRelay.ts::planRelay` routes live in-tab
        when the target is the active tab, else headless via the endpoint; the panel relays to its
        focused conversation even off-tab, and the deck has a target picker (over `listConversations`).
- [x] **`survey_conversations`** — shipped (`0.21.115`), **lean / model-free**. Enumerates
      recent sessions (`list_recent_conversations`) and enriches each with its own **rolling
      summary** (`conversation_summary_storage.get_summary` — already a digest) when present,
      else the first/last snippet; the ambassador composes the **application-wide summary**
      from that block (its own model, already in the loop). Read-only (Postgres list + Redis
      GET), never-raise, no new coupling, **zero extra model calls**. The summary's payoff is
      concentrated in long/aged conversations (short ones degrade to the snippet ≈
      `list_conversations`). (`agent/ambassador_tools.py`; tests
      `AmbassadorServiceTest.test_survey_conversations_*`.) Follow-ons:
  - [x] **Per-conversation goals** — shipped (`0.21.118`). `GoalMemory.add_goal` now persists
        `conversation_id` on the `(:Goal)` node (the value already flowed per-run via the memory
        facade — `core.py` sets `memory.conversation_id`; only the CREATE Cypher needed it),
        backed by Neo4j index `goal_conversation` (migration `0004`). New
        `get_goals_for_conversation` (any channel) feeds a best-effort `goals:` line in
        `survey_conversations` (never-raise — a down/disabled Neo4j degrades to no line). **Backfill
        caveat:** only goals created after this carry `conversation_id`; older goals won't appear.
  - [x] **Aide swarm** — shipped (`0.21.137`). `agent/aide_swarm.py::AideService` (mirrors
        `ToolOutputCompressor`) fans out cheap **aide** model calls — each condenses ONE conversation
        read-only into a short digest, so the ambassador ingests digests, never raw transcripts
        (map-reduce: aides map, the ambassador reduces). Wired into `survey_conversations`
        (un-summarized convs get a parallel digest instead of a thin snippet — summarized ones stay
        zero-extra-cost) and `summarize`/`explore` (digest instead of dumping the transcript);
        `read_conversation` stays raw as the drill-in path. Bounded (`asyncio.Semaphore`
        `max_parallel` + per-aide `wait_for` timeout + `max_per_survey` cap), **never-raise**, and
        OFF ⇒ today's behavior. Digests cached in the sidecar (`amb_aide:` via `ambassador_storage`,
        fingerprinted on message_count+last_at — INV-2 holds). Cheap tier defaults to the haiku floor
        (the doc's `consolidation.feature_default_model` never existed). Metered under usage source
        `aide`. Config `ambassador.aide.*` (default-on, Settings → Ambassador opt-out).
        **Deferred:** aide-condensing `read_conversation`; promoting the thread to a durable store.
- [x] **Dispatch seam (write-side) v1** — shipped (`0.21.138`). The orchestration write-side —
      the ambassador handing a task to a worker. `POST /api/agent/ambassador/dispatch` `{agent_id,
      text}` mints a **brand-new conversation** and runs the chosen worker **headless** on the task
      as its first **user** turn (`enqueue_background_chat`) — you authored it, so INV-2 holds
      (`AmbassadorService` never writes a transcript as itself). Confirm-first per the roadmap: the
      user picks any agent in a new **Dispatch** composer mode (`AmbassadorPanel`), the ambassador
      drafts a self-contained task (`draft_relay_message(fresh=True)`), then sends; the client opens
      the new conversation on-ready (polls past the async-worker 404 window). Gated by
      `ambassador.dispatch.enabled` (default on, Settings opt-out). The `{agent_id}` seam is
      target-extensible. **Deferred → resolved in the `0.21.235` close-out (see the v3 list):**
      ambassador-*proposed* targeting, dispatch into an **existing** conversation, and the
      instant task echo all shipped; fully **autonomous** (confirm-less) dispatch and
      `AlloyExecutor.delegate` start/steer stay parked on purpose (doctrine — a future per-user
      policy setting). The survey/aide-swarm is the read-side of that same world.

**Ambassador v3 — context state, conversation management & analysis (proposed 2026-07-16)**

> Belt review verdict: the v2 reads are strong (transcript/survey/roster + aide digests) but
> she's blind to everything *around* the transcript — the structured conversation state, live
> runs, memory, and spend — and discovery is recency-only (no search). The expansion keeps the
> Slice-3 doctrine (**reads auto-run, writes confirm**) and every new tool must be classified
> on purpose in `test_tool_belt_write_surface_is_pinned`.

- [x] **`read_conversation_state` (context state)** — **shipped (`0.21.233`)**. SELECT over the
      Slice-1a structured working memory (`render_state_block`); the deep survey gained the
      one-line `state:` hint (open-threads count + freshest decision, `_conversation_state_line`,
      goals-line doctrine).
- [x] **`search_conversations`** — **shipped (`0.21.233`)**. Lexical FTS via Alembic **0008**
      (expression GIN on `to_tsvector('simple', content)` — no table rewrite);
      `conversation_history.search_conversation_logs` groups hits per conversation with
      markdown-bold `ts_headline` snippets (`websearch_to_tsquery('simple')` — multi-word AND +
      phrases, language-neutral; CJK degrades, pg_trgm is the additive later fix; semantic later).
- [x] **`list_active_runs`** — **shipped (`0.21.233`)**. Merges `chat_run.store.list_runs` +
      `list_background_chats` (each half degrades independently); agent via `latest_agent_name`;
      needed `user_id` plumbed through the service → `execute_tool` (from `_bg_user_id`).
      The ambassador's own runs are un-indexed so they never self-report.
- [x] **`usage_report`** — **shipped (`0.21.233`)**. The `/metrics/usage` SQL extracted into
      `usage_ledger.aggregate_usage(days, conversation_id=None)` (the view delegates — no
      duplicated SQL); tool renders totals + top-3 by model/agent/source, `"this"` scopes to
      the focused conversation.
- [x] **`recall_memory`** — **shipped (`0.21.233`)**. `AgentMemory.remember` wrapped never-raise
      (it embeds + can raise, unlike the other sources); default `_global` channel with an
      explicit `channel` escape hatch.
- [x] **Conversation management (the confirmed-write belt)** — **shipped (`0.21.233`)**.
      `rename_conversation`/`archive_conversation`/`delete_conversation` are **proposal-only**
      (`proposal_for`; delete requires an explicit id — never the focused default): the loop
      emits `ambassador_tool_proposal` + stamps the chip, the client renders a warning-tone
      confirm strip (latest-entry-only; delete rides `ConfirmDialog` with title + message
      count) and executes via the new `PATCH /memory/conversations/{id}/meta` over the
      Alembic **0007** `conversation_meta` table (durable custom title + archived flag,
      read-through in both listing paths, archived default-excluded w/ `include_archived`)
      or the existing DELETE (now also drops the meta row). Voice re-opened the deferred
      Slice-3 `{action:'tool'}` strip (`VoiceBar`); the conversation list write-throughs
      rename/auto-title/archive to the server store so the local meta and the durable overlay
      stay in step. Pinned test grows the documented `confirmed_conversation_meta_writes` class.
- [x] **`read_conversation_results`** — **shipped (`0.21.235`, close-out)**. The 16.7-deferred
      exhibits/sources read: SELECT-only over the persisted synthetic `present_exhibit` tool
      turns (`conversation_history.load_exhibit_calls` — no bibliography plumbing needed; the
      turns ARE the record). `_render_results` parses via `exhibit_from_present_call`, keeps
      the latest call per exhibit id (revise-in-place), lists citation sources label + URL;
      tables/images/diagrams as titled mentions. Classified `read_only` in the pinned test.
- [x] **Dispatch orchestration (confirm-first `dispatch_task`)** — **shipped (`0.21.235`,
      close-out)**. The 16.7-deferred *ambassador-proposed targeting*, *dispatch into an
      existing conversation*, and *instant task echo*, all in one slice: `dispatch_task`
      joins `CONFIRMED_WRITE_TOOLS` (proposal-only — loop stamping/SSE/voice ride the set for
      free); `proposal_for` resolves the worker (`_resolve_worker`: agent_id else exact name,
      agents-only) into `{action: dispatch, agent_id, agent_name, task, conversation_id?}`;
      the endpoint takes optional `conversation_id` (unknown → 404, relay guard); the client
      confirm strip executes via `POST /dispatch` and the open-on-ready poll became an
      **instant echo** (`openDispatchedTab`: open-tab append / seeded restore / fresh seeded
      tab). Conversation DELETE now clears the state/summary/ambassador sidecar families
      (`deleted.sidecars`). **Parked on purpose** (the close-out's non-goals): fully
      unconfirmed autonomous dispatch (breaches reads-auto/writes-confirm — a future per-user
      policy setting), `AlloyExecutor.delegate` start/steer from the ambassador, Inquiry-thread
      durability + Deck scope (still open questions below), Factory UI.
- [x] **Polish close-out (grounded voice-agent prompts, think-strip, module seam)** —
      **shipped (`0.21.236`)**. The final Ambassador slice before agent orgs. The
      **grounded-fact doctrine** defeats the chain's sequential lossiness (conversation →
      aide → ambassador → voice): aide digests end with a **verbatim `Facts:` tail**
      (survey clip is facts-aware via `_clip_digest`); the answer persona gains GROUNDED
      FACTS / THE-MAP-IS-NOT-THE-TERRITORY (drill down via the belt when a digest can't
      answer; quote verbatim on "exactly") / natural PROVENANCE / lead-with-the-answer;
      the voice answer leg adds `_SPOKEN_DELIVERY_NOTE`; the voice router was rewritten
      (concrete say-backs) and became a first-class **overridable persona**
      (`AmbassadorConfig.voice_persona`, persona-defaults `voice` key, fourth editor
      field). **Think-strip**: `ThinkBlockStripper`/`strip_think_blocks` suppress
      reasoning blocks at the stream, persist, TTS, and router-parse layers (+ the client
      back-compat strip for pre-fix records) — the live-caught `<think>` leak is dead.
      **Deck**: entry-module registry (`entryModules.tsx` — tool chips + proposal strips
      are the first two; future aide-delegation/exhibit/form/report surfaces are one
      registry entry each), five starter chips with icons, and the portal z-fix
      (DropdownMenu — and Popover as convention — joined `z-[2400]`; Deck menus no longer
      open behind the surface). Rolling-summary prompt deliberately untouched (main-agent
      compaction territory; the drill-down doctrine covers it).
- [x] **Client — promote the Command Deck (and Memory) to first-class desktop tabs** —
      **shipped (`0.21.234`)**. TopBar gains a **surface-pill group** after Agents (Deck +
      Memory — no new `PageId`s; selected state *derives* from `useModal().isOpen`, the Dock
      lesson applied by deriving rather than duplicating open-state). True tab semantics: on
      desktop both surfaces render **below the 56px bar** (`top: 56px`; the Deck joined
      `FULLSCREEN_SURFACES`/`SELF_CLOSING` and owns its chrome — inline close in the panel
      header, Escape, scroll lock), the lit pill toggle-closes, and navigating to a page
      dismisses the open surface (the page pill goes quiet while a tab-surface holds the
      selection). Deck pill carries a live pulse while background runs are active (30s poll,
      quiet once open). Pills are hidden ≤600px — the palette entries remain the mobile path;
      mobile keeps full-viewport surfaces. Cross-ref:
      [backlog/conversation-context.md](../backlog/conversation-context.md) Memory Area UX.

**Stability & invariants (apply across all slices)** — ✅ closed out
- [x] No-pollution regression tests extended to the thread + tool loop:
      `test_agentic_loop_writes_only_the_ambassador_sidecar` (a full tool round + answer
      writes only `amb_thread:`/`ambassador:`/`amb_aide:`/`amb_user:` keys — never
      `conv_summary:`; no PG connection exists in the test env, so a `conversation_logs`
      write would raise) + `test_tool_belt_write_surface_is_pinned` (structural
      SELECT-only allowlist — a new tool fails the test until classified on purpose;
      `rename_inquiry` is the documented lone self-scoped write).
- [x] Never-raise tests: empty conversation + no provider (already covered), **tool
      error** (`test_execute_tool_degrades_when_a_read_fails` at the belt boundary +
      `test_agentic_loop_survives_a_failing_tool` mid-loop), and **no conversation at
      all** (`test_deck_thread_with_no_conversation_stays_clean` — the server-side
      "no active tab") — each degrades to a clean notice.
- [x] Docs: `CLAUDE.md` ambassador one-liner rewritten for v2 (tool belt, Inquiries,
      deck, relay/dispatch); `OpenApi.yaml` + `endpoints.md` now carry the
      `ambassador_tool_call`/`_result` SSE events, a v2 intro, and the missing
      `/relay` + `/threads` bullets; the Development-Notes endpoint table gained
      `/relay`, `/dispatch`, `/threads` rows.

**Open questions (decide before building the relevant slice)**
- Thread persistence depth: keep the full ambassador thread in Redis (TTL'd like
  today) vs. promote to a durable store for the command-deck history? (Recording
  lifecycle from 16.6 is the sibling decision.)
- Tool-belt surface: do tool calls show as chips in the text thread (transparency) or
  stay invisible (just the answer)? Lean transparent — they're the proof of grounding.
- Command-deck scope: all conversations, or a user-pinned working set?

### Design Notes

- `agent_id` (Docker-style, e.g., "bold-cosmic-falcon") = formal routing identifier
- `name` (e.g., "Claude", "NodeManager") = flexible display name
- `Message.name` field carries `agent_id` on assistant messages — no provider schema changes
- Extend existing `agent/chat/stream` with optional `target_agent_id` — no new endpoints
- Memory already supports this: each agent recalls from `[channel, _self_{agent_id}, _global]`


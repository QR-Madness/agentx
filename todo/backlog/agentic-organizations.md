# Agentic Organizations — Org Manager, Chain of Command, Persistent Delegation Threads

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)
> Supersedes the "Manager agents (agent → team delegation)" bullet in
> [phase-16-multi-agent.md](../phases/phase-16-multi-agent.md) (16.x Deferred) and realizes the
> deferred Factory-canvas item at org scope.

> **The idea:** Agent Teams grow into **Agentic Organizations** — a real hierarchy with a strict
> **chain of command**: **managers** delegate to the **leads** of the teams they own, leads work
> their team's **members**, and members can escalate back up to their own lead. **Executives** (a
> tier above managers) are named and reserved but explicitly parked. A full-screen **Organization
> Manager** panel (React Flow canvas + roster + docked editors) replaces the Agent Teams modal,
> and every delegated run becomes a durable, openable **conversation thread** instead of an
> ephemeral card. The **ambassador** is hardened as the built-in apex: non-deletable, pinned atop
> the org canvas (its §16.7 "top-level agent" framing made visible).

**Terminology.** "Org manager" here always means the **agent tier** — unrelated to the
*deployment* manager (`manager/`, ADR-10); new backend modules use `org_`/`chain_` prefixes, never
a bare `manager.py`. **Lead** is the shipped user-facing name for the internal `supervisor` role
(strings-only rename `[v0.21.170]`; internals keep `supervisor`/`specialist`). **Chain of
command** is the user-facing name for the strict adjacency-only delegation filter.

---

### Design pillars

- **Tier on the profile, structure on the workflow, layout in its own store.** The org is
  *derivable from the two existing stores* — no org-membership database. A profile declares its
  tier; a workflow declares its owner; everything else (edges, roster grouping, canvas) derives.
- **`kind` stays binary.** Managers/leads/members are all `kind="agent"` (routable, chattable);
  the hierarchy is an orthogonal `org_level` field. Ambassadors stay excluded from chat/routing/
  delegation exactly as today (phase-18.14) — nothing in this design touches those filters.
- **Chain of command is enforced twice**: at the `delegate_to` tool enum (the model can only name
  legal targets) and at executor validation (reality check). Both read one derivation module, so
  prompt, enum, and enforcement can never disagree.
- **Org-free installs are untouched.** Agents outside any org keep today's flat opt-in ad-hoc
  roster (`available_for_delegation`); the chain only applies to org participants.
- **Delegation runs become real conversations** — durable child threads linked both ways to the
  parent turn's delegation card, openable like any conversation. The transcript is no longer a
  place where work disappears.
- **Visualize-first canvas.** v1 renders the org and persists node positions; structural editing
  stays in docked forms over the existing CRUD. Canvas *editing* (drawing edges) comes later.

### Slice 1 — Org data model, Role picker, ambassador hardening

> Additive fields + one guard; ships standalone value (roles visible/editable, ambassador
> system-owned) and unblocks every later slice.

- [ ] **`AgentProfile.org_level`** — `Literal["executive","manager","lead","agent"]`, default
      `"agent"`, in `agent/models.py` beside `available_for_delegation`. Orthogonal to `kind`;
      `"executive"` is reserved in the enum *now* (pydantic rejects unknown values at YAML load,
      so pre-reserving avoids a data migration when executives unpark) but never offered in UI.
- [ ] **`Workflow.manager_agent_id: str | None`** — in `alloy/models.py` (+ `to_dict`), parsed in
      `alloy/manager.py::_parse_workflow` and `views.py::_parse_workflow_payload`. Semantics: a
      manager owns *teams*, not agents — manager→leads derives as "supervisors of workflows where
      `manager_agent_id == X`"; one team has ≤1 manager, one manager may own many teams.
- [ ] **Persistence round-trip test** — `org_level` and `manager_agent_id` must be added to
      `profiles.py::save_config`'s hand-enumerated field list **and** the four hand-picked profile
      serializer blocks in `views.py`, or they're silently dropped on restart (the documented
      Phase-18.2 regression class). The test asserts a full save→load→serialize round trip.
- [ ] **Soft validation** in `manager._validate`: `manager_agent_id` must resolve to a known
      profile, must not equal `supervisor_agent_id`, must not appear in `members`. Tier mismatches
      (manager profile whose `org_level != "manager"`, supervisor not `"lead"`) are **log-warn
      only** (like the routes warning) — hard rules would break existing `workflows.yaml` on boot.
- [ ] **Profile editor Role picker** — agent-kind profiles only: Agent (member) / Lead / Manager;
      Executive omitted; ambassador profiles show no picker (`kind` stays read-only in the
      editor). Hint copy: once an agent is in an org, chain of command supersedes
      `available_for_delegation` (which keeps governing only the flat roster).
- [ ] **Ambassador delete guard** — `ProfileManager.delete_profile` refuses the default ambassador
      (`kind=='ambassador' and is_default_ambassador`) with `ValueError` → 4xx from the DELETE
      handler; client hides the delete affordance. Today deletion "works" and the boot reconciler
      resurrects a fresh one — the guard makes the system-owned contract explicit and immediate;
      `_ensure_ambassador_defaults` stays as the safety net.
- [ ] **Docs**: `OpenApi.yaml` + `endpoints.md` (profile + workflow schemas), version +
      Release-Notes bump with the slice.

### Slice 2 — Chain of command (the strict delegation gate)

> One pure derivation module, consumed at both existing choke points. Adjacency-only by
> construction — no level-skipping, no peer-to-peer.

- [ ] **`alloy/org_chart.py`** — pure functions over `get_profile_manager()` +
      `get_workflow_manager()`: `teams_managed_by`, `teams_led_by`, `teams_of_member`, `in_org`,
      and the single source of truth `chain_targets(agent_id) → [(agent_id, name, hint,
      relation)]` with relation ∈ {`down_lead` (manager→lead of owned team), `down_member`
      (lead→own team member), `up_lead` (member→own lead, escalation)}. Unit-tested standalone.
- [ ] **Tool-enum gate** — `list_adhoc_delegation_targets`: if `in_org(self_agent_id)` and the new
      knob `alloy.chain_of_command` (default **on**) is set, return `chain_targets(...)` (still
      excluding self and non-`kind=='agent'`); otherwise today's flat opted-in roster unchanged.
      The roster prompt block shares this function, so prompt and enum stay in lockstep for free.
      Deliberate asymmetry, documented: **chain edges ignore `available_for_delegation`** —
      adjacency implies delegability (exactly as workflow supervisors already delegate to
      specialists regardless of the flag); the flag keeps governing only the flat roster.
- [ ] **Executor gate** — `AlloyExecutor._validate_target`: in ad-hoc mode, when the delegator is
      in-org + knob on, the target must be in `chain_targets(delegator)`. Workflow mode is already
      adjacency-only (supervisor→specialists) and stays as is. Defense in depth: the enum
      constrains the model, the executor constrains reality.
- [ ] **Loop guard** — thread a `delegation_path: list[agent_id]` parameter down
      `AlloyExecutor.delegate` → nested executors (a parameter like `depth`, not instance state,
      so concurrent fan-out branches don't race); `_validate_target` rejects any target already in
      the path — kills manager→lead→member→lead cycles categorically.
- [ ] **Downward nesting** — when a delegated specialist is itself a **lead of a team** (knob on),
      its ephemeral Agent gets a `delegate_to` scoped to its `down_member` targets at `depth+1`
      (`max_delegation_depth=3` already fits manager(0)→lead(1)→member(2)). Members in-chain still
      get **no** delegation tool.
- [ ] **Upward escalation** — *top-level* member chats get an `up_lead`-scoped `delegate_to`
      (uniform tooling: same tool, same SSE events, same cards — zero new client work). *Mid-run*
      escalation is a structured-result convention (`[NEEDS ESCALATION] …` block the awaiting
      lead's own loop reacts to) — a distinct `escalate` verb/tool with its own UX is an open
      question below.
- [ ] **Docs**: Decisions.md new INV candidate — *"chain of command is enforced at the tool enum
      AND the executor; adjacency only"*; existing delegation tests in `tests.py` updated.

### Slice 3 — Persistent delegation threads

> Delegated runs stop being ephemeral: each becomes a durable child conversation, linked both ways
> to the parent's delegation card. Extends `AlloyExecutor` in place — **not**
> `enqueue_background_chat` (fire-and-forget, no SSE backchannel, slated for retirement per
> ADR-12); the background-chat seam stays the precedent for the persistence *shape* and the
> ready-made path for follow-up messages into a finished thread.

- [ ] **Child conversation minting** — `child_conversation_id = uuid4()` alongside
      `delegation_id`; added to `delegation_start`/`delegation_complete` SSE payloads (additive —
      old clients ignore it).
- [ ] **Turn persistence** — on completion, persist two durable turns via the existing audit write
      path `episodic.store_turn_log`: turn 0 `user` = the task text, turn 1 `assistant` = the
      accumulated specialist output; both with `conversation_id=child_conversation_id`,
      `agent_id=target`, the **same channel as today** (shared `_alloy_{workflow_id}` or ad-hoc —
      INV-7 unchanged, no new scope), and `metadata.delegation = {delegation_id,
      parent_conversation_id, delegator_agent_id, depth}` + `agent_name` (so the worker is
      recovered for follow-ups). No Session object is created — opening the thread hydrates from
      history like any restored conversation. Knob `alloy.persist_delegation_threads` (default on).
- [ ] **Parent-side linkage** — `delegation_raw[tc.id]` gains `conversation_id` → rides
      `result_entry["delegation"]` → persisted on the delegator's tool turn → reloaded cards carry
      the link. Both directions stamped. The shared-channel memory Turn (`[Delegated to X]…`)
      **stays** (recall/attribution, per ADR-6) and gains the child id in its metadata. Cost
      rollup unchanged: `usage_ledger` stays keyed to the parent conversation.
- [ ] **Sidebar hygiene** — child conversations appear in `list_recent_conversations`
      automatically; the lister gains a `delegation_of` field, the client badges children and
      ships a default-on "hide delegated runs" filter (full grouping lands in Slice 5).
- [ ] **Client links** — `DelegationBubble` gets an "Open thread" affordance when
      `delegation.conversation_id` is present → `restoreConversation` into a new tab; old
      persisted cards lack the field and render exactly as today (zero back-compat cost);
      `AlloyRunTraceModal`/`alloyTrace.ts` surface the same links per run.
- [ ] **Docs**: `OpenApi.yaml` (SSE events + conversations-list field) + `endpoints.md`;
      Decisions.md ADR-6 note (delegation transcripts are conversation records) + an explicit
      INV-2 non-impact statement — these writes are the *supervisor's* delegation flow, never the
      ambassador; dispatch stays `enqueue_background_chat`-only.

### Slice 4 — Organization Manager panel (visualize-first)

> Replaces `AlloyFactoryModal` outright — the "Factory Canvas — coming soon" banner finally pays
> off, at org scope. Needs Slice 1 only.

- [ ] **Org layout store** — new `alloy/organization.py` (WorkflowManager-shaped load/save) owning
      `data/organization.yaml`: **layout only** (canvas node positions keyed `agent:{agent_id}` /
      `team:{workflow_id}` — INV-3-safe; membership/edges always derive). Exposed as
      `GET/PUT /api/alloy/organization`. `Workflow.canvas` stays untouched/reserved for the future
      per-team factory editor.
- [ ] **OrgManagerModal** — full-screen takeover: left roster sidebar (Ambassador / Managers /
      Leads / Members / Unaffiliated, delegable badges); center React Flow canvas (precedent:
      `MemoryGraphView.tsx`; `@xyflow/react` already a dep) — **ambassador apex pinned**
      top-center, non-draggable; reserved empty **executive band** below it (label only); managers;
      **team subflow groups** containing lead + members; unaffiliated delegable agents in a side
      lane; solid chain edges (manager→lead, lead→member) with the member→lead escalation rendered
      as a dashed up-chevron on the same edge; a subtle ambassador **dispatch halo** — visually
      distinct because dispatch is *not* a chain edge; right contextual dock — team editor (the
      absorbed AlloyFactory form over the existing `/api/alloy/workflows` CRUD), profile
      quick-edit (role, Specialty, delegability, "open full editor"), read-only edge inspector.
      Node drags debounce-PUT positions.
- [ ] **Surface rewiring** — repoint `SURFACES.teams` → `{id:'org-manager',
      component:'orgManager', size:'full'}`; add `orgManager` to `MODAL_REGISTRY` +
      `SELF_CLOSING` + `FULLSCREEN_SURFACES` in `ModalPortal.tsx` (the UnifiedSettings takeover
      pattern — AlloyFactory never was); extend the `open-teams` command keywords with
      `['organization','org','a2a','a2a manager','agent teams','manager']` (palette aliases:
      "Agent Teams", "A2A Manager"); `AgentSelectorDropdown.handleManageWorkflows()` follows via
      the shared surface. Delete `AlloyFactoryModal.tsx`.
- [ ] **Docs**: `OpenApi.yaml` + `endpoints.md` (new organization endpoint); phase-16 Factory-
      canvas cross-link.

### Slice 5 — Live org activity & thread grouping

> The org panel becomes a window into work, not just structure. Depends on Slices 3 + 4.

- [ ] Delegated-runs strip in the panel (recent/live child threads, status, cost)
- [ ] Open-thread directly from canvas nodes (agent node → its recent delegation threads)
- [ ] Sidebar grouping of child conversations under their parent
- [ ] Edge pulse on active delegations (live `delegation_*` events animate the chain edge)

### Parked — Executive tier

- [-] **Executive semantics** — an executive delegates across multiple managers/teams (phase-16
      tiering note). Reserved now: the `"executive"` enum value (so future YAML loads today) and
      the empty canvas band. **No delegation semantics, no picker option, no validation, no build**
      until unparked.

### Stability & invariants (apply across all slices)

- **Old data keeps loading**: every new field is optional with a default — `workflows.yaml`
  without `manager_agent_id` and `agent_profiles.yaml` without `org_level` load unchanged;
  validation additions are soft-warn only. Explicit boot-with-old-YAML test per store.
- **Hand-enumeration hazard**: `profiles.py::save_config` + the four `views.py` serializer blocks
  silently drop unlisted fields — Slice 1's round-trip test guards the whole class.
- **`kind` stays binary; ambassadors stay out of chat** (phase-18.14): `org_level` is orthogonal;
  `list_adhoc_delegation_targets` keeps `kind=='agent'`; no routing-filter changes.
- **INV-2 / ADR-2 untouched**: child-thread turns are written by the supervisor's delegation flow
  (user-initiated work), never by the ambassador; ambassador dispatch stays
  `enqueue_background_chat`-only.
- **INV-3**: every org reference (`manager_agent_id`, canvas keys, chain edges) is an `agent_id` —
  renames are aliases. **INV-7**: child threads ride the existing shared alloy channel; no new
  scope.
- **Seeded defaults**: no new one-time seeds; the ambassador reconciler is unchanged — the delete
  guard complements it.
- **Back-compat matrix**: old client + new API → additive fields/SSE ignored; new client + old
  data → cards without `conversation_id` and profiles without `org_level` degrade to today's
  rendering; org-free installs → flat roster behavior byte-identical.
- **Fan-out amplification**: nested executors each own a `Semaphore(max_parallel_delegations)` —
  manager→N leads→M members multiplies concurrency (N×M). v1 mitigation is the depth cap + this
  documented warning; a global budget is an open question.
- **Docs travel with the work**: each slice updates `OpenApi.yaml`/`endpoints.md` (+
  `Decisions.md` where flagged) and bumps version + Release-Notes per repo convention.

### Open questions (decide before building the relevant slice)

1. **Escalation UX** — distinct `escalate` tool + user-facing notification vs the uniform upward
   `delegate_to`; how a mid-run `[NEEDS ESCALATION]` surfaces beyond the awaiting lead's loop.
2. **Completion notification** — delegations are awaited in-parent today; for timeout/backgrounded
   runs: bg-chat inbox, toast, or org-panel badge?
3. **Team-as-target delegation** (the phase-16 workflow-id enum sketch) — still wanted once
   manager→lead nesting exists, or pure sugar? (Delegating to the *lead* already engages the team.)
4. **Tool-turn parity in child threads** — persist the specialist's tool_call/tool_result rows as
   child turns (reload parity in-thread) or keep user+assistant only?
5. **Org store growth** — does `organization.yaml` later own explicit edges (multi-manager teams,
   matrix reporting) or stay layout-only forever?
6. **Nested fan-out budget** — global semaphore vs per-level caps.
7. **Gate scope** — chain of command never constrains @-mentions or ambassador dispatch
   (recommendation: no — human dual-entry stays unrestricted per §16.7).
8. **Follow-ups to finished threads** — plain chat in the child, or a re-delegation that stamps
   results back to the parent card?
9. **Executive semantics when unparked** — multi-manager delegation; the apex relationship to the
   ambassador.

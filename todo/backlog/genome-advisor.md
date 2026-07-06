# Agent Genome & Settings Advisor (the meta-layer)

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

### Agent Genome & Cognitive Evolution (intelligence-focused)

> External idea (Copilot, codebase-blind) evaluated against the actual code. The genome's real value
> is **unification + wiring**: consolidating our scattered cognitive knobs (reasoning strategy,
> ToT branching, Reflection, temperature, delegation config, tool gating) into one tunable per-profile
> struct read per task. The JSON schema is trivial; wiring each gene to a real lever — and giving the
> vague ones (`abstraction_level`, `evidence_strictness`, `tool_bias`) a concrete meaning — is the
> work. Dependency-ordered; the evolution loop is a *research bet*, not an engineering task.

- [ ] **(1, foundation) Reasoning-quality scoring (LLM-as-judge)** — score an agent's reasoning trace
      on coherence / groundedness / foresight / abstraction / self-correction, stored per task. The
      existing `eval_consolidation` harness is **memory-only**, so this is new; reuse the provider layer
      + Reflection's critique-prompt patterns. Independently valuable (powers the **Context Inspector**
      + dashboards) even if evolution never ships. Build this first.
- [ ] **(2) Agent genome — unify cognitive knobs on `AgentProfile`** — a tunable struct
      (`planning_depth`, `branching_factor`, `abstraction_level`, `self_critique_strength`,
      `evidence_strictness`, `delegation_aggressiveness`, `tool_bias`) read per task. **Wire genes to
      existing levers**: `planning_depth`→reasoning strategy + ToT depth / `planner.max_subtasks`;
      `branching_factor`→ToT beam width; `self_critique_strength`→**Reflection** passes (already exists);
      `delegation_aggressiveness`→`alloy.*` thresholds. **Operationalize the unwired genes**
      (`abstraction_level`, `evidence_strictness`→a verification/fact-confidence pass, `tool_bias`→
      tool-choice prompting). Half maps to machinery we have; the value is one coherent control surface.
- [ ] **(3) Context-adaptive genome expression** — modulate genes by derived signals (uncertainty,
      time/risk, tool availability): e.g. high uncertainty → deeper planning, high risk → stricter
      evidence. Downstream of (2); needs uncertainty/risk signals we'd have to derive (not free).
- [ ] **(4) Genome presets = "thinking styles"** — named bundles (careful-analyst, creative-strategist,
      fast-executor) extending the existing `DEFAULT_PROFILES`; Alloy can assign a style to a specialist.
      Falls out of (2) cheaply.
- [ ] **(5, EXPLORATORY — research bet, gate it) Offline genome evolution + intelligence control loop**
      — actor (AgentX) / critic (LLM judge from #1) / environment (a *reasoning* eval harness) →
      store task→trace→score→genome, mutate, keep top-K, discard worst; plus an SLO controller that
      nudges genes when the rolling score drifts. **Risks to respect:** LLM-judge scores are noisy +
      gameable, and auto-tuning a controller off them invites oscillation / reward-hacking. Treat as a
      time-boxed experiment with a **kill criterion** (must beat a fixed-genome baseline on held-out
      tasks), not a shippable feature. Depends on (1)+(2). *(Note: the "online self-critique" half of
      Copilot's #7 already exists as the Reflection strategy.)*

### Settings Advisor + Settings Manifest (the control-plane interface)

> Conceptual frame — the **family model**: **parents** = the Settings Advisor *and* evolution as one
> governance layer with standing authority over the **children** (agents), who act only within the
> config/genome the parents give them (children may *petition* — failures, low reasoning scores,
> uncertainty — but the parents decide). The **user is an associate of the parents** — a *peer*, not a
> boss and not a child: co-decides, gets explanations, sets the **bounds** the parents may act within,
> and keeps ultimate veto. So evolution is not a separate machine — it's **the parents doing long-term
> child-rearing autonomously *within those bounds***; the Advisor is the same governance acting in the
> moment / with the associate. Both run one primitive: *propose a config/genome diff → validate against
> the manifest → (optionally) eval its effect → apply (auto if within bounds, else escalate)*.
> The Advisor's voice follows from "associate": transparent peer — "here's what I see, here's what I'd
> do, your call" — never subservient, never commanding.

- [ ] **(keystone) Settings Manifest** — a canonical registry of every config key
      (`{path, type, default, range, description, "how it works abstractly", affected feature}`).
      Today this knowledge is scattered as inline comments in `config.py` + ad-hoc UI hints. One
      manifest collapses **four** items into itself: it feeds the **Settings Advisor**, lets the
      **settings-overhaul panel** auto-generate a clean UI, supplies the **"document every feature
      in-UI"** + **Memory Area cleanup** descriptions, and gives `/api/config/update` real validation.
      Build this first.
  - [x] **v1 registry + API** `[v0.21.160]` — `agentx_ai/settings_manifest.py` +
        `GET /api/settings/manifest`: both stores (memory `Settings` fields + `DEFAULT_CONFIG`
        leaves) with per-key `{store, type, default, value (secrets redacted), writable_via,
        role_member/role}`; write routing mirrors `config_update`'s section handlers
        (`_CONFIG_WRITE_ROUTES` — extend in lockstep) and the two memory endpoints' whitelists.
  - [ ] **v2**: prose `description`/"how it works" per key, validation ranges, dead-knob flags —
        then `/api/config/update` validation + auto-generated UI hints hang off it.
- [ ] **`@Settings` Advisor agent** — a built-in agent profile addressed via the shipped @-mention
      routing (16.5). Free-rein **read** access: the Settings Manifest, the docs-site (a docs-search
      tool), and a **conversation-diagnostic** tool (transcript + the **Context Inspector** + logs/
      metrics) so it can answer "**why did X happen**" and pinpoint the setting responsible. Proposes
      fixes as a **confirmed `form`/`choice` exhibit** that writes via `/api/config/update` —
      **read-broad, write-gated** (user confirms; never silent writes). Uses a **long-context model
      (Opus 1M)** to swallow a whole conversation for diagnosis; budget its own context carefully
      (reuse `assemble_turn_context`). *(Depends on: Settings Manifest; the `form` exhibit element for
      rich apply-a-fix UI — `choice` covers simple toggles until then. This agent is the consumer that
      makes the observability cluster — Context Inspector, SSE status, reasoning scoring — pay off.)*
- [ ] **Shared "control-plane change" primitive** — a single path that takes a config/genome **diff**,
      validates it against the manifest, applies it, and (optionally) evals its effect. The Advisor
      drives it human-confirmed; the evolution subsystem (above) drives it autonomously within bounds.
      Unifying these means evolution is just "the Advisor on auto, gated" — not a separate machine.
- [ ] **Autonomy envelope (the safety keystone)** — a per-system policy object the *associate* (user)
      grants the *parents*: which genes/settings may be auto-tuned and within which ranges, what is
      always escalate-and-confirm (cost, API keys, destructive resets, model swaps), and the
      log/notify behavior. This is what makes evolution **bounded child-rearing** rather than an
      unsupervised mutation loop, and gives the Advisor its collegial-but-empowered footing. Low-risk →
      act + log; high-risk → escalate to the associate. Every control-plane change is checked against it.
- [ ] **Child→parent petition channel** — agents emit governance signals (repeated failures, low
      reasoning scores, high uncertainty, tool errors) that the parents consume as inputs for tuning a
      child. The children do the work and surface what's hurting them; the parents decide the fix.


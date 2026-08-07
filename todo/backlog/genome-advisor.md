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
        role_member/role}`; write routing mirrored `config_update`'s section handlers
        (`_CONFIG_WRITE_ROUTES` — "extend in lockstep") and the two memory endpoints' whitelists.
  - [x] **v2 + the registry it hangs off** `[v0.21.263]` — Genome Foundation Wave 1. The mirror is
        gone: `settings_registry.py` is the one declaration, and the write path, the manifest, and
        the docs all derive from it (ADR-17). Ships two-phase writes (validate-all-then-apply-all),
        per-key `constraints`/`tier`/`ui_section`/`nullable`/`empty_means`/`write_mode`, authored
        `help` in `settings_help.yaml`, opt-in write validation, and unified secret classification.
        Fixed by construction: `research`/`web_research` under-reported, `ambassador`'s voice keys
        over-reported-then-dropped, `search.source_policy` per-leaf clobbering. Retired the
        unvalidated `trajectory_compression_*` bridge on `/api/memory/settings`.
  - [x] **Settings UI reads the manifest** `[v0.21.263]` — field kit v2 (default chips, reset,
        first-class help popovers, the a11y label fixes), the Overview landing (what you've changed,
        computed from the manifest), per-section error boundaries, and the golden Recall refit as
        the help-content exemplar.
  - [x] **Generated Settings Reference** `[v0.21.263]` — `scripts/gen_settings_reference.py` renders
        the docs-site page from the same declarations, gated by `task docs:check`. Settings and docs
        can no longer drift.
  - [x] **Wave 2 — reach the setting** `[v0.21.264]` — search matches individual settings (keys,
        humanized names, and authored help) via the manifest, not just the 21 section labels;
        results jump to the control with a scroll + flash (`lib/scrollToAnchor.ts`, generalized from
        the Plans-drawer recipe) using `data-setting` anchors the field kit emits; the modal props
        chain carries `initialSection`/`focusSetting` end-to-end (fixed `stubs.tsx`, `surfaces`,
        `open()` arity, and RootLayout's duplicated ⌘, descriptor); registry-driven `Settings: <x>`
        palette commands, `searchOnly` so they don't bury the resting list.
  - [ ] **Wave 3 — help cadence** (in progress): write up the remaining sections to the golden
        standard, section by section, binding each to the manifest as it goes. A section joins
        `SettingsHelpTest.DOCUMENTED_SECTIONS` when refit, which then holds its coverage.
        **208 writable settings; 45 documented at v0.21.265.** Sliced by subject, not by
        section — several remaining screens are 5–9 keys and don't justify a PR each.
    - [x] **Memory → Recall** `[v0.21.263]` — the golden section; 24 keys.
    - [x] **Memory → Conversation Context** `[v0.21.265]` — 20 keys across five config roots
          (`context`, `session.rolling_summary`, `trajectory_compression`, `compression`,
          `memory`); constraints + tiers declared, all 17 controls manifest-bound.
    - [x] **Where to start** `[v0.21.266]` — the two things every later slice needed, plus the
          Overview. Section blurbs: a `sections` store in `settings_help.yaml` keyed by screen
          id, all 21 authored, served by manifest **v3** (`sections` block: label, blurb,
          writable count) and rendered on the Overview tiles, each screen's header, and as the
          section intro in the generated reference — which is now laid out by screen in nav
          order. `ModelPickerField` joined the field kit's chrome contract (it sat outside it,
          so ~25 model-valued keys had no anchor and search could find them but not land on
          them); Recall's and Conversation Context's five pickers backfilled. Overview gained
          a search hero sharing the nav's query, a setup checklist read from the manifest, and
          tiles carrying blurb + changed-count. Fixed: every table in the generated reference
          rendered as literal pipe text; two humanizers named the same key differently.
    - [x] **Web Search + Research Mode** `[v0.21.267]` — 31 keys, written around the cost
          story: the per-turn budget window (call count *and* dollar ceiling, whichever runs
          out first; interactive turns only — delegated work isn't metered), Tavily credit
          arithmetic, and `source_policy`'s hard-floor/soft-preference asymmetry. Fixed a
          mis-section the refit caught — `search.research_per_turn_limit` claimed the Web
          Search screen but renders on Research Mode — and surfaced 6 keys that were writable
          over the API with no control anywhere (`brave_context_max_tokens_per_url` plus the
          whole deep-research tuning group, including the budget weight the budget help points
          at).
    - [x] **How the agent thinks** `[v0.21.268]` — 26 keys across Thinking Patterns, Task
          Planner and Agent Teams, written around what each pattern *costs*: chain-of-thought
          rides in the same call, step-back adds a pre-call before anything streams,
          reflection spends tokens on a draft you never read, and consensus multiplies by k.
          Two more mis-sections fixed — `planner.prompt_override` and
          `prompt_enhancement.system_prompt` are edited on **Feature Prompts**, so that screen
          moved out of the "owns no settings" list — plus a wrong `empty_means`:
          `step_back_model`/`sc_model` fall back to the conversation's own model, not a role.
          Surfaced 3 more control-less keys (`step_back_timeout_seconds`, `max_subtasks`,
          `non_blocking_delegations`). **102 of 208 documented.**
    - [x] **Consolidation** `[v0.21.269]` — the memory twin of Recall; 50 keys across the
          extraction pipeline (extraction → relevance filter → contradiction → correction →
          combined) and the background work (procedural distillation, reflex/salient cores,
          entity linking, promotion, four job intervals). Thresholds get their measured
          meaning: `fact_confidence_threshold` is explained against the calibration it filters
          (0.95 stated / 0.85 implied / 0.70 inferred / 0.50 hedged), `semantic_duplicate_threshold`
          against what merges at 0.92 vs 0.85. Prerequisite shipped with it: **memory keys can
          now declare a screen** — `_apply_spec` honours `KeySpec.ui_section` for both stores,
          which moved `extraction_system_prompt` and `relevance_filter_prompt` to the Feature
          Prompts screen that actually edits them. **152 of 208 documented.**
    - [ ] Ambassador (18) + Images & Audio (6) · Model Providers (12) + Model Limits (8) +
          Model Roles (3) + Prompt Enhancement (4) + Feature Prompts (1 left) + the 4 strays.
    - [ ] Sweep the ad-hoc Advanced/Experimental groupings onto the declared tiers.
    - [ ] **Invert the coverage gate** (the closing move) — `DOCUMENTED_SECTIONS` becomes an
          `UNDOCUMENTED_ALLOWLIST` that must be empty, so a new setting can't ship without
          help unless someone writes its key in and defends it in review.
  - [ ] **Dead-knob flags** — declared-but-unread keys. `scripts/check_config_keys.py` already has the
        inventory half; the registry makes its output authoritative enough to gate on.
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
- [ ] **The Regulator (enforcement arm — naming candidate)** — the parents' *enforcement* duties get
      a name: tune, quarantine (pull from rosters), or disable a misbehaving agent, riding the shared
      control-plane-change primitive within the autonomy envelope. Needs a first-class agent
      **quarantine/disable flag** — none exists today (`AgentProfile` has no active/disabled field;
      gating is only `available_for_delegation` + the tool lists). Detection stays with the Agency
      (ambassador + aides, read-only by construction) — see [cognitive-os.md](cognitive-os.md)
      Pillar 8 for the separation of powers.


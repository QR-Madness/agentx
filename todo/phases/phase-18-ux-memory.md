# Phase 18 — UX Improvements & Memory Tuning

> Part of the AgentX TODO — index: [Todo.md](../../Todo.md)

---

## Phase 18: UX Improvements & Optimization and Memory Tuning (In Progress, ~90%)

> Polish the client and tune the memory pipeline. Shipped waves moved to
> [roadmap.md](../../docs-site/src/content/docs/roadmap.md):
> **18.1** Wave 1 fixes (provider settings, mobile topbar) · **18.2** Toolkit (MCP server CRUD +
> tool browser, tags/groups/`allowed_agent_ids`, per-agent `allowed_tools`/`blocked_tools`) ·
> **18.3** Relay module (background-run inbox, "No Memorization" toggle) · **18.4** model metadata +
> `ModelPickerModal` (OpenRouter/Vercel capabilities + pricing) · **18.5** per-tab context bar +
> per-turn cost chip · **18.6** extraction tuning (entity resolution, `refines_fact_id` supersedure,
> scope context, `eval_consolidation` harness) · **18.8** Wave 2 fixes (KaTeX, table HTML, plan-step
> restore, editable cached servers, MCP auto-connect) · **18.9** memory tuning (`recall_user_history`,
> token-budget header, `checkpoint` tool + badge UI) · **18.10** plan/streaming reliability (token
> clamp, Plans drawer + step annotation, detached chat runs) · **18.11** client error contract +
> foundation cleanup (`ApiError`/toasts/`useApi`, Tailwind v4 + `ui/` primitives, god-component /
> `lib/api` / `ConversationContext` splits) · **18.11.x** cancel-CSRF + gate-page chrome fixes ·
> **18.12** Wave 3 entry-surface UX (Start recents, renamable conversations, selector redesigns,
> splash, README trim).

### 18.x Shipped — moved to [roadmap.md](../../docs-site/src/content/docs/roadmap.md)

> Dashboard redesign + usage metrics (`[v0.21.9]`/`[v0.21.10]`), extraction eval-harness + cleanup
> (18.6: procedural cases, snapshot/restore, persisted eval runs, `dedupe_entities`), working-memory
> follow-ups (`scratchpad_note`, `forget`/`remember_this`/provenance, cached recap), and the
> per-profile internal-tool gating UI (18.9.x). All shipped — see roadmap.

### 18.17 Context integrity (INV-CTX-1) — ✅ Complete `[v0.21.173]`

> Fix for total mid-conversation recall loss: history dropped at 70% of the window while the
> rolling summary was reactive (post-turn) — the first over-budget turn lost turns with zero
> coverage, and a cold rehydrate before any summary persisted lost them entirely.

- [x] Verbatim ceiling 0.7 → **0.9** (compress just before the limit), decoupled from the new
      post-turn pre-warm trigger `context.summary_trigger_ratio` (0.85).
- [x] **JIT pre-assembly summarization** (`views.py::_ensure_summary_coverage`): summary sized to
      exactly what `fit_history` drops this turn, registered the same turn; deterministic
      `history_digest` block fallback when the summarizer is unavailable.
- [x] Rehydration: `context.rehydrate_max_turns` wired (was dead); `history_overflow` flag when
      the row cap is hit uncovered.
- [x] `done` carries `context_summarized`/`context_dropped_turns`; composer **context chip**
      (hidden <50%, % ≥50%, warn + summarization hint ≥75%) replaced the header usage bar.

### 18.13 Layered Prompt Composer ("Prompt Stack") — ✅ Complete `[v0.21.47]`

> Replace the single-base-prompt model with a durable, block-based stack of editable
> prompt **layers**. Each built-in layer ships a `default` (sidecar, owned by the app,
> updated by releases) + an optional user `override`; effective = override ?? default,
> so untouched layers keep getting release improvements while edits are pinned and
> never silently overwritten. Locked decisions: **global stack only** (per-agent prompt
> stays in the profile editor, read-only in preview), **debounced autosave**, **no named
> presets in v1** (retire legacy prompt-profiles into the stack). Fixes the original
> durability bug — the old global/sections edits were in-memory only and lost on restart.

- [x] **1a — model + store** `[v0.21.43]`: `PromptLayer` (default/override/`default_version`/
      `base_version`; `effective`/`modified`/`update_available`); `BUILTIN_LAYERS` (global
      prompt split into versioned blocks); `LayerStore` durable via `ConfigManager`
      (`prompts.layers`, write-through) — override/reset/acknowledge/enable/reorder + custom
      CRUD + `compose()`. Tests: precedence, override persists, default-change → update/ack/
      reset, custom CRUD/reorder/compose.
- [x] **1b — wire + migrate** `[v0.21.44]`: `PromptManager.compose_prompt` now sources the
      global content from `LayerStore.compose()` (lights up all 3 live sites — `core.py`,
      streaming `views.py`, `alloy/executor.py` — unchanged). Folded the default "General"
      profile's sections (structured-thinking/concise-output/safety-constraints) into
      `BUILTIN_LAYERS` so the stack is the *complete* default prompt (byte-parity), and dropped
      the default-profile auto-attach (`is_default` guard) to avoid double-injection. One-time
      legacy migration (`_ensure_layers_migrated`, guarded by `prompts.layers_migrated`) imports
      any customized legacy global into a reserved `legacy-global` custom layer. `/prompts/global*`
      kept as back-compat shims over the store (`set_singleton_override` upsert). Scope: governs
      only the conversational prompt — `SystemPromptLoader` feature prompts untouched. Tests:
      parity, no-duplication, override→live, shim, singleton idempotency.
- [x] **2 — layer API** `[v0.21.45]`: `GET/POST /api/prompts/layers` (list `{layers, composed}`
      / create custom), `PATCH/DELETE /api/prompts/layers/{id}` (content→override, title,
      enabled / delete custom), `POST /{id}/reset`, `POST /{id}/acknowledge`, `POST
      /layers/reorder`. Typed client (`promptsApi.{list,create,update,delete,reset,acknowledge,
      reorder}PromptLayer(s)` + `PromptLayer` type). Docs: CLAUDE.md table, OpenApi (`PromptLayer`
      schema + paths, spec lints clean), endpoints.md. Tests: `PromptLayerApiTest` (list/create/
      patch/delete/reset/reorder/404 via RequestFactory). Compose-preview enrichment (dynamic
      injections + active agent) deferred to the editor (Phase 3).
- [x] **3 — block-stack editor UI** `[v0.21.46]` (Settings → Intelligence → **System Prompt**):
      side-by-side two-pane composer — draggable layer cards (`@dnd-kit`, collapse via
      framer-motion, Built-in/Custom badge, enable Switch, ● edited / ▲ update dots), inline
      `Textarea` edit w/ 600ms debounced autosave + ~token count, **reset-to-default**, **diff
      modal** (`diff` lib; update: Keep[ack] / Adopt[reset] / Load-default merge-assist; edited:
      Reset), live composed preview (client `composeStack` mirrors backend, unit-tested) + custom
      layer add/delete. Narrow widths collapse to one column + Preview dialog. New: `SystemPromptSection`
      + `prompt-stack/{LayerCard,ComposedPreview,LayerDiffModal}` + `PromptStack.css` +
      `lib/promptStack.ts`(+test). Libs added: `@dnd-kit/*`, `diff`.
- [x] **4 — snippet library** `[v0.21.47]`: reframed `template_manager` as the **Prompt Library** —
      snippets **insert-as-layer** (reuse `PromptLibraryModal` `mode='insert'` → `createPromptLayer`)
      and the enhancer rewrites a layer **in place** (`Enhance` button → `/api/prompts/enhance`,
      one-click undo). Extended `onInsert(content, name?)` to seed the layer title. Dropped the
      misleading "replaces PromptProfile/PromptSection" docstring (it's a building-block library,
      not a replacement — composition is owned by `LayerStore`).

### 18.14 Prompt System Round 2 + Ambassador-as-profile-kind — ✅ Complete `[v0.21.48–54]`
- [x] **Dialog z-index fix** `[v0.21.48]`: `ui/Dialog` raised above the legacy app modals
      (z-1000/1001) → "Insert from library"/diff/preview popups no longer render behind Settings.
- [x] **Per-agent prompt editor** `[v0.21.49]`: reusable `common/PromptEditor` (tokens + in-place
      Enhance/undo + library-insert-replaces) + `common/EffectivePromptPreview` (name → agent prompt
      → global stack, true backend order).
- [x] **Ambassador is its own profile `kind`** `[v0.21.50 backend / .51 client]`: `AgentProfile.kind`
      (`agent`|`ambassador`) + `is_default_ambassador`; separate defaults (`get/set_default_ambassador`,
      agent-default never an ambassador); engine-level chat exclusion (default-agent/routing/delegation);
      `AmbassadorConfig` persona overrides (briefing/qa/draft) over code defaults + `system_prompt` =
      Communications voice; migration seeds a default ambassador, never converts the default agent.
      Endpoint `set-default-ambassador` + `ambassador/persona-defaults`. Editor adapts by kind
      (`OverridablePromptField` default/override/reset/diff); Settings → Ambassador slimmed to a
      default-ambassador picker + New/Edit.
- [x] **Diff coloring everywhere + memory-prompt diff** `[v0.21.52]`: moved `.layer-diff` CSS into
      `LayerDiffModal.css` on `--feedback-success/-error` tokens; memory extraction/relevance
      `PromptField` gains a Diff button (default vs override).
- [x] **More avatar icons** `[v0.21.53]`: curated set 11 → ~57 (static/tree-shaken).
- [x] **Autosave** `[v0.21.54]`: agent profiles + memory recall/consolidation settings autosave
      (debounced, baseline-diff = no save-loops, hydration keyed on `profile.id` = preserves cursor).

### 18.15 Prompt/Ambassador follow-ups (observed while in the code) — open
- [ ] **Finish moving ambassador settings onto the profile**: `ambassador.model` +
      `max_context_turns` still live in global `config.ambassador.*` (split-brain). Move per-ambassador
      knobs onto the ambassador profile (the user's "ambassadors will have lots of settings" intent).
- [ ] **Retire the vestigial PromptProfile/PromptSection path**: the "Select Base Template"
      (`prompt_profile_id`) control + legacy `/prompts/profiles*`/`/sections*` are now confusing
      alongside the layer stack + library-insert. Plan a deprecation/removal once nothing depends on it.
- [ ] **Client tests for the new logic**: autosave baseline-diff, `EffectivePromptPreview` compose
      order, `OverridablePromptField` reset/diff — only `promptStack` has unit coverage today.
- [x] ~~Theme leak straggler (`.profile-avatar-option.selected` → `--cosmic-violet`)~~ — **not a leak.**
      `--cosmic-violet` is a theme-adaptive *alias* (defined per theme = the accent), so it follows
      themes correctly. Optional cosmetic cleanup: rename the legacy alias to `--accent-primary`
      app-wide (15+ files) — zero visual change, low priority.
- [x] **Prompt placeholders** `[v0.21.55]`: whitelist `{agent_name}`/`{date}`/`{time}` substituted
      at compose time (`prompts/placeholders.py`, applied in `compose_system_prompt` + the ambassador
      persona builders — so an override's `{agent_name}` now resolves). Client: `PromptEditor` "Insert
      placeholder" menu (`lib/promptPlaceholders.ts`) + preview highlighting (`HighlightedPrompt`).
      Tests: `PromptPlaceholderTest`. Also raised `DropdownMenu` z-index above the legacy modals
      (same class of bug as the Dialog fix). **Follow-up:** add the insert-placeholder affordance to
      the global `LayerCard` editor too (substitution + highlight already work there).
- [ ] **`SKILL.md` standardization (parked, user-flagged)**: adopt the standard skills-file pattern;
      let the prompt/layer machinery host composable skills; guard so profile-typing/layering never
      gates a user's skill access. Its own initiative.

### 18.16 Agent-profile control-center + icon picker — ✅ Complete `[v0.21.56]`
- [x] **Redesigned editor**: hero identity header (AvatarPicker tile + `agentAccent` aura, inline
      name, `CopyChip` agent-id, kind/default badges, tags, description) over a `ControlCard` grid
      per tab (Model / Generation / System Prompt-full / Delegation|Ambassador / Tools / Memory),
      each with an at-a-glance header summary. All field logic + `useProfileEditorState` autosave
      preserved (presentation-only re-house).
- [x] **Consolidated icon picker** (`common/AvatarPicker`): searchable, categorized modal (recents
      + live preview + stagger), with a disabled **Generate** seam for AI icons next. Catalog grown
      to ~95 icons with `category`/`keywords` (`lib/avatars.ts`).
- [x] **New reusable primitives**: `ui/SegmentedControl` (reasoning + picker tabs; sliding motion,
      a11y radiogroup), `ui/CopyChip`, `common/ControlCard`, `lib/agentAccent.ts` (deterministic
      per-agent color — foundation for chat/Alloy identity later). Gradient temperature slider +
      dynamic label. Tests: `agentAccent`, `avatars`. Reduced-motion respected.
- [ ] **Follow-up**: generated icons (the Generate tab); a light `ProfileNav` polish (apply the
      agent accent to the active item); retire now-dead `.profile-section-card`/accordion CSS.

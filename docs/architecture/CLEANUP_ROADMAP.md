# Backend Cleanup Roadmap

Deferred, higher-risk structural goals for the advanced backend (agent memory,
extraction/consolidation, providers, MCP, reasoning, streaming). These are
**planned but not yet executed** — they change behavior, signatures, or
architecture and should be taken on deliberately, not as part of a mechanical
cleanup pass.

A first, behavior-preserving cleanup pass has already landed: provider/Redis
deduplication, channel-filter and entity-embedding helpers, config extraction
(recall confidence, connection pooling, LM Studio timeout), exception/logging
hygiene, ruff `7 → 0`, and several genuine bug fixes (see "Already done"). This
document covers what was intentionally left out.

## How to use this doc

Each item lists: the problem, a verified location, the proposed shape, the risk,
and rough effort. Items are independent unless noted. Nothing here is required
for current functionality — these reduce the cost of building the next layer of
features and burn down the residual type-checker debt.

---

## 1. Decompose god functions — ✅ consolidation done (Pass 5)

**Problem.** A few functions mix many responsibilities, which hurts testability
and makes the control flow hard to follow.

**Done (Pass 5).** `consolidate_episodic_to_semantic()` (was **614 lines**,
`kit/agent_memory/consolidation/jobs.py`) is now a ~125-line thin coordinator
over per-stage helpers: `_fetch_pending_conversations`,
`_extract_from_conversation` (→ `_ConvExtraction`), `_store_conversation_entities`,
`_store_facts_with_verification` (→ `_FactStoreResult`),
`_link_facts_and_relationships`, plus `_consolidate_user_conversation` /
`_consolidate_assistant_conversation` for the two phases. Helpers share the open
Neo4j session and mutate `metrics`/`errors` in place (the existing
`_resolve_and_prepare_entities` convention) — a pure relocation. Backfilled
`ConsolidationPipelineTest` (5 behavior-pinning tests written **before** the
refactor) proves parity; the brittle source-grep test was repointed at the
helper that now owns the defensive try/except. pyright held at 4, ruff 0.

**Still deferred.**
- `streaming_tool_loop()` — `streaming/tool_loop.py`, ~250 lines. Extract chunk
  processing, tool execution, and compression steps.
- `RecallLayer.recall()` — `kit/agent_memory/memory/recall.py`. Already
  decomposed in an earlier pass (orchestrates 5 techniques via helpers).

**Risk:** medium — pure refactor. **Effort:** ~0.5 day for the remaining
`streaming_tool_loop()`.

---

## 2. Tighten loose typing (`dict[str, Any]` / `Any`)

**Problem.** Facts, entities, traces, and tool I/O flow through the system as
untyped dicts. This is the single largest source of pyright findings —
`reportAttributeAccessIssue` + `reportArgumentType` dominate the remaining
count.

**Shape.** Introduce `TypedDict`/dataclass models for the fact, entity, trace,
and tool-event shapes (the pydantic `Fact`/`Entity` models in
`kit/agent_memory/models.py` are a starting point; the dict forms returned by
Neo4j queries are the divergent ones). Thread these through `RecallLayer`,
`SemanticMemory`, and the consolidation pipeline.

**Risk:** medium — touches many call sites; do it module-by-module.
**Effort:** ~2 days. Tracked against the pyright baseline (see "Type-check
baseline" below) — this item is what drives the count toward zero.

---

## 3. Lazy-load-returns-`None` properties

**Problem.** `Agent.memory`, `Agent.mcp_client`, `Agent.drafting`
(`agent/core.py`) lazily return `Optional[...]`, conflating "disabled" with
"failed to initialize" and forcing `if self.memory:` guards everywhere. These
Optionals are the direct source of most `reportOptionalMemberAccess` findings.

**Shape.** Distinguish disabled vs. failed (e.g. an explicit sentinel or a
`MemoryUnavailable` state), and centralize access so callers don't each
re-check. Pairs naturally with item 5.

**Risk:** medium. **Effort:** ~0.5–1 day.

---

## 4. Dependency injection for singletons

**Problem.** `providers/registry.py` (`get_registry`) and `config.py`
(`get_config_manager`) are process-global singletons. Tests must patch globals,
and isolation between tests is fragile.

**Shape.** Accept an optional instance in constructors / call sites, defaulting
to the global. No behavior change in production; large testability win.

**Risk:** low–medium. **Effort:** ~0.5 day.

---

## 5. Memory decoupling — ✅ done (Pass 4)

**Problem.** Memory calls were scattered through `Agent.run()`/`Agent.chat()`
(`agent/core.py`), each wrapped in its own try/except. Goal operations in
`kit/agent_memory/memory/interface.py` bypassed the sub-modules and hit Neo4j
directly, so they couldn't be mocked independently.

**Done (Pass 4).** New `agent/hooks.py` (`TaskOutcome`, `AgentHooks` base,
`MemoryRecorder`): the agent now emits `on_task_complete`/`on_task_error`/
`on_turn`/`on_tool_use`/`on_goal_complete` via a single `Agent._dispatch` (which
isolates subscriber failures) and `MemoryRecorder` subscribes — the ~7 scattered
write blocks moved out of `core.py`/`plan_executor.py`. `remember()` stays a
direct call (it's a retrieval, not a lifecycle event). New
`kit/agent_memory/memory/goal.py` (`GoalMemory`) holds the goal Cypher mirroring
`SemanticMemory`; the facade delegates (embedding stays facade-side). Goal-test
gaps backfilled; `GoalAccessControlTest` still green (proves the security Cypher
survived the move). Behavior-parity refactor — no pyright movement (4), ruff 0.

---

## 6. Provider resource lifecycle — ✅ provider half done (Pass 3)

**Problem.** OpenAI/Anthropic providers created `AsyncOpenAI`/`AsyncAnthropic`
clients that were never closed; OpenRouter/Vercel/LM Studio already close per
request. `ProviderRegistry.reload()` evicted cached providers without closing
them → leaked pools on every config change.

**Done (Pass 3).** Added default `async close()` to `ModelProvider`; implemented
it on OpenAI/Anthropic (close + reset the cached client); added
`ProviderRegistry.aclose()` and made `reload()` close evicted providers (bridged
via `utils/async_bridge.run_coro_sync`).

**Still deferred.** Neo4j retry/backoff and a unified `health_check()` across the
three stores (`kit/agent_memory/connections.py`) — the memory-store half of this
item. **Effort:** ~0.5 day.

---

## 7. Async-to-sync bridge

**Problem.** `streaming/trajectory_compression.py:136` and the newly added
`kit/agent_memory/memory/recall.py` `_run_coro_sync` both wrap `asyncio.run` in
a `ThreadPoolExecutor` to call async provider methods from sync contexts. This
works but is duplicated and fragile (nested loops, thread overhead).

**Shape.** Factor a single, well-tested `run_coro_sync(coro, timeout)` helper
(e.g. `agentx_ai/utils/async_bridge.py`) and have all sync→async call sites use
it; longer term, make the sync recall/compression paths async-native.

**Risk:** medium. **Effort:** ~0.5 day.

---

## 8. LM Studio timeout sentinel — ✅ done (Pass 3)

**Problem.** `providers/lmstudio_provider.py` treated a configured timeout equal
to `ProviderConfig`'s default (60.0) as "unset" and bumped it to 300.0, so an
*explicit* 60.0 was indistinguishable from unset.

**Done (Pass 3).** `ProviderConfig.timeout` is now `Optional[float] = None`.
LM Studio bumps only `None` → 300.0; cloud providers resolve `None` → 60.0 at the
client-build site; the YAML loader passes `None` when unset. An explicit value
(including 60.0) is honored everywhere.

---

## 9. Custom exception hierarchy + Result types — partial (Pass 3 scoped slice)

**Problem.** Broad `except Exception` blocks remain across consolidation and the
agent core. `mcp/tool_executor.py` `discover_tools()` and `mcp/client.py`
resource discovery returned `[]` on failure, indistinguishable from "no
tools/resources available".

**Done (Pass 3, scoped).** Discovery failures are now observable without changing
return types: `ToolExecutor.get_discovery_error()` /
`MCPClientManager.get_resource_discovery_error()` record the last failure (cleared
on a successful discovery), logged at `error`/`warning`, and surfaced on
`/api/mcp/servers` as `tool_discovery_error` / `resource_discovery_error` so the
client can tell "failed" from "0".

**Still deferred.** The `AgentError`/`MemoryError`/`ToolExecutionError` hierarchy
and tightening the broad excepts across `agent/core.py` (many are *intentionally*
defensive best-effort memory swallows) / consolidation. Judgment-heavy and
sprawling — its own pass. **Effort:** ~1 day.

---

## 10. Streaming constants → ConfigManager (optional)

**Problem.** `streaming/constants.py` and `_MAX_RESULT_CHARS_IN_PROMPT` in
`streaming/trajectory_compression.py` are module constants. They are a clean
single source of truth today (not duplication), and `constants.py` notes they
"can be moved to ConfigManager for hot-reloading if needed."

**Shape.** If runtime tuning becomes desirable, move them into the
`trajectory_compression` / a new `streaming` ConfigManager section, defaulting
to today's values. `rounds_to_text()` would need `config` threaded in.

**Risk:** low. **Effort:** ~0.5 day. **Priority:** only if hot-reload is wanted.

---

## 11. LM Studio message conversion divergence — ✅ done (Pass 3)

**Problem.** `providers/lmstudio_provider._convert_messages` omitted the
`tool_calls` passthrough on assistant messages that the shared
`convert_messages_to_openai_format` (used by OpenAI/OpenRouter/Vercel) includes,
dropping prior tool calls in multi-turn tool conversations against LM Studio.

**Done (Pass 3).** Confirmed it was a latent bug (not an LM Studio quirk) and
switched `_convert_messages` to delegate to the shared helper. **Still needs
verification against a live LM Studio instance** (the change alters request
payloads); unit-covered for the `tool_calls` passthrough in the meantime.

---

## 12. Multi-user auth

**Problem.** `views.py` hardcodes `DEFAULT_USER_ID = "default"` (the inline TODO
notes "Replace with actual auth when multi-user is implemented"). The memory
subsystem already scopes by `user_id`, so the data layer is mostly ready.

**Shape.** Wire real authentication and propagate the authenticated user id
through the request → agent → memory path. Large, product-level change.

**Risk:** high (cross-cutting). **Effort:** multi-day, own initiative.

---

## Type-check baseline (guardrail)

Two passes reduced pyright from **169 → 4** errors (Passes 3–5 held at 4). A baseline guardrail
(`scripts/check_pyright_baseline.py`, run via `task check:types:python:baseline`)
fails CI if the count *rises* above `api/.pyright-baseline` (currently **4**), so
the debt can only shrink. When an item above lands, lower the baseline to lock
in the gain.

The remaining **4** are documented false positives (left in the baseline):
`apps.py` Django `default_auto_field` override idiom, and pydantic
`AgentProfile(...)` constructions (`agent/profiles.py`, `tests.py`) where pyright
does not recognize `Field(None, ...)` positional defaults.

---

## Already done

### Pass 1 — behavior-preserving cleanup
- **Bug:** `recall.py` provider imports used the wrong relative depth *and*
  called the async `complete()` without awaiting — HyDE and self-query were
  silently broken since inception. Fixed + regression test.
- Provider dedup: shared `convert_messages_to_openai_format` /
  `parse_openai_tool_calls` in `providers/base.py`.
- `ModelProvider.health_check` made `async` to match all overrides + callers.
- Possibly-unbound variables fixed (`agent/core.py`, `drafting/speculative.py`,
  `views.py`).
- Redis blob storage deduped into `agent/redis_blob_storage.py`.
- Channel-filter Cypher + `Entity.embedding_text()` helpers.
- Config extraction: `recall_min_confidence`, Neo4j/Postgres pool settings,
  LM Studio timeout default reference.
- Exception/logging hygiene in `connections.py`, `providers/base.py`.
- ruff `7 → 0`.

### Pass 2 — type-debt burndown (this pass; pyright 93 → 4)
The dominant "type debt" turned out to be the **same un-awaited-async bug class**
as recall/job_run, plus false positives — not loose `dict` typing.
- **Bug:** the whole reasoning subsystem (CoT/ToT/ReAct/Reflection +
  orchestrator) was sync and called the async `provider.complete()` without
  awaiting, so `Agent.run()` reasoning **silently always returned `status=FAILED`**
  (swallowed by the orchestrator's broad `except`). Made reasoning **async-native**
  (item ~2/3/7) and bridged at the one sync caller; added AsyncMock regression
  tests. `ReasoningStatus.FAILED` enum fixed.
- **Bug:** `extraction/service.py` `extract_entities/facts/relationships` had the
  same un-awaited `extract_all` — bridged.
- **Bug:** `consolidation/jobs.py` used `memory._semantic` (no such attr →
  `AttributeError`); corrected to `.semantic`.
- **Bug:** `mcp/internal_tools.py` imported `get_agent_memory` from the wrong
  module (always failed → "Memory system unavailable"); fixed.
- **Bug:** `semantic.py` supersede called `audit_logger.log_operation` (no such
  method); corrected to `log_write`.
- **Item 7 (async bridge):** consolidated the duplicated sync→async bridges into
  `agentx_ai/utils/async_bridge.run_coro_sync` (used by reasoning bridge,
  recall, trajectory compression, extraction wrappers, planner bridge).
- **Item 3 (lazy-None):** narrowed `Optional` access in `agent/core.py`,
  `plan_executor.py`, `streaming/tool_loop.py`, `views.py`, `mcp/client.py`.
- False positives: Django `self.style` (per-file pragma on 3 management
  commands), `ExtractionService._settings` test seam (now a real override hook —
  the test mocks were previously ignored), redis `ResponseT` casts.

### Pass 3 — robustness / correctness (provider + MCP edge)
Continued the bug-hunting trend on the provider/MCP boundary; no pyright movement
(the 4 are documented false positives), ruff stays 0.
- **Bug (item 11):** LM Studio `_convert_messages` dropped assistant `tool_calls`,
  breaking multi-turn tool use — switched to the shared
  `convert_messages_to_openai_format`. *Live LM Studio verification still pending.*
- **Item 6 (provider half):** added `ModelProvider.close()` (default no-op),
  real impls on OpenAI/Anthropic (close + reset cached client),
  `ProviderRegistry.aclose()`, and close-on-`reload()` (bridged via
  `run_coro_sync`) so config reloads stop leaking connection pools.
- **Item 8:** `ProviderConfig.timeout` → `Optional[float] = None` sentinel; each
  provider resolves its own default. An explicit `60.0` is no longer mistaken for
  "unset" against LM Studio.
- **Item 9 (scoped slice):** tool/resource discovery failures are now recorded
  (`get_discovery_error` / `get_resource_discovery_error`) and surfaced on
  `/api/mcp/servers`, distinguishing "failed" from "0 tools/resources".
- Added `ProviderRobustnessTest` + `ToolDiscoveryErrorTest` (9 tests).

### Pass 4 — memory decoupling (roadmap item 5)
Structural step toward multi-agent: the agent no longer calls memory inline.
Behavior-parity refactor; pyright stays 4, ruff 0.
- **AgentHooks seam:** `agent/hooks.py` (`TaskOutcome`, `AgentHooks`,
  `MemoryRecorder`). `Agent._dispatch` fires lifecycle events to subscribers and
  isolates a broken subscriber; `MemoryRecorder` owns the per-op best-effort
  guards. Removed the duplicated `reflect`/`complete_goal`/`store_turn`/
  tool-usage blocks from `core.py` and `plan_executor.py`.
- **GoalMemory sub-module:** `kit/agent_memory/memory/goal.py`; the four goal
  methods on `AgentMemory` are now thin delegators (embedding stays facade-side).
- Tests: `MemoryRecorderTest`, `AgentHookDispatchTest`, `GoalMemoryTest` (+ the
  existing `GoalAccessControlTest` still green).

### Pass 5 — consolidation god-function decomposition (roadmap item 1)
Largest remaining readability debt; behavior-parity refactor (pyright 4, ruff 0).
- **Test-first:** new `ConsolidationPipelineTest` (5 tests) drives
  `consolidate_episodic_to_semantic` end-to-end against mocked Neo4j/extraction/
  memory and pins the observable contract (storage calls, `consolidated` /
  `self_consolidated` flags, metrics, return dict). Written + green **before** the
  refactor so the relocation is provably behavior-preserving.
- **Decomposition:** the 614-line function → ~125-line coordinator over
  `_fetch_pending_conversations`, `_extract_from_conversation`,
  `_store_conversation_entities`, `_store_facts_with_verification`,
  `_link_facts_and_relationships`, `_consolidate_user_conversation`,
  `_consolidate_assistant_conversation` (+ `_ConvExtraction` / `_FactStoreResult`
  dataclasses). Helpers share the session and mutate `metrics`/`errors` in place.
- Repointed the brittle `inspect.getsource` test at the helper that now owns the
  per-fact try/except.

Still deferred: item 1's remaining `streaming_tool_loop()` split, item 4 (DI),
item 6 (memory-store retry/health half), item 9 (exception hierarchy +
broad-except tightening), item 10, item 12.

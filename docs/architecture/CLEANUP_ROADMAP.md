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

## 1. Decompose god functions

**Problem.** A few functions mix many responsibilities, which hurts testability
and makes the control flow hard to follow.

- `consolidate_episodic_to_semantic()` — **~614 lines**, `kit/agent_memory/
  consolidation/jobs.py:758`. Does extraction, validation, contradiction
  checking, entity resolution, fact/relationship linking, storage, and metrics
  in one body.
  - **Shape:** split into `_extract_from_conversation` → `_validate_and_verify`
    → `_resolve_and_store_entities` → `_link_facts_and_relationships`, with the
    top-level function as a thin coordinator. The private helpers
    (`_handle_contradiction`, `_resolve_and_prepare_entities`, etc.) already
    exist and show the seams.
- `streaming_tool_loop()` — `streaming/tool_loop.py`, ~250 lines. Extract chunk
  processing, tool execution, and compression steps.
- `RecallLayer.recall()` — `kit/agent_memory/memory/recall.py`. Orchestrates 5
  techniques inline; extract a `_run_all_techniques` helper.

**Risk:** medium — pure refactor, but consolidation is central and under-tested
at the integration level. Land behind the existing `tests_memory` suite plus new
unit tests for each extracted helper.
**Effort:** ~1–1.5 days.

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

## 5. Memory decoupling

**Problem.** Memory calls are scattered through `Agent.run()`/`Agent.chat()`
(`agent/core.py`), each wrapped in its own try/except. Goal operations in
`kit/agent_memory/memory/interface.py` bypass the sub-modules and hit Neo4j
directly, so they cannot be mocked independently.

**Shape.** Introduce an `AgentHooks`/`MemoryRecorder` seam (on_task_start /
on_task_complete / on_error) so the agent emits events and memory subscribes.
Extract a `GoalMemory` sub-module mirroring `SemanticMemory`/`EpisodicMemory`.

**Risk:** medium. **Effort:** ~1–1.5 days.

---

## 6. Provider resource lifecycle

**Problem.** OpenAI, Anthropic, and Vercel providers create HTTP/SDK clients
(`AsyncOpenAI`/`AsyncAnthropic`) that are never closed; OpenRouter and LM Studio
already create-and-close per request. Long-running processes accumulate
connection pools.

**Shape.** Add an `async close()` to `ModelProvider`, implement it in each
provider, and give the registry ownership of the lifecycle (close on shutdown).
While here: add Neo4j retry/backoff and a unified `health_check()` across the
three stores (`kit/agent_memory/connections.py`).

**Risk:** medium — touches connection management. **Effort:** ~1 day.

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

## 8. LM Studio timeout sentinel

**Problem.** `providers/lmstudio_provider.py` treats a configured timeout equal
to `ProviderConfig`'s default (60.0) as "unset" and bumps it to 300.0. An
*explicit* 60.0 is therefore indistinguishable from unset. (The cleanup pass
made this reference the dataclass default instead of a magic literal, but the
ambiguity remains.)

**Shape.** Make `ProviderConfig.timeout` an `Optional[float] = None` sentinel so
"unset" is explicit, and update all providers. Cross-provider change.

**Risk:** low–medium (touches every provider). **Effort:** ~0.5 day.

---

## 9. Custom exception hierarchy + Result types

**Problem.** Broad `except Exception` blocks remain across consolidation and the
agent core. `mcp/tool_executor.py` `discover_tools()` returns `[]` on failure,
indistinguishable from "no tools available".

**Shape.** Introduce `AgentError` / `MemoryError` / `ToolExecutionError`, catch
specifically, and give tool discovery a Result type (or raise) so callers can
tell "empty" from "failed".

**Risk:** medium — changes error propagation. **Effort:** ~1 day.

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

## 11. LM Studio message conversion divergence

**Problem.** `providers/lmstudio_provider._convert_messages` diverges from the
shared `convert_messages_to_openai_format` (now used by OpenAI/OpenRouter/
Vercel): it omits the `tool_calls` passthrough on assistant messages. This may
drop prior tool calls in multi-turn tool conversations against LM Studio.

**Shape.** Confirm whether the omission is intentional (LM Studio quirk) or a
latent bug; if a bug, switch LM Studio to the shared helper. Not done in the
cleanup pass because it would change LM Studio request payloads.

**Risk:** low, but behavior-affecting for LM Studio tool use. **Effort:** ~2h
incl. verification against a live LM Studio instance.

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

Two passes reduced pyright from **169 → 4** errors. A baseline guardrail
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

Still deferred: **item 1 (god-function decomposition)**, item 4 (DI), item 5
(memory decoupling), item 6 (provider lifecycle), item 8 (timeout sentinel),
item 9 (exception hierarchy), items 10–12.

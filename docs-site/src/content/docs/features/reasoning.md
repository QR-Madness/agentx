# Thinking Patterns & Reasoning

AgentX has two reasoning systems, and it's worth knowing which one you're using:

- **Thinking Patterns** — how agents reason **in chat**. A pattern is compiled into the
  live streamed turn (directives, hidden pre-calls, multi-pass streams) so tools,
  steering, and the thinking bubble all keep working. This is the system you interact
  with every day.
- **The offline reasoning kit** — the classic CoT / ToT / ReAct / Reflection strategy
  classes behind `POST /api/agent/run` (blocking, non-streaming). Kept for programmatic
  task execution.

Models with **native reasoning** (they emit their own `<think>` deltas) stream that
thinking live into the chat's thinking bubble regardless of pattern — patterns shape
*how* that thinking happens, and give non-reasoning models a visible thinking process
of their own.

## The patterns

| Pattern | What it does | Extra LM calls |
|---------|--------------|----------------|
| **Auto** | Picks per message: instant keyword heuristics; an optional bounded LLM tiebreak when unsure. Never stacks a scaffold on a native reasoner. | 0 (rarely 1 tiny) |
| **Native** | The model thinks freely — no scaffold, raised output floor so thinking can't starve the answer. | 0 |
| **Step-by-step** (`cot`) | Explicit numbered reasoning steps inside the thinking block — gives non-reasoning models a verifiable chain. | 0 |
| **Step-back** (`step_back`) | A hidden pre-call distills the governing principles first; the turn then applies them explicitly. | 1 small |
| **Reflection** (`reflection`) | One completion structured as draft → self-critique → improved final answer (critique stays in thinking). | 0 |
| **Deep reflection** (`deep_reflection`) | True multi-pass: the draft and critique stream **live into the thinking bubble**, then the improved final answer streams with tools. | 2 |
| **Consensus** (`self_consistency`) | k independent samples (2–5, tool-less) surface as thinking; a judged final answer streams. Auto only picks it for short calculation/logic turns with no tools. | k (+judge rides the final turn) |

Legacy values keep working with honest degradations in chat: **Tree-of-Thought** runs
step-by-step (full ToT stays on `/agent/run`), and **ReAct** maps to native thinking
with tool narration — chat's tool loop already *is* reason+act with real function
calls. A one-line status tells you when a degradation applied.

## Choosing a pattern

Resolution per turn: **Thinking mode** (a per-conversation override in the [Relay](chat.md)) →
**agent profile** (`reasoning_strategy`, the
"Thinking pattern" control in the profile editor) → **global default**
(`preferences.default_reasoning_strategy`) → **Auto**.

The **Thinking mode** picker unifies the patterns with **Research Mode** as one choice —
they're mutually exclusive by design (a research turn keeps its own rigorous prompt and
never stacks a pattern), so picking Research in the same menu makes that rule visible
instead of letting a chosen pattern silently no-op. Options the settings disable drop out
of the menu. The turn's mode is reported on the done event (`thinking_pattern` +
`research`), stamped on the persisted turn, and badged on the message. Multi-pass phases
surface as live status lines ("Drafting…", "Critiquing the draft…", "Sampling 3
independent solutions…").

## Settings → Intelligence → Thinking Patterns

- **Patterns** — master kill-switch + per-pattern availability (both explicit selection
  and Auto respect these).
- **Auto selection** — the LLM tiebreak toggle, its model (empty = the Fast Utility
  role), and the minimum message length below which it never fires.
- **Pattern models & budgets** — step-back and consensus-sampling models (empty = the
  conversation's own model), consensus `k`, and the thinking output floor
  (`0` = automatic: floored whenever a pattern is active or the model reasons natively,
  so thinking tokens can't starve the visible answer).

Config lives under `reasoning.*` (`/api/config/update`, allowlisted). The classifier is
a Fast Utility role member; `step_back_model`/`sc_model` inherit the active turn model.

## The offline kit (`/api/agent/run`)

`ReasoningOrchestrator` classifies the task (shared heuristics with chat Auto —
`reasoning/selection.py`) and runs the full strategy classes: Chain-of-Thought
(zero-shot/few-shot), Tree-of-Thought (BFS/DFS/beam over branching thoughts), ReAct
(reason+act with registered tools), Reflection (iterative critique/revision cycles,
prompts in `system_prompts.yaml`). Strategy execution is wall-clock bounded
(`OrchestratorConfig.timeout_seconds`), falls back to Chain-of-Thought on failure, and
resolves models through the provider fallback chain like every other feature.

```
POST /api/agent/run
{"task": "…", "reasoning_strategy": "tot"}
```

Chat-first pattern values (`native`, `step_back`, `deep_reflection`,
`self_consistency`) alias to their nearest kit strategy on this endpoint.

## Advanced: multi-model drafting

Thinking patterns shape how *one* model reasons. **Drafting** goes wider — orchestrating
**several models** on a single generation to trade speed, quality, and cost. It's an
advanced, opt-in layer: **off by default** (`enable_drafting = false`), with strategies
defined in `drafting/drafting_strategies.yaml` and picked per task type. Three families:

### Speculative decoding

A fast **draft** model generates tokens that a stronger **target** model verifies,
accepting or rejecting each batch — cheaper tokens whenever the two agree. See the
[speculative-decoding flow](../architecture/system-design.md#speculative-decoding) on the
System Design page.

| Config | Description |
|--------|-------------|
| `draft_model` | The fast proposer |
| `target_model` | The strong verifier |
| `draft_tokens` | Tokens per draft batch (20–30) |
| `acceptance_threshold` | Minimum score to accept (0.7–0.8) |
| `max_iterations` | Maximum draft-verify cycles |

Pre-configured: `fast_accurate`, `local_cloud`, `claude_fast`.

### Pipeline

Multi-stage generation where each stage uses a different model, prompt, and temperature for
a specific role — analyze, critique/review, refine, summarize. Pre-configured: `code_review`
(generate → review → refine), `writing_pipeline` (outline → draft → edit → polish),
`analysis_pipeline` (decompose → research → synthesize).

### Candidate generation

Generate several candidates and pick the best with a scoring method — `majority_vote` (most
common answer wins), `verifier` (a separate model scores each), or `length_preference`.
Pre-configured: `consensus` (multi-model vote), `best_of_n` (N candidates + verifier),
`diverse_ensemble` (varied models), `self_consistency` (same model, multiple samples).

A run returns a **`DraftResult`**:

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Final output |
| `strategy` | string | Strategy name |
| `status` | DraftStatus | `"complete"` or `"failed"` |
| `draft_tokens` / `accepted_tokens` | int | Tokens drafted / accepted (speculative) |
| `models_used` | list[string] | Every model involved |
| `stages_completed` / `candidates_generated` | int | Pipeline stages · candidates produced |
| `estimated_cost` | float | Estimated USD cost |
| `total_time_ms` | float | Elapsed time |

Task types map to a default strategy via the `defaults` block — `general → fast_accurate`,
`code → code_review`, `writing → writing_pipeline`, `analysis → analysis_pipeline`,
`consensus → consensus`. Full config lives in
`api/agentx_ai/drafting/drafting_strategies.yaml`; the result schema is in
[API Models](../api/models.md).

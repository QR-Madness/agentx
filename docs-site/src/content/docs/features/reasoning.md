# Thinking Patterns

A **thinking pattern** is how an agent works through a problem before it answers: thinking
freely, reasoning step by step, starting from principles, or drafting an answer and then
critiquing it. You can let AgentX choose one for each message, or pick one for an agent or a
single conversation. Whichever pattern runs, you can watch the agent think in the chat's thinking
bubble as it works.

Models with built-in reasoning stream their own thinking into the bubble whatever the pattern. A
pattern shapes how that thinking goes, and gives models without built-in reasoning a visible
thinking process of their own.

## The patterns

| Pattern | What it does | Extra model calls |
|---------|--------------|-------------------|
| **Auto** | Picks a pattern for each message, using quick rules and, when unsure, a short check with a small model. It never adds structure on top of a model that reasons natively. | 0 (rarely 1 small) |
| **Native** | The model thinks freely, with no added structure. AgentX reserves room for the answer so the thinking can't crowd it out. | 0 |
| **Step-by-step** | Numbered reasoning steps, so a model without built-in reasoning shows a chain you can check. | 0 |
| **Step-back** | A hidden first call distills the governing principles; the answer then applies them explicitly. | 1 small |
| **Reflection** | One answer written as a draft, a self-critique, then an improved final answer. The critique stays in the thinking. | 0 |
| **Deep reflection** | Separate passes: the draft and the critique stream into the thinking bubble as you watch, then the improved answer streams, with tools available. | 2 |
| **Consensus** | Several independent attempts (2–5, without tools) appear as thinking, then a judged final answer. Auto picks it only for short calculation or logic questions that need no tools. | One per attempt |

## Choosing a pattern

AgentX uses the first of these that is set:

1. **This conversation:** the **Thinking mode** in the
   [Relay](chat.md#the-relay--the-conversations-command-center).
2. **This agent:** the **Thinking pattern** in its [profile](agent-profiles.md#thinking-pattern).
3. **Auto**, unless your server sets a different default.

The Thinking mode menu also offers **Research Mode**. A research turn follows its own rigorous
approach and never combines with a pattern, so the two share one menu to make that clear. Options
turned off in Settings drop out of the menu. Each message carries a badge showing the mode it ran
with, and the multi-pass patterns show their progress as they go: "Drafting…", "Critiquing the
draft…", "Sampling 3 independent solutions…".

Two older pattern names still work, mapped to their nearest equivalent in chat:
**Tree-of-Thought** runs as step-by-step, and **ReAct** runs as native thinking with the tools
narrated (chat's tool loop already reasons and acts with real tools). A one-line status says when
this happens.

## Settings → Intelligence → Thinking Patterns

- **Patterns:** turn patterns off altogether, or one at a time. Both your own choices and Auto
  respect these.
- **Auto selection:** turn the small-model check on or off, choose its model (left empty, it uses
  the Fast Utility role), and set the shortest message it considers.
- **Pattern models & budgets:** the models for step-back and consensus (left empty, they use the
  conversation's own model), how many consensus attempts to make, and the floor on thinking
  output. `0` sets the floor automatically whenever a pattern is active or the model reasons
  natively, so thinking can't crowd out the visible answer.

## For developers

### Configuration and values

Thinking-pattern settings live under `reasoning.*` and are writable through
`/api/config/update` (allowlisted). The classifier is a member of the Fast Utility role, and
`step_back_model` and `sc_model` inherit the turn's model when empty.

In requests and profiles, patterns use these values: `auto`, `native`, `cot` (step-by-step),
`step_back`, `reflection`, `deep_reflection` and `self_consistency` (consensus). An agent's
pattern is its profile's `reasoning_strategy`. The server-wide default is
`preferences.default_reasoning_strategy` (shipped as `auto`); no settings screen shows it, so set
it through `/api/config/update` or `data/config.json`. Each turn reports its mode on the stream's
`done` event (`thinking_pattern` and `research`) and on the stored turn.

### The offline reasoning kit (`/api/agent/run`)

Besides the chat patterns, AgentX keeps the classic strategy classes for blocking,
non-streaming task execution. `ReasoningOrchestrator` classifies the task (with the same
heuristics as chat's Auto, in `reasoning/selection.py`) and runs the full strategies:
Chain-of-Thought (zero-shot/few-shot), Tree-of-Thought (BFS/DFS/beam over branching thoughts),
ReAct (reason+act with registered tools), and Reflection (iterative critique/revision cycles,
prompts in `system_prompts.yaml`). Strategy execution is wall-clock bounded
(`OrchestratorConfig.timeout_seconds`), falls back to Chain-of-Thought on failure, and resolves
models through the provider fallback chain like every other feature.

```
POST /api/agent/run
{"task": "…", "reasoning_strategy": "tot"}
```

The chat-first values (`native`, `step_back`, `deep_reflection`, `self_consistency`) alias to
their nearest kit strategy on this endpoint, and full Tree-of-Thought is available only here.

### Multi-model drafting

Thinking patterns shape how *one* model reasons. **Drafting** goes wider, orchestrating
**several models** on a single generation to trade speed, quality and cost. It's an opt-in agent
option, **off by default** (`enable_drafting = false` on the agent config), with strategies
defined in `drafting/drafting_strategies.yaml` and picked per task type. Three families:

#### Speculative decoding

A fast **draft** model generates tokens that a stronger **target** model verifies, accepting or
rejecting each batch, so tokens are cheaper whenever the two agree. See the
[speculative-decoding flow](../architecture/system-design.md#speculative-decoding) on the System
Design page.

| Config | Description |
|--------|-------------|
| `draft_model` | The fast proposer |
| `target_model` | The strong verifier |
| `draft_tokens` | Tokens per draft batch (20–30) |
| `acceptance_threshold` | Minimum score to accept (0.7–0.8) |
| `max_iterations` | Maximum draft-verify cycles |

Pre-configured: `fast_accurate`, `local_cloud`, `claude_fast`.

#### Pipeline

Multi-stage generation where each stage uses a different model, prompt and temperature for a
specific role: analyze, critique/review, refine, summarize. Pre-configured: `code_review`
(generate → review → refine), `writing_pipeline` (outline → draft → edit → polish),
`analysis_pipeline` (decompose → research → synthesize).

#### Candidate generation

Generate several candidates and pick the best with a scoring method: `majority_vote` (the most
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

Task types map to a default strategy via the `defaults` block: `general → fast_accurate`,
`code → code_review`, `writing → writing_pipeline`, `analysis → analysis_pipeline`,
`consensus → consensus`. Full config lives in `api/agentx_ai/drafting/drafting_strategies.yaml`;
the result schema is in [API Models](../api/models.md).

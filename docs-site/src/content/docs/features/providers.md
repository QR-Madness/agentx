# Model Providers

AgentX is **model-agnostic** — bring your own. It speaks to local runtimes and cloud APIs
through one unified interface, so you can run a small model on your own machine, call Claude or
GPT directly, or reach hundreds of models through an aggregator, and switch between them per
conversation or per agent.

## Bring your own model

Five backends ship out of the box. Add a key (in `.env` or **Settings**) and the models become
selectable:

| Provider | What it is | Key |
|----------|------------|-----|
| **LM Studio** | Local models over an OpenAI-compatible API — private, no per-token cost | `LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`) |
| **Anthropic** | Claude models, direct | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT models, direct | `OPENAI_API_KEY` |
| **OpenRouter** | Aggregator — hundreds of models behind one key | `OPENROUTER_API_KEY` |
| **Vercel AI Gateway** | Aggregator gateway with automatic failover | `AI_GATEWAY_API_KEY` |

Pick a model anywhere you choose one (the Relay, an agent profile) with **`provider:model`** —
`lmstudio:llama-3.2` for a local model, `anthropic:claude-sonnet-5` for Claude direct, or an
aggregator route like `openrouter:<vendor>/<model>` to reach the long tail. Available models are
discovered live from each provider's API, so the picker reflects what you can actually run.

## Which model runs a turn

The active model is resolved top-down: an **explicit choice for the turn** (the Relay's Model
tile or a per-request override) → the **agent profile's** default model → the **server default**.
That server default is itself layered — the **`DEFAULT_MODEL`** environment variable, then your
**global preference** in Settings, then a built-in local-model floor. What `models.yaml` does
*not* do is pin the runtime default; it holds provider settings only.

## Model roles

Not every internal job deserves your best model. A quick auto-classification or a compaction
summary can run on something faster and cheaper. **Model roles** let you assign a model to a
named system role — a **Fast Utility** role (short classifications, the auto thinking-pattern
tiebreak) and a **Summarizer** role (context compaction) are the main ones.

Role members default to **inherit** (left empty), so a role follows the conversation's own model
until you deliberately set one. That default matters: pinning a concrete model as a role's
built-in default would quietly bypass the roles overlay, so roles stay empty-by-default on
purpose. Configure them in **Settings → Intelligence**.

## Fallback — never hard-fail a turn

If a model's provider is unreachable or errors mid-feature, AgentX **falls back** to another
capable model rather than failing the turn. It's on by default (`models.fallback_enabled`) and
watches real call outcomes, so a provider that just failed is briefly deprioritized. Features
like reasoning and delegation resolve through this same chain, which is why a flaky provider
degrades gracefully instead of breaking a conversation.

## Model Limits & the `:latest` gotcha

Aggregator routes pinned to **`:latest`** don't report their context window, so AgentX has to
assume a conservative ~8k tokens — which triggers premature context compaction (memory that
feels forgetful) and can break image generation. Two guards catch this:

- The **model picker warns** on a `:latest` route and suggests pinning a concrete version.
- **Settings → Model Limits** lets you set a **per-model context-window override** (an escape
  hatch for any provider) alongside local-model context and output caps. An override **wins**
  over whatever the provider reports.

!!! tip "Pin the version"
    When you can, choose a concrete model version (e.g. `…-20250219`) over `:latest`. It resolves
    the real context window, keeps memory working, and won't surprise you when the aggregator
    re-points the alias.

## Under the hood

A lazy `ProviderRegistry` resolves each `provider:model` name to its backend, loads provider
configs from `providers/models.yaml` (**provider settings only** — per-model capabilities are
fetched from each provider's API, not hand-listed), and creates providers on demand from
environment variables or `data/config.json`. The OpenRouter and Vercel providers additionally
extract per-model metadata and pricing (`providers/pricing.py`) for live cost estimation. See the
[provider-resolution diagram](../architecture/system-design.md#provider-resolution) on the System
Design page.

Provider settings can also be changed at runtime in **Settings**, and the programmatic surface —
listing providers, models with capabilities, and health — is in the
[API Reference](../api/endpoints.md#providers).

## Related

- [Reasoning](reasoning.md) — model roles power auto pattern-selection
- [Agent Profiles](agent-profiles.md) — set an agent's default model and temperature
- [API Models: Provider](../api/models.md#provider-models) — Message, CompletionResult, ModelCapabilities

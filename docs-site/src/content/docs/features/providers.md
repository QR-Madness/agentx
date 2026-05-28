# Model Providers

The provider system provides a unified interface for interacting with different LLM backends. All providers implement the `ModelProvider` abstract class.

## Provider Registry

```mermaid
graph TD
    R[ProviderRegistry] --> |resolves model → provider| LMS[LM Studio]
    R --> ANT[Anthropic]
    R --> OAI[OpenAI]
    R --> ORT[OpenRouter]
    R --> VRC[Vercel AI Gateway]
    R --> |loads| MY[models.yaml]

    LMS --> |OpenAI-compatible API| LOCAL[Local Models]
    ANT --> |Anthropic API| CLAUDE[Claude Models]
    OAI --> |OpenAI API| GPT[GPT Models]
    ORT --> |aggregator API| MANY[100+ models]
    VRC --> |gateway API| MANY
```

`ProviderRegistry` is a lazy singleton that:
- Loads model definitions from `providers/models.yaml`
- Resolves model names to their provider (e.g., `"claude-3-5-sonnet-latest"` → Anthropic)
- Creates provider instances on demand with config from env vars or `data/config.json`

```python
registry = get_registry()
provider, model_id = registry.get_provider_for_model("claude-3-5-sonnet-latest")
result = await provider.complete(messages, model_id)
```

## Implementations

| Provider | Class | API | Config |
|----------|-------|-----|--------|
| LM Studio | `LMStudioProvider` | OpenAI-compatible (local) | `LMSTUDIO_BASE_URL` (default: `http://localhost:1234/v1`) |
| Anthropic | `AnthropicProvider` | Anthropic API | `ANTHROPIC_API_KEY` |
| OpenAI | `OpenAIProvider` | OpenAI API | `OPENAI_API_KEY` |
| OpenRouter | `OpenRouterProvider` | OpenAI-compatible aggregator (100+ models) | `OPENROUTER_API_KEY` |
| Vercel AI Gateway | `VercelProvider` | Cloud aggregator gateway (100+ models) | `AI_GATEWAY_API_KEY` |

All providers support the same interface:

| Method | Description |
|--------|-------------|
| `complete(messages, model, **kwargs)` | Full completion (async) |
| `stream(messages, model, **kwargs)` | Streaming completion → `AsyncIterator[StreamChunk]` |
| `get_capabilities(model)` | Returns `ModelCapabilities` for a model |
| `list_models()` | Returns available model names |
| `health_check()` | Tests connectivity |

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | `0.7` | Sampling temperature |
| `max_tokens` | int | `null` | Maximum output tokens |
| `tools` | list[dict] | `null` | Function calling tools |
| `tool_choice` | string/dict | `null` | Tool selection (`"auto"`, `"none"`, or specific) |
| `stop` | list[string] | `null` | Stop sequences |

## Model Configuration

Providers (and their default models) are defined in `providers/models.yaml`. The file
holds **provider configs only** — per-model capabilities are fetched dynamically from each
provider's API, not hand-listed. Select a model with `provider:model` syntax:

```yaml
providers:
  lmstudio:
    base_url_env: LMSTUDIO_BASE_URL
    timeout: 300.0       # local models may be slower
    max_retries: 1
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    timeout: 60.0
    max_retries: 3
  openai:     { api_key_env: OPENAI_API_KEY,     timeout: 60.0, max_retries: 3 }
  openrouter: { api_key_env: OPENROUTER_API_KEY, timeout: 60.0, max_retries: 3 }
  vercel:     { api_key_env: AI_GATEWAY_API_KEY, timeout: 60.0, max_retries: 3 }

# Default model per use case (provider:model format)
defaults:
  chat:      anthropic:claude-3-5-sonnet-latest
  reasoning: anthropic:claude-3-5-sonnet-latest
  code:      lmstudio:deepseek-coder-v2
  fast:      anthropic:claude-3-5-haiku-latest
  local:     lmstudio:llama3.2
```

The registry discovers available models dynamically from each provider's API (e.g. LM
Studio's `/v1/models`). The OpenRouter and Vercel providers additionally extract per-model
metadata and pricing, surfaced via `providers/pricing.py` for cost estimation.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEFAULT_MODEL` | Default model for chat (default: `"llama-3.2-1b-instruct"`) |
| `LMSTUDIO_BASE_URL` | LM Studio API URL |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key (100+ models) |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway API key (100+ models) |
| `DEBUG_LOG_LLM_REQUESTS` | Log full request payloads when set to `1`/`true` |

Provider settings can also be set at runtime via `POST /api/config/update`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/providers` | GET | List configured providers |
| `/api/providers/models` | GET | List models with capabilities |
| `/api/providers/health` | GET | Health check all providers |

See [API Endpoints](../api/endpoints.md#providers) for full details.

## Related

- [API Models: Provider](../api/models.md#provider-models) — Message, CompletionResult, ModelCapabilities schemas
- Config file: `api/agentx_ai/providers/models.yaml`

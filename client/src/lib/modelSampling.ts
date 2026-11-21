/**
 * Which models honour the temperature a profile sets.
 *
 * Mirrors the API. `AnthropicProvider` sends no `temperature`, `top_p` or `top_k`
 * (v0.21.290): current Claude models reject a temperature with HTTP 400, and API
 * version 2026-10-21 rejects one on every model. Those models always run on their
 * default sampling, whatever the profile says.
 *
 * The route is the `provider:` prefix of the model id, split the way the API's
 * `get_provider_for_model` splits it (first colon, case-sensitive). Claude reached
 * through OpenRouter or Vercel (`openrouter:anthropic/…`, `vercel:anthropic/…`) runs
 * through another provider and still takes the temperature.
 *
 * Keep this in step with the API: when a provider starts or stops sending sampling
 * parameters, update the map below.
 */

/** Provider id (the `provider:` prefix) → display name, for providers that send no sampling parameters. */
const DEFAULT_SAMPLING_PROVIDERS: ReadonlyMap<string, string> = new Map([
  ['anthropic', 'Anthropic'],
]);

/**
 * The display name of the model's provider when that provider ignores the
 * temperature, otherwise `null`.
 *
 * An empty id (inherit the system default) returns `null`: the effective model is
 * resolved server-side, so the editor cannot tell which provider it lands on.
 */
export function temperatureIgnoredBy(modelId: string | null | undefined): string | null {
  if (!modelId) return null;
  const sep = modelId.indexOf(':');
  if (sep <= 0) return null;
  return DEFAULT_SAMPLING_PROVIDERS.get(modelId.slice(0, sep)) ?? null;
}

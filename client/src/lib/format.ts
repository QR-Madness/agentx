/**
 * Shared formatting helpers for model usage metrics (tokens, cost, latency).
 *
 * Extracted from `components/chat/MetadataBar.tsx` so the per-turn metadata bar
 * and the Dashboard usage card render figures identically.
 */

const CURRENCY_SYMBOLS: Record<string, string> = {
  USD: '$',
  EUR: '€',
  GBP: '£',
};

/** Format a monetary amount with the currency symbol, scaling precision down for tiny costs. */
export function formatCost(amount: number, currency = 'USD'): string {
  const symbol = CURRENCY_SYMBOLS[currency] ?? '';
  const suffix = symbol ? '' : ` ${currency}`;
  if (amount < 0.0001) return `<${symbol}0.0001${suffix}`;
  if (amount < 0.01) return `${symbol}${amount.toFixed(4)}${suffix}`;
  if (amount < 1) return `${symbol}${amount.toFixed(3)}${suffix}`;
  return `${symbol}${amount.toFixed(2)}${suffix}`;
}

/** Format milliseconds as `ms` under a second, otherwise seconds with 2 decimals. */
export function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/** Strip provider prefixes like `anthropic/claude-3-sonnet` → `claude-3-sonnet`. */
export function getModelShortName(modelName: string): string {
  const parts = modelName.split('/');
  return parts[parts.length - 1];
}

/** Compact large token counts: 1234 → `1.2K`, 1_200_000 → `1.2M`. */
export function formatCompact(n: number): string {
  if (n < 1000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}K`;
  return `${(n / 1_000_000).toFixed(n < 10_000_000 ? 1 : 0)}M`;
}

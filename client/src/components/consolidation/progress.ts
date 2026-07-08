/**
 * Pure helpers that turn raw SSE progress into drawer display values.
 */

/** Parse a backend `"N of M"` conversation string → fraction 0–1, or null. */
export function parseNofM(s: unknown): number | null {
  if (typeof s !== 'string') return null;
  const m = s.match(/^\s*(\d+)\s+of\s+(\d+)\s*$/i);
  if (!m) return null;
  const n = Number(m[1]);
  const total = Number(m[2]);
  if (!Number.isFinite(n) || !Number.isFinite(total) || total <= 0) return null;
  return Math.max(0, Math.min(1, n / total));
}

/** Compact number for chips: 0–999 as-is, then 1.2K / 45.3K / 3.4M. */
export function compactNumber(n: number): string {
  if (!Number.isFinite(n)) return '0';
  const abs = Math.abs(n);
  if (abs < 1000) return String(Math.round(n));
  if (abs < 1_000_000) return `${(n / 1000).toFixed(abs < 10_000 ? 1 : 0)}K`.replace('.0K', 'K');
  return `${(n / 1_000_000).toFixed(1)}M`.replace('.0M', 'M');
}

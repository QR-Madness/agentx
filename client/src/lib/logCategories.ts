/**
 * Client mirror of the backend log-category contract
 * (`api/agentx_ai/logging_kit/categories.py`). Keep the keys in sync with that
 * file — same discipline as `promptPlaceholders.ts`. The panel can also hydrate
 * labels/emojis from `GET /api/logs/categories`; this static map provides the
 * theme-aware colors and a synchronous fallback.
 */

export interface LogCategoryMeta {
  key: string;
  label: string;
  emoji: string;
  /** A CSS color for the badge — chosen to read on any theme. */
  color: string;
}

export const LOG_CATEGORIES: Record<string, LogCategoryMeta> = {
  provider: { key: 'provider', label: 'PROVIDER', emoji: '🧠', color: '#22d3ee' },
  stream: { key: 'stream', label: 'STREAM', emoji: '📡', color: '#e879f9' },
  plan: { key: 'plan', label: 'PLAN', emoji: '🗺️', color: '#fb923c' },
  reason: { key: 'reason', label: 'REASON', emoji: '💭', color: '#a78bfa' },
  ambassador: { key: 'ambassador', label: 'AMBASS', emoji: '🤝', color: '#2dd4bf' },
  memory: { key: 'memory', label: 'MEMORY', emoji: '🧩', color: '#facc15' },
  mcp: { key: 'mcp', label: 'MCP', emoji: '🔌', color: '#60a5fa' },
  jobs: { key: 'jobs', label: 'JOBS', emoji: '⚙️', color: '#9ca3af' },
  agent: { key: 'agent', label: 'AGENT', emoji: '🤖', color: '#34d399' },
  translation: { key: 'translation', label: 'TRANS', emoji: '🌐', color: '#4ade80' },
  logkit: { key: 'logkit', label: 'LOGS', emoji: '🪵', color: '#9ca3af' },
  core: { key: 'core', label: 'CORE', emoji: '•', color: '#cbd5e1' },
};

export function categoryMeta(key: string): LogCategoryMeta {
  return (
    LOG_CATEGORIES[key] ?? {
      key,
      label: key.toUpperCase(),
      emoji: '•',
      color: 'var(--text-muted)',
    }
  );
}

export const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const;

export const LOG_LEVEL_COLORS: Record<string, string> = {
  DEBUG: '#6b7280',
  INFO: '#3b82f6',
  WARNING: '#f59e0b',
  ERROR: '#ef4444',
  CRITICAL: '#dc2626',
};

export function levelColor(level: string): string {
  return LOG_LEVEL_COLORS[level.toUpperCase()] ?? 'var(--text-muted)';
}

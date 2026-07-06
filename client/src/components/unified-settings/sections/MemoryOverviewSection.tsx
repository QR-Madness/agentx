/**
 * MemoryOverviewSection — Memory & storage overview
 *
 * Live memory stats (GET /api/memory/stats), settings-file health from
 * GET /api/memory/settings (`settings_file_status` — this section is the only
 * UI surfacing of that field), plus the storage-technology rows.
 */

import { Database, RefreshCw, TriangleAlert } from 'lucide-react';
import { api } from '../../../lib/api';
import { useApi } from '../../../lib/hooks';
import { Card, SectionHeader } from '../../ui';

const STATS: { key: 'entities' | 'facts' | 'turns' | 'strategies'; label: string }[] = [
  { key: 'entities', label: 'Entities' },
  { key: 'facts', label: 'Facts' },
  { key: 'turns', label: 'Turns' },
  { key: 'strategies', label: 'Strategies' },
];

export default function MemoryOverviewSection() {
  const { data: stats, loading: statsLoading, error: statsError } =
    useApi(() => api.getMemoryStats(), []);
  // Loaded only for `settings_file_status` (GET-only health of the overrides file).
  const { data: consolidation, loading: fileLoading } =
    useApi(() => api.getConsolidationSettings(), []);

  const fileStatus = consolidation?.settings_file_status;
  const statsUnavailable = !!statsError || !!stats?.unavailable;

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Database size={20} />}
        title="Memory & Storage"
        description="What the agent has remembered, and where it lives."
      />

      {/* Settings-file health — a corrupt overrides file silently falls back to defaults,
          so a parse error gets a loud card; the healthy path is a one-liner. */}
      {fileStatus?.error ? (
        <Card className="mb-4 border-warning/40 bg-warning/10">
          <div className="flex items-start gap-3 p-1">
            <TriangleAlert size={18} className="mt-0.5 shrink-0 text-warning" />
            <div className="min-w-0">
              <p className="font-semibold text-fg">
                Your memory settings file is corrupt — defaults are in effect.
              </p>
              <p className="mt-1 text-sm text-fg-secondary break-words">{fileStatus.error}</p>
              <p className="mt-1 text-xs font-mono text-fg-muted break-all">{fileStatus.path}</p>
              <p className="mt-2 text-sm text-fg-secondary">
                Saving any memory setting rewrites the file; or fix it by hand at the path above.
              </p>
            </div>
          </div>
        </Card>
      ) : fileStatus ? (
        <p className="mb-4 text-sm text-fg-muted">
          Settings file: {fileStatus.exists ? 'OK' : 'using defaults (no overrides file yet)'}
          <span className="ml-2 font-mono text-xs">{fileStatus.path}</span>
        </p>
      ) : fileLoading ? null : (
        <p className="mb-4 text-sm text-fg-muted">Settings file: status unavailable</p>
      )}

      <Card className="mb-4">
        <p className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">
          Stored memories
        </p>
        {statsLoading ? (
          <div className="flex items-center gap-2 py-3 text-sm text-fg-muted">
            <RefreshCw size={14} className="spin" />
            <span>Loading memory statistics...</span>
          </div>
        ) : statsUnavailable || !stats ? (
          <p className="py-3 text-sm text-fg-muted">
            Memory statistics unavailable — the memory databases may be offline.
          </p>
        ) : (
          <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {STATS.map(({ key, label }) => (
              <div key={key} className="rounded-lg bg-surface-sunken px-3 py-2">
                <div className="text-xl font-semibold text-fg">
                  {stats.totals[key].toLocaleString()}
                </div>
                <div className="text-xs text-fg-muted">{label}</div>
              </div>
            ))}
          </div>
        )}
        {!statsLoading && !statsUnavailable && stats && (
          <p className="mt-3 text-xs text-fg-muted">
            Across {Object.keys(stats.by_channel).length.toLocaleString()} memory channel
            {Object.keys(stats.by_channel).length === 1 ? '' : 's'}.
          </p>
        )}
      </Card>

      <Card>
        <p className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">Storage</p>
        <div className="mt-2 flex flex-col divide-y divide-line-subtle">
          <div className="flex items-center justify-between gap-4 py-2">
            <span className="text-sm text-fg-secondary">Session Storage</span>
            <span className="text-sm text-fg">Local (Browser)</span>
          </div>
          <div className="flex items-center justify-between gap-4 py-2">
            <span className="text-sm text-fg-secondary">Server Data</span>
            <span className="text-sm text-fg">PostgreSQL + Neo4j</span>
          </div>
          <div className="flex items-center justify-between gap-4 py-2">
            <span className="text-sm text-fg-secondary">Cache</span>
            <span className="text-sm text-fg">Redis</span>
          </div>
        </div>
      </Card>
    </div>
  );
}

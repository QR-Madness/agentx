// ─── Overview / stats home ─────────────────────────────────────────────────
// Landing area for the Memory Workbench: totals + a per-channel breakdown.
// Clicking a channel row scopes the workbench to that channel's Entities.

import { Users, FileText, Zap, MessagesSquare, Database, RefreshCw } from 'lucide-react';
import type { MemoryStats } from '../../lib/api';

const TILES = [
  { key: 'entities', label: 'Entities', icon: Users },
  { key: 'facts', label: 'Facts', icon: FileText },
  { key: 'strategies', label: 'Strategies', icon: Zap },
  { key: 'turns', label: 'Turns', icon: MessagesSquare },
] as const;

export function OverviewPanel({
  stats, loading, onOpenChannel,
}: {
  stats: MemoryStats | null | undefined;
  loading: boolean;
  onOpenChannel: (channel: string) => void;
}) {
  if (loading && !stats) {
    return <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading overview...</p></div>;
  }

  if (!stats || stats.unavailable) {
    return (
      <div className="memory-empty">
        <Database size={32} />
        <p>Memory statistics are unavailable</p>
        <p className="hint">The memory databases may be offline — start them with <code>task db:up</code>.</p>
      </div>
    );
  }

  const channels = Object.entries(stats.by_channel ?? {}).sort(
    (a, b) => (b[1].entities + b[1].facts) - (a[1].entities + a[1].facts),
  );

  return (
    <div className="mem-overview">
      <div className="mem-overview-tiles">
        {TILES.map(({ key, label, icon: Icon }) => (
          <div key={key} className="mem-stat-tile">
            <span className="mem-stat-icon"><Icon size={18} /></span>
            <span className="mem-stat-value">{(stats.totals[key] ?? 0).toLocaleString()}</span>
            <span className="mem-stat-label">{label}</span>
          </div>
        ))}
      </div>

      <div className="mem-overview-section">
        <h3 className="mem-overview-heading">Channels</h3>
        {channels.length === 0 ? (
          <p className="mem-overview-empty">No channels yet.</p>
        ) : (
          <div className="mem-channel-table" role="table">
            <div className="mem-channel-row mem-channel-head" role="row">
              <span role="columnheader">Channel</span>
              <span role="columnheader">Entities</span>
              <span role="columnheader">Facts</span>
              <span role="columnheader">Strategies</span>
              <span role="columnheader">Turns</span>
            </div>
            {channels.map(([name, counts]) => (
              <button
                key={name}
                type="button"
                className="mem-channel-row"
                role="row"
                onClick={() => onOpenChannel(name)}
                title={`Browse ${name} entities`}
              >
                <span className="mem-channel-name" role="cell">{name}</span>
                <span role="cell">{counts.entities.toLocaleString()}</span>
                <span role="cell">{counts.facts.toLocaleString()}</span>
                <span role="cell">{counts.strategies.toLocaleString()}</span>
                <span role="cell">{counts.turns.toLocaleString()}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

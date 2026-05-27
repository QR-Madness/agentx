// ─── Strategy list view ────────────────────────────────────────────────────

import { Zap, RefreshCw } from 'lucide-react';
import type { MemoryStrategy, ApiError } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';

export function StrategyListView({
  strategies, total, loading, error, selectedStrategyId, onSelectStrategy,
}: {
  strategies: MemoryStrategy[];
  total: number;
  loading: boolean;
  error: ApiError | null;
  selectedStrategyId: string | null;
  onSelectStrategy: (strategy: MemoryStrategy | null) => void;
}) {
  if (loading) return <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading strategies...</p></div>;
  if (error) return <div className="memory-error"><p>Failed to load strategies: {error.message}</p></div>;
  if (strategies.length === 0) return (
    <div className="memory-empty">
      <Zap size={32} /><p>No strategies found</p>
      <p className="hint">Strategies are learned from successful tool usage patterns</p>
    </div>
  );

  return (
    <div className="memory-list">
      <div className="memory-list-header strategies-header">
        <span>Description</span><span>Tool Sequence</span><span>Success Rate</span><span>Channel</span><span>Last Used</span>
      </div>
      {strategies.map(strategy => (
        <div
          key={strategy.id}
          className={`memory-row strategy-row${selectedStrategyId === strategy.id ? ' selected' : ''}`}
          onClick={() => onSelectStrategy(selectedStrategyId === strategy.id ? null : strategy)}
        >
          <span className="strategy-description">{strategy.description}</span>
          <span className="strategy-tools">
            <div className="tool-sequence">
              {strategy.tool_sequence.map((tool, i) => <span key={i} className="tool-chip">{tool}</span>)}
            </div>
          </span>
          <span className="strategy-success">
            <div className="success-bar">
              <div className="success-fill" style={{ width: `${(strategy.success_rate || 0) * 100}%` }} />
            </div>
            <span>{((strategy.success_rate || 0) * 100).toFixed(0)}%</span>
          </span>
          <span className="strategy-channel">{strategy.channel}</span>
          <span className="strategy-used">{formatTimestamp(strategy.last_used)}</span>
        </div>
      ))}
      <div className="memory-list-footer">Showing {strategies.length} of {total} strategies</div>
    </div>
  );
}

// ─── Fact list view ────────────────────────────────────────────────────────

import { FileText, RefreshCw, ArrowUpRight } from 'lucide-react';
import type { MemoryFact, ApiError } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';

export function FactListView({
  facts, total, loading, error, selectedFactId, onSelectFact,
}: {
  facts: MemoryFact[];
  total: number;
  loading: boolean;
  error: ApiError | null;
  selectedFactId: string | null;
  onSelectFact: (fact: MemoryFact | null) => void;
}) {
  if (loading) return <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading facts...</p></div>;
  if (error) return <div className="memory-error"><p>Failed to load facts: {error.message}</p></div>;
  if (facts.length === 0) return <div className="memory-empty"><FileText size={32} /><p>No facts found</p></div>;

  return (
    <div className="memory-list">
      <div className="memory-list-header facts-header">
        <span>Claim</span><span>Confidence</span><span>Source</span><span>Channel</span><span>Created</span>
      </div>
      {facts.map(fact => (
        <div
          key={fact.id}
          className={`memory-row fact-row${selectedFactId === fact.id ? ' selected' : ''}`}
          onClick={() => onSelectFact(selectedFactId === fact.id ? null : fact)}
        >
          <span className="fact-claim">
            {fact.claim}
            {fact.promoted_from && (
              <span className="promoted-badge"><ArrowUpRight size={12} />promoted</span>
            )}
          </span>
          <span className="fact-confidence">
            <div className="confidence-bar">
              <div className="confidence-fill" style={{ width: `${(fact.confidence || 0) * 100}%` }} />
            </div>
            <span>{((fact.confidence || 0) * 100).toFixed(0)}%</span>
          </span>
          <span className="fact-source badge">{fact.source}</span>
          <span className="fact-channel">{fact.channel}</span>
          <span className="fact-created">{formatTimestamp(fact.created_at)}</span>
        </div>
      ))}
      <div className="memory-list-footer">Showing {facts.length} of {total} facts</div>
    </div>
  );
}

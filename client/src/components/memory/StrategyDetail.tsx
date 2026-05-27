// ─── Strategy detail panel ─────────────────────────────────────────────────

import React from 'react';
import { X } from 'lucide-react';
import type { MemoryStrategy } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';
import { Button } from '../ui';

export function StrategyDetail({
  strategy, onClose,
}: {
  strategy: MemoryStrategy;
  onClose: () => void;
}) {
  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>Strategy</h3>
        <Button variant="ghost" onClick={onClose}><X size={18} /></Button>
      </div>

      <p className="fact-claim-full">{strategy.description}</p>

      <div className="entity-section">
        <h4>Tool Sequence</h4>
        <div className="tool-sequence-detail">
          {strategy.tool_sequence.map((tool, i) => (
            <React.Fragment key={i}>
              <span className="tool-chip">{tool}</span>
              {i < strategy.tool_sequence.length - 1 && <span className="tool-arrow">→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="entity-section">
        <h4>Performance</h4>
        <div className="strategy-metrics-grid">
          <div className="strategy-metric">
            <span className="metric-value success-value">{strategy.success_count}</span>
            <span className="metric-label">Successes</span>
          </div>
          <div className="strategy-metric">
            <span className="metric-value error-value">{strategy.failure_count}</span>
            <span className="metric-label">Failures</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12 }}>
          <div style={{ flex: 1, height: 6, background: 'var(--bg-void)', borderRadius: 3, overflow: 'hidden' }}>
            <div className="success-fill" style={{ width: `${(strategy.success_rate || 0) * 100}%`, height: '100%' }} />
          </div>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)', minWidth: 40 }}>
            {((strategy.success_rate || 0) * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <div className="entity-info">
        <div className="info-row"><span className="label">Channel</span><span className="value">{strategy.channel}</span></div>
        <div className="info-row"><span className="label">Last Used</span><span className="value">{formatTimestamp(strategy.last_used)}</span></div>
      </div>
    </div>
  );
}

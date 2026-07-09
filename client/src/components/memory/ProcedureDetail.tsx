// ─── Procedure detail panel ────────────────────────────────────────────────
// Read-only (the procedures API has no patch/delete route).

import { X } from 'lucide-react';
import type { MemoryProcedure } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';
import { procedureHeadline } from './procedureLabel';
import { Button } from '../ui';

export function ProcedureDetail({
  procedure, onClose,
}: {
  procedure: MemoryProcedure;
  onClose: () => void;
}) {
  const convRefs = (procedure.evidence_refs ?? []).filter(r => r.startsWith('conv:'));

  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>Procedure</h3>
        <Button variant="ghost" onClick={onClose}><X size={18} /></Button>
      </div>

      <p className="fact-claim-full">{procedureHeadline(procedure.trigger, procedure.body)}</p>

      {procedure.body && (
        <div className="entity-section">
          <h4>Action</h4>
          <p className="procedure-body">{procedure.body}</p>
        </div>
      )}

      {procedure.rationale && (
        <div className="entity-section">
          <h4>Rationale</h4>
          <p className="procedure-body">{procedure.rationale}</p>
        </div>
      )}

      <div className="entity-section">
        <h4>Strength</h4>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ flex: 1, height: 6, background: 'var(--bg-void)', borderRadius: 3, overflow: 'hidden' }}>
            <div className="success-fill" style={{ width: `${Math.min(procedure.strength * 100, 100)}%`, height: '100%' }} />
          </div>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)', minWidth: 40 }}>
            {procedure.strength.toFixed(2)}
          </span>
        </div>
      </div>

      {(procedure.signal_kinds ?? []).length > 0 && (
        <div className="entity-section">
          <h4>Signals</h4>
          <div className="tool-sequence">
            {procedure.signal_kinds.map((s, i) => <span key={i} className="tool-chip">{s}</span>)}
          </div>
        </div>
      )}

      <div className="entity-info">
        <div className="info-row"><span className="label">Scope</span><span className="value badge">{procedure.scope}</span></div>
        {procedure.agent_id && (
          <div className="info-row"><span className="label">Agent</span><span className="value">{procedure.agent_id}</span></div>
        )}
        <div className="info-row"><span className="label">Channel</span><span className="value">{procedure.channel}</span></div>
        <div className="info-row"><span className="label">Evidence</span><span className="value">{convRefs.length} conversation{convRefs.length === 1 ? '' : 's'}</span></div>
        <div className="info-row"><span className="label">Last Reinforced</span><span className="value">{formatTimestamp(procedure.last_reinforced)}</span></div>
      </div>
    </div>
  );
}

// ─── Procedure list view ───────────────────────────────────────────────────
// Distilled procedural memory (read-only, mirrors StrategyListView). The API
// (/api/memory/procedures) is list-only — no edit/delete.

import { ListChecks, RefreshCw } from 'lucide-react';
import type { MemoryProcedure, ApiError } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';
import { procedureHeadline } from './procedureLabel';

export function ProcedureListView({
  procedures, total, loading, error, selectedProcedureId, onSelectProcedure,
}: {
  procedures: MemoryProcedure[];
  total: number;
  loading: boolean;
  error: ApiError | null;
  selectedProcedureId: string | null;
  onSelectProcedure: (procedure: MemoryProcedure | null) => void;
}) {
  if (loading) return <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading procedures...</p></div>;
  if (error) return <div className="memory-error"><p>Failed to load procedures: {error.message}</p></div>;
  if (procedures.length === 0) return (
    <div className="memory-empty">
      <ListChecks size={32} /><p>No procedures found</p>
      <p className="hint">Procedures are distilled from repeated corrections and rules</p>
    </div>
  );

  return (
    <div className="memory-list">
      <div className="memory-list-header procedures-header">
        <span>Trigger</span><span>Scope</span><span>Strength</span><span>Signals</span><span>Last Reinforced</span>
      </div>
      {procedures.map(procedure => (
        <div
          key={procedure.id}
          className={`memory-row procedure-row${selectedProcedureId === procedure.id ? ' selected' : ''}`}
          onClick={() => onSelectProcedure(selectedProcedureId === procedure.id ? null : procedure)}
        >
          <span className="procedure-trigger">{procedureHeadline(procedure.trigger, procedure.body)}</span>
          <span className="procedure-scope badge">{procedure.scope}</span>
          <span className="procedure-strength">
            <div className="success-bar">
              <div className="success-fill" style={{ width: `${Math.min(procedure.strength * 100, 100)}%` }} />
            </div>
            <span>{procedure.strength.toFixed(2)}</span>
          </span>
          <span className="procedure-signals">
            <div className="tool-sequence">
              {(procedure.signal_kinds ?? []).map((s, i) => <span key={i} className="tool-chip">{s}</span>)}
            </div>
          </span>
          <span className="procedure-reinforced">{formatTimestamp(procedure.last_reinforced)}</span>
        </div>
      ))}
      <div className="memory-list-footer">Showing {procedures.length} of {total} procedures</div>
    </div>
  );
}

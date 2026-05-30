// ─── Fact detail panel ─────────────────────────────────────────────────────

import { useState, useCallback, useEffect } from 'react';
import { RefreshCw, X, ArrowUpRight, Edit2, Trash2, Check, AlertTriangle, Star, Archive, History } from 'lucide-react';
import {
  useUpdateMemoryFact,
  useDeleteMemoryFact,
  useRememberMemoryFact,
  useForgetMemoryFact,
  useFactProvenance,
} from '../../lib/hooks';
import type { MemoryFact, FactProvenance } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';
import { Button } from '../ui';

export function FactDetail({
  fact, onClose, onDeleted, onUpdated,
}: {
  fact: MemoryFact;
  onClose: () => void;
  onDeleted: () => void;
  onUpdated: (f: MemoryFact) => void;
}) {
  const { mutate: updateFact, loading: saving } = useUpdateMemoryFact();
  const { mutate: deleteFact, loading: deleting } = useDeleteMemoryFact();
  const { mutate: rememberFact, loading: remembering } = useRememberMemoryFact();
  const { mutate: forgetFact, loading: forgetting } = useForgetMemoryFact();
  const { fetch: fetchProvenance, loading: loadingProvenance } = useFactProvenance();

  const [editMode, setEditMode] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [forgetConfirm, setForgetConfirm] = useState(false);
  const [remembered, setRemembered] = useState(false);
  const [provenance, setProvenance] = useState<FactProvenance | null>(null);
  const [editClaim, setEditClaim] = useState(fact.claim);
  const [editConfidence, setEditConfidence] = useState(Math.round((fact.confidence || 0) * 100));
  const [editSource, setEditSource] = useState(fact.source);
  const [editTemporal, setEditTemporal] = useState('');

  useEffect(() => {
    setEditClaim(fact.claim);
    setEditConfidence(Math.round((fact.confidence || 0) * 100));
    setEditSource(fact.source);
    setEditTemporal('');
    setEditMode(false);
    setDeleteConfirm(false);
    setForgetConfirm(false);
    setRemembered(false);
    setProvenance(null);
  }, [fact.id]);

  const handleSave = useCallback(async () => {
    const result = await updateFact(fact.id, {
      claim: editClaim.trim(),
      confidence: editConfidence / 100,
      source: editSource.trim(),
      temporal_context: (editTemporal as 'current' | 'past' | 'future') || null,
    });
    if (result) {
      setEditMode(false);
      onUpdated(result);
    }
  }, [fact.id, editClaim, editConfidence, editSource, editTemporal, updateFact, onUpdated]);

  const handleDelete = useCallback(async () => {
    const ok = await deleteFact(fact.id);
    if (ok) onDeleted();
  }, [fact.id, deleteFact, onDeleted]);

  const handleRemember = useCallback(async () => {
    const result = await rememberFact(fact.id);
    if (result) {
      setRemembered(true);
      onUpdated(result);
    }
  }, [fact.id, rememberFact, onUpdated]);

  const handleForget = useCallback(async () => {
    const result = await forgetFact(fact.id, false);
    if (result?.success) {
      setForgetConfirm(false);
      if (result.fact) onUpdated(result.fact);
      else onDeleted();
    }
  }, [fact.id, forgetFact, onUpdated, onDeleted]);

  const handleProvenance = useCallback(async () => {
    if (provenance) { setProvenance(null); return; }
    const result = await fetchProvenance(fact.id);
    if (result) setProvenance(result);
  }, [fact.id, provenance, fetchProvenance]);

  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>{editMode ? 'Edit Fact' : 'Fact Detail'}</h3>
        <Button variant="ghost" onClick={onClose}><X size={18} /></Button>
      </div>

      {editMode ? (
        <div className="edit-form">
          <div>
            <label>Claim</label>
            <textarea value={editClaim} onChange={e => setEditClaim(e.target.value)} rows={5} />
          </div>
          <div>
            <label>Confidence: {editConfidence}%</label>
            <input
              type="range" min="0" max="100" value={editConfidence}
              onChange={e => setEditConfidence(Number(e.target.value))}
              style={{ accentColor: 'var(--cosmic-violet)', width: '100%' }}
            />
          </div>
          <div>
            <label>Source</label>
            <input value={editSource} onChange={e => setEditSource(e.target.value)} />
          </div>
          <div>
            <label>Temporal Context</label>
            <select value={editTemporal} onChange={e => setEditTemporal(e.target.value)}>
              <option value="">None</option>
              <option value="current">Current</option>
              <option value="past">Past</option>
              <option value="future">Future</option>
            </select>
          </div>
          <div className="edit-actions">
            <Button variant="primary" onClick={handleSave} disabled={saving}>
              {saving ? <><RefreshCw size={14} className="spin" /> Saving...</> : <><Check size={14} /> Save</>}
            </Button>
            <Button variant="ghost" onClick={() => setEditMode(false)} disabled={saving}>Cancel</Button>
          </div>
        </div>
      ) : (
        <>
          <p className="fact-claim-full">{fact.claim}</p>
          {fact.promoted_from && (
            <div style={{ marginBottom: 12 }}>
              <span className="promoted-badge"><ArrowUpRight size={12} />promoted</span>
            </div>
          )}

          <div className="entity-info">
            <div className="info-row">
              <span className="label">Confidence</span>
              <span className="value" style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
                <div style={{ flex: 1, height: 6, background: 'var(--bg-void)', borderRadius: 3, overflow: 'hidden' }}>
                  <div className="confidence-fill" style={{ width: `${(fact.confidence || 0) * 100}%`, height: '100%' }} />
                </div>
                {((fact.confidence || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <div className="info-row"><span className="label">Source</span><span className="value badge">{fact.source}</span></div>
            <div className="info-row"><span className="label">Channel</span><span className="value">{fact.channel}</span></div>
            <div className="info-row"><span className="label">Created</span><span className="value">{formatTimestamp(fact.created_at)}</span></div>
          </div>

          <div className="detail-actions">
            <Button variant="secondary" onClick={() => setEditMode(true)}>
              <Edit2 size={14} /> Edit
            </Button>
            <Button variant="ghost" onClick={handleRemember} disabled={remembering}>
              <Star size={14} /> {remembered ? 'Remembered' : 'Remember'}
            </Button>
            <Button variant="ghost" onClick={handleProvenance} disabled={loadingProvenance}>
              <History size={14} /> Source
            </Button>
            <Button variant="secondary" onClick={() => setForgetConfirm(true)} disabled={forgetting}>
              <Archive size={14} /> Forget
            </Button>
            <Button variant="danger" onClick={() => setDeleteConfirm(true)} disabled={deleting}>
              <Trash2 size={14} /> Delete
            </Button>
          </div>

          {provenance && (
            <div className="provenance-panel">
              {provenance.origin ? (
                <>
                  <div className="info-row">
                    <span className="label">Learned in</span>
                    <span className="value">{formatTimestamp(provenance.origin.timestamp)}</span>
                  </div>
                  <p className="provenance-snippet">“{provenance.origin.snippet}”</p>
                </>
              ) : (
                <p className="provenance-empty">No recorded source conversation for this fact.</p>
              )}
            </div>
          )}

          {forgetConfirm && (
            <div className="delete-confirm">
              <AlertTriangle size={14} />
              <span>Forget this fact? It’s retired from recall but kept for provenance.</span>
              <Button variant="secondary" onClick={handleForget} disabled={forgetting}>
                {forgetting ? 'Forgetting...' : <><Check size={14} /> Confirm</>}
              </Button>
              <Button variant="ghost" onClick={() => setForgetConfirm(false)}>Cancel</Button>
            </div>
          )}

          {deleteConfirm && (
            <div className="delete-confirm">
              <AlertTriangle size={14} />
              <span>Delete this fact?</span>
              <Button variant="danger" onClick={handleDelete} disabled={deleting}>
                {deleting ? 'Deleting...' : <><Check size={14} /> Confirm</>}
              </Button>
              <Button variant="ghost" onClick={() => setDeleteConfirm(false)}>Cancel</Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

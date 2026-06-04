// ─── Fact detail panel ─────────────────────────────────────────────────────

import { useState, useCallback, useEffect } from 'react';
import { RefreshCw, X, ArrowUpRight, Edit2, Trash2, Check, AlertTriangle, Star, Archive, History, Link2, Plus, Search } from 'lucide-react';
import {
  useUpdateMemoryFact,
  useDeleteMemoryFact,
  useRememberMemoryFact,
  useForgetMemoryFact,
  useFactProvenance,
  useLinkFactEntity,
  useUnlinkFactEntity,
  useMemoryEntities,
} from '../../lib/hooks';
import type { MemoryFact, MemoryFactEntity, FactProvenance } from '../../lib/api';
import { formatTimestamp } from './formatTimestamp';
import { Button } from '../ui';

// Searchable entity picker — mounts only while the user is adding a link, so
// the entities fetch doesn't fire on every fact selection. Scoped to the fact's
// channel (the API also folds in _global); excludes already-linked entities.
function EntityPicker({
  channel, exclude, onPick, onCancel, busy,
}: {
  channel: string;
  exclude: Set<string>;
  onPick: (entityId: string) => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const [search, setSearch] = useState('');
  const { entities, loading } = useMemoryEntities(channel, 1, search);
  const candidates = entities.filter(e => !exclude.has(e.id)).slice(0, 8);

  return (
    <div className="entity-picker">
      <div className="entity-picker-search">
        <Search size={14} />
        <input
          autoFocus
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search entities to link…"
        />
        <Button variant="ghost" onClick={onCancel}><X size={14} /></Button>
      </div>
      <div className="entity-picker-results">
        {loading ? (
          <div className="entity-picker-empty"><RefreshCw size={14} className="spin" /></div>
        ) : candidates.length === 0 ? (
          <div className="entity-picker-empty">No matching entities</div>
        ) : candidates.map(e => (
          <button
            key={e.id}
            className="entity-picker-item"
            disabled={busy}
            onClick={() => onPick(e.id)}
          >
            <span className="entity-chip-name">{e.name}</span>
            <span className="entity-chip-type">{e.type}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function FactDetail({
  fact, onClose, onDeleted, onUpdated, onNavigateEntity,
}: {
  fact: MemoryFact;
  onClose: () => void;
  onDeleted: () => void;
  onUpdated: (f: MemoryFact) => void;
  onNavigateEntity: (entityId: string) => void;
}) {
  const { mutate: updateFact, loading: saving } = useUpdateMemoryFact();
  const { mutate: deleteFact, loading: deleting } = useDeleteMemoryFact();
  const { mutate: rememberFact, loading: remembering } = useRememberMemoryFact();
  const { mutate: forgetFact, loading: forgetting } = useForgetMemoryFact();
  const { fetch: fetchProvenance, loading: loadingProvenance } = useFactProvenance();
  const { mutate: linkEntity, loading: linking } = useLinkFactEntity();
  const { mutate: unlinkEntity, loading: unlinking } = useUnlinkFactEntity();

  const [editMode, setEditMode] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [forgetConfirm, setForgetConfirm] = useState(false);
  const [remembered, setRemembered] = useState(false);
  const [provenance, setProvenance] = useState<FactProvenance | null>(null);
  const [picking, setPicking] = useState(false);

  const entities: MemoryFactEntity[] = fact.entities ?? [];
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
    setPicking(false);
  }, [fact.id]);

  // Link/unlink update the fact's entity list; the endpoint returns the fresh
  // list, so we optimistically push it back up via onUpdated (no single-fact GET).
  const applyEntities = useCallback((next: MemoryFactEntity[]) => {
    onUpdated({ ...fact, entities: next, entity_ids: next.map(e => e.id) });
  }, [fact, onUpdated]);

  const handleLink = useCallback(async (entityId: string) => {
    const next = await linkEntity(fact.id, entityId);
    if (next) { applyEntities(next); setPicking(false); }
  }, [fact.id, linkEntity, applyEntities]);

  const handleUnlink = useCallback(async (entityId: string) => {
    const next = await unlinkEntity(fact.id, entityId);
    if (next) applyEntities(next);
  }, [fact.id, unlinkEntity, applyEntities]);

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

          <div className="mentioned-entities">
            <div className="mentioned-entities-head">
              <span className="label"><Link2 size={13} /> Mentioned entities</span>
              {!picking && (
                <Button variant="ghost" onClick={() => setPicking(true)} disabled={linking}>
                  <Plus size={14} /> Link
                </Button>
              )}
            </div>
            {entities.length === 0 && !picking ? (
              <p className="mentioned-entities-empty">No entities linked to this fact.</p>
            ) : (
              <div className="entity-chips">
                {entities.map(e => (
                  <span key={e.id} className="entity-chip">
                    <button
                      className="entity-chip-link"
                      onClick={() => onNavigateEntity(e.id)}
                      title={`View ${e.name}`}
                    >
                      <span className="entity-chip-name">{e.name}</span>
                      <span className="entity-chip-type">{e.type}</span>
                    </button>
                    <button
                      className="entity-chip-remove"
                      onClick={() => handleUnlink(e.id)}
                      disabled={unlinking}
                      title="Unlink"
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            {picking && (
              <EntityPicker
                channel={fact.channel}
                exclude={new Set(entities.map(e => e.id))}
                onPick={handleLink}
                onCancel={() => setPicking(false)}
                busy={linking}
              />
            )}
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

// ─── Entity detail panel ───────────────────────────────────────────────────

import { useState, useCallback, useEffect } from 'react';
import { RefreshCw, X, Edit2, Trash2, Check, AlertTriangle } from 'lucide-react';
import { useEntityGraph, useUpdateMemoryEntity, useDeleteMemoryEntity } from '../../lib/hooks';
import { Button, Input, Textarea } from '../ui';

export function EntityDetail({
  entityId, onClose, onDeleted, onRefreshList,
}: {
  entityId: string;
  onClose: () => void;
  onDeleted: () => void;
  onRefreshList: () => void;
}) {
  const { graph, loading, error } = useEntityGraph(entityId);
  const { mutate: updateEntity, loading: saving } = useUpdateMemoryEntity();
  const { mutate: deleteEntity, loading: deleting } = useDeleteMemoryEntity();

  const [editMode, setEditMode] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [editName, setEditName] = useState('');
  const [editType, setEditType] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editAliases, setEditAliases] = useState('');

  useEffect(() => {
    if (graph?.entity) {
      setEditName(graph.entity.name);
      setEditType(graph.entity.type);
      setEditDescription(graph.entity.description || '');
      setEditAliases((graph.entity.aliases || []).join(', '));
      setEditMode(false);
      setDeleteConfirm(false);
    }
  }, [graph?.entity?.id]);

  const handleSave = useCallback(async () => {
    if (!graph?.entity) return;
    const result = await updateEntity(entityId, {
      name: editName.trim(),
      type: editType.trim(),
      description: editDescription.trim() || null,
      aliases: editAliases.split(',').map(s => s.trim()).filter(Boolean),
    });
    if (result) {
      setEditMode(false);
      onRefreshList();
    }
  }, [entityId, editName, editType, editDescription, editAliases, updateEntity, onRefreshList, graph?.entity]);

  const handleDelete = useCallback(async () => {
    const ok = await deleteEntity(entityId);
    if (ok) onDeleted();
  }, [entityId, deleteEntity, onDeleted]);

  if (loading) return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>Entity Details</h3>
        <Button variant="ghost" onClick={onClose}><X size={18} /></Button>
      </div>
      <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading...</p></div>
    </div>
  );

  if (error || !graph?.entity) return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>Entity Details</h3>
        <Button variant="ghost" onClick={onClose}><X size={18} /></Button>
      </div>
      <div className="memory-error"><p>Failed to load entity details</p></div>
    </div>
  );

  const { entity, facts, relationships } = graph;

  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>{editMode ? 'Edit Entity' : entity.name}</h3>
        <Button variant="ghost" onClick={onClose}><X size={18} /></Button>
      </div>

      {editMode ? (
        <div className="edit-form">
          <div>
            <label>Name</label>
            <Input value={editName} onChange={e => setEditName(e.target.value)} />
          </div>
          <div>
            <label>Type</label>
            <Input value={editType} onChange={e => setEditType(e.target.value)} />
          </div>
          <div>
            <label>Description</label>
            <Textarea value={editDescription} onChange={e => setEditDescription(e.target.value)} rows={3} />
          </div>
          <div>
            <label>Aliases (comma-separated)</label>
            <Input value={editAliases} onChange={e => setEditAliases(e.target.value)} placeholder="alias1, alias2" />
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
          <div className="entity-info">
            <div className="info-row"><span className="label">Type</span><span className="value badge">{entity.type}</span></div>
            <div className="info-row"><span className="label">Channel</span><span className="value">{entity.channel}</span></div>
            <div className="info-row">
              <span className="label">Salience</span>
              <span className="value">{((entity.salience || 0) * 100).toFixed(0)}%</span>
            </div>
            {entity.description && (
              <div className="info-row"><span className="label">Description</span><span className="value">{entity.description}</span></div>
            )}
            {entity.aliases && entity.aliases.length > 0 && (
              <div className="info-row">
                <span className="label">Aliases</span>
                <span className="value">
                  <div className="tool-sequence">
                    {entity.aliases.map((a, i) => <span key={i} className="tool-chip">{a}</span>)}
                  </div>
                </span>
              </div>
            )}
          </div>

          <div className="detail-actions">
            <Button variant="secondary" onClick={() => setEditMode(true)}>
              <Edit2 size={14} /> Edit
            </Button>
            <Button variant="danger" onClick={() => setDeleteConfirm(true)} disabled={deleting}>
              <Trash2 size={14} /> Delete
            </Button>
          </div>

          {deleteConfirm && (
            <div className="delete-confirm">
              <AlertTriangle size={14} />
              <span>Delete this entity?</span>
              <Button variant="danger" onClick={handleDelete} disabled={deleting}>
                {deleting ? 'Deleting...' : <><Check size={14} /> Confirm</>}
              </Button>
              <Button variant="ghost" onClick={() => setDeleteConfirm(false)}>Cancel</Button>
            </div>
          )}

          {facts.length > 0 && (
            <div className="entity-section">
              <h4>Connected Facts ({facts.length})</h4>
              <div className="connected-facts">
                {facts.map(fact => (
                  <div key={fact.id} className="connected-fact">
                    <span className="fact-text">{fact.claim}</span>
                    <span className="fact-meta">{((fact.confidence || 0) * 100).toFixed(0)}% confidence</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {relationships.length > 0 && (
            <div className="entity-section">
              <h4>Relationships ({relationships.length})</h4>
              <div className="entity-relationships">
                {relationships.map((rel, i) => (
                  <div key={i} className="relationship">
                    <span className="rel-type">{rel.type}</span>
                    <span className="rel-arrow">→</span>
                    <span className="rel-target">
                      <span className="target-name">{rel.target.name}</span>
                      <span className="target-type badge">{rel.target.type}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

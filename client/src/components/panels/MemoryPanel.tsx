/**
 * Memory Explorer — Split-pane data browser with inline edit/delete and graph visualization.
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Database, Users, FileText, Zap, Search, RefreshCw,
  ChevronRight, X, ArrowUpRight, ChevronLeft, Clock,
  Edit2, Trash2, Check, AlertTriangle, GitBranch
} from 'lucide-react';
import {
  ReactFlow, Background, Controls, Handle, Position,
  useNodesState, useEdgesState,
  type NodeProps, type Node, type Edge
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  useMemoryEntities, useMemoryFacts, useMemoryStrategies,
  useMemoryStats, useEntityGraph, useConsolidate,
  useUpdateMemoryFact, useDeleteMemoryFact,
  useUpdateMemoryEntity, useDeleteMemoryEntity
} from '../../lib/hooks';
import type { MemoryEntity, MemoryFact, MemoryStrategy, ApiError } from '../../lib/api';
import { JobsPanel } from '../JobsPanel';
import '../../styles/MemoryPanel.css';

type MemorySection = 'entities' | 'facts' | 'strategies' | 'graph' | 'jobs';

function formatTimestamp(timestamp: string | undefined): string {
  if (!timestamp) return 'Never';
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  } catch { return 'Unknown'; }
}

// ─── React Flow — custom entity node (module-level to avoid re-creation) ────

type EntityNodeData = { label: string; typeLabel: string; isCenter: boolean } & Record<string, unknown>;

function EntityNodeComponent({ data }: NodeProps) {
  const d = data as EntityNodeData;
  return (
    <div className={`graph-node${d.isCenter ? ' center' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <span className="graph-node-name">{d.label}</span>
      <span className="graph-node-type">{d.typeLabel}</span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { entityNode: EntityNodeComponent };

// ─── List Views ──────────────────────────────────────────────────────────────

function EntityListView({
  entities, total, loading, error, selectedEntityId, onSelectEntity,
}: {
  entities: MemoryEntity[];
  total: number;
  loading: boolean;
  error: ApiError | null;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
}) {
  if (loading) return <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading entities...</p></div>;
  if (error) return <div className="memory-error"><p>Failed to load entities: {error.message}</p></div>;
  if (entities.length === 0) return <div className="memory-empty"><Users size={32} /><p>No entities found</p></div>;

  return (
    <div className="memory-list">
      <div className="memory-list-header">
        <span>Name</span><span>Type</span><span>Channel</span><span>Salience</span><span>Last Accessed</span>
      </div>
      {entities.map(entity => (
        <div
          key={entity.id}
          className={`memory-row${selectedEntityId === entity.id ? ' selected' : ''}`}
          onClick={() => onSelectEntity(selectedEntityId === entity.id ? null : entity.id)}
        >
          <span className="entity-name">{entity.name}</span>
          <span className="entity-type badge">{entity.type}</span>
          <span className="entity-channel">{entity.channel}</span>
          <span className="entity-salience">
            <div className="salience-bar">
              <div className="salience-fill" style={{ width: `${(entity.salience || 0) * 100}%` }} />
            </div>
            <span>{((entity.salience || 0) * 100).toFixed(0)}%</span>
          </span>
          <span className="entity-accessed">{formatTimestamp(entity.last_accessed)}</span>
        </div>
      ))}
      <div className="memory-list-footer">Showing {entities.length} of {total} entities</div>
    </div>
  );
}

function FactListView({
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

function StrategyListView({
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

// ─── Detail Panels ────────────────────────────────────────────────────────────

function EntityDetail({
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
        <button className="button-ghost" onClick={onClose}><X size={18} /></button>
      </div>
      <div className="memory-loading"><RefreshCw size={24} className="spin" /><p>Loading...</p></div>
    </div>
  );

  if (error || !graph?.entity) return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>Entity Details</h3>
        <button className="button-ghost" onClick={onClose}><X size={18} /></button>
      </div>
      <div className="memory-error"><p>Failed to load entity details</p></div>
    </div>
  );

  const { entity, facts, relationships } = graph;

  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>{editMode ? 'Edit Entity' : entity.name}</h3>
        <button className="button-ghost" onClick={onClose}><X size={18} /></button>
      </div>

      {editMode ? (
        <div className="edit-form">
          <div>
            <label>Name</label>
            <input value={editName} onChange={e => setEditName(e.target.value)} />
          </div>
          <div>
            <label>Type</label>
            <input value={editType} onChange={e => setEditType(e.target.value)} />
          </div>
          <div>
            <label>Description</label>
            <textarea value={editDescription} onChange={e => setEditDescription(e.target.value)} rows={3} />
          </div>
          <div>
            <label>Aliases (comma-separated)</label>
            <input value={editAliases} onChange={e => setEditAliases(e.target.value)} placeholder="alias1, alias2" />
          </div>
          <div className="edit-actions">
            <button className="button-primary" onClick={handleSave} disabled={saving}>
              {saving ? <><RefreshCw size={14} className="spin" /> Saving...</> : <><Check size={14} /> Save</>}
            </button>
            <button className="button-ghost" onClick={() => setEditMode(false)} disabled={saving}>Cancel</button>
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
            <button className="button-secondary" onClick={() => setEditMode(true)}>
              <Edit2 size={14} /> Edit
            </button>
            <button className="button-secondary danger" onClick={() => setDeleteConfirm(true)} disabled={deleting}>
              <Trash2 size={14} /> Delete
            </button>
          </div>

          {deleteConfirm && (
            <div className="delete-confirm">
              <AlertTriangle size={14} />
              <span>Delete this entity?</span>
              <button className="button-secondary danger" onClick={handleDelete} disabled={deleting}>
                {deleting ? 'Deleting...' : <><Check size={14} /> Confirm</>}
              </button>
              <button className="button-ghost" onClick={() => setDeleteConfirm(false)}>Cancel</button>
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

function FactDetail({
  fact, onClose, onDeleted, onUpdated,
}: {
  fact: MemoryFact;
  onClose: () => void;
  onDeleted: () => void;
  onUpdated: (f: MemoryFact) => void;
}) {
  const { mutate: updateFact, loading: saving } = useUpdateMemoryFact();
  const { mutate: deleteFact, loading: deleting } = useDeleteMemoryFact();

  const [editMode, setEditMode] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
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

  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>{editMode ? 'Edit Fact' : 'Fact Detail'}</h3>
        <button className="button-ghost" onClick={onClose}><X size={18} /></button>
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
            <button className="button-primary" onClick={handleSave} disabled={saving}>
              {saving ? <><RefreshCw size={14} className="spin" /> Saving...</> : <><Check size={14} /> Save</>}
            </button>
            <button className="button-ghost" onClick={() => setEditMode(false)} disabled={saving}>Cancel</button>
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
            <button className="button-secondary" onClick={() => setEditMode(true)}>
              <Edit2 size={14} /> Edit
            </button>
            <button className="button-secondary danger" onClick={() => setDeleteConfirm(true)} disabled={deleting}>
              <Trash2 size={14} /> Delete
            </button>
          </div>

          {deleteConfirm && (
            <div className="delete-confirm">
              <AlertTriangle size={14} />
              <span>Delete this fact?</span>
              <button className="button-secondary danger" onClick={handleDelete} disabled={deleting}>
                {deleting ? 'Deleting...' : <><Check size={14} /> Confirm</>}
              </button>
              <button className="button-ghost" onClick={() => setDeleteConfirm(false)}>Cancel</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StrategyDetail({
  strategy, onClose,
}: {
  strategy: MemoryStrategy;
  onClose: () => void;
}) {
  return (
    <div className="split-detail-inner">
      <div className="detail-header">
        <h3>Strategy</h3>
        <button className="button-ghost" onClick={onClose}><X size={18} /></button>
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

// ─── Graph View ───────────────────────────────────────────────────────────────

function MemoryGraphView() {
  const [graphEntityId, setGraphEntityId] = useState<string | null>(null);
  const { entities, loading: entitiesLoading } = useMemoryEntities('_all', 1, '');
  const { graph, loading: graphLoading } = useEntityGraph(graphEntityId);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    if (!graph) {
      setNodes([]);
      setEdges([]);
      return;
    }
    const n = Math.max(graph.relationships.length, 1);
    const center: Node = {
      id: graph.entity.id,
      type: 'entityNode',
      data: { label: graph.entity.name, typeLabel: graph.entity.type, isCenter: true },
      position: { x: 0, y: 0 },
    };
    const neighbors: Node[] = graph.relationships.map((rel, i) => ({
      id: rel.target.id,
      type: 'entityNode',
      data: { label: rel.target.name, typeLabel: rel.target.type, isCenter: false },
      position: {
        x: Math.cos((i * 2 * Math.PI) / n) * 260,
        y: Math.sin((i * 2 * Math.PI) / n) * 260,
      },
    }));
    const newEdges: Edge[] = graph.relationships.map((rel, i) => ({
      id: `e-${i}`,
      source: graph.entity.id,
      target: rel.target.id,
      label: rel.type,
      animated: true,
    }));
    setNodes([center, ...neighbors]);
    setEdges(newEdges);
  }, [graph]);

  return (
    <div className="graph-view">
      <div className="graph-entity-list">
        <div className="graph-list-header">Entities</div>
        {entitiesLoading ? (
          <div className="graph-list-loading"><RefreshCw size={16} className="spin" /></div>
        ) : entities.map(e => (
          <button
            key={e.id}
            className={`graph-entity-btn${graphEntityId === e.id ? ' active' : ''}`}
            onClick={() => setGraphEntityId(graphEntityId === e.id ? null : e.id)}
          >
            <span className="graph-btn-name">{e.name}</span>
            <span className="graph-btn-type">{e.type}</span>
          </button>
        ))}
      </div>
      <div className="graph-canvas">
        {!graphEntityId ? (
          <div className="graph-empty">
            <GitBranch size={32} />
            <p>Select an entity to explore its relationships</p>
          </div>
        ) : graphLoading ? (
          <div className="graph-empty"><RefreshCw size={24} className="spin" /></div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
        )}
      </div>
    </div>
  );
}

// ─── Pagination ───────────────────────────────────────────────────────────────

function Pagination({
  page, hasNext, onPageChange,
}: {
  page: number;
  hasNext: boolean;
  onPageChange: (p: number) => void;
}) {
  return (
    <div className="pagination">
      <button className="button-ghost" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
        <ChevronLeft size={16} />Previous
      </button>
      <span className="page-info">Page {page}</span>
      <button className="button-ghost" disabled={!hasNext} onClick={() => onPageChange(page + 1)}>
        Next<ChevronRight size={16} />
      </button>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export const MemoryPanel: React.FC = () => {
  const [activeSection, setActiveSection] = useState<MemorySection>('entities');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [confidenceFilter, setConfidenceFilter] = useState(0);

  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedFact, setSelectedFact] = useState<MemoryFact | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<MemoryStrategy | null>(null);

  const [consolidateMessage, setConsolidateMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  const { stats, loading: statsLoading, refresh: refreshStats } = useMemoryStats();
  const { consolidate, loading: consolidating } = useConsolidate();

  const {
    entities, total: entitiesTotal, hasNext: entitiesHasNext,
    loading: entitiesLoading, error: entitiesError, refresh: refreshEntities,
  } = useMemoryEntities('_all', currentPage, searchQuery);

  const {
    facts, total: factsTotal, hasNext: factsHasNext,
    loading: factsLoading, error: factsError, refresh: refreshFacts,
  } = useMemoryFacts('_all', currentPage, confidenceFilter / 100, searchQuery);

  const {
    strategies, total: strategiesTotal, hasNext: strategiesHasNext,
    loading: strategiesLoading, error: strategiesError,
  } = useMemoryStrategies('_all', currentPage);

  const hasNext = useMemo(() => {
    switch (activeSection) {
      case 'entities': return entitiesHasNext ?? false;
      case 'facts': return factsHasNext ?? false;
      case 'strategies': return strategiesHasNext ?? false;
      default: return false;
    }
  }, [activeSection, entitiesHasNext, factsHasNext, strategiesHasNext]);

  const selectedId = selectedEntityId ?? selectedFact?.id ?? selectedStrategy?.id ?? null;

  const memorySections = [
    { id: 'entities' as const, label: 'Entities', icon: <Users size={18} /> },
    { id: 'facts' as const, label: 'Facts', icon: <FileText size={18} /> },
    { id: 'strategies' as const, label: 'Strategies', icon: <Zap size={18} /> },
    { id: 'graph' as const, label: 'Graph', icon: <GitBranch size={18} /> },
    { id: 'jobs' as const, label: 'Jobs', icon: <Clock size={18} /> },
  ];

  const handleSectionChange = (section: MemorySection) => {
    setActiveSection(section);
    setCurrentPage(1);
    setSelectedEntityId(null);
    setSelectedFact(null);
    setSelectedStrategy(null);
    setSearchQuery('');
  };

  const handleConsolidate = async () => {
    try {
      const result = await consolidate();
      const totalEntities = result.results?.consolidate?.entities ?? 0;
      const totalFacts = result.results?.consolidate?.facts ?? 0;
      const totalRelationships = result.results?.consolidate?.relationships ?? 0;
      setConsolidateMessage({
        type: 'success',
        text: `Extracted ${totalEntities} entities, ${totalFacts} facts, ${totalRelationships} relationships`,
      });
      refreshStats();
      setTimeout(() => setConsolidateMessage(null), 5000);
    } catch (err) {
      setConsolidateMessage({ type: 'error', text: `Consolidation failed: ${(err as Error).message}` });
      setTimeout(() => setConsolidateMessage(null), 5000);
    }
  };

  return (
    <div className="memory-tab">
      {/* Header */}
      <div className="memory-header fade-in">
        <div className="header-title-row">
          <h1 className="page-title">
            <Database className="page-icon-svg" />
            <span>Memory Explorer</span>
          </h1>
          <div className="header-actions">
            <button
              className="button-primary consolidate-button"
              onClick={handleConsolidate}
              disabled={consolidating}
              title="Run consolidation to extract entities and facts"
            >
              {consolidating
                ? <><RefreshCw size={16} className="spin" /> Consolidating...</>
                : <><Zap size={16} /> Consolidate Now</>}
            </button>
            <button className="button-ghost" onClick={refreshStats} disabled={statsLoading}>
              <RefreshCw size={18} className={statsLoading ? 'spin' : ''} />
            </button>
          </div>
        </div>
        <p className="page-subtitle">Browse and inspect stored memories</p>

        {consolidateMessage && (
          <div className={`consolidate-message ${consolidateMessage.type}`}>
            {consolidateMessage.type === 'success' ? <Zap size={16} /> : <X size={16} />}
            <span>{consolidateMessage.text}</span>
            <button className="dismiss-btn" onClick={() => setConsolidateMessage(null)}><X size={14} /></button>
          </div>
        )}

        <div className="memory-stats-bar">
          <span className="stat-badge"><Users size={14} /> {stats?.totals.entities ?? 0} Entities</span>
          <span className="stat-badge"><FileText size={14} /> {stats?.totals.facts ?? 0} Facts</span>
          <span className="stat-badge"><Zap size={14} /> {stats?.totals.strategies ?? 0} Strategies</span>
        </div>
      </div>

      <div className="memory-layout">
        {/* Sidebar Navigation */}
        <nav className="memory-nav card">
          {memorySections.map(section => (
            <button
              key={section.id}
              className={`nav-item${activeSection === section.id ? ' active' : ''}`}
              onClick={() => handleSectionChange(section.id)}
            >
              <span className="nav-icon">{section.icon}</span>
              <span className="nav-label">{section.label}</span>
              <ChevronRight size={16} className="nav-arrow" />
            </button>
          ))}
        </nav>

        {/* Content Area */}
        <div className="memory-content">
          {activeSection === 'jobs' ? (
            <div className="memory-list-container card"><JobsPanel /></div>
          ) : activeSection === 'graph' ? (
            <div className="memory-list-container card full-height"><MemoryGraphView /></div>
          ) : (
            <div className={`memory-split${selectedId ? ' detail-open' : ''}`}>
              {/* Left: filters + list + pagination */}
              <div className="split-list">
                <div className="memory-filters card">
                  <div className="filter-group search always-visible">
                    <Search size={16} />
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchQuery}
                      onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                    />
                  </div>
                  {activeSection === 'facts' && (
                    <div className="filter-group confidence">
                      <label>Min Confidence: {confidenceFilter}%</label>
                      <input
                        type="range" min="0" max="100" value={confidenceFilter}
                        onChange={e => { setConfidenceFilter(Number(e.target.value)); setCurrentPage(1); }}
                      />
                    </div>
                  )}
                </div>

                <div className="memory-list-container card">
                  {activeSection === 'entities' && (
                    <EntityListView
                      entities={entities}
                      total={entitiesTotal ?? 0}
                      loading={entitiesLoading}
                      error={entitiesError}
                      selectedEntityId={selectedEntityId}
                      onSelectEntity={setSelectedEntityId}
                    />
                  )}
                  {activeSection === 'facts' && (
                    <FactListView
                      facts={facts}
                      total={factsTotal ?? 0}
                      loading={factsLoading}
                      error={factsError}
                      selectedFactId={selectedFact?.id ?? null}
                      onSelectFact={setSelectedFact}
                    />
                  )}
                  {activeSection === 'strategies' && (
                    <StrategyListView
                      strategies={strategies}
                      total={strategiesTotal ?? 0}
                      loading={strategiesLoading}
                      error={strategiesError}
                      selectedStrategyId={selectedStrategy?.id ?? null}
                      onSelectStrategy={setSelectedStrategy}
                    />
                  )}
                </div>

                <Pagination page={currentPage} hasNext={hasNext} onPageChange={setCurrentPage} />
              </div>

              {/* Right: detail panel */}
              <div className={`split-detail${selectedId ? ' is-open' : ''}`}>
                {activeSection === 'entities' && selectedEntityId ? (
                  <EntityDetail
                    entityId={selectedEntityId}
                    onClose={() => setSelectedEntityId(null)}
                    onDeleted={() => { setSelectedEntityId(null); refreshEntities(); }}
                    onRefreshList={refreshEntities}
                  />
                ) : activeSection === 'facts' && selectedFact ? (
                  <FactDetail
                    fact={selectedFact}
                    onClose={() => setSelectedFact(null)}
                    onDeleted={() => { setSelectedFact(null); refreshFacts(); }}
                    onUpdated={updated => { setSelectedFact(updated); refreshFacts(); }}
                  />
                ) : activeSection === 'strategies' && selectedStrategy ? (
                  <StrategyDetail
                    strategy={selectedStrategy}
                    onClose={() => setSelectedStrategy(null)}
                  />
                ) : (
                  <div className="detail-placeholder">
                    <ChevronRight size={32} />
                    <p>Select an item to view details</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MemoryPanel;

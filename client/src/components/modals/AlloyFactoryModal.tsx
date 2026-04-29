/**
 * AlloyFactoryModal — Manage Agent Alloy workflows.
 *
 * Full-screen layout: workflow list in the left sidebar, editor on the right.
 * v1 is form-based; the visual Factory canvas is on the roadmap and will read
 * and write the same workflow records (workflow.canvas blob).
 */

import { useEffect, useMemo, useState } from 'react';
import {
  Workflow as WorkflowIcon,
  X,
  Plus,
  Trash2,
  Save,
  Sparkles,
} from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import type {
  AgentProfile,
  AlloyWorkflow,
  AlloyWorkflowMember,
} from '../../lib/api';
import './AlloyFactoryModal.css';

interface AlloyFactoryModalProps {
  onClose: () => void;
  editWorkflowId?: string;
  isNew?: boolean;
}

// 'list' = empty state, 'edit-<id>' = editing existing, 'new' = creating
type Selection = { kind: 'empty' } | { kind: 'edit'; id: string } | { kind: 'new' };

function slugify(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}

interface SpecialistDraft {
  agentId: string;
  enabled: boolean;
  delegationHint: string;
}

export function AlloyFactoryModal({ onClose, editWorkflowId, isNew }: AlloyFactoryModalProps) {
  const { profiles } = useAgentProfile();
  const {
    workflows,
    createWorkflow,
    updateWorkflow,
    deleteWorkflow,
    getWorkflowById,
  } = useAlloyWorkflow();

  const [selection, setSelection] = useState<Selection>(
    editWorkflowId ? { kind: 'edit', id: editWorkflowId } :
    isNew ? { kind: 'new' } :
    { kind: 'empty' }
  );

  const profilesByAgentId = useMemo(() => {
    const m = new Map<string, AgentProfile>();
    for (const p of profiles) m.set(p.agentId, p);
    return m;
  }, [profiles]);

  const handleDelete = async (id: string) => {
    if (!confirm(`Delete workflow "${id}"? This cannot be undone.`)) return;
    await deleteWorkflow(id);
    setSelection(prev => (prev.kind === 'edit' && prev.id === id ? { kind: 'empty' } : prev));
  };

  const editingWorkflow =
    selection.kind === 'edit' ? getWorkflowById(selection.id) : null;

  return (
    <div className="alloy-factory-modal full">
      <div className="alloy-header">
        <div className="alloy-title-group">
          <div className="alloy-title-icon">
            <WorkflowIcon size={18} />
          </div>
          <div>
            <h2>Agent Alloy — Factory</h2>
            <div className="alloy-subtitle">
              Compose multi-agent workflows with a supervisor and specialists
            </div>
          </div>
        </div>
        <button type="button" className="alloy-close-btn" onClick={onClose} title="Close">
          <X size={18} />
        </button>
      </div>

      <div className="alloy-shell">
        {/* Sidebar */}
        <aside className="alloy-sidebar">
          <button
            className="alloy-btn alloy-btn-primary alloy-sidebar-new"
            onClick={() => setSelection({ kind: 'new' })}
          >
            <Plus size={14} />
            New workflow
          </button>

          {workflows.length === 0 ? (
            <div className="alloy-sidebar-empty">No workflows yet.</div>
          ) : (
            <div className="alloy-sidebar-list">
              {workflows.map(w => {
                const supervisor = profilesByAgentId.get(w.supervisorAgentId);
                const specialistCount = w.members.filter(m => m.role === 'specialist').length;
                const isActive = selection.kind === 'edit' && selection.id === w.id;
                return (
                  <button
                    key={w.id}
                    className={`alloy-sidebar-item ${isActive ? 'active' : ''}`}
                    onClick={() => setSelection({ kind: 'edit', id: w.id })}
                  >
                    <div className="sidebar-item-icon">
                      <WorkflowIcon size={14} />
                    </div>
                    <div className="sidebar-item-text">
                      <div className="sidebar-item-name">{w.name}</div>
                      <div className="sidebar-item-meta">
                        {supervisor?.name ?? w.supervisorAgentId} · {specialistCount} specialist{specialistCount === 1 ? '' : 's'}
                      </div>
                    </div>
                    <span
                      className="sidebar-item-delete"
                      role="button"
                      tabIndex={0}
                      title="Delete workflow"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(w.id);
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          e.stopPropagation();
                          handleDelete(w.id);
                        }
                      }}
                    >
                      <Trash2 size={12} />
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </aside>

        {/* Main pane */}
        <main className="alloy-pane">
          <div className="alloy-canvas-banner">
            <Sparkles size={16} className="banner-icon" />
            <div>
              <strong>Factory Canvas — coming soon.</strong> A visual node graph for
              designing delegation flows is on the roadmap. Workflows you save here
              will load directly into the canvas when it ships.
            </div>
          </div>

          {selection.kind === 'empty' ? (
            <EmptyState onNew={() => setSelection({ kind: 'new' })} />
          ) : (
            <WorkflowEditorView
              key={selection.kind === 'edit' ? `edit-${selection.id}` : 'new'}
              initial={editingWorkflow}
              profiles={profiles}
              existingIds={new Set(workflows.map(w => w.id))}
              onSubmit={async (payload, isUpdate) => {
                if (isUpdate && selection.kind === 'edit') {
                  await updateWorkflow(selection.id, {
                    name: payload.name,
                    description: payload.description,
                    supervisorAgentId: payload.supervisorAgentId,
                    members: payload.members,
                  });
                } else {
                  const created = await createWorkflow({
                    id: payload.id,
                    name: payload.name,
                    description: payload.description,
                    supervisorAgentId: payload.supervisorAgentId,
                    members: payload.members,
                  });
                  setSelection({ kind: 'edit', id: created.id });
                }
              }}
            />
          )}
        </main>
      </div>
    </div>
  );
}

// ---------- empty state ----------

function EmptyState({ onNew }: { onNew: () => void }) {
  return (
    <div className="alloy-empty-state">
      <WorkflowIcon size={48} />
      <h3>No workflow selected</h3>
      <p>Pick a workflow from the sidebar to edit it, or create a new one to get started.</p>
      <button className="alloy-btn alloy-btn-primary" onClick={onNew}>
        <Plus size={14} />
        Create new workflow
      </button>
    </div>
  );
}

// ---------- editor view ----------

interface EditorPayload {
  id: string;
  name: string;
  description?: string;
  supervisorAgentId: string;
  members: AlloyWorkflowMember[];
}

interface EditorProps {
  initial: AlloyWorkflow | null;
  profiles: AgentProfile[];
  existingIds: Set<string>;
  onSubmit: (payload: EditorPayload, isUpdate: boolean) => Promise<void>;
}

function WorkflowEditorView({ initial, profiles, existingIds, onSubmit }: EditorProps) {
  const isEditing = initial !== null;

  const [name, setName] = useState(initial?.name ?? '');
  const [id, setId] = useState(initial?.id ?? '');
  const [idDirty, setIdDirty] = useState(isEditing);
  const [description, setDescription] = useState(initial?.description ?? '');
  const [supervisorId, setSupervisorId] = useState<string>(initial?.supervisorAgentId ?? '');

  const [specialistDrafts, setSpecialistDrafts] = useState<Map<string, SpecialistDraft>>(() => {
    const m = new Map<string, SpecialistDraft>();
    if (initial) {
      for (const member of initial.members) {
        if (member.role === 'specialist') {
          m.set(member.agentId, {
            agentId: member.agentId,
            enabled: true,
            delegationHint: member.delegationHint ?? '',
          });
        }
      }
    }
    return m;
  });

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-slug id from name until the user touches the id field
  useEffect(() => {
    if (!idDirty && !isEditing) {
      setId(slugify(name));
    }
  }, [name, idDirty, isEditing]);

  const toggleSpecialist = (agentId: string) => {
    setSpecialistDrafts(prev => {
      const next = new Map(prev);
      const existing = next.get(agentId);
      if (existing) {
        next.set(agentId, { ...existing, enabled: !existing.enabled });
      } else {
        next.set(agentId, { agentId, enabled: true, delegationHint: '' });
      }
      return next;
    });
  };

  const setHint = (agentId: string, hint: string) => {
    setSpecialistDrafts(prev => {
      const next = new Map(prev);
      const existing = next.get(agentId) ?? { agentId, enabled: true, delegationHint: '' };
      next.set(agentId, { ...existing, delegationHint: hint });
      return next;
    });
  };

  const handleSave = async () => {
    setError(null);

    if (!name.trim()) {
      setError('Name is required.');
      return;
    }
    if (!id.trim()) {
      setError('ID is required.');
      return;
    }
    if (!/^[a-z0-9][a-z0-9-]*$/.test(id)) {
      setError('ID must be lowercase letters, numbers, or hyphens (e.g. "research-team").');
      return;
    }
    if (!isEditing && existingIds.has(id)) {
      setError(`A workflow with id "${id}" already exists.`);
      return;
    }
    if (!supervisorId) {
      setError('Pick a supervisor agent.');
      return;
    }

    const enabledSpecialists = [...specialistDrafts.values()]
      .filter(s => s.enabled && s.agentId !== supervisorId);

    const members: AlloyWorkflowMember[] = [
      { agentId: supervisorId, role: 'supervisor' },
      ...enabledSpecialists.map(s => ({
        agentId: s.agentId,
        role: 'specialist' as const,
        delegationHint: s.delegationHint.trim() || undefined,
      })),
    ];

    setSaving(true);
    try {
      await onSubmit(
        {
          id,
          name: name.trim(),
          description: description.trim() || undefined,
          supervisorAgentId: supervisorId,
          members,
        },
        isEditing
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save workflow.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="alloy-form">
      {error && <div className="alloy-error">{error}</div>}

      <div className="alloy-form-grid">
        <div className="alloy-form-row">
          <label htmlFor="alloy-name">Name</label>
          <input
            id="alloy-name"
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Research Team"
          />
        </div>

        <div className="alloy-form-row">
          <label htmlFor="alloy-id">ID</label>
          <input
            id="alloy-id"
            type="text"
            value={id}
            onChange={e => {
              setIdDirty(true);
              setId(e.target.value);
            }}
            disabled={isEditing}
            placeholder="research-team"
          />
          <span className="form-hint">Lowercase letters, digits, hyphens.</span>
        </div>
      </div>

      <div className="alloy-form-row">
        <label htmlFor="alloy-desc">Description (optional)</label>
        <textarea
          id="alloy-desc"
          rows={2}
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="What this workflow is for"
        />
      </div>

      <div className="alloy-form-row">
        <label htmlFor="alloy-supervisor">Supervisor</label>
        <select
          id="alloy-supervisor"
          value={supervisorId}
          onChange={e => setSupervisorId(e.target.value)}
        >
          <option value="">— Pick an agent —</option>
          {profiles.map(p => (
            <option key={p.agentId} value={p.agentId}>
              {p.name} ({p.agentId})
            </option>
          ))}
        </select>
        <span className="form-hint">
          The supervisor owns the conversation and decides when to delegate.
          The supervisor <strong>must</strong> be a high-quality model that's
          tested in multi-agent orchestration; actionable results aren't
          guaranteed otherwise.
        </span>
      </div>

      <div className="alloy-form-row">
        <label>Specialists</label>
        <span className="form-hint">
          Toggle which agents the supervisor can delegate to. Hints are passed
          to the supervisor so it can choose the right specialist for each task.
        </span>
        <div className="alloy-specialists">
          {profiles.length === 0 && (
            <div className="alloy-list-empty">No agent profiles yet. Create one first.</div>
          )}
          {profiles.map(p => {
            const isSupervisor = p.agentId === supervisorId;
            const draft = specialistDrafts.get(p.agentId);
            const enabled = !isSupervisor && (draft?.enabled ?? false);
            return (
              <div
                key={p.agentId}
                className={`alloy-specialist-row ${isSupervisor ? 'is-supervisor' : ''} ${enabled ? 'enabled' : ''}`}
              >
                <label className="specialist-toggle">
                  <input
                    type="checkbox"
                    checked={enabled}
                    disabled={isSupervisor}
                    onChange={() => toggleSpecialist(p.agentId)}
                    aria-label={`Include ${p.name} as specialist`}
                  />
                  <div className="specialist-info">
                    <div className="specialist-name">
                      {p.name}
                      {isSupervisor && <span className="specialist-badge">supervisor</span>}
                    </div>
                    <div className="specialist-id">{p.agentId}</div>
                  </div>
                </label>
                <input
                  type="text"
                  className="specialist-hint"
                  placeholder={enabled ? 'When to delegate to this agent…' : 'Enable to add a hint'}
                  value={draft?.delegationHint ?? ''}
                  disabled={!enabled}
                  onChange={e => setHint(p.agentId, e.target.value)}
                />
              </div>
            );
          })}
        </div>
      </div>

      <div className="alloy-actions">
        <span className="form-hint">
          {isEditing
            ? 'Changes apply to new conversations using this workflow.'
            : 'You can edit specialists and hints any time.'}
        </span>
        <div className="alloy-actions-right">
          <button
            className="alloy-btn alloy-btn-primary"
            onClick={handleSave}
            disabled={saving}
          >
            <Save size={14} />
            {saving ? 'Saving…' : isEditing ? 'Save changes' : 'Create workflow'}
          </button>
        </div>
      </div>
    </div>
  );
}

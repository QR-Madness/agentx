/**
 * AlloyFactoryModal — Manage Agent Alloy workflows.
 * User-facing name: "Agent Teams" (internal: Alloy) — precedent: Workspaces→Projects.
 * UI terminology: workflow → Team, supervisor → Lead, specialists → Members.
 *
 * Desktop: team list in the left sidebar, editor on the right.
 * Mobile: master-detail — the team list, then tap through to a full-width
 * editor with a Back button (the sidebar/pane don't co-exist on a phone).
 *
 * Members are built additively: pick a Lead, then "+ Add member" opens a
 * searchable picker (dropdown on desktop, bottom sheet on mobile) of agents
 * not already on the team. Each member gets an editable delegation hint whose
 * placeholder is the agent's own profile Specialty — leaving it blank falls
 * back to that specialty server-side (see alloy/delegation_tool.py).
 *
 * v1 is form-based; the visual Factory canvas is on the roadmap and will read
 * and write the same workflow records (workflow.canvas blob).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Workflow as WorkflowIcon,
  X,
  Plus,
  Trash2,
  Save,
  Sparkles,
  ChevronLeft,
  Search,
  AlertCircle,
} from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { apiErrorMessage } from '../../lib/api';
import type {
  AgentProfile,
  AlloyWorkflow,
  AlloyWorkflowMember,
} from '../../lib/api';
import { useConfirm } from '../ui/ConfirmDialog';
import { Input, Textarea } from '../ui/Field';
import { Button } from '../ui/Button';
import { IconButton } from '../ui/IconButton';
import { DropdownPortal } from '../ui/DropdownPortal';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/Select';
import { useIsMobile } from '../../lib/hooks';
import './AlloyFactoryModal.css';

interface AlloyFactoryModalProps {
  onClose: () => void;
  editWorkflowId?: string;
  isNew?: boolean;
}

// 'empty' = no team selected, 'edit' = editing existing, 'new' = creating
type Selection = { kind: 'empty' } | { kind: 'edit'; id: string } | { kind: 'new' };

function slugify(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}

function initialLetter(name: string): string {
  return (name.trim()[0] || '?').toUpperCase();
}

/** A team member the lead can delegate to (ordered, insertion order preserved). */
interface MemberDraft {
  agentId: string;
  delegationHint: string;
}

export function AlloyFactoryModal({ onClose, editWorkflowId, isNew }: AlloyFactoryModalProps) {
  const confirm = useConfirm();
  const isMobile = useIsMobile();
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
    const ok = await confirm({
      title: `Delete team "${id}"?`,
      body: 'This cannot be undone.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    await deleteWorkflow(id);
    setSelection(prev => (prev.kind === 'edit' && prev.id === id ? { kind: 'empty' } : prev));
  };

  const editingWorkflow =
    selection.kind === 'edit' ? getWorkflowById(selection.id) : null;

  // Master-detail on mobile: the list and the editor never co-exist. Desktop
  // shows both columns at once.
  const showList = !isMobile || selection.kind === 'empty';
  const showEditor = !isMobile || selection.kind !== 'empty';

  return (
    <div className="alloy-factory-modal full">
      <div className="alloy-header">
        <div className="alloy-title-group">
          <div className="alloy-title-icon">
            <WorkflowIcon size={18} />
          </div>
          <div>
            <h2>Agent Teams</h2>
            <div className="alloy-subtitle">
              Compose teams of agents — a lead who delegates to members
            </div>
          </div>
        </div>
        <button type="button" className="alloy-close-btn" onClick={onClose} title="Close">
          <X size={18} />
        </button>
      </div>

      <div className={`alloy-shell${isMobile ? ' is-mobile' : ''}`}>
        {/* Sidebar / team list */}
        {showList && (
          <aside className="alloy-sidebar">
            <Button
              variant="primary"
              size="sm"
              className="alloy-sidebar-new"
              onClick={() => setSelection({ kind: 'new' })}
            >
              <Plus size={14} />
              New team
            </Button>

            {workflows.length === 0 ? (
              <div className="alloy-sidebar-empty">No teams yet.</div>
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
                          {supervisor?.name ?? w.supervisorAgentId} · {specialistCount} member{specialistCount === 1 ? '' : 's'}
                        </div>
                      </div>
                      <span
                        className="sidebar-item-delete"
                        role="button"
                        tabIndex={0}
                        title="Delete team"
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
        )}

        {/* Main pane / editor */}
        {showEditor && (
          <main className="alloy-pane">
            {isMobile && (
              <button
                type="button"
                className="alloy-back-btn"
                onClick={() => setSelection({ kind: 'empty' })}
              >
                <ChevronLeft size={16} />
                Teams
              </button>
            )}

            {!isMobile && (
              <div className="alloy-canvas-banner">
                <Sparkles size={16} className="banner-icon" />
                <div>
                  <strong>Factory Canvas — coming soon.</strong> A visual node graph for
                  designing delegation flows is on the roadmap. Teams you save here
                  will load directly into the canvas when it ships.
                </div>
              </div>
            )}

            {selection.kind === 'empty' ? (
              <EmptyState onNew={() => setSelection({ kind: 'new' })} />
            ) : (
              <WorkflowEditorView
                key={selection.kind === 'edit' ? `edit-${selection.id}` : 'new'}
                initial={editingWorkflow}
                profiles={profiles}
                profilesByAgentId={profilesByAgentId}
                existingIds={new Set(workflows.map(w => w.id))}
                onSubmit={async (payload, isUpdate) => {
                  if (isUpdate && selection.kind === 'edit') {
                    await updateWorkflow(selection.id, {
                      name: payload.name,
                      description: payload.description,
                      supervisorAgentId: payload.supervisorAgentId,
                      managerAgentId: payload.managerAgentId,
                      members: payload.members,
                    });
                  } else {
                    const created = await createWorkflow({
                      id: payload.id,
                      name: payload.name,
                      description: payload.description,
                      supervisorAgentId: payload.supervisorAgentId,
                      managerAgentId: payload.managerAgentId,
                      members: payload.members,
                    });
                    setSelection({ kind: 'edit', id: created.id });
                  }
                }}
              />
            )}
          </main>
        )}
      </div>
    </div>
  );
}

// ---------- empty state ----------

function EmptyState({ onNew }: { onNew: () => void }) {
  return (
    <div className="alloy-empty-state">
      <WorkflowIcon size={48} />
      <h3>No team selected</h3>
      <p>Pick a team from the list to edit it, or create a new one to get started.</p>
      <Button variant="primary" onClick={onNew}>
        <Plus size={14} />
        Create new team
      </Button>
    </div>
  );
}

// ---------- editor view ----------

interface EditorPayload {
  id: string;
  name: string;
  description?: string;
  supervisorAgentId: string;
  managerAgentId?: string | null;
  members: AlloyWorkflowMember[];
}

interface EditorProps {
  initial: AlloyWorkflow | null;
  profiles: AgentProfile[];
  profilesByAgentId: Map<string, AgentProfile>;
  existingIds: Set<string>;
  onSubmit: (payload: EditorPayload, isUpdate: boolean) => Promise<void>;
}

function WorkflowEditorView({ initial, profiles, profilesByAgentId, existingIds, onSubmit }: EditorProps) {
  const isEditing = initial !== null;

  const [name, setName] = useState(initial?.name ?? '');
  const [id, setId] = useState(initial?.id ?? '');
  const [idDirty, setIdDirty] = useState(isEditing);
  const [description, setDescription] = useState(initial?.description ?? '');
  const [supervisorId, setSupervisorId] = useState<string>(initial?.supervisorAgentId ?? '');
  // Agentic Organizations: the manager that owns this team ('' = org-free team).
  const [managerAgentId, setManagerAgentId] = useState<string>(initial?.managerAgentId ?? '');

  // Ordered team members (specialists). Seed from the workflow, preserving order.
  const [members, setMembers] = useState<MemberDraft[]>(() =>
    (initial?.members ?? [])
      .filter(m => m.role === 'specialist')
      .map(m => ({ agentId: m.agentId, delegationHint: m.delegationHint ?? '' }))
  );

  const [pickerOpen, setPickerOpen] = useState(false);
  const addBtnRef = useRef<HTMLButtonElement>(null);

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-slug id from name until the user touches the id field
  useEffect(() => {
    if (!idDirty && !isEditing) {
      setId(slugify(name));
    }
  }, [name, idDirty, isEditing]);

  // Agents eligible to be added: real agents (never ambassadors), not the lead,
  // not the owning manager, not already on the team.
  const candidates = useMemo(
    () =>
      profiles.filter(
        p =>
          p.kind === 'agent' &&
          p.agentId !== supervisorId &&
          p.agentId !== managerAgentId &&
          !members.some(m => m.agentId === p.agentId)
      ),
    [profiles, supervisorId, managerAgentId, members]
  );

  // Leads must be real agents (ambassadors are never chat/delegation agents).
  const leadOptions = useMemo(() => profiles.filter(p => p.kind === 'agent'), [profiles]);

  // Manager candidates: agents that aren't the lead or a member (server hard
  // rules mirrored); manager-tier profiles listed first.
  const managerOptions = useMemo(
    () =>
      profiles
        .filter(
          p =>
            p.kind === 'agent' &&
            p.agentId !== supervisorId &&
            !members.some(m => m.agentId === p.agentId)
        )
        .sort((a, b) =>
          Number(b.orgLevel === 'manager') - Number(a.orgLevel === 'manager')
        ),
    [profiles, supervisorId, members]
  );

  const addMember = (agentId: string) => {
    setMembers(prev =>
      prev.some(m => m.agentId === agentId) ? prev : [...prev, { agentId, delegationHint: '' }]
    );
  };

  const removeMember = (agentId: string) => {
    setMembers(prev => prev.filter(m => m.agentId !== agentId));
  };

  const setHint = (agentId: string, hint: string) => {
    setMembers(prev => prev.map(m => (m.agentId === agentId ? { ...m, delegationHint: hint } : m)));
  };

  // Picking a lead that's currently a member removes it from the roster — an
  // agent can't be both the lead and one of its own members (and never the
  // team's manager either).
  const handleLeadChange = (value: string) => {
    setSupervisorId(value);
    setMembers(prev => prev.filter(m => m.agentId !== value));
    if (value && value === managerAgentId) setManagerAgentId('');
  };

  // Radix SelectItem can't carry value="" — sentinel for "no manager".
  const NO_MANAGER = '__none__';
  const handleManagerChange = (value: string) => {
    setManagerAgentId(value === NO_MANAGER ? '' : value);
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
      setError(`A team with id "${id}" already exists.`);
      return;
    }
    if (!supervisorId) {
      setError('Pick a lead agent.');
      return;
    }

    const memberList: AlloyWorkflowMember[] = [
      { agentId: supervisorId, role: 'supervisor' },
      ...members
        .filter(m => m.agentId !== supervisorId)
        .map(m => ({
          agentId: m.agentId,
          role: 'specialist' as const,
          delegationHint: m.delegationHint.trim() || undefined,
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
          // null (not undefined) so a PATCH can clear the manager (org-free).
          managerAgentId: managerAgentId || null,
          members: memberList,
        },
        isEditing
      );
    } catch (e) {
      setError(apiErrorMessage(e) || 'Failed to save team.');
    } finally {
      setSaving(false);
    }
  };

  const poolExhausted = candidates.length === 0;

  return (
    <div className="alloy-form">
      {error && (
        <div className="alloy-error">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <div className="alloy-form-grid">
        <div className="alloy-form-row">
          <label htmlFor="alloy-name">Name</label>
          <Input
            id="alloy-name"
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Research Team"
          />
        </div>

        <div className="alloy-form-row">
          <label htmlFor="alloy-id">ID</label>
          <Input
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
        <Textarea
          id="alloy-desc"
          rows={2}
          value={description}
          onChange={e => setDescription(e.target.value)}
          placeholder="What this team is for"
        />
      </div>

      <div className="alloy-form-row">
        <label>Lead</label>
        <Select value={supervisorId} onValueChange={handleLeadChange}>
          <SelectTrigger aria-label="Lead agent">
            <SelectValue placeholder="— Pick an agent —" />
          </SelectTrigger>
          <SelectContent>
            {leadOptions.map(p => (
              <SelectItem key={p.agentId} value={p.agentId}>
                {p.name} ({p.agentId})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="form-hint">
          The lead owns the conversation and decides when to delegate.
          The lead <strong>must</strong> be a high-quality model that's
          tested in multi-agent orchestration; actionable results aren't
          guaranteed otherwise.
        </span>
      </div>

      <div className="alloy-form-row">
        <label>Manager (optional)</label>
        <Select value={managerAgentId || NO_MANAGER} onValueChange={handleManagerChange}>
          <SelectTrigger aria-label="Owning manager">
            <SelectValue placeholder="— No manager (org-free team) —" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NO_MANAGER}>No manager — org-free team</SelectItem>
            {managerOptions.map(p => (
              <SelectItem key={p.agentId} value={p.agentId}>
                {p.name} ({p.agentId}){p.orgLevel === 'manager' ? ' · manager' : ''}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="form-hint">
          Setting a manager puts this team in the organization: the manager
          delegates to the lead, the lead works the members, and the chain of
          command replaces the flat roster for everyone on the team.
        </span>
      </div>

      <div className="alloy-form-row">
        <div className="alloy-members-head">
          <span className="alloy-eyebrow">Members ({members.length})</span>
          <button
            ref={addBtnRef}
            type="button"
            className="alloy-add-btn"
            disabled={poolExhausted}
            onClick={() => setPickerOpen(true)}
          >
            <Plus size={14} />
            Add member
          </button>
        </div>
        <span className="form-hint">
          The lead can delegate to these agents. Hints tell it which member fits each task.
        </span>

        {members.length === 0 ? (
          <div className="alloy-list-empty">
            {profiles.length === 0
              ? 'No agent profiles yet. Create one first.'
              : 'No members yet — add agents the lead can delegate to.'}
          </div>
        ) : (
          <div className="alloy-members">
            {members.map(m => {
              const profile = profilesByAgentId.get(m.agentId);
              const specialty = profile?.delegationHint?.trim() || '';
              const usingFallback = !m.delegationHint.trim() && !!specialty;
              return (
                <div key={m.agentId} className="alloy-member-card">
                  <div className="alloy-member-top">
                    <span className="alloy-avatar" aria-hidden>{initialLetter(profile?.name ?? m.agentId)}</span>
                    <div className="alloy-member-id-group">
                      <div className="alloy-member-name">{profile?.name ?? m.agentId}</div>
                      <div className="alloy-member-id">{m.agentId}</div>
                    </div>
                    <IconButton
                      size="sm"
                      tone="danger"
                      aria-label={`Remove ${profile?.name ?? m.agentId}`}
                      onClick={() => removeMember(m.agentId)}
                    >
                      <Trash2 size={14} />
                    </IconButton>
                  </div>
                  {!profile && (
                    <div className="alloy-member-warn">
                      <AlertCircle size={12} />
                      Profile missing — this agent no longer exists.
                    </div>
                  )}
                  <Textarea
                    className="alloy-member-hint"
                    rows={2}
                    maxLength={200}
                    placeholder={specialty || 'When to delegate to this agent…'}
                    value={m.delegationHint}
                    onChange={e => setHint(m.agentId, e.target.value)}
                  />
                  {usingFallback && (
                    <span className="form-hint">Blank → uses this agent's profile specialty.</span>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {poolExhausted && members.length > 0 && (
          <span className="form-hint">All agents added.</span>
        )}

        <AddMemberPicker
          isOpen={pickerOpen}
          onClose={() => setPickerOpen(false)}
          anchorRef={addBtnRef}
          candidates={candidates}
          onPick={addMember}
        />
      </div>

      <div className="alloy-actions">
        <span className="form-hint">
          {isEditing
            ? 'Changes apply to new conversations using this team.'
            : 'You can edit members and hints any time.'}
        </span>
        <div className="alloy-actions-right">
          <Button variant="primary" onClick={handleSave} loading={saving}>
            <Save size={14} />
            {saving ? 'Saving…' : isEditing ? 'Save changes' : 'Create team'}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------- add-member picker ----------

interface AddMemberPickerProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  candidates: AgentProfile[];
  onPick: (agentId: string) => void;
}

/**
 * Searchable "add a teammate" picker. Desktop = anchored dropdown; mobile =
 * thumb-reachable bottom sheet (mirrors RelayMenu). Only offers agents not
 * already on the team (the caller filters `candidates`).
 */
function AddMemberPicker({ isOpen, onClose, anchorRef, candidates, onPick }: AddMemberPickerProps) {
  const isMobile = useIsMobile();
  const [query, setQuery] = useState('');

  useEffect(() => {
    if (!isOpen) setQuery('');
  }, [isOpen]);

  // Escape closes only the picker — not the whole Team Builder modal. The modal
  // shell (ModalDialog) listens on `document` in the bubble phase, so we grab
  // Escape first on `window` in the capture phase and stop it there. This one
  // handler covers both the mobile sheet and the desktop dropdown (and pre-empts
  // DropdownPortal's own Escape, which would otherwise let the modal close too).
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.stopImmediatePropagation();
      e.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [isOpen, onClose]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return candidates;
    return candidates.filter(
      p => p.name.toLowerCase().includes(q) || p.agentId.toLowerCase().includes(q)
    );
  }, [candidates, query]);

  const body = (
    <div
      className={`alloy-picker${isMobile ? ' alloy-picker--sheet' : ''}`}
      role="dialog"
      aria-label="Add member"
    >
      <div className="alloy-picker-header">
        <Search size={14} className="alloy-picker-search-icon" />
        <input
          autoFocus
          className="alloy-picker-search"
          placeholder="Search agents…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <button className="alloy-picker-close" onClick={onClose} aria-label="Close">
          <X size={14} />
        </button>
      </div>
      <div className="alloy-picker-list">
        {filtered.length === 0 ? (
          <div className="alloy-picker-empty">
            {candidates.length === 0 ? 'All agents are already on the team.' : 'No matching agents.'}
          </div>
        ) : (
          filtered.map(p => (
            <button
              key={p.agentId}
              type="button"
              className="alloy-picker-item"
              onClick={() => { onPick(p.agentId); onClose(); }}
            >
              <span className="alloy-avatar" aria-hidden>{initialLetter(p.name)}</span>
              <span className="alloy-picker-item-body">
                <span className="alloy-picker-item-name">{p.name}</span>
                <span className="alloy-picker-item-id">{p.agentId}</span>
                {p.delegationHint && (
                  <span className="alloy-picker-item-hint">{p.delegationHint}</span>
                )}
              </span>
              <Plus size={15} className="alloy-picker-item-add" />
            </button>
          ))
        )}
      </div>
    </div>
  );

  if (isMobile) {
    if (!isOpen) return null;
    return createPortal(
      <div className="alloy-sheet-backdrop" onClick={onClose}>
        <div onClick={e => e.stopPropagation()}>{body}</div>
      </div>,
      document.body,
    );
  }

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="bottom"
      align="end"
      estimatedHeight={360}
    >
      {body}
    </DropdownPortal>
  );
}

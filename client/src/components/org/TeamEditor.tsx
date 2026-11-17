/**
 * TeamEditor — create or edit one team (internally an Alloy workflow) in the
 * Organization dock. Terminology: workflow → Team, supervisor → Lead,
 * specialists → Members.
 *
 * Members are built additively: pick a Lead, then "Add member" opens an inline
 * search list of the agents not already on the team, each with where it
 * already sits. Every member has an editable delegation hint whose placeholder
 * is the agent's own profile Specialty — left blank, the server falls back to
 * that specialty (alloy/delegation_tool.py).
 *
 * Saving is explicit. The editor reports whether it holds unsaved edits so the
 * surface can ask before discarding them.
 */

import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, Plus, Save, Search, Trash2, X } from 'lucide-react';
import { apiErrorMessage } from '../../lib/api';
import type { AgentProfile, AlloyWorkflow, AlloyWorkflowMember } from '../../lib/api';
import { affiliations, type OrgModel } from '../../lib/orgChart';
import { AgentAvatar } from '../common/AgentAvatar';
import { Button } from '../ui/Button';
import { Input, Textarea } from '../ui/Field';
import { IconButton } from '../ui/IconButton';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/Select';

export interface TeamPayload {
  id: string;
  name: string;
  description?: string;
  supervisorAgentId: string;
  managerAgentId: string | null;
  members: AlloyWorkflowMember[];
}

interface TeamEditorProps {
  /** The team being edited, or null to create one. */
  team: AlloyWorkflow | null;
  /** Manager to pre-select on a new team (created from a manager's panel). */
  defaultManagerAgentId?: string;
  profiles: AgentProfile[];
  model: OrgModel;
  existingIds: Set<string>;
  onSave: (payload: TeamPayload, isUpdate: boolean) => Promise<void>;
  /** Existing teams only. */
  onDelete?: () => void;
  onDirtyChange: (dirty: boolean) => void;
  onOpenAgent: (agentId: string) => void;
}

interface MemberDraft {
  agentId: string;
  delegationHint: string;
}

// Radix SelectItem can't carry value="" — sentinel for "no manager".
const NO_MANAGER = '__none__';

function slugify(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}

function initialMembers(team: AlloyWorkflow | null): MemberDraft[] {
  return (team?.members ?? [])
    .filter((m) => m.role === 'specialist')
    .map((m) => ({ agentId: m.agentId, delegationHint: m.delegationHint ?? '' }));
}

export function TeamEditor({
  team,
  defaultManagerAgentId,
  profiles,
  model,
  existingIds,
  onSave,
  onDelete,
  onDirtyChange,
  onOpenAgent,
}: TeamEditorProps) {
  const isEditing = team !== null;

  const [name, setName] = useState(team?.name ?? '');
  const [id, setId] = useState(team?.id ?? '');
  const [idTouched, setIdTouched] = useState(isEditing);
  const [description, setDescription] = useState(team?.description ?? '');
  const [leadId, setLeadId] = useState(team?.supervisorAgentId ?? '');
  // '' = a team outside the organization (no manager).
  const [managerId, setManagerId] = useState(team?.managerAgentId ?? defaultManagerAgentId ?? '');
  const [members, setMembers] = useState<MemberDraft[]>(() => initialMembers(team));

  const [pickerOpen, setPickerOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const profilesById = useMemo(() => {
    const map = new Map<string, AgentProfile>();
    for (const p of profiles) map.set(p.agentId, p);
    return map;
  }, [profiles]);

  // Unsaved edits = the draft differs from what the team (or a blank form) holds.
  const baseline = useMemo(
    () =>
      JSON.stringify({
        name: team?.name ?? '',
        id: team?.id ?? '',
        description: team?.description ?? '',
        leadId: team?.supervisorAgentId ?? '',
        managerId: team?.managerAgentId ?? defaultManagerAgentId ?? '',
        members: initialMembers(team),
      }),
    [team, defaultManagerAgentId],
  );
  // A new team's id is derived from its name until touched, so on its own it is not an edit.
  const draftId = isEditing ? team.id : idTouched ? id : '';
  const dirty =
    JSON.stringify({ name, id: draftId, description, leadId, managerId, members }) !== baseline;

  useEffect(() => {
    onDirtyChange(dirty);
  }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);

  // Derive the id from the name until the id field is edited by hand.
  useEffect(() => {
    if (!idTouched && !isEditing) setId(slugify(name));
  }, [name, idTouched, isEditing]);

  const agents = useMemo(() => profiles.filter((p) => p.kind === 'agent'), [profiles]);

  // Eligible members: agents that are not the lead, the manager, or already on the team.
  const candidates = useMemo(
    () =>
      agents.filter(
        (p) =>
          p.agentId !== leadId &&
          p.agentId !== managerId &&
          !members.some((m) => m.agentId === p.agentId),
      ),
    [agents, leadId, managerId, members],
  );

  // Manager candidates mirror the server's hard rules (not the lead, not a
  // member); profiles already holding the Manager role come first.
  const managerOptions = useMemo(
    () =>
      agents
        .filter((p) => p.agentId !== leadId && !members.some((m) => m.agentId === p.agentId))
        .sort((a, b) => Number(b.orgLevel === 'manager') - Number(a.orgLevel === 'manager')),
    [agents, leadId, members],
  );

  const optionNote = (p: AgentProfile): string => affiliations(p.agentId, model, team?.id).join(' · ');

  // An agent can't be both the lead and one of its own members, nor the manager.
  const changeLead = (value: string) => {
    setLeadId(value);
    setMembers((prev) => prev.filter((m) => m.agentId !== value));
    if (value && value === managerId) setManagerId('');
  };

  const addMember = (agentId: string) => {
    setMembers((prev) =>
      prev.some((m) => m.agentId === agentId) ? prev : [...prev, { agentId, delegationHint: '' }],
    );
  };
  const removeMember = (agentId: string) => setMembers((prev) => prev.filter((m) => m.agentId !== agentId));
  const setHint = (agentId: string, hint: string) =>
    setMembers((prev) => prev.map((m) => (m.agentId === agentId ? { ...m, delegationHint: hint } : m)));

  const save = async () => {
    setError(null);
    const teamId = isEditing ? team.id : id.trim();
    if (!name.trim()) return setError('Name is required.');
    if (!teamId) return setError('ID is required.');
    if (!/^[a-z0-9][a-z0-9-]*$/.test(teamId)) {
      return setError('ID must be lowercase letters, numbers, or hyphens (e.g. "research-team").');
    }
    if (!isEditing && existingIds.has(teamId)) return setError(`A team with id "${teamId}" already exists.`);
    if (!leadId) return setError('Pick a lead agent.');

    const memberList: AlloyWorkflowMember[] = [
      { agentId: leadId, role: 'supervisor' },
      ...members
        .filter((m) => m.agentId !== leadId)
        .map((m) => ({
          agentId: m.agentId,
          role: 'specialist' as const,
          delegationHint: m.delegationHint.trim() || undefined,
        })),
    ];

    setSaving(true);
    try {
      await onSave(
        {
          id: teamId,
          name: name.trim(),
          description: description.trim() || undefined,
          supervisorAgentId: leadId,
          // null (not undefined) so a PATCH can clear the manager.
          managerAgentId: managerId || null,
          members: memberList,
        },
        isEditing,
      );
    } catch (e) {
      setError(apiErrorMessage(e) || 'Failed to save the team.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="org-form">
      {error && (
        <div className="org-form-error" role="alert">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <div className="org-form-row">
        <label htmlFor="org-team-name">Name</label>
        <Input
          id="org-team-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Research"
        />
      </div>

      {/* An existing team's id never changes; the dock shows it under the title. */}
      {!isEditing && (
        <div className="org-form-row">
          <label htmlFor="org-team-id">ID</label>
          <Input
            id="org-team-id"
            type="text"
            value={id}
            onChange={(e) => {
              setIdTouched(true);
              setId(e.target.value);
            }}
            placeholder="research"
          />
          <span className="org-form-hint">Permanent once created: lowercase letters, digits, hyphens.</span>
        </div>
      )}

      <div className="org-form-row">
        <label htmlFor="org-team-desc">Description (optional)</label>
        <Textarea
          id="org-team-desc"
          rows={2}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What this team is for"
        />
      </div>

      <div className="org-form-row">
        <label>Lead</label>
        <Select value={leadId} onValueChange={changeLead}>
          <SelectTrigger aria-label="Lead agent">
            {/* The trigger shows the name alone; where each agent sits is in the list. */}
            <SelectValue placeholder="Pick an agent">{profilesById.get(leadId)?.name ?? leadId}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {agents.map((p) => (
              <SelectItem key={p.agentId} value={p.agentId}>
                {p.name}
                {optionNote(p) && <span className="org-option-note"> · {optionNote(p)}</span>}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="org-form-hint">
          The lead owns the conversation and decides when to delegate. Give it a model you trust
          with multi-agent orchestration.
        </span>
      </div>

      <div className="org-form-row">
        <label>Manager (optional)</label>
        <Select value={managerId || NO_MANAGER} onValueChange={(v) => setManagerId(v === NO_MANAGER ? '' : v)}>
          <SelectTrigger aria-label="Owning manager">
            <SelectValue>
              {managerId ? (profilesById.get(managerId)?.name ?? managerId) : 'No manager — outside the organization'}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NO_MANAGER}>No manager — outside the organization</SelectItem>
            {managerOptions.map((p) => (
              <SelectItem key={p.agentId} value={p.agentId}>
                {p.name}
                <span className="org-option-note">
                  {p.orgLevel === 'manager' ? ' · Manager role' : ''}
                  {optionNote(p) ? ` · ${optionNote(p)}` : ''}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="org-form-hint">
          A manager puts this team in the organization: the manager delegates to the lead, the lead
          works the members, and the chain of command replaces the flat roster for everyone on it.
        </span>
      </div>

      <div className="org-form-row">
        <div className="org-members-head">
          <span className="org-eyebrow">Members ({members.length})</span>
          <Button
            variant="secondary"
            size="sm"
            disabled={candidates.length === 0}
            aria-expanded={pickerOpen}
            onClick={() => setPickerOpen((open) => !open)}
          >
            <Plus size={14} />
            Add member
          </Button>
        </div>
        <span className="org-form-hint">
          The lead can delegate to these agents. A hint tells it which member fits which task.
        </span>

        {pickerOpen && (
          <MemberPicker
            candidates={candidates}
            note={optionNote}
            onPick={addMember}
            onClose={() => setPickerOpen(false)}
          />
        )}

        {members.length === 0 ? (
          <div className="org-list-empty">
            {agents.length === 0
              ? 'No agent profiles yet. Create one first.'
              : 'No members yet. Add the agents the lead can delegate to.'}
          </div>
        ) : (
          <div className="org-members">
            {members.map((m) => {
              const profile = profilesById.get(m.agentId);
              const specialty = profile?.delegationHint?.trim() || '';
              const label = profile?.name ?? m.agentId;
              return (
                <div key={m.agentId} className="org-member-card">
                  <div className="org-member-top">
                    <span className="org-avatar" aria-hidden>
                      <AgentAvatar avatar={profile?.avatar} size={16} fill />
                    </span>
                    <div className="org-member-ident">
                      {profile ? (
                        <button type="button" className="org-member-name" onClick={() => onOpenAgent(m.agentId)}>
                          {label}
                        </button>
                      ) : (
                        <span className="org-member-name">{label}</span>
                      )}
                      <span className="org-member-id">{m.agentId}</span>
                    </div>
                    <IconButton size="sm" tone="danger" aria-label={`Remove ${label}`} onClick={() => removeMember(m.agentId)}>
                      <Trash2 size={14} />
                    </IconButton>
                  </div>
                  {!profile && (
                    <div className="org-member-warn">
                      <AlertCircle size={12} />
                      No agent has this id any more. Remove it, or it is skipped at run time.
                    </div>
                  )}
                  <Textarea
                    className="org-member-hint"
                    rows={2}
                    maxLength={200}
                    aria-label={`When to delegate to ${label}`}
                    placeholder={specialty || 'When to delegate to this agent…'}
                    value={m.delegationHint}
                    onChange={(e) => setHint(m.agentId, e.target.value)}
                  />
                  {!m.delegationHint.trim() && specialty && (
                    <span className="org-form-hint">Blank uses this agent's profile specialty.</span>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="org-form-actions">
        {onDelete ? (
          <Button variant="danger" size="sm" onClick={onDelete} disabled={saving}>
            <Trash2 size={14} />
            Delete team
          </Button>
        ) : (
          <span />
        )}
        <Button variant="primary" size="sm" onClick={save} loading={saving} disabled={isEditing && !dirty}>
          <Save size={14} />
          {saving ? 'Saving…' : isEditing ? 'Save changes' : 'Create team'}
        </Button>
      </div>
      {isEditing && <span className="org-form-hint">Changes apply to new conversations that use this team.</span>}
    </div>
  );
}

/** Inline search list of the agents that can still join the team. */
function MemberPicker({
  candidates,
  note,
  onPick,
  onClose,
}: {
  candidates: AgentProfile[];
  note: (p: AgentProfile) => string;
  onPick: (agentId: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return candidates;
    return candidates.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.agentId.toLowerCase().includes(q) ||
        (p.delegationHint ?? '').toLowerCase().includes(q),
    );
  }, [candidates, query]);

  return (
    <div className="org-picker" role="group" aria-label="Add member">
      <div className="org-picker-header">
        <Input
          autoFocus
          className="org-picker-search"
          icon={<Search size={14} />}
          placeholder="Search agents…"
          aria-label="Search agents to add"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            // Escape closes the picker only, not the whole surface.
            if (e.key === 'Escape') {
              e.preventDefault();
              e.stopPropagation();
              onClose();
            }
          }}
        />
        <IconButton size="sm" onClick={onClose} aria-label="Close the member list">
          <X size={14} />
        </IconButton>
      </div>
      <div className="org-picker-list">
        {filtered.length === 0 ? (
          <div className="org-picker-empty">
            {candidates.length === 0 ? 'Every agent is already on this team.' : 'No agent matches.'}
          </div>
        ) : (
          filtered.map((p) => (
            <button key={p.agentId} type="button" className="org-picker-item" onClick={() => onPick(p.agentId)}>
              <span className="org-avatar" aria-hidden>
                <AgentAvatar avatar={p.avatar} size={16} fill />
              </span>
              <span className="org-picker-item-body">
                <span className="org-picker-item-name">{p.name}</span>
                {note(p) && <span className="org-picker-item-note">{note(p)}</span>}
                {p.delegationHint && <span className="org-picker-item-hint">{p.delegationHint}</span>}
              </span>
              <Plus size={15} className="org-picker-item-add" />
            </button>
          ))
        )}
      </div>
    </div>
  );
}

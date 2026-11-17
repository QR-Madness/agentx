/**
 * AgentSelectorDropdown — pick the agent profile or Team (Alloy workflow) for the
 * active conversation.
 *
 * The list is org-aware: the Agents segment is grouped by the chain of command
 * (`lib/orgPlacement.groupRoster` — a crowned manager, then each team lead-first,
 * then Independent agents), and every row carries the info that used to be
 * invisible here (tier, friendly model, a one-line role). The Teams segment
 * groups workflows under their owning manager and shows each team's lead + size.
 * A footer link opens the full two-pane Roster (org chart + dossier).
 *
 * Layout: an [ Agents | Teams ] toggle over a single full-height scroll list, a
 * per-segment search, a context footer, and ↑/↓/Enter keyboard nav. Opens on
 * whichever segment is currently active.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot, Settings, Check, Plus, Workflow as WorkflowIcon, Crown, Search, Users, LayoutGrid,
} from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useConversation } from '../../contexts/ConversationContext';
import { useModal } from '../../contexts/ModalContext';
import { SURFACES } from '../../lib/surfaces';
import { groupRoster, type RosterGroup } from '../../lib/orgPlacement';
import { modelShortLabel } from '../../lib/modelLabel';
import type { AgentProfile, AlloyWorkflow } from '../../lib/api/types';
import { AgentAvatar } from '../common/AgentAvatar';
import { DropdownPortal } from '../ui/DropdownPortal';
import './AgentSelectorDropdown.css';

// Show a segment's search field only once its list is long enough to warrant it.
const SEARCH_THRESHOLD = 6;

type Segment = 'agents' | 'workflows';
type Tier = 'manager' | 'lead';

interface AgentSelectorDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
}

/** The one-line role shown under an agent — what you'd hand them. */
function roleLine(p: AgentProfile): string {
  return (p.delegationHint || p.description || '').trim();
}

function matchesQuery(p: AgentProfile, q: string): boolean {
  return (
    p.name.toLowerCase().includes(q) ||
    (p.defaultModel ?? '').toLowerCase().includes(q) ||
    roleLine(p).toLowerCase().includes(q) ||
    (p.tags ?? []).some(t => t.toLowerCase().includes(q))
  );
}

/** A team's one-line subtitle: who leads it and how big it is. */
function teamSubtitle(wf: AlloyWorkflow, nameOf: (agentId: string) => string | null): string {
  const lead = nameOf(wf.supervisorAgentId);
  const specialists = wf.members.filter(m => m.role === 'specialist').length;
  const size = `${specialists} specialist${specialists === 1 ? '' : 's'}`;
  return lead ? `Led by ${lead} · ${size}` : size;
}

export function AgentSelectorDropdown({ isOpen, onClose, anchorRef }: AgentSelectorDropdownProps) {
  const { profiles, activeProfile, setActiveProfile } = useAgentProfile();
  const { workflows } = useAlloyWorkflow();
  const { activeTab, setActiveTabWorkflow } = useConversation();
  const { openModal } = useModal();

  const activeWorkflowId = activeTab?.workflowId ?? null;
  const activeWorkflow = activeWorkflowId
    ? workflows.find(w => w.id === activeWorkflowId) ?? null
    : null;
  const supervisorAgentId = activeWorkflow?.supervisorAgentId ?? null;

  const [segment, setSegment] = useState<Segment>(activeWorkflowId ? 'workflows' : 'agents');
  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState(0);

  // On each open, land on the active context and reset transient UI state.
  useEffect(() => {
    if (!isOpen) return;
    setSegment(activeWorkflowId ? 'workflows' : 'agents');
    setQuery('');
    setHighlight(0);
  }, [isOpen, activeWorkflowId]);

  const switchSegment = (next: Segment) => {
    setSegment(next);
    setQuery('');
    setHighlight(0);
  };

  // agentId → display name, and agentId → declared tier (manager wins over lead).
  const nameByAgentId = useMemo(() => {
    const m = new Map<string, string>();
    for (const p of profiles) m.set(p.agentId, p.name);
    return m;
  }, [profiles]);
  const tierByAgentId = useMemo(() => {
    const m = new Map<string, Tier>();
    for (const wf of workflows) if (wf.managerAgentId) m.set(wf.managerAgentId, 'manager');
    for (const wf of workflows) if (!m.has(wf.supervisorAgentId)) m.set(wf.supervisorAgentId, 'lead');
    return m;
  }, [workflows]);

  // ---- agents (org-grouped) ----
  const agentsLocked = activeWorkflow !== null; // agent is dictated by the workflow
  const agentGroups = useMemo(() => {
    // Ambassadors never join a chat — drop that group entirely.
    const groups = groupRoster(profiles, workflows).filter(g => g.kind !== 'ambassador');
    const q = query.trim().toLowerCase();
    if (!q) return groups;
    return groups
      .map(g => ({ ...g, profiles: g.profiles.filter(p => matchesQuery(p, q)) }))
      .filter(g => g.profiles.length > 0);
  }, [profiles, workflows, query]);
  const flatAgents = useMemo(() => agentGroups.flatMap(g => g.profiles), [agentGroups]);

  // ---- workflows (grouped: org teams under their manager, then independent) ----
  const filteredWorkflows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return workflows;
    return workflows.filter(w =>
      w.name.toLowerCase().includes(q) ||
      (w.description ?? '').toLowerCase().includes(q),
    );
  }, [workflows, query]);
  const workflowGroups = useMemo(() => {
    const byManager = new Map<string, AlloyWorkflow[]>();
    const independent: AlloyWorkflow[] = [];
    for (const wf of filteredWorkflows) {
      if (wf.managerAgentId) {
        (byManager.get(wf.managerAgentId) ?? byManager.set(wf.managerAgentId, []).get(wf.managerAgentId)!).push(wf);
      } else independent.push(wf);
    }
    const groups: { label: string; managed: boolean; teams: AlloyWorkflow[] }[] = [];
    for (const [mgrId, teams] of byManager) {
      groups.push({ label: `${nameByAgentId.get(mgrId) ?? 'Manager'}’s org`, managed: true, teams });
    }
    if (independent.length) groups.push({ label: 'Independent teams', managed: false, teams: independent });
    return groups;
  }, [filteredWorkflows, nameByAgentId]);

  const handleSelect = (profileId: string) => { setActiveProfile(profileId); onClose(); };
  const handleSelectWorkflow = (workflowId: string | null) => { setActiveTabWorkflow(workflowId); onClose(); };

  const handleEdit = (e: React.MouseEvent, profileId: string) => {
    e.stopPropagation();
    openModal({ id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full', props: { initialProfileId: profileId } });
    onClose();
  };
  const handleCreateNew = () => {
    openModal({ id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor', size: 'full', props: { isNew: true } });
    onClose();
  };
  const handleManageWorkflows = () => { openModal(SURFACES.teams); onClose(); };
  const handleBrowseRoster = () => { openModal(SURFACES.roster); onClose(); };

  // Flat list of keyboard-selectable actions for the active segment.
  const items = useMemo<Array<() => void>>(() => {
    if (segment === 'agents') {
      if (agentsLocked) return [];
      return flatAgents.map(p => () => handleSelect(p.id));
    }
    return [() => handleSelectWorkflow(null), ...filteredWorkflows.map(w => () => handleSelectWorkflow(w.id))];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segment, agentsLocked, flatAgents, filteredWorkflows]);

  useEffect(() => { setHighlight(h => Math.min(h, Math.max(0, items.length - 1))); }, [items.length]);

  const showSearch = segment === 'agents'
    ? profiles.length > SEARCH_THRESHOLD
    : workflows.length > SEARCH_THRESHOLD;

  const searchRef = useRef<HTMLInputElement>(null);
  useEffect(() => { if (isOpen && showSearch) searchRef.current?.focus(); }, [isOpen, segment, showSearch]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setHighlight(h => Math.min(h + 1, items.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setHighlight(h => Math.max(h - 1, 0)); }
    else if (e.key === 'Enter') { e.preventDefault(); items[highlight]?.(); }
    else if (e.key === 'ArrowLeft') { switchSegment('agents'); }
    else if (e.key === 'ArrowRight') { switchSegment('workflows'); }
  };

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="top"
      align="start"
      estimatedHeight={460}
    >
      <div className="agent-selector-dropdown" onKeyDown={onKeyDown}>
        {/* Segmented control */}
        <div className="agent-selector-segments" role="tablist">
          <button
            role="tab"
            aria-selected={segment === 'agents'}
            className={`agent-selector-segment ${segment === 'agents' ? 'active' : ''}`}
            onClick={() => switchSegment('agents')}
          >
            <Bot size={13} />
            Agents
          </button>
          <button
            role="tab"
            aria-selected={segment === 'workflows'}
            className={`agent-selector-segment ${segment === 'workflows' ? 'active' : ''}`}
            onClick={() => switchSegment('workflows')}
          >
            <WorkflowIcon size={13} />
            Teams
            {activeWorkflow && <Crown size={11} className="segment-active-dot" />}
          </button>
        </div>

        {/* Per-segment search */}
        {showSearch && (
          <div className="agent-selector-search">
            <Search size={13} />
            <input
              ref={searchRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={segment === 'agents' ? 'Search agents…' : 'Search teams…'}
              aria-label={segment === 'agents' ? 'Search agents' : 'Search teams'}
            />
          </div>
        )}

        {/* Single scroll list */}
        <div className="agent-selector-list" role="listbox">
          {segment === 'agents' ? renderAgents() : renderWorkflows()}
        </div>

        {/* Context footer */}
        <div className="agent-selector-footer">
          {segment === 'agents' ? (
            <>
              <button className="agent-create-button" onClick={handleCreateNew}>
                <Plus size={14} />
                <span>Create New Agent</span>
              </button>
              <button className="agent-roster-link" onClick={handleBrowseRoster}>
                <LayoutGrid size={13} />
                <span>Browse the full roster</span>
              </button>
            </>
          ) : (
            <button className="agent-create-button" onClick={handleManageWorkflows}>
              <Settings size={14} />
              <span>Manage Teams…</span>
            </button>
          )}
        </div>
      </div>
    </DropdownPortal>
  );

  function groupHeader(group: RosterGroup) {
    if (group.kind === 'manager') return <><Crown size={11} /> Manager</>;
    if (group.kind === 'org-free') return <><Bot size={11} /> Independent</>;
    return <><Users size={11} /> {group.label}</>;
  }

  function renderAgentRow(profile: AgentProfile, i: number, group: RosterGroup, interactive: boolean) {
    const isActive = interactive
      ? profile.id === activeProfile?.id
      : profile.agentId === supervisorAgentId;
    const tier = tierByAgentId.get(profile.agentId);
    const model = modelShortLabel(profile.defaultModel);
    const role = roleLine(profile);
    // The ad-hoc delegation flag only means something outside an org — inside one,
    // the chain of command supersedes it, so the dot would mislead there.
    const showDot = group.kind === 'org-free' && profile.availableForDelegation;
    return (
      <div
        key={profile.id}
        role="option"
        aria-selected={isActive}
        className={`agent-selector-item ${isActive ? 'active' : ''} ${interactive && i === highlight ? 'kb-active' : ''} ${interactive ? '' : 'locked'}`}
        onClick={interactive ? () => handleSelect(profile.id) : undefined}
        onMouseEnter={interactive ? () => setHighlight(i) : undefined}
      >
        <div className="agent-item-avatar"><AgentAvatar avatar={profile.avatar} size={15} fill /></div>
        <div className="agent-item-info">
          <span className="agent-item-name">
            <span className="agent-item-name-text">{profile.name}</span>
            {tier && <span className={`agent-item-tier agent-item-tier--${tier}`}>{tier === 'manager' ? 'MGR' : 'LEAD'}</span>}
            {profile.isDefault && <span className="agent-item-badge">default</span>}
            {showDot && <span className="agent-item-deleg" title="Available for delegation" aria-hidden />}
          </span>
          {model && <span className="agent-item-model">{model}</span>}
          {role && <span className="agent-item-role">{role}</span>}
        </div>
        {isActive && (interactive ? <Check size={14} className="agent-item-check" /> : <Crown size={13} className="agent-item-check" />)}
        {interactive && (
          <button className="agent-item-edit" onClick={(e) => handleEdit(e, profile.id)} title="Edit profile">
            <Settings size={12} />
          </button>
        )}
      </div>
    );
  }

  function renderAgents() {
    if (agentsLocked) {
      // A team is active — the agent is fixed. Show the roster read-only, with the
      // team's lead crowned, so the structure is still legible.
      return (
        <>
          <div className="agent-selector-locked-hint">
            Agent is set by the active team. Switch to <strong>Teams → No team</strong> to choose one.
          </div>
          {agentGroups.map(group => (
            <div key={group.label} className="agent-selector-group">
              <div className="agent-selector-group-label">{groupHeader(group)}</div>
              {group.profiles.map((p) => renderAgentRow(p, -1, group, false))}
            </div>
          ))}
        </>
      );
    }

    if (flatAgents.length === 0) {
      return <div className="agent-selector-empty">No agents match “{query}”.</div>;
    }

    let cursor = 0;
    return agentGroups.map(group => (
      <div key={group.label} className="agent-selector-group">
        <div className="agent-selector-group-label">{groupHeader(group)}</div>
        {group.profiles.map((profile) => renderAgentRow(profile, cursor++, group, true))}
      </div>
    ));
  }

  function renderWorkflows() {
    return (
      <>
        <div
          role="option"
          aria-selected={activeWorkflowId === null}
          className={`agent-selector-item ${activeWorkflowId === null ? 'active' : ''} ${highlight === 0 ? 'kb-active' : ''}`}
          onClick={() => handleSelectWorkflow(null)}
          onMouseEnter={() => setHighlight(0)}
        >
          <div className="agent-item-avatar agent-item-avatar-ghost"><Bot size={14} /></div>
          <div className="agent-item-info">
            <span className="agent-item-name"><span className="agent-item-name-text">No team</span></span>
            <span className="agent-item-model">Single-agent chat</span>
          </div>
          {activeWorkflowId === null && <Check size={14} className="agent-item-check" />}
        </div>

        {filteredWorkflows.length === 0 && query.trim() !== '' && (
          <div className="agent-selector-empty">No teams match “{query}”.</div>
        )}

        {(() => {
          let idx = 1; // row 0 is "No team"
          return workflowGroups.map(group => (
            <div key={group.label} className="agent-selector-group">
              <div className="agent-selector-group-label">
                {group.managed ? <Crown size={11} /> : <WorkflowIcon size={11} />} {group.label}
              </div>
              {group.teams.map(w => {
                const i = idx++;
                return (
                  <div
                    key={w.id}
                    role="option"
                    aria-selected={activeWorkflowId === w.id}
                    className={`agent-selector-item ${activeWorkflowId === w.id ? 'active' : ''} ${i === highlight ? 'kb-active' : ''}`}
                    onClick={() => handleSelectWorkflow(w.id)}
                    onMouseEnter={() => setHighlight(i)}
                  >
                    <div className="agent-item-avatar"><WorkflowIcon size={14} /></div>
                    <div className="agent-item-info">
                      <span className="agent-item-name"><span className="agent-item-name-text">{w.name}</span></span>
                      <span className="agent-item-role">{teamSubtitle(w, id => nameByAgentId.get(id) ?? null)}</span>
                    </div>
                    {activeWorkflowId === w.id && <Check size={14} className="agent-item-check" />}
                  </div>
                );
              })}
            </div>
          ));
        })()}
      </>
    );
  }
}

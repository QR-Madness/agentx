/**
 * AgentSelectorDropdown — pick the agent profile or Alloy workflow for the
 * active conversation.
 *
 * Segmented layout: an [ Agents | Workflows ] toggle over a single full-height
 * scroll list (so neither list is ever squished), a per-segment search, a
 * context footer, and ↑/↓/Enter keyboard nav. Opens on whichever segment is
 * currently active.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Settings, Check, Plus, Workflow as WorkflowIcon, Crown, Search } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useConversation } from '../../contexts/ConversationContext';
import { useModal } from '../../contexts/ModalContext';
import { SURFACES } from '../../lib/surfaces';
import { AgentAvatar } from '../common/AgentAvatar';
import { DropdownPortal } from '../ui/DropdownPortal';
import './AgentSelectorDropdown.css';

// Show a segment's search field only once its list is long enough to warrant it.
const SEARCH_THRESHOLD = 6;

type Segment = 'agents' | 'workflows';

interface AgentSelectorDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
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

  // ---- agents ----
  const agentsLocked = activeWorkflow !== null; // agent is dictated by the workflow
  const filteredProfiles = useMemo(() => {
    // Ambassadors are not chat agents — they only ever brief, never join.
    const agents = profiles.filter(p => p.kind !== 'ambassador');
    const q = query.trim().toLowerCase();
    if (!q) return agents;
    return agents.filter(p =>
      p.name.toLowerCase().includes(q) ||
      (p.defaultModel ?? '').toLowerCase().includes(q) ||
      (p.tags ?? []).some(t => t.toLowerCase().includes(q)),
    );
  }, [profiles, query]);

  // ---- workflows ----
  const filteredWorkflows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return workflows;
    return workflows.filter(w => w.name.toLowerCase().includes(q));
  }, [workflows, query]);

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
  const handleManageWorkflows = () => {
    openModal(SURFACES.teams);
    onClose();
  };

  // Flat list of keyboard-selectable rows for the active segment.
  const items = useMemo<Array<() => void>>(() => {
    if (segment === 'agents') {
      if (agentsLocked) return [];
      return filteredProfiles.map(p => () => handleSelect(p.id));
    }
    return [() => handleSelectWorkflow(null), ...filteredWorkflows.map(w => () => handleSelectWorkflow(w.id))];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segment, agentsLocked, filteredProfiles, filteredWorkflows]);

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
      estimatedHeight={440}
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
          {segment === 'agents'
            ? renderAgents()
            : renderWorkflows()}
        </div>

        {/* Context footer */}
        <div className="agent-selector-footer">
          {segment === 'agents' ? (
            <button className="agent-create-button" onClick={handleCreateNew}>
              <Plus size={14} />
              <span>Create New Agent</span>
            </button>
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

  function renderAgents() {
    if (agentsLocked) {
      return (
        <>
          <div className="agent-selector-locked-hint">
            Agent is set by the active team. Switch to <strong>Teams → No team</strong> to choose one.
          </div>
          {profiles.map(profile => {
            const isSupervisor = profile.agentId === supervisorAgentId;
            return (
              <div key={profile.id} className={`agent-selector-item locked ${isSupervisor ? 'active' : ''}`}>
                <div className="agent-item-avatar"><AgentAvatar avatar={profile.avatar} size={15} fill /></div>
                <div className="agent-item-info">
                  <span className="agent-item-name">
                    {profile.name}
                    {isSupervisor && <span className="agent-item-badge">lead</span>}
                  </span>
                  <span className="agent-item-model">{profile.defaultModel || 'Default model'}</span>
                  {profile.tags && profile.tags.length > 0 && (
                    <span className="agent-item-tags">
                      {profile.tags.map(tag => (
                        <span key={tag} className="agent-item-tag">{tag}</span>
                      ))}
                    </span>
                  )}
                </div>
                {isSupervisor && <Crown size={13} className="agent-item-check" />}
              </div>
            );
          })}
        </>
      );
    }

    if (filteredProfiles.length === 0) {
      return <div className="agent-selector-empty">No agents match “{query}”.</div>;
    }
    return filteredProfiles.map((profile, i) => {
      const isActive = profile.id === activeProfile?.id;
      return (
        <div
          key={profile.id}
          role="option"
          aria-selected={isActive}
          className={`agent-selector-item ${isActive ? 'active' : ''} ${i === highlight ? 'kb-active' : ''}`}
          onClick={() => handleSelect(profile.id)}
          onMouseEnter={() => setHighlight(i)}
        >
          <div className="agent-item-avatar"><AgentAvatar avatar={profile.avatar} size={15} fill /></div>
          <div className="agent-item-info">
            <span className="agent-item-name">
              {profile.name}
              {profile.isDefault && <span className="agent-item-badge">default</span>}
            </span>
            <span className="agent-item-model">{profile.defaultModel || 'Default model'}</span>
            {profile.tags && profile.tags.length > 0 && (
              <span className="agent-item-tags">
                {profile.tags.map(tag => (
                  <span key={tag} className="agent-item-tag">{tag}</span>
                ))}
              </span>
            )}
          </div>
          {isActive && <Check size={14} className="agent-item-check" />}
          <button className="agent-item-edit" onClick={(e) => handleEdit(e, profile.id)} title="Edit profile">
            <Settings size={12} />
          </button>
        </div>
      );
    });
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
            <span className="agent-item-name">No team</span>
            <span className="agent-item-model">Single-agent chat</span>
          </div>
          {activeWorkflowId === null && <Check size={14} className="agent-item-check" />}
        </div>

        {filteredWorkflows.length === 0 && query.trim() !== '' && (
          <div className="agent-selector-empty">No teams match “{query}”.</div>
        )}

        {filteredWorkflows.map((w, i) => {
          const specialistCount = w.members.filter(m => m.role === 'specialist').length;
          const idx = i + 1; // row 0 is "No workflow"
          return (
            <div
              key={w.id}
              role="option"
              aria-selected={activeWorkflowId === w.id}
              className={`agent-selector-item ${activeWorkflowId === w.id ? 'active' : ''} ${idx === highlight ? 'kb-active' : ''}`}
              onClick={() => handleSelectWorkflow(w.id)}
              onMouseEnter={() => setHighlight(idx)}
            >
              <div className="agent-item-avatar"><WorkflowIcon size={14} /></div>
              <div className="agent-item-info">
                <span className="agent-item-name">{w.name}</span>
                <span className="agent-item-model">
                  {specialistCount} member{specialistCount === 1 ? '' : 's'}
                </span>
              </div>
              {activeWorkflowId === w.id && <Check size={14} className="agent-item-check" />}
            </div>
          );
        })}
      </>
    );
  }
}

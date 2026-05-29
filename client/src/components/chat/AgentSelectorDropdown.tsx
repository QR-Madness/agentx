/**
 * AgentSelectorDropdown — Select and manage agent profiles
 * Renders via portal for proper z-index handling
 */

import { useMemo, useState } from 'react';
import { Bot, Settings, Check, Plus, ChevronUp, Workflow as WorkflowIcon, Crown, Search } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useConversation } from '../../contexts/ConversationContext';
import { useModal } from '../../contexts/ModalContext';
import { getAvatarIcon } from '../../lib/avatars';
import { DropdownPortal } from '../ui/DropdownPortal';
import './AgentSelectorDropdown.css';

// Show the profile search field only once the list is long enough to warrant it.
const PROFILE_SEARCH_THRESHOLD = 6;

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

  const [profileQuery, setProfileQuery] = useState('');
  const showProfileSearch = profiles.length > PROFILE_SEARCH_THRESHOLD;
  const filteredProfiles = useMemo(() => {
    const q = profileQuery.trim().toLowerCase();
    if (!q) return profiles;
    return profiles.filter(p =>
      p.name.toLowerCase().includes(q) ||
      (p.defaultModel ?? '').toLowerCase().includes(q),
    );
  }, [profiles, profileQuery]);

  const handleSelect = (profileId: string) => {
    setActiveProfile(profileId);
    onClose();
  };

  const handleEdit = (e: React.MouseEvent, profileId: string) => {
    e.stopPropagation();
    openModal({
      id: 'profile-editor',
      type: 'modal',
      component: 'unifiedProfileEditor',
      size: 'full',
      props: { initialProfileId: profileId },
    });
    onClose();
  };

  const handleCreateNew = () => {
    openModal({
      id: 'profile-editor',
      type: 'modal',
      component: 'unifiedProfileEditor',
      size: 'full',
      props: { isNew: true },
    });
    onClose();
  };

  const activeWorkflowId = activeTab?.workflowId ?? null;
  const activeWorkflow = activeWorkflowId
    ? workflows.find(w => w.id === activeWorkflowId) ?? null
    : null;
  const supervisorProfile = activeWorkflow
    ? profiles.find(p => p.agentId === activeWorkflow.supervisorAgentId) ?? null
    : null;

  const handleSelectWorkflow = (workflowId: string | null) => {
    setActiveTabWorkflow(workflowId);
    onClose();
  };

  const handleManageWorkflows = () => {
    openModal({
      id: 'alloy-factory',
      type: 'modal',
      component: 'alloyFactory',
      size: 'full',
    });
    onClose();
  };

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="top"
      align="start"
      estimatedHeight={540}
    >
    <div className="agent-selector-dropdown">
      <div className="agent-selector-header">
        <Bot size={14} />
        <span>{activeWorkflow ? 'Supervisor' : 'Select Agent'}</span>
        {activeWorkflow ? (
          <Crown size={12} className="header-icon" />
        ) : (
          <ChevronUp size={14} className="header-icon" />
        )}
      </div>

      {activeWorkflow ? (
        <>
          <div className="agent-selector-list">
            {supervisorProfile ? (
              <div className="agent-selector-item locked active">
                <div className="agent-item-avatar">
                  {(() => {
                    const SupAvatar = getAvatarIcon(supervisorProfile.avatar);
                    return <SupAvatar size={16} />;
                  })()}
                </div>
                <div className="agent-item-info">
                  <span className="agent-item-name">{supervisorProfile.name}</span>
                  <span className="agent-item-model">
                    {supervisorProfile.defaultModel || 'Default model'}
                  </span>
                </div>
                <Crown size={12} className="agent-item-check" />
              </div>
            ) : (
              <div className="agent-selector-item locked">
                <div className="agent-item-avatar">
                  <Bot size={16} />
                </div>
                <div className="agent-item-info">
                  <span className="agent-item-name">{activeWorkflow.supervisorAgentId}</span>
                  <span className="agent-item-model">Supervisor profile not found</span>
                </div>
              </div>
            )}
          </div>
          <div className="agent-selector-locked-hint">
            The agent is set by the active workflow. Switch workflow to "No workflow" to choose a different agent.
          </div>
        </>
      ) : (
        <>
          {showProfileSearch && (
            <div className="agent-selector-search">
              <Search size={13} />
              <input
                type="text"
                value={profileQuery}
                onChange={(e) => setProfileQuery(e.target.value)}
                placeholder="Search agents…"
                aria-label="Search agents"
              />
            </div>
          )}
          <div className="agent-selector-list">
            {filteredProfiles.length === 0 ? (
              <div className="agent-selector-empty">No agents match “{profileQuery}”.</div>
            ) : filteredProfiles.map(profile => {
              const AvatarIcon = getAvatarIcon(profile.avatar);
              const isActive = profile.id === activeProfile?.id;
              return (
                <div
                  key={profile.id}
                  className={`agent-selector-item ${isActive ? 'active' : ''}`}
                  onClick={() => handleSelect(profile.id)}
                >
                  <div className="agent-item-avatar">
                    <AvatarIcon size={16} />
                  </div>
                  <div className="agent-item-info">
                    <span className="agent-item-name">
                      {profile.name}
                      {profile.isDefault && <span className="agent-item-badge">default</span>}
                    </span>
                    <span className="agent-item-model">{profile.defaultModel || 'Default model'}</span>
                  </div>
                  {isActive && <Check size={14} className="agent-item-check" />}
                  <button
                    className="agent-item-edit"
                    onClick={(e) => handleEdit(e, profile.id)}
                    title="Edit profile"
                  >
                    <Settings size={12} />
                  </button>
                </div>
              );
            })}
          </div>

          <div className="agent-selector-footer">
            <button className="agent-create-button" onClick={handleCreateNew}>
              <Plus size={14} />
              <span>Create New Agent</span>
            </button>
          </div>
        </>
      )}

      <div className="agent-selector-section-divider" />

      <div className="agent-selector-header">
        <WorkflowIcon size={14} />
        <span>Alloy Workflow</span>
      </div>

      <div className="agent-selector-list">
        <div
          className={`agent-selector-item ${activeWorkflowId === null ? 'active' : ''}`}
          onClick={() => handleSelectWorkflow(null)}
        >
          <div className="agent-item-avatar" style={{ background: 'transparent', border: '1px dashed var(--border-color)' }}>
            <Bot size={14} />
          </div>
          <div className="agent-item-info">
            <span className="agent-item-name">No workflow</span>
            <span className="agent-item-model">Single-agent chat</span>
          </div>
          {activeWorkflowId === null && <Check size={14} className="agent-item-check" />}
        </div>

        {workflows.map(w => {
          const specialistCount = w.members.filter(m => m.role === 'specialist').length;
          return (
            <div
              key={w.id}
              className={`agent-selector-item ${activeWorkflowId === w.id ? 'active' : ''}`}
              onClick={() => handleSelectWorkflow(w.id)}
            >
              <div className="agent-item-avatar">
                <WorkflowIcon size={14} />
              </div>
              <div className="agent-item-info">
                <span className="agent-item-name">{w.name}</span>
                <span className="agent-item-model">
                  {specialistCount} specialist{specialistCount === 1 ? '' : 's'}
                </span>
              </div>
              {activeWorkflowId === w.id && <Check size={14} className="agent-item-check" />}
            </div>
          );
        })}
      </div>

      <div className="agent-selector-footer">
        <button className="agent-create-button" onClick={handleManageWorkflows}>
          <Settings size={14} />
          <span>Manage Workflows…</span>
        </button>
      </div>
    </div>
    </DropdownPortal>
  );
}

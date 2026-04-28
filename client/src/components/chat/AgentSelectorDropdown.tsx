/**
 * AgentSelectorDropdown — Select and manage agent profiles
 * Renders via portal for proper z-index handling
 */

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Bot, Settings, Check, Plus, ChevronUp, Workflow as WorkflowIcon, Crown } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { useConversation } from '../../contexts/ConversationContext';
import { useModal } from '../../contexts/ModalContext';
import { getAvatarIcon } from '../../lib/avatars';
import './AgentSelectorDropdown.css';

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
  const [position, setPosition] = useState({ bottom: 0, left: 0 });
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Calculate position based on anchor element (opens upward)
  useEffect(() => {
    if (!isOpen || !anchorRef.current) return;

    const updatePosition = () => {
      const rect = anchorRef.current?.getBoundingClientRect();
      if (rect) {
        setPosition({
          bottom: window.innerHeight - rect.top + 8,
          left: rect.left,
        });
      }
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    return () => window.removeEventListener('resize', updatePosition);
  }, [isOpen, anchorRef]);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(e.target as Node)
      ) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose, anchorRef]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

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

  const dropdown = (
    <div
      className="agent-selector-dropdown"
      ref={dropdownRef}
      style={{ bottom: position.bottom, left: position.left }}
    >
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
          <div className="agent-selector-list">
            {profiles.map(profile => {
              const AvatarIcon = getAvatarIcon(profile.avatar);
              return (
                <div
                  key={profile.id}
                  className={`agent-selector-item ${profile.id === activeProfile?.id ? 'active' : ''}`}
                  onClick={() => handleSelect(profile.id)}
                >
                  <div className="agent-item-avatar">
                    <AvatarIcon size={16} />
                  </div>
                  <div className="agent-item-info">
                    <span className="agent-item-name">{profile.name}</span>
                    <span className="agent-item-model">{profile.defaultModel || 'Default model'}</span>
                  </div>
                  {profile.id === activeProfile?.id && (
                    <Check size={14} className="agent-item-check" />
                  )}
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
  );

  return createPortal(dropdown, document.body);
}

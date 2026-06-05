/**
 * ActiveAgentsDropdown — Shows list of active/streaming conversations
 * Rendered as a portal dropdown anchored to the brain icon in TopBar
 */

import { MessageSquare, Plus, Radio } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import { DropdownPortal } from '../ui/DropdownPortal';
import './ActiveAgentsDropdown.css';

interface ActiveAgentsDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
}

export function ActiveAgentsDropdown({ isOpen, onClose, anchorRef }: ActiveAgentsDropdownProps) {
  const { tabs, activeTabId, switchTab, addTab } = useConversation();
  const { profiles, activeProfile } = useAgentProfile();

  const activeTabs = tabs.filter(t => t.isStreaming);
  const inactiveTabs = tabs.filter(t => !t.isStreaming);

  const handleSelect = (tabId: string) => {
    switchTab(tabId);
    onClose();
  };

  const handleNew = () => {
    addTab();
    onClose();
  };

  // Resolve the agent profile driving a given tab (falling back to the active
  // profile) so each row shows the right avatar + trait/role tags.
  const profileForTab = (profileId: string | null) =>
    (profileId ? profiles.find(p => p.id === profileId) : null) ?? activeProfile;

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="bottom"
      align="start"
      estimatedHeight={400}
    >
    <div className="active-agents-dropdown">
      <button className="agents-new" onClick={handleNew}>
        <Plus size={14} />
        <span>New conversation</span>
      </button>

      {/* Active/streaming conversations */}
      {activeTabs.length > 0 && (
        <>
          <div className="agents-section-header">
            <Radio size={12} className="section-icon active" />
            <span>Active</span>
          </div>
          {activeTabs.map(tab => {
            const tabProfile = profileForTab(tab.profileId);
            const TabAvatar = getAvatarIcon(tabProfile?.avatar);
            return (
              <button
                key={tab.id}
                className={`agents-item ${tab.id === activeTabId ? 'current' : ''}`}
                onClick={() => handleSelect(tab.id)}
              >
                <div className="agents-item-avatar streaming">
                  <TabAvatar size={14} />
                </div>
                <div className="agents-item-info">
                  <span className="agents-item-title">{tab.title}</span>
                  <span className="agents-item-status">Streaming...</span>
                  {tabProfile?.tags && tabProfile.tags.length > 0 && (
                    <span className="agents-item-tags">
                      {tabProfile.tags.map(tag => (
                        <span key={tag} className="agents-item-tag">{tag}</span>
                      ))}
                    </span>
                  )}
                </div>
                <span className="agents-item-pulse" />
              </button>
            );
          })}
        </>
      )}

      {/* Inactive conversations */}
      {inactiveTabs.length > 0 && (
        <>
          <div className="agents-section-header">
            <MessageSquare size={12} className="section-icon" />
            <span>Conversations</span>
          </div>
          {inactiveTabs.map(tab => {
            const tabProfile = profileForTab(tab.profileId);
            const TabAvatar = getAvatarIcon(tabProfile?.avatar);
            return (
              <button
                key={tab.id}
                className={`agents-item ${tab.id === activeTabId ? 'current' : ''}`}
                onClick={() => handleSelect(tab.id)}
              >
                <div className="agents-item-avatar">
                  <TabAvatar size={14} />
                </div>
                <div className="agents-item-info">
                  <span className="agents-item-title">{tab.title}</span>
                  {tab.sessionId && (
                    <span className="agents-item-session">{tab.sessionId.slice(0, 8)}</span>
                  )}
                  {tabProfile?.tags && tabProfile.tags.length > 0 && (
                    <span className="agents-item-tags">
                      {tabProfile.tags.map(tag => (
                        <span key={tag} className="agents-item-tag">{tag}</span>
                      ))}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </>
      )}

      {tabs.length === 0 && (
        <div className="agents-empty">No conversations yet</div>
      )}
    </div>
    </DropdownPortal>
  );
}

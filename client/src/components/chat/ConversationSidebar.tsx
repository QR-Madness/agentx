/**
 * ConversationSidebar — the desktop chat-page rail that replaced the cramped
 * TopBar tab strip as the primary conversation navigator.
 *
 * Presentation over the existing ConversationContext (the tab/session/streaming
 * model is unchanged): "open conversations" are the restyled rows at the top of
 * the list, with past/server conversations below — all via the shared
 * ConversationList. Collapsible to a thin avatar rail; state persisted.
 *
 * This rail is the deliberate home for the conversation-management tools coming
 * next (multi-select + bulk actions, pin, archive) — the seam lives in
 * ConversationList's rows.
 */

import { useState, useCallback } from 'react';
import { Plus, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { getAvatarIcon } from '../../lib/avatars';
import { ConversationList } from './ConversationList';
import './ConversationSidebar.css';

const COLLAPSE_KEY = 'agentx:conv-sidebar-collapsed';

export function ConversationSidebar() {
  const { tabs, activeTabId, switchTab, addTab } = useConversation();
  const { profiles } = useAgentProfile();
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(COLLAPSE_KEY) === '1'; } catch { return false; }
  });

  const setCollapsedPersist = useCallback((next: boolean) => {
    setCollapsed(next);
    try { localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0'); } catch { /* ignore */ }
  }, []);

  const avatarFor = (profileId: string | null) =>
    getAvatarIcon(profiles.find(p => p.id === profileId)?.avatar);

  if (collapsed) {
    return (
      <aside className="conversation-sidebar conversation-sidebar--collapsed" aria-label="Conversations">
        <button className="conv-rail-btn" onClick={() => setCollapsedPersist(false)} title="Expand conversations">
          <PanelLeftOpen size={18} />
        </button>
        <button className="conv-rail-btn conv-rail-new" onClick={() => addTab()} title="New conversation">
          <Plus size={18} />
        </button>
        <div className="conv-rail-tabs">
          {tabs.map(tab => {
            const Avatar = avatarFor(tab.profileId);
            return (
              <button
                key={tab.id}
                className={`conv-rail-tab ${tab.id === activeTabId ? 'active' : ''}`}
                onClick={() => switchTab(tab.id)}
                title={tab.title}
              >
                <Avatar size={16} />
                {tab.isStreaming && <span className="conv-rail-streaming" />}
              </button>
            );
          })}
        </div>
      </aside>
    );
  }

  return (
    <aside className="conversation-sidebar" aria-label="Conversations">
      <div className="conv-sidebar-header">
        <span className="conv-sidebar-title">Conversations</span>
        <div className="conv-sidebar-header-actions">
          <button className="conv-rail-btn" onClick={() => addTab()} title="New conversation">
            <Plus size={16} />
          </button>
          <button className="conv-rail-btn" onClick={() => setCollapsedPersist(true)} title="Collapse">
            <PanelLeftClose size={16} />
          </button>
        </div>
      </div>
      <div className="conv-sidebar-body">
        <ConversationList onActivated={() => {}} autoFocusSearch={false} />
      </div>
    </aside>
  );
}

/**
 * ConversationTabBar — Browser-style tabs for managing conversations
 */

import { useState, useRef } from 'react';
import { Plus, X, History } from 'lucide-react';
import { useConversation } from '../contexts/ConversationContext';
import { ConversationHistoryDropdown } from '../components/chat/ConversationHistoryDropdown';
import './ConversationTabBar.css';

interface ConversationTabBarProps {
  visible: boolean;
}

export function ConversationTabBar({ visible }: ConversationTabBarProps) {
  const { tabs, activeTabId, addTab, closeTab, switchTab, renameTab } = useConversation();
  const [showHistory, setShowHistory] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const historyButtonRef = useRef<HTMLButtonElement>(null);

  if (!visible) {
    return null;
  }

  const handleAddTab = () => {
    addTab();
  };

  const handleCloseTab = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    closeTab(id);
  };

  const startEditing = (id: string, currentTitle: string) => {
    setEditingId(id);
    setDraftTitle(currentTitle);
  };

  const commitEditing = () => {
    if (editingId) {
      const next = draftTitle.trim();
      if (next) renameTab(editingId, next);
    }
    setEditingId(null);
  };

  const handleEditKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitEditing();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setEditingId(null);
    }
  };

  return (
    <div className="conversation-tab-bar">
      <div className="tabs-container">
        {tabs.map(tab =>
          editingId === tab.id ? (
            <div key={tab.id} className="tab-button active tab-editing">
              <input
                className="tab-title-input"
                value={draftTitle}
                autoFocus
                onChange={e => setDraftTitle(e.target.value)}
                onBlur={commitEditing}
                onKeyDown={handleEditKeyDown}
                aria-label="Rename conversation"
              />
            </div>
          ) : (
            <button
              key={tab.id}
              className={`tab-button ${activeTabId === tab.id ? 'active' : ''}`}
              onClick={() => switchTab(tab.id)}
              onDoubleClick={() => startEditing(tab.id, tab.title)}
              title={`${tab.title} — double-click to rename`}
            >
              <span className="tab-title">{tab.title}</span>
              {tab.isStreaming && <span className="tab-streaming" />}
              <span
                className="tab-close"
                onClick={e => handleCloseTab(e, tab.id)}
                role="button"
                aria-label="Close tab"
              >
                <X size={12} />
              </span>
            </button>
          )
        )}
      </div>

      <div className="tabs-actions">
        <button
          className="tab-action-button"
          onClick={handleAddTab}
          title="New conversation"
        >
          <Plus size={16} />
        </button>

        <div className="history-dropdown-container">
          <button
            ref={historyButtonRef}
            className={`tab-action-button ${showHistory ? 'active' : ''}`}
            onClick={() => setShowHistory(!showHistory)}
            title="Conversation history"
          >
            <History size={16} />
          </button>

          <ConversationHistoryDropdown
            isOpen={showHistory}
            onClose={() => setShowHistory(false)}
            anchorRef={historyButtonRef}
          />
        </div>
      </div>
    </div>
  );
}

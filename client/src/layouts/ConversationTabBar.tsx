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
  const { tabs, activeTabId, addTab, closeTab, switchTab } = useConversation();
  const [showHistory, setShowHistory] = useState(false);
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

  return (
    <div className="conversation-tab-bar">
      <div className="tabs-container">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTabId === tab.id ? 'active' : ''}`}
            onClick={() => switchTab(tab.id)}
            title={tab.title}
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
        ))}
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

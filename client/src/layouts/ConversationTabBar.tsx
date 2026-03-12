/**
 * ConversationTabBar — Browser-style tabs for managing conversations
 */

import { useState, useRef, useEffect } from 'react';
import { Plus, X, History, MessageSquare } from 'lucide-react';
import { useConversation } from '../contexts/ConversationContext';
import { getRecentChats, type RecentChat } from '../lib/storage';
import './ConversationTabBar.css';

interface ConversationTabBarProps {
  visible: boolean;
}

export function ConversationTabBar({ visible }: ConversationTabBarProps) {
  const { tabs, activeTabId, addTab, closeTab, switchTab } = useConversation();
  const [showHistory, setShowHistory] = useState(false);
  const [recentChats, setRecentChats] = useState<RecentChat[]>([]);
  const historyRef = useRef<HTMLDivElement>(null);

  // Load recent chats when history is opened
  useEffect(() => {
    if (showHistory) {
      setRecentChats(getRecentChats());
    }
  }, [showHistory]);

  // Close history dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    }

    if (showHistory) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showHistory]);

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

  const handleHistorySelect = (_chat: RecentChat) => {
    // Create a new tab with the selected conversation
    // For now, just create a new tab (full restoration in later phase)
    // TODO: Restore chat messages from _chat.messages
    addTab();
    setShowHistory(false);
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

        <div className="history-dropdown-container" ref={historyRef}>
          <button
            className={`tab-action-button ${showHistory ? 'active' : ''}`}
            onClick={() => setShowHistory(!showHistory)}
            title="Conversation history"
          >
            <History size={16} />
          </button>

          {showHistory && (
            <div className="history-dropdown">
              <div className="history-header">Recent Conversations</div>
              {recentChats.length === 0 ? (
                <div className="history-empty">No recent conversations</div>
              ) : (
                <div className="history-list">
                  {recentChats.map(chat => (
                    <button
                      key={chat.id}
                      className="history-item"
                      onClick={() => handleHistorySelect(chat)}
                    >
                      <MessageSquare size={14} />
                      <div className="history-item-content">
                        <span className="history-item-title">{chat.title}</span>
                        <span className="history-item-preview">{chat.preview}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

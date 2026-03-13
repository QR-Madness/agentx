/**
 * ConversationHistoryDropdown — Browse and switch between past conversations
 */

import { useState, useRef, useEffect } from 'react';
import { Clock, Search, Trash2, MessageSquare, X } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import './ConversationHistoryDropdown.css';

interface ConversationHistoryDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement>;
}

export function ConversationHistoryDropdown({ isOpen, onClose, anchorRef }: ConversationHistoryDropdownProps) {
  const { tabs, activeTabId, switchTab, closeTab } = useConversation();
  const [searchQuery, setSearchQuery] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

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

  // Focus search on open
  useEffect(() => {
    if (isOpen && searchRef.current) {
      searchRef.current.focus();
    }
  }, [isOpen]);

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

  // Filter tabs by search query
  const filteredTabs = tabs.filter(tab =>
    tab.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Sort by last message (most recent first)
  const sortedTabs = [...filteredTabs].sort((a, b) => {
    const aTime = new Date(a.lastMessageAt).getTime();
    const bTime = new Date(b.lastMessageAt).getTime();
    return bTime - aTime;
  });

  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  const handleSelect = (tabId: string) => {
    switchTab(tabId);
    onClose();
  };

  const handleDelete = (e: React.MouseEvent, tabId: string) => {
    e.stopPropagation();
    closeTab(tabId);
  };

  return (
    <div className="history-dropdown" ref={dropdownRef}>
      <div className="history-header">
        <Clock size={16} />
        <span>Conversation History</span>
        <button className="close-button" onClick={onClose}>
          <X size={14} />
        </button>
      </div>

      <div className="history-search">
        <Search size={14} />
        <input
          ref={searchRef}
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search conversations..."
        />
        {searchQuery && (
          <button className="clear-search" onClick={() => setSearchQuery('')}>
            <X size={12} />
          </button>
        )}
      </div>

      <div className="history-list">
        {sortedTabs.length === 0 ? (
          <div className="history-empty">
            {searchQuery ? (
              <>No conversations match "{searchQuery}"</>
            ) : (
              <>No conversations yet</>
            )}
          </div>
        ) : (
          sortedTabs.map(tab => (
            <div
              key={tab.id}
              className={`history-item ${tab.id === activeTabId ? 'active' : ''}`}
              onClick={() => handleSelect(tab.id)}
            >
              <div className="history-item-icon">
                <MessageSquare size={14} />
              </div>
              <div className="history-item-info">
                <span className="history-item-title">{tab.title}</span>
                <span className="history-item-meta">
                  {tab.messages.length} messages · {formatDate(tab.lastMessageAt)}
                </span>
              </div>
              <button
                className="history-item-delete"
                onClick={(e) => handleDelete(e, tab.id)}
                title="Delete conversation"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))
        )}
      </div>

      <div className="history-footer">
        <span>{tabs.length} conversation{tabs.length !== 1 ? 's' : ''}</span>
      </div>
    </div>
  );
}

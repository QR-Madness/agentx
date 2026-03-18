/**
 * ConversationHistoryDropdown — Browse and switch between past conversations
 * Shows open tabs and server-side conversation history
 * Renders via portal to escape overflow constraints
 */

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Clock, Search, Trash2, MessageSquare, X, Download, Loader2 } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import './ConversationHistoryDropdown.css';

interface ConversationHistoryDropdownProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
}

export function ConversationHistoryDropdown({ isOpen, onClose, anchorRef }: ConversationHistoryDropdownProps) {
  const {
    tabs, activeTabId, switchTab, closeTab,
    serverConversations, isLoadingHistory, restoreConversation, refreshHistory,
    deleteConversation, deleteServerConversation,
  } = useConversation();
  const [searchQuery, setSearchQuery] = useState('');
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [position, setPosition] = useState({ top: 0, right: 0 });
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Calculate position based on anchor element
  useEffect(() => {
    if (!isOpen || !anchorRef.current) return;

    const updatePosition = () => {
      const rect = anchorRef.current?.getBoundingClientRect();
      if (rect) {
        setPosition({
          top: rect.bottom + 8,
          right: window.innerWidth - rect.right,
        });
      }
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    return () => window.removeEventListener('resize', updatePosition);
  }, [isOpen, anchorRef]);

  // Refresh history when opened
  useEffect(() => {
    if (isOpen) {
      refreshHistory();
    }
  }, [isOpen, refreshHistory]);

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

  const query = searchQuery.toLowerCase();

  // Filter and sort open tabs
  const filteredTabs = tabs
    .filter(tab => tab.title.toLowerCase().includes(query))
    .sort((a, b) => new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime());

  // Get session IDs of open tabs to exclude from server history
  const openSessionIds = new Set(tabs.map(t => t.sessionId).filter(Boolean));

  // Filter server conversations: exclude those already open as tabs
  const pastConversations = serverConversations
    .filter(c => !openSessionIds.has(c.conversation_id))
    .filter(c => c.title.toLowerCase().includes(query) || c.preview.toLowerCase().includes(query));

  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return '';
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

  const handleClose = (e: React.MouseEvent, tabId: string) => {
    e.stopPropagation();
    closeTab(tabId);
  };

  const handleDeleteTab = async (e: React.MouseEvent, tabId: string, title: string) => {
    e.stopPropagation();
    const confirmed = window.confirm(
      `Permanently delete "${title}"?\n\nThis will remove the conversation from the server and cannot be undone.`
    );
    if (!confirmed) return;

    setDeletingId(tabId);
    try {
      await deleteConversation(tabId);
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteServerConversation = async (e: React.MouseEvent, conversationId: string, title: string) => {
    e.stopPropagation();
    const confirmed = window.confirm(
      `Permanently delete "${title}"?\n\nThis will remove the conversation from the server and cannot be undone.`
    );
    if (!confirmed) return;

    setDeletingId(conversationId);
    try {
      await deleteServerConversation(conversationId);
    } catch {
      // Delete failed silently
    } finally {
      setDeletingId(null);
    }
  };

  const handleRestore = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    setRestoringId(conversationId);
    try {
      await restoreConversation(conversationId);
      onClose();
    } catch {
      // Restore failed silently
    } finally {
      setRestoringId(null);
    }
  };

  const totalCount = filteredTabs.length + pastConversations.length;

  const dropdown = (
    <div
      className="history-dropdown-portal"
      ref={dropdownRef}
      style={{ top: position.top, right: position.right }}
    >
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
        {/* Open Tabs Section */}
        {filteredTabs.length > 0 && (
          <>
            <div className="history-section-label">Open Tabs</div>
            {filteredTabs.map(tab => (
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
                <div className="history-item-actions">
                  {deletingId === tab.id ? (
                    <div className="history-item-loading">
                      <Loader2 size={12} className="spin" />
                    </div>
                  ) : (
                    <>
                      <button
                        className="history-item-action"
                        onClick={(e) => handleClose(e, tab.id)}
                        title="Close tab (keeps on server)"
                      >
                        <X size={12} />
                      </button>
                      <button
                        className="history-item-action history-item-action-danger"
                        onClick={(e) => handleDeleteTab(e, tab.id, tab.title)}
                        title="Delete conversation (removes from server)"
                      >
                        <Trash2 size={12} />
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </>
        )}

        {/* Past Conversations Section (from server) */}
        {pastConversations.length > 0 && (
          <>
            <div className="history-section-label">Past Conversations</div>
            {pastConversations.map(conv => (
              <div
                key={conv.conversation_id}
                className="history-item history-item-server"
                onClick={(e) => handleRestore(e, conv.conversation_id)}
              >
                <div className="history-item-icon">
                  <Download size={14} />
                </div>
                <div className="history-item-info">
                  <span className="history-item-title">{conv.title}</span>
                  <span className="history-item-meta">
                    {conv.message_count} messages · {formatDate(conv.last_message_at)}
                  </span>
                  {conv.preview && (
                    <span className="history-item-preview">{conv.preview}</span>
                  )}
                </div>
                <div className="history-item-actions">
                  {restoringId === conv.conversation_id || deletingId === conv.conversation_id ? (
                    <div className="history-item-loading">
                      <Loader2 size={12} className="spin" />
                    </div>
                  ) : (
                    <button
                      className="history-item-action history-item-action-danger"
                      onClick={(e) => handleDeleteServerConversation(e, conv.conversation_id, conv.title)}
                      title="Delete conversation"
                    >
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </>
        )}

        {/* Loading indicator */}
        {isLoadingHistory && pastConversations.length === 0 && filteredTabs.length === 0 && (
          <div className="history-empty">
            <Loader2 size={16} className="spin" />
            <span>Loading history...</span>
          </div>
        )}

        {/* Empty state */}
        {!isLoadingHistory && totalCount === 0 && (
          <div className="history-empty">
            {searchQuery ? (
              <>No conversations match &quot;{searchQuery}&quot;</>
            ) : (
              <>No conversations yet</>
            )}
          </div>
        )}
      </div>

      <div className="history-footer">
        <span>
          {tabs.length} open{pastConversations.length > 0 ? ` · ${pastConversations.length} past` : ''}
        </span>
      </div>
    </div>
  );

  return createPortal(dropdown, document.body);
}

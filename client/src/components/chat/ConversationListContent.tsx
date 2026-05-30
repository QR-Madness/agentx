/**
 * ConversationListContent — the searchable list body shared by the desktop
 * history dropdown (ConversationHistoryDropdown) and the mobile Conversations
 * drawer (ConversationsDrawerContent).
 *
 * Owns: resume-running runs, open tabs, and past (server) conversations grouped
 * by recency, plus search / restore / rename / delete. It reads everything from
 * ConversationContext; the host only supplies `onActivated`, called after a
 * conversation is switched/restored/resumed so the host can dismiss itself.
 *
 * Mounts fresh each time the host opens (both hosts unmount it while closed),
 * so the on-mount effect doubles as fetch-on-open.
 */

import { useState, useRef, useEffect, useMemo } from 'react';
import { Search, Trash2, MessageSquare, X, Download, Loader2, Radio, Pencil } from 'lucide-react';
import { useConversation } from '../../contexts/ConversationContext';
import { useNotify } from '../../contexts/NotificationContext';
import { orphanedRuns } from '../../contexts/conversation/orphanedRuns';
import { api, type ActiveChatRun } from '../../lib/api';
import { getDisplayTitle, setTitleOverride } from '../../lib/conversationTitles';
import './ConversationHistoryDropdown.css';

/** Bucket a date into a coarse recency group for section labelling. */
function recencyBucket(dateStr: string | null): 'Today' | 'Yesterday' | 'This week' | 'Older' {
  if (!dateStr) return 'Older';
  const days = Math.floor((Date.now() - new Date(dateStr).getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return 'This week';
  return 'Older';
}

interface ConversationListContentProps {
  /** Called after a conversation is switched/restored/resumed so the host dismisses. */
  onActivated: () => void;
  /** Focus the search input on mount. Default true; pass false to avoid popping
   *  the on-screen keyboard when the drawer opens on mobile. */
  autoFocusSearch?: boolean;
}

export function ConversationListContent({ onActivated, autoFocusSearch = true }: ConversationListContentProps) {
  const {
    tabs, activeTabId, switchTab, closeTab, renameTab,
    serverConversations, isLoadingHistory, restoreConversation, refreshHistory,
    deleteConversation, deleteServerConversation, resumeRun,
  } = useConversation();
  const { notifyError } = useNotify();
  const [searchQuery, setSearchQuery] = useState('');
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [runs, setRuns] = useState<ActiveChatRun[]>([]);
  // Inline rename: `editingId` is a tab id or a conversation_id; `renameTick`
  // forces a re-render after a server-conversation override is written.
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [, setRenameTick] = useState(0);
  const searchRef = useRef<HTMLInputElement>(null);

  // Refresh history + detached runs + focus search on mount. Fetched on open
  // only (no polling) — the Relay inbox owns the live cadence.
  useEffect(() => {
    refreshHistory();
    api.listChatRuns().then(r => setRuns(r.runs)).catch(() => setRuns([]));
    if (autoFocusSearch) searchRef.current?.focus();
  }, [refreshHistory, autoFocusSearch]);

  // Runs still going whose owning tab is closed — offered as "Resume Running".
  const liveRuns = useMemo(() => orphanedRuns(runs, tabs), [runs, tabs]);

  const query = searchQuery.toLowerCase();

  // Filter and sort open tabs
  const filteredTabs = tabs
    .filter(tab => tab.title.toLowerCase().includes(query))
    .sort((a, b) => new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime());

  // Get session IDs of open tabs + live runs to exclude from server history,
  // so a running conversation isn't also listed under "Past Conversations".
  const openSessionIds = new Set([
    ...tabs.map(t => t.sessionId).filter(Boolean),
    ...liveRuns.map(r => r.session_id).filter(Boolean),
  ]);

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
    onActivated();
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
    } catch (err) {
      notifyError(err, 'Failed to delete conversation');
    } finally {
      setDeletingId(null);
    }
  };

  const handleRestore = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    setRestoringId(conversationId);
    try {
      await restoreConversation(conversationId);
      onActivated();
    } catch (err) {
      notifyError(err, 'Failed to restore conversation');
    } finally {
      setRestoringId(null);
    }
  };

  const handleResume = async (e: React.MouseEvent, run: ActiveChatRun) => {
    e.stopPropagation();
    setRestoringId(run.run_id);
    try {
      await resumeRun(run);
      onActivated();
    } catch (err) {
      notifyError(err, 'Failed to resume run');
    } finally {
      setRestoringId(null);
    }
  };

  const startRename = (e: React.MouseEvent, id: string, currentTitle: string) => {
    e.stopPropagation();
    setEditingId(id);
    setDraftTitle(currentTitle);
  };

  // Commit an inline rename. Open tabs go through `renameTab`; server
  // conversations get a client-side title override.
  const commitRename = (kind: 'tab' | 'server') => {
    if (editingId) {
      const next = draftTitle.trim();
      if (next) {
        if (kind === 'tab') renameTab(editingId, next);
        else { setTitleOverride(editingId, next); setRenameTick(t => t + 1); }
      }
    }
    setEditingId(null);
  };

  const renameKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, kind: 'tab' | 'server') => {
    if (e.key === 'Enter') { e.preventDefault(); commitRename(kind); }
    else if (e.key === 'Escape') { e.preventDefault(); setEditingId(null); }
  };

  // Group past conversations by coarse recency for sub-section labels.
  const pastByBucket = useMemo(() => {
    const order: Array<'Today' | 'Yesterday' | 'This week' | 'Older'> = ['Today', 'Yesterday', 'This week', 'Older'];
    const groups = new Map<string, typeof pastConversations>();
    for (const c of pastConversations) {
      const b = recencyBucket(c.last_message_at);
      (groups.get(b) ?? groups.set(b, []).get(b)!).push(c);
    }
    return order.filter(b => groups.has(b)).map(b => [b, groups.get(b)!] as const);
  }, [pastConversations]);

  const totalCount = filteredTabs.length + pastConversations.length + liveRuns.length;

  return (
    <>
      <div className="history-search">
        <Search size={14} />
        <input
          ref={searchRef}
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => {
            // Enter opens the first visible result (open tab, else past conv).
            if (e.key !== 'Enter') return;
            if (filteredTabs.length > 0) handleSelect(filteredTabs[0].id);
            else if (pastConversations.length > 0) handleRestore(e as unknown as React.MouseEvent, pastConversations[0].conversation_id);
          }}
          placeholder="Search conversations..."
        />
        {searchQuery && (
          <button className="clear-search" onClick={() => setSearchQuery('')}>
            <X size={12} />
          </button>
        )}
      </div>

      <div className="history-list">
        {/* Resume Running Section — runs whose owning tab was closed */}
        {liveRuns.filter(r => r.message.toLowerCase().includes(query)).length > 0 && (
          <>
            <div className="history-section-label">Resume Running</div>
            {liveRuns
              .filter(r => r.message.toLowerCase().includes(query))
              .map(run => (
                <div
                  key={run.run_id}
                  className="history-item history-item-running"
                  onClick={(e) => handleResume(e, run)}
                  title="Reopen this run and continue streaming"
                >
                  <div className="history-item-icon">
                    {restoringId === run.run_id
                      ? <Loader2 size={14} className="spin" />
                      : <Radio size={14} className="history-running-dot" />}
                  </div>
                  <div className="history-item-info">
                    <span className="history-item-title">
                      {run.message.slice(0, 60) || 'Running conversation'}
                    </span>
                    <span className="history-item-meta">running · {formatDate(run.updated_at)}</span>
                  </div>
                </div>
              ))}
          </>
        )}

        {/* Open Tabs Section */}
        {filteredTabs.length > 0 && (
          <>
            <div className="history-section-label">Open Tabs</div>
            {filteredTabs.map(tab => (
              <div
                key={tab.id}
                className={`history-item ${tab.id === activeTabId ? 'active' : ''}`}
                onClick={() => editingId === tab.id ? undefined : handleSelect(tab.id)}
              >
                <div className="history-item-icon">
                  <MessageSquare size={14} />
                </div>
                <div className="history-item-info">
                  {editingId === tab.id ? (
                    <input
                      className="history-item-rename"
                      value={draftTitle}
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setDraftTitle(e.target.value)}
                      onBlur={() => commitRename('tab')}
                      onKeyDown={(e) => renameKeyDown(e, 'tab')}
                      aria-label="Rename conversation"
                    />
                  ) : (
                    <span
                      className="history-item-title"
                      onDoubleClick={(e) => startRename(e, tab.id, tab.title)}
                    >
                      {tab.title}
                    </span>
                  )}
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
                        onClick={(e) => startRename(e, tab.id, tab.title)}
                        title="Rename"
                      >
                        <Pencil size={12} />
                      </button>
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

        {/* Past Conversations Section (from server), grouped by recency */}
        {pastByBucket.map(([bucket, convs]) => (
          <div key={bucket}>
            <div className="history-section-label">{bucket}</div>
            {convs.map(conv => {
              const displayTitle = getDisplayTitle(conv.conversation_id, conv.title);
              return (
                <div
                  key={conv.conversation_id}
                  className="history-item history-item-server"
                  onClick={(e) => editingId === conv.conversation_id ? undefined : handleRestore(e, conv.conversation_id)}
                >
                  <div className="history-item-icon">
                    <Download size={14} />
                  </div>
                  <div className="history-item-info">
                    {editingId === conv.conversation_id ? (
                      <input
                        className="history-item-rename"
                        value={draftTitle}
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setDraftTitle(e.target.value)}
                        onBlur={() => commitRename('server')}
                        onKeyDown={(e) => renameKeyDown(e, 'server')}
                        aria-label="Rename conversation"
                      />
                    ) : (
                      <span
                        className="history-item-title"
                        onDoubleClick={(e) => startRename(e, conv.conversation_id, displayTitle)}
                      >
                        {displayTitle}
                      </span>
                    )}
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
                      <>
                        <button
                          className="history-item-action"
                          onClick={(e) => startRename(e, conv.conversation_id, displayTitle)}
                          title="Rename"
                        >
                          <Pencil size={12} />
                        </button>
                        <button
                          className="history-item-action history-item-action-danger"
                          onClick={(e) => handleDeleteServerConversation(e, conv.conversation_id, displayTitle)}
                          title="Delete conversation"
                        >
                          <Trash2 size={12} />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}

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
    </>
  );
}

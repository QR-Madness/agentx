/**
 * useConversationList — the data + handlers behind the conversation list.
 *
 * Extracted from the old ConversationListContent so the desktop Conversations
 * sidebar and the mobile Conversations drawer share one source of truth for:
 * open tabs, resume-running runs, past (server) conversations grouped by
 * recency, plus search / restore / rename / delete / resume. Everything is read
 * from ConversationContext; hosts supply `onActivated`, called after a
 * conversation is switched/restored/resumed so the host can dismiss itself.
 */

import { useState, useRef, useEffect, useMemo } from 'react';
import { useConversation } from '../contexts/ConversationContext';
import { useNotify } from '../contexts/NotificationContext';
import { orphanedRuns } from '../contexts/conversation/orphanedRuns';
import { api, type ActiveChatRun } from '../lib/api';
import { getDisplayTitle, setTitleOverride } from '../lib/conversationTitles';

export type RecencyBucket = 'Today' | 'Yesterday' | 'This week' | 'Older';

/** Bucket a date into a coarse recency group for section labelling. */
export function recencyBucket(dateStr: string | null): RecencyBucket {
  if (!dateStr) return 'Older';
  const days = Math.floor((Date.now() - new Date(dateStr).getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return 'This week';
  return 'Older';
}

export interface UseConversationListOptions {
  /** Called after a conversation is switched/restored/resumed so the host dismisses. */
  onActivated: () => void;
  /** Focus the search input on mount (default true). */
  autoFocusSearch?: boolean;
}

export function useConversationList({ onActivated, autoFocusSearch = true }: UseConversationListOptions) {
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

  // Refresh history + detached runs + focus search on mount (fetch-on-open).
  useEffect(() => {
    refreshHistory();
    api.listChatRuns().then(r => setRuns(r.runs)).catch(() => setRuns([]));
    if (autoFocusSearch) searchRef.current?.focus();
  }, [refreshHistory, autoFocusSearch]);

  const query = searchQuery.toLowerCase();

  // Runs still going whose owning tab is closed — offered as "Resume Running".
  const liveRuns = useMemo(() => orphanedRuns(runs, tabs), [runs, tabs]);

  const filteredTabs = tabs
    .filter(tab => tab.title.toLowerCase().includes(query))
    .sort((a, b) => new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime());

  // Exclude open tabs / live runs from server history so a running conversation
  // isn't also listed under "Past Conversations".
  const openSessionIds = new Set([
    ...tabs.map(t => t.sessionId).filter(Boolean),
    ...liveRuns.map(r => r.session_id).filter(Boolean),
  ]);

  const pastConversations = serverConversations
    .filter(c => !openSessionIds.has(c.conversation_id))
    .filter(c => c.title.toLowerCase().includes(query) || c.preview.toLowerCase().includes(query));

  const filteredLiveRuns = liveRuns.filter(r => r.message.toLowerCase().includes(query));

  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
    if (days === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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
    if (!window.confirm(`Permanently delete "${title}"?\n\nThis will remove the conversation from the server and cannot be undone.`)) return;
    setDeletingId(tabId);
    try {
      await deleteConversation(tabId);
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteServerConversation = async (e: React.MouseEvent, conversationId: string, title: string) => {
    e.stopPropagation();
    if (!window.confirm(`Permanently delete "${title}"?\n\nThis will remove the conversation from the server and cannot be undone.`)) return;
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
    const order: RecencyBucket[] = ['Today', 'Yesterday', 'This week', 'Older'];
    const groups = new Map<RecencyBucket, typeof pastConversations>();
    for (const c of pastConversations) {
      const b = recencyBucket(c.last_message_at);
      (groups.get(b) ?? groups.set(b, []).get(b)!).push(c);
    }
    return order.filter(b => groups.has(b)).map(b => [b, groups.get(b)!] as const);
  }, [pastConversations]);

  const totalCount = filteredTabs.length + pastConversations.length + filteredLiveRuns.length;

  return {
    // state
    searchQuery, setSearchQuery, query, searchRef,
    activeTabId, isLoadingHistory,
    restoringId, deletingId, editingId, draftTitle, setDraftTitle,
    // derived data
    filteredTabs, filteredLiveRuns, pastByBucket,
    openCount: tabs.length, pastCount: pastConversations.length, totalCount,
    // handlers
    handleSelect, handleClose, handleDeleteTab, handleDeleteServerConversation,
    handleRestore, handleResume, startRename, commitRename, renameKeyDown,
    formatDate, getDisplayTitle,
  };
}

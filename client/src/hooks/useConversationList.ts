/**
 * useConversationList — the data + handlers behind the conversation list/sidebar
 * and the mobile drawer. Owns: normalization of open tabs + past (server)
 * conversations into one `ConversationItem[]`, meta-driven partitioning
 * (pinned / groups / open / past / archived), search, inline rename, restore /
 * resume / delete, multi-select + bulk actions, and the management mutators
 * (pin / archive / group / icon / color) over `conversationMeta`.
 *
 * Hosts supply `onActivated`, called after a conversation is
 * switched/restored/resumed so the host can dismiss itself.
 */

import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useConversation } from '../contexts/ConversationContext';
import { useNotify } from '../contexts/NotificationContext';
import { useConfirm } from '../components/ui/ConfirmDialog';
import { orphanedRuns } from '../contexts/conversation/orphanedRuns';
import { api, type ActiveChatRun } from '../lib/api';
import {
  type ConversationMeta,
  useConversationMeta, getMeta, patchMeta, setTitleOverride, listGroups,
} from '../lib/conversationMeta';

export type RecencyBucket = 'Today' | 'Yesterday' | 'This week' | 'Older';

export interface ConversationItem {
  key: string;                 // session/conversation id (the meta key)
  kind: 'tab' | 'server';
  title: string;               // display title (meta override applied)
  meta: ConversationMeta;
  lastMessageAt: string | null;
  messageCount: number;
  preview?: string;
  isStreaming?: boolean;
  tabId?: string;              // kind==='tab'
  conversationId?: string;     // server id (tab.sessionId for tabs)
}

const GROUPS_COLLAPSED_KEY = 'agentx:conv-groups-collapsed';
const ARCHIVED_COLLAPSED_KEY = 'agentx:conv-archived-collapsed';

function recencyBucket(dateStr: string | null): RecencyBucket {
  if (!dateStr) return 'Older';
  const days = Math.floor((Date.now() - new Date(dateStr).getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return 'This week';
  return 'Older';
}

function readSet(key: string): Set<string> {
  try {
    const raw = localStorage.getItem(key);
    return new Set(raw ? (JSON.parse(raw) as string[]) : []);
  } catch { return new Set(); }
}
function writeSet(key: string, set: Set<string>): void {
  try { localStorage.setItem(key, JSON.stringify([...set])); } catch { /* ignore */ }
}

export interface UseConversationListOptions {
  onActivated: () => void;
  autoFocusSearch?: boolean;
}

export function useConversationList({ onActivated, autoFocusSearch = true }: UseConversationListOptions) {
  const {
    tabs, activeTabId, switchTab, closeTab, renameTab,
    serverConversations, isLoadingHistory, restoreConversation, refreshHistory,
    deleteConversation, deleteServerConversation, resumeRun,
  } = useConversation();
  const { notifyError } = useNotify();
  const confirm = useConfirm();
  useConversationMeta(); // re-render on any meta change

  const [searchQuery, setSearchQuery] = useState('');
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [runs, setRuns] = useState<ActiveChatRun[]>([]);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);

  // Multi-select
  const [selectionMode, setSelectionMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // Collapsible sections (persisted)
  const [groupsCollapsed, setGroupsCollapsed] = useState<Set<string>>(() => readSet(GROUPS_COLLAPSED_KEY));
  const [archivedCollapsed, setArchivedCollapsed] = useState<boolean>(() => {
    try { return localStorage.getItem(ARCHIVED_COLLAPSED_KEY) !== '0'; } catch { return true; }
  });

  useEffect(() => {
    refreshHistory();
    api.listChatRuns().then(r => setRuns(r.runs)).catch(() => setRuns([]));
    if (autoFocusSearch) searchRef.current?.focus();
  }, [refreshHistory, autoFocusSearch]);

  const query = searchQuery.toLowerCase();
  const liveRuns = useMemo(() => orphanedRuns(runs, tabs), [runs, tabs]);
  const filteredLiveRuns = liveRuns.filter(r => r.message.toLowerCase().includes(query));

  // Open-tab / live-run session ids excluded from server history (no dupes).
  const openSessionIds = new Set([
    ...tabs.map(t => t.sessionId).filter(Boolean),
    ...liveRuns.map(r => r.session_id).filter(Boolean),
  ]);

  // --- Normalize into one item list ---
  const items = useMemo<ConversationItem[]>(() => {
    const out: ConversationItem[] = [];
    for (const tab of tabs) {
      const key = tab.sessionId ?? tab.id;
      const meta = getMeta(key);
      out.push({
        key, kind: 'tab', title: meta.title ?? tab.title, meta,
        lastMessageAt: tab.lastMessageAt, messageCount: tab.messages.length,
        isStreaming: tab.isStreaming, tabId: tab.id, conversationId: tab.sessionId ?? undefined,
      });
    }
    for (const conv of serverConversations) {
      if (openSessionIds.has(conv.conversation_id)) continue;
      const meta = getMeta(conv.conversation_id);
      out.push({
        key: conv.conversation_id, kind: 'server', title: meta.title ?? conv.title, meta,
        lastMessageAt: conv.last_message_at, messageCount: conv.message_count,
        preview: conv.preview, conversationId: conv.conversation_id,
      });
    }
    return out;
    // openSessionIds is derived from tabs+liveRuns; depend on those.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tabs, serverConversations, liveRuns]);

  const matches = (it: ConversationItem) =>
    it.title.toLowerCase().includes(query) || (it.preview ?? '').toLowerCase().includes(query);

  const visible = items.filter(matches);

  // --- Partition by meta ---
  const pinned = visible
    .filter(it => it.meta.pinned)
    .sort((a, b) => new Date(b.lastMessageAt ?? 0).getTime() - new Date(a.lastMessageAt ?? 0).getTime());

  const archived = visible.filter(it => it.meta.archived && !it.meta.pinned);

  const groupable = visible.filter(it => !it.meta.pinned && !it.meta.archived);

  const groups = useMemo(() => {
    const byGroup = new Map<string, ConversationItem[]>();
    for (const it of groupable) {
      if (!it.meta.group) continue;
      (byGroup.get(it.meta.group) ?? byGroup.set(it.meta.group, []).get(it.meta.group)!).push(it);
    }
    return [...byGroup.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [groupable]);

  const openItems = groupable
    .filter(it => it.kind === 'tab' && !it.meta.group)
    .sort((a, b) => new Date(b.lastMessageAt ?? 0).getTime() - new Date(a.lastMessageAt ?? 0).getTime());

  const pastByBucket = useMemo(() => {
    const order: RecencyBucket[] = ['Today', 'Yesterday', 'This week', 'Older'];
    const m = new Map<RecencyBucket, ConversationItem[]>();
    for (const it of groupable) {
      if (it.kind !== 'server' || it.meta.group) continue;
      const b = recencyBucket(it.lastMessageAt);
      (m.get(b) ?? m.set(b, []).get(b)!).push(it);
    }
    return order.filter(b => m.has(b)).map(b => [b, m.get(b)!] as const);
  }, [groupable]);

  const itemByKey = useMemo(() => new Map(items.map(it => [it.key, it])), [items]);

  const totalCount = visible.length + filteredLiveRuns.length;

  const formatDate = (dateStr: string | null): string => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
    if (days === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  // --- Activation ---
  const openItem = (it: ConversationItem) => {
    if (it.kind === 'tab' && it.tabId) { switchTab(it.tabId); onActivated(); return; }
    if (it.kind === 'server') { void handleRestore(it); }
  };

  const handleRestore = async (it: ConversationItem) => {
    setRestoringId(it.key);
    try {
      await restoreConversation(it.key);
      onActivated();
    } catch (err) {
      notifyError(err, 'Failed to restore conversation');
    } finally {
      setRestoringId(null);
    }
  };

  const handleResume = async (run: ActiveChatRun) => {
    setRestoringId(run.run_id);
    try { await resumeRun(run); onActivated(); }
    catch (err) { notifyError(err, 'Failed to resume run'); }
    finally { setRestoringId(null); }
  };

  const closeOpenTab = (it: ConversationItem) => { if (it.tabId) closeTab(it.tabId); };

  // --- Delete (themed confirm) ---
  const deleteItem = async (it: ConversationItem) => {
    const ok = await confirm({
      title: 'Delete conversation?',
      body: `"${it.title}" will be permanently removed from the server. This can't be undone.`,
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    setDeletingId(it.key);
    try {
      if (it.kind === 'tab' && it.tabId) await deleteConversation(it.tabId);
      else await deleteServerConversation(it.key);
    } catch (err) {
      notifyError(err, 'Failed to delete conversation');
    } finally {
      setDeletingId(null);
    }
  };

  // --- Inline rename ---
  const startRename = (it: ConversationItem) => { setEditingKey(it.key); setDraftTitle(it.title); };
  const commitRename = (it: ConversationItem) => {
    const next = draftTitle.trim();
    if (next) {
      if (it.kind === 'tab' && it.tabId) renameTab(it.tabId, next);
      else setTitleOverride(it.key, next);
    }
    setEditingKey(null);
  };
  const renameKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, it: ConversationItem) => {
    if (e.key === 'Enter') { e.preventDefault(); commitRename(it); }
    else if (e.key === 'Escape') { e.preventDefault(); setEditingKey(null); }
  };

  // --- Management mutators ---
  const togglePin = (it: ConversationItem) => patchMeta(it.key, { pinned: !it.meta.pinned });
  const toggleArchive = (it: ConversationItem) => patchMeta(it.key, { archived: !it.meta.archived });
  const setGroup = (it: ConversationItem, group: string | undefined) => patchMeta(it.key, { group });
  const setGroupByKey = (key: string, group: string | undefined) => patchMeta(key, { group });
  const setIcon = (key: string, icon: string | undefined) => patchMeta(key, { icon });
  const setColor = (key: string, color: string | undefined) => patchMeta(key, { color });

  // --- Multi-select + bulk ---
  const toggleSelect = (key: string) =>
    setSelected(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; });
  const enterSelection = (key?: string) => {
    setSelectionMode(true);
    if (key) setSelected(new Set([key]));
  };
  const clearSelection = useCallback(() => { setSelectionMode(false); setSelected(new Set()); }, []);

  const bulkPin = () => { selected.forEach(k => patchMeta(k, { pinned: true })); clearSelection(); };
  const bulkArchive = () => { selected.forEach(k => patchMeta(k, { archived: true })); clearSelection(); };
  const bulkDelete = async () => {
    const keys = [...selected];
    if (keys.length === 0) return;
    const ok = await confirm({
      title: `Delete ${keys.length} conversation${keys.length > 1 ? 's' : ''}?`,
      body: 'They will be permanently removed from the server. This can\'t be undone.',
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    for (const key of keys) {
      const it = itemByKey.get(key);
      if (!it) continue;
      setDeletingId(key);
      try {
        if (it.kind === 'tab' && it.tabId) await deleteConversation(it.tabId);
        else await deleteServerConversation(it.key);
      } catch (err) {
        notifyError(err, 'Failed to delete conversation');
      }
    }
    setDeletingId(null);
    clearSelection();
  };

  // --- Section collapse ---
  const toggleGroupCollapse = (group: string) =>
    setGroupsCollapsed(prev => {
      const n = new Set(prev); n.has(group) ? n.delete(group) : n.add(group);
      writeSet(GROUPS_COLLAPSED_KEY, n); return n;
    });
  const toggleArchivedCollapse = () =>
    setArchivedCollapsed(prev => {
      const n = !prev; try { localStorage.setItem(ARCHIVED_COLLAPSED_KEY, n ? '1' : '0'); } catch { /* ignore */ }
      return n;
    });

  return {
    // search + state
    searchQuery, setSearchQuery, searchRef, query,
    activeTabId, isLoadingHistory, restoringId, deletingId,
    editingKey, draftTitle, setDraftTitle,
    // data
    pinned, groups, openItems, pastByBucket, archived, filteredLiveRuns,
    totalCount, openCount: tabs.length,
    groupsCollapsed, archivedCollapsed,
    existingGroups: listGroups(),
    // selection
    selectionMode, selected, toggleSelect, enterSelection, clearSelection,
    bulkPin, bulkArchive, bulkDelete,
    // handlers
    openItem, handleResume, closeOpenTab, deleteItem,
    startRename, commitRename, renameKeyDown,
    togglePin, toggleArchive, setGroup, setGroupByKey, setIcon, setColor,
    toggleGroupCollapse, toggleArchivedCollapse,
    formatDate,
  };
}

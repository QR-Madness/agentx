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
  workspaceId?: string;        // the project it belongs to (server membership, meta fallback)
}

/** Stable, row-facing handler bundle passed to a memoized `ConversationRow`. */
export interface RowHandlers {
  openItem: (it: ConversationItem) => void;
  closeOpenTab: (it: ConversationItem) => void;
  deleteItem: (it: ConversationItem) => void | Promise<void>;
  startRename: (it: ConversationItem) => void;
  commitRename: (it: ConversationItem) => void;
  renameKeyDown: (e: React.KeyboardEvent<HTMLInputElement>, it: ConversationItem) => void;
  setDraftTitle: (v: string) => void;
  togglePin: (it: ConversationItem) => void;
  toggleArchive: (it: ConversationItem) => void;
  setGroup: (it: ConversationItem, group: string | undefined) => void;
  setProject: (it: ConversationItem, workspaceId: string | undefined) => void;
  setColor: (key: string, color: string | undefined) => void;
  toggleSelect: (key: string) => void;
  enterSelection: (key?: string) => void;
  existingGroups: string[];
  existingProjects: { id: string; name: string }[];
  formatDate: (dateStr: string | null) => string;
}

const GROUPS_COLLAPSED_KEY = 'agentx:conv-groups-collapsed';
const ARCHIVED_COLLAPSED_KEY = 'agentx:conv-archived-collapsed';

/** Home is a personal media space, never rendered as a project section. */
const HOME_WORKSPACE_ID = 'ws_home';

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
  const metaVersion = useConversationMeta(); // bump on any meta change → recompute items

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

  // Project (workspace) names for the sidebar's project sections + move menu.
  const [projectNames, setProjectNames] = useState<Map<string, string>>(new Map());

  useEffect(() => {
    refreshHistory();
    api.listChatRuns().then(r => setRuns(r.runs)).catch(() => setRuns([]));
    api.listWorkspaces()
      .then(r => setProjectNames(new Map(
        r.workspaces.filter(w => w.id !== HOME_WORKSPACE_ID).map(w => [w.id, w.name]))))
      .catch(() => setProjectNames(new Map()));
    if (autoFocusSearch) searchRef.current?.focus();
  }, [refreshHistory, autoFocusSearch]);

  const query = searchQuery.toLowerCase();

  // Cheap projection signature over only the tab fields the sidebar reads. It
  // stays byte-identical across streaming content tokens (`updateMessage` grows
  // a message body but touches neither the title, message count, lastMessageAt
  // nor isStreaming), so keying the memos below on `tabSig` instead of the
  // `tabs` reference stops the whole list from rebuilding on every token.
  const tabSig = tabs
    .map(t => `${t.id}|${t.sessionId ?? ''}|${t.title}|${t.lastMessageAt ?? ''}|${t.messages.length}|${t.isStreaming ? 1 : 0}`)
    .join('§');

  // eslint-disable-next-line react-hooks/exhaustive-deps -- reads tabs, keyed on tabSig
  const liveRuns = useMemo(() => orphanedRuns(runs, tabs), [runs, tabSig]);
  const filteredLiveRuns = useMemo(
    () => liveRuns.filter(r => r.message.toLowerCase().includes(query)),
    [liveRuns, query],
  );

  // --- Normalize into one item list ---
  const items = useMemo<ConversationItem[]>(() => {
    // Open-tab / live-run session ids excluded from server history (no dupes).
    const openSessionIds = new Set<string>([
      ...tabs.map(t => t.sessionId).filter((s): s is string => Boolean(s)),
      ...liveRuns.map(r => r.session_id).filter((s): s is string => Boolean(s)),
    ]);
    // Server membership by conversation id (covers open tabs too — the raw
    // fetch includes them even though they're skipped as list items below).
    const wsByConv = new Map<string, string>();
    for (const conv of serverConversations) {
      if (conv.workspace_id) wsByConv.set(conv.conversation_id, conv.workspace_id);
    }
    const out: ConversationItem[] = [];
    for (const tab of tabs) {
      const key = tab.sessionId ?? tab.id;
      const meta = getMeta(key);
      out.push({
        key, kind: 'tab', title: meta.title ?? tab.title, meta,
        lastMessageAt: tab.lastMessageAt, messageCount: tab.messages.length,
        isStreaming: tab.isStreaming, tabId: tab.id, conversationId: tab.sessionId ?? undefined,
        workspaceId: (tab.sessionId ? wsByConv.get(tab.sessionId) : undefined) ?? meta.workspaceId,
      });
    }
    for (const conv of serverConversations) {
      if (openSessionIds.has(conv.conversation_id)) continue;
      const meta = getMeta(conv.conversation_id);
      out.push({
        key: conv.conversation_id, kind: 'server', title: meta.title ?? conv.title, meta,
        lastMessageAt: conv.last_message_at, messageCount: conv.message_count,
        preview: conv.preview, conversationId: conv.conversation_id,
        workspaceId: conv.workspace_id ?? meta.workspaceId,
      });
    }
    return out;
    // tabSig captures the relevant tab fields; metaVersion re-pulls getMeta()
    // so pin/archive/icon/color/group changes reflect immediately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tabSig, serverConversations, liveRuns, metaVersion]);

  // --- Partition by meta (memoized so identity is stable across re-renders) ---
  // Precedence (each item has exactly one home): archived > project > group >
  // pinned > open/past. So moving an item always visibly relocates it even if
  // it's pinned (the pin flag still renders on the row), and archiving always
  // hides it. Projects only claim items whose workspace still exists (a deleted
  // project's conversations fall back to the plain sections).
  const visible = useMemo(
    () => items.filter(it =>
      it.title.toLowerCase().includes(query) || (it.preview ?? '').toLowerCase().includes(query)),
    [items, query],
  );
  const archived = useMemo(() => visible.filter(it => it.meta.archived), [visible]);
  const live = useMemo(() => visible.filter(it => !it.meta.archived), [visible]);
  const isProjectItem = useCallback((it: ConversationItem) =>
    Boolean(it.workspaceId && it.workspaceId !== HOME_WORKSPACE_ID && projectNames.has(it.workspaceId)),
    [projectNames]);
  const projected = useMemo(() => live.filter(isProjectItem), [live, isProjectItem]);
  const grouped = useMemo(() => live.filter(it => !isProjectItem(it) && it.meta.group), [live, isProjectItem]);
  const ungrouped = useMemo(() => live.filter(it => !isProjectItem(it) && !it.meta.group), [live, isProjectItem]);

  const pinned = useMemo(() => ungrouped
    .filter(it => it.meta.pinned)
    .sort((a, b) => new Date(b.lastMessageAt ?? 0).getTime() - new Date(a.lastMessageAt ?? 0).getTime()),
    [ungrouped]);

  const groups = useMemo(() => {
    const byGroup = new Map<string, ConversationItem[]>();
    for (const it of grouped) {
      (byGroup.get(it.meta.group!) ?? byGroup.set(it.meta.group!, []).get(it.meta.group!)!).push(it);
    }
    // Within a group, pinned items float to the top, then most-recent.
    for (const list of byGroup.values()) {
      list.sort((a, b) =>
        (Number(!!b.meta.pinned) - Number(!!a.meta.pinned)) ||
        (new Date(b.lastMessageAt ?? 0).getTime() - new Date(a.lastMessageAt ?? 0).getTime()),
      );
    }
    return [...byGroup.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [grouped]);

  // Project sections: [workspaceId, name, items], sorted by name; items sort
  // like groups (pinned first, then most-recent).
  const projects = useMemo(() => {
    const byWs = new Map<string, ConversationItem[]>();
    for (const it of projected) {
      (byWs.get(it.workspaceId!) ?? byWs.set(it.workspaceId!, []).get(it.workspaceId!)!).push(it);
    }
    for (const list of byWs.values()) {
      list.sort((a, b) =>
        (Number(!!b.meta.pinned) - Number(!!a.meta.pinned)) ||
        (new Date(b.lastMessageAt ?? 0).getTime() - new Date(a.lastMessageAt ?? 0).getTime()),
      );
    }
    return [...byWs.entries()]
      .map(([id, list]) => ({ id, name: projectNames.get(id) ?? id, items: list }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [projected, projectNames]);

  const openItems = useMemo(() => ungrouped
    .filter(it => it.kind === 'tab' && !it.meta.pinned)
    .sort((a, b) => new Date(b.lastMessageAt ?? 0).getTime() - new Date(a.lastMessageAt ?? 0).getTime()),
    [ungrouped]);

  const pastByBucket = useMemo(() => {
    const order: RecencyBucket[] = ['Today', 'Yesterday', 'This week', 'Older'];
    const m = new Map<RecencyBucket, ConversationItem[]>();
    for (const it of ungrouped) {
      if (it.kind !== 'server' || it.meta.pinned) continue;
      const b = recencyBucket(it.lastMessageAt);
      (m.get(b) ?? m.set(b, []).get(b)!).push(it);
    }
    return order.filter(b => m.has(b)).map(b => [b, m.get(b)!] as const);
  }, [ungrouped]);

  const itemByKey = useMemo(() => new Map(items.map(it => [it.key, it])), [items]);

  const totalCount = visible.length + filteredLiveRuns.length;
  const existingGroups = useMemo(() => listGroups(), [metaVersion]);
  const existingProjects = useMemo(
    () => [...projectNames.entries()]
      .map(([id, name]) => ({ id, name }))
      .sort((a, b) => a.name.localeCompare(b.name)),
    [projectNames]);

  // Inline rename reads the live draft through a ref so the commit callbacks stay
  // identity-stable across keystrokes (only the editing row re-renders, not the
  // whole list). `setDraft` keeps the controlled input value + the ref in sync.
  const draftTitleRef = useRef('');
  const setDraft = useCallback((v: string) => { draftTitleRef.current = v; setDraftTitle(v); }, []);

  const formatDate = useCallback((dateStr: string | null): string => {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
    if (days === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }, []);

  // --- Activation ---
  const handleRestore = useCallback(async (it: ConversationItem) => {
    setRestoringId(it.key);
    try {
      await restoreConversation(it.key);
      onActivated();
    } catch (err) {
      notifyError(err, 'Failed to restore conversation');
    } finally {
      setRestoringId(null);
    }
  }, [restoreConversation, onActivated, notifyError]);

  const openItem = useCallback((it: ConversationItem) => {
    if (it.kind === 'tab' && it.tabId) { switchTab(it.tabId); onActivated(); return; }
    if (it.kind === 'server') { void handleRestore(it); }
  }, [switchTab, onActivated, handleRestore]);

  const handleResume = useCallback(async (run: ActiveChatRun) => {
    setRestoringId(run.run_id);
    try { await resumeRun(run); onActivated(); }
    catch (err) { notifyError(err, 'Failed to resume run'); }
    finally { setRestoringId(null); }
  }, [resumeRun, onActivated, notifyError]);

  const closeOpenTab = useCallback((it: ConversationItem) => { if (it.tabId) closeTab(it.tabId); }, [closeTab]);

  // --- Delete (themed confirm) ---
  const deleteItem = useCallback(async (it: ConversationItem) => {
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
  }, [confirm, deleteConversation, deleteServerConversation, notifyError]);

  // --- Inline rename ---
  const startRename = useCallback((it: ConversationItem) => {
    setEditingKey(it.key); setDraft(it.title);
  }, [setDraft]);
  const commitRename = useCallback((it: ConversationItem) => {
    const next = draftTitleRef.current.trim();
    if (next) {
      if (it.kind === 'tab' && it.tabId) renameTab(it.tabId, next);
      else setTitleOverride(it.key, next);
    }
    setEditingKey(null);
  }, [renameTab]);
  const renameKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>, it: ConversationItem) => {
    if (e.key === 'Enter') { e.preventDefault(); commitRename(it); }
    else if (e.key === 'Escape') { e.preventDefault(); setEditingKey(null); }
  }, [commitRename]);

  // --- Management mutators (patchMeta is module-level, so these are stable) ---
  const togglePin = useCallback((it: ConversationItem) => patchMeta(it.key, { pinned: !it.meta.pinned }), []);
  const toggleArchive = useCallback((it: ConversationItem) => patchMeta(it.key, { archived: !it.meta.archived }), []);
  const setGroup = useCallback((it: ConversationItem, group: string | undefined) => patchMeta(it.key, { group }), []);

  // Move a conversation into/out of a project. Meta updates immediately (drives
  // the badge + next turn's workspace_id); the server link makes it durable for
  // real conversations (pre-session tabs persist membership on first message —
  // link upserts, so moving between projects needs no explicit unlink).
  const setProject = useCallback((it: ConversationItem, workspaceId: string | undefined) => {
    const previous = it.workspaceId;
    patchMeta(it.key, { workspaceId });
    void (async () => {
      try {
        if (!it.conversationId) return;
        if (workspaceId) await api.linkConversation(workspaceId, it.conversationId);
        else if (previous) await api.unlinkConversation(previous, it.conversationId);
        await refreshHistory();
      } catch (err) {
        notifyError(err, 'Could not move conversation');
      }
    })();
  }, [refreshHistory, notifyError]);
  const setGroupByKey = useCallback((key: string, group: string | undefined) => patchMeta(key, { group }), []);
  const setIcon = useCallback((key: string, icon: string | undefined) => patchMeta(key, { icon }), []);
  const setColor = useCallback((key: string, color: string | undefined) => patchMeta(key, { color }), []);

  // --- Multi-select + bulk ---
  const toggleSelect = useCallback((key: string) =>
    setSelected(prev => { const n = new Set(prev); n.has(key) ? n.delete(key) : n.add(key); return n; }), []);
  const enterSelection = useCallback((key?: string) => {
    setSelectionMode(true);
    if (key) setSelected(new Set([key]));
  }, []);
  const clearSelection = useCallback(() => { setSelectionMode(false); setSelected(new Set()); }, []);

  const bulkPin = useCallback(() => { selected.forEach(k => patchMeta(k, { pinned: true })); clearSelection(); }, [selected, clearSelection]);
  const bulkArchive = useCallback(() => { selected.forEach(k => patchMeta(k, { archived: true })); clearSelection(); }, [selected, clearSelection]);
  const bulkDelete = useCallback(async () => {
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
  }, [selected, confirm, itemByKey, deleteConversation, deleteServerConversation, notifyError, clearSelection]);

  // --- Section collapse ---
  const toggleGroupCollapse = useCallback((group: string) =>
    setGroupsCollapsed(prev => {
      const n = new Set(prev); n.has(group) ? n.delete(group) : n.add(group);
      writeSet(GROUPS_COLLAPSED_KEY, n); return n;
    }), []);
  const toggleArchivedCollapse = useCallback(() =>
    setArchivedCollapsed(prev => {
      const n = !prev; try { localStorage.setItem(ARCHIVED_COLLAPSED_KEY, n ? '1' : '0'); } catch { /* ignore */ }
      return n;
    }), []);

  // Stable bundle of the row-facing handlers + the few values rows read, so a
  // memoized `ConversationRow` only re-renders when its own item/flags change
  // (not on every parent render during streaming).
  const rowHandlers = useMemo<RowHandlers>(() => ({
    openItem, closeOpenTab, deleteItem,
    startRename, commitRename, renameKeyDown, setDraftTitle: setDraft,
    togglePin, toggleArchive, setGroup, setProject, setColor,
    toggleSelect, enterSelection,
    existingGroups, existingProjects, formatDate,
  }), [
    openItem, closeOpenTab, deleteItem, startRename, commitRename, renameKeyDown,
    setDraft, togglePin, toggleArchive, setGroup, setProject, setColor, toggleSelect,
    enterSelection, existingGroups, existingProjects, formatDate,
  ]);

  return {
    // search + state
    searchQuery, setSearchQuery, searchRef, query,
    activeTabId, isLoadingHistory, restoringId, deletingId,
    editingKey, draftTitle, setDraftTitle: setDraft,
    // data
    pinned, projects, groups, openItems, pastByBucket, archived, filteredLiveRuns,
    totalCount, openCount: tabs.length,
    groupsCollapsed, archivedCollapsed,
    existingGroups, existingProjects,
    // selection
    selectionMode, selected, toggleSelect, enterSelection, clearSelection,
    bulkPin, bulkArchive, bulkDelete,
    // handlers
    openItem, handleResume, closeOpenTab, deleteItem,
    startRename, commitRename, renameKeyDown,
    togglePin, toggleArchive, setGroup, setGroupByKey, setProject, setIcon, setColor,
    toggleGroupCollapse, toggleArchivedCollapse,
    formatDate,
    // stable bundle for ConversationRow
    rowHandlers,
  };
}

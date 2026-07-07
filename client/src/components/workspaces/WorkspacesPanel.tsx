/**
 * WorkspacesPanel — the **Projects** hub (user-facing name; internal naming stays
 * `workspace`). Left: project list (create / select / delete) with the reserved
 * Home media space pinned at the bottom. Right: the selected project's hub —
 * name + description, custom instructions (debounced autosave; injected into
 * every turn server-side), files with ingestion status + click-to-upload (drop
 * is best-effort — the Tauri webview doesn't deliver dropped files), and the
 * project's conversations (durable server membership) with "new chat here".
 * Polls while any document is still ingesting (pending).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Check, FilePlus2, FileText, FolderKanban, Home, Link2, Link2Off, Loader2, MessageSquare,
  MessageSquarePlus, Pencil, Plus, RotateCcw, Terminal, Trash2, Upload, X,
} from 'lucide-react';
import { api, toApiError, type ConversationSummary, type Workspace, type WorkspaceDocument } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { useConversation } from '../../contexts/ConversationContext';
import { getDisplayTitle, getMeta, patchMeta, useConversationMeta } from '../../lib/conversationMeta';
import { useConfirm } from '../ui/ConfirmDialog';
import { Badge, Button, IconButton, Input, SegmentedControl, Textarea } from '../ui';
import { WorkspaceContainerCard } from './WorkspaceContainerCard';
import { DocumentPreviewModal } from './DocumentPreviewModal';

/** Reserved personal media space — files only, never a project. */
const HOME_ID = 'ws_home';

const DESCRIPTION_MAX = 500;
const INSTRUCTIONS_MAX = 8000;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
  if (days === 0) return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

// Documents are keyed by a flat filename; a leading `folder/` prefix (e.g.
// `generated/`, `uploads/`) is a convention used to group scratch media so it
// can be viewed and bulk-cleared apart from curated documents.
const FOLDER_LABELS: Record<string, string> = { generated: 'Generated', uploads: 'Uploaded' };
const ROOT_FOLDER = '';
// Folders whose contents are agent/scratch media — safe to offer a bulk "Clear".
const CLEARABLE_FOLDERS = new Set(['generated', 'uploads']);

function folderOf(filename: string): string {
  const slash = filename.indexOf('/');
  return slash === -1 ? ROOT_FOLDER : filename.slice(0, slash);
}

function leafName(filename: string, folder: string): string {
  return folder === ROOT_FOLDER ? filename : filename.slice(folder.length + 1);
}

interface DocGroup { folder: string; label: string; docs: WorkspaceDocument[]; }

function groupDocuments(documents: WorkspaceDocument[]): DocGroup[] {
  const byFolder = new Map<string, WorkspaceDocument[]>();
  for (const doc of documents) {
    const folder = folderOf(doc.filename);
    (byFolder.get(folder) ?? byFolder.set(folder, []).get(folder)!).push(doc);
  }
  // Root ("Files") first, then named folders alphabetically.
  return Array.from(byFolder.entries())
    .sort(([a], [b]) => (a === ROOT_FOLDER ? -1 : b === ROOT_FOLDER ? 1 : a.localeCompare(b)))
    .map(([folder, docs]) => ({
      folder,
      label: folder === ROOT_FOLDER ? 'Files' : (FOLDER_LABELS[folder] ?? folder),
      docs,
    }));
}

export function WorkspacesPanel({
  onClose,
  initialWorkspaceId,
}: {
  onClose?: () => void;
  initialWorkspaceId?: string;
}) {
  const notify = useNotify();
  const confirm = useConfirm();
  const { activeTab, tabs, addTab, switchTab, restoreConversation, refreshHistory } = useConversation();
  useConversationMeta();  // re-render when the conversation↔project tag changes
  const convKey = activeTab ? (activeTab.sessionId ?? activeTab.id) : null;
  const attachedId = convKey ? getMeta(convKey).workspaceId : undefined;

  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(initialWorkspaceId ?? null);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [creating, setCreating] = useState(false);
  const [draftName, setDraftName] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [editingDesc, setEditingDesc] = useState(false);
  const [descDraft, setDescDraft] = useState('');
  const [instrDraft, setInstrDraft] = useState('');
  const [instrStatus, setInstrStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
  const [preview, setPreview] = useState<{ doc: WorkspaceDocument; edit: boolean } | null>(null);
  const [namingDoc, setNamingDoc] = useState(false);
  const [docNameDraft, setDocNameDraft] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshWorkspaces = useCallback(async () => {
    try {
      const { workspaces } = await api.listWorkspaces();
      setWorkspaces(workspaces);
      // Prefer the first real project over Home for the default selection.
      setSelectedId(prev =>
        prev ?? workspaces.find(w => w.id !== HOME_ID)?.id ?? workspaces[0]?.id ?? null);
    } catch (err) {
      notify.notifyError(err, 'Could not load projects');
    } finally {
      setLoading(false);
    }
  }, [notify]);

  const refreshDocuments = useCallback(async (workspaceId: string) => {
    try {
      const { documents } = await api.listDocuments(workspaceId);
      setDocuments(documents);
    } catch (err) {
      notify.notifyError(err, 'Could not load files');
    }
  }, [notify]);

  const refreshConversations = useCallback(async (workspaceId: string) => {
    try {
      const { conversations } = await api.listWorkspaceConversations(workspaceId);
      setConversations(conversations);
    } catch {
      setConversations([]);  // older server without membership — degrade quietly
    }
  }, []);

  useEffect(() => { void refreshWorkspaces(); }, [refreshWorkspaces]);
  useEffect(() => {
    if (selectedId) void refreshDocuments(selectedId);
    else setDocuments([]);
  }, [selectedId, refreshDocuments]);
  useEffect(() => {
    if (selectedId && selectedId !== HOME_ID) void refreshConversations(selectedId);
    else setConversations([]);
  }, [selectedId, refreshConversations]);

  // Poll while any document is still ingesting.
  useEffect(() => {
    if (!selectedId || !documents.some(d => d.status === 'pending')) return;
    const t = setInterval(() => { void refreshDocuments(selectedId); }, 3000);
    return () => clearInterval(t);
  }, [selectedId, documents, refreshDocuments]);

  // --- Instructions: debounced autosave (the pending write carries its own
  // workspace id, so switching projects mid-debounce still saves correctly). ---
  const instrTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const instrPending = useRef<{ id: string; text: string } | null>(null);

  const flushInstructions = useCallback(async () => {
    if (instrTimer.current) { clearTimeout(instrTimer.current); instrTimer.current = null; }
    const pending = instrPending.current;
    instrPending.current = null;
    if (!pending) return;
    try {
      await api.updateWorkspace(pending.id, { instructions: pending.text });
      setInstrStatus('saved');
      setWorkspaces(prev =>
        prev.map(w => (w.id === pending.id ? { ...w, instructions: pending.text } : w)));
    } catch (err) {
      setInstrStatus('idle');
      notify.notifyError(err, 'Could not save instructions');
    }
  }, [notify]);

  const onInstructionsChange = useCallback((workspaceId: string, text: string) => {
    setInstrDraft(text);
    instrPending.current = { id: workspaceId, text };
    setInstrStatus('saving');
    if (instrTimer.current) clearTimeout(instrTimer.current);
    instrTimer.current = setTimeout(() => { void flushInstructions(); }, 800);
  }, [flushInstructions]);

  // Flush any pending edit when the panel unmounts (drawer closed mid-debounce).
  useEffect(() => () => { void flushInstructions(); }, [flushInstructions]);

  const selected = workspaces.find(w => w.id === selectedId) ?? null;
  const isHome = selected?.id === HOME_ID;

  // Seed the editors when the selection lands on a different project.
  const seededFor = useRef<string | null>(null);
  useEffect(() => {
    if (!selected || seededFor.current === selected.id) return;
    void flushInstructions();  // save the previous project's pending edit first
    seededFor.current = selected.id;
    setInstrDraft(selected.instructions ?? '');
    setInstrStatus('idle');
    setEditingDesc(false);
    setRenaming(false);
  }, [selected, flushInstructions]);

  const createWorkspace = useCallback(async () => {
    const name = draftName.trim();
    if (!name) { setCreating(false); return; }
    try {
      const { workspace } = await api.createWorkspace(name);
      setCreating(false); setDraftName('');
      await refreshWorkspaces();
      setSelectedId(workspace.id);
    } catch (err) {
      notify.notifyError(err, 'Could not create project');
    }
  }, [draftName, notify, refreshWorkspaces]);

  const renameWorkspace = useCallback(async () => {
    const name = draftName.trim();
    if (!selectedId || !name) { setRenaming(false); return; }
    try {
      await api.renameWorkspace(selectedId, name);
      setRenaming(false); setDraftName('');
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not rename project');
    }
  }, [draftName, selectedId, notify, refreshWorkspaces]);

  const saveDescription = useCallback(async () => {
    if (!selectedId) { setEditingDesc(false); return; }
    const description = descDraft.trim();
    setEditingDesc(false);
    if (description === (selected?.description ?? '')) return;
    try {
      await api.updateWorkspace(selectedId, { description });
      setWorkspaces(prev =>
        prev.map(w => (w.id === selectedId ? { ...w, description } : w)));
    } catch (err) {
      notify.notifyError(err, 'Could not save description');
    }
  }, [descDraft, selected, selectedId, notify]);

  const toggleShell = useCallback(async (ws: Workspace) => {
    try {
      await api.setWorkspaceShell(ws.id, !ws.allow_shell);
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not change shell access');
    }
  }, [notify, refreshWorkspaces]);

  const setBackend = useCallback(async (ws: Workspace, backend: 'bubblewrap' | 'container') => {
    if (ws.shell_backend === backend) return;
    try {
      await api.setWorkspaceShellBackend(ws.id, backend);
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not change shell backend');
    }
  }, [notify, refreshWorkspaces]);

  const deleteWorkspace = useCallback(async (ws: Workspace) => {
    const ok = await confirm({
      title: `Delete "${ws.name}"?`,
      body: 'This removes the project and all its files. Conversations stay, but leave the project. This cannot be undone.',
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteWorkspace(ws.id);
      if (selectedId === ws.id) setSelectedId(null);
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not delete project');
    }
  }, [confirm, notify, refreshWorkspaces, selectedId]);

  // Attach/detach the active conversation. Meta drives the next turn's
  // workspace_id immediately; the server link makes membership durable (skipped
  // for Home — it's a media dump, not a project — and for pre-session tabs,
  // which persist membership on their first message instead).
  const toggleAttach = useCallback(async (workspaceId: string) => {
    if (!convKey) return;
    const detaching = attachedId === workspaceId;
    patchMeta(convKey, { workspaceId: detaching ? undefined : workspaceId });
    const sessionId = activeTab?.sessionId;
    if (!sessionId || workspaceId === HOME_ID) return;
    try {
      if (detaching) await api.unlinkConversation(workspaceId, sessionId);
      else await api.linkConversation(workspaceId, sessionId);
      void refreshConversations(workspaceId);
      void refreshHistory();
    } catch (err) {
      notify.notifyError(err, detaching ? 'Could not remove from project' : 'Could not add to project');
    }
  }, [convKey, attachedId, activeTab, refreshConversations, refreshHistory, notify]);

  const newChatInProject = useCallback(() => {
    if (!selected || isHome) return;
    const tab = addTab();
    // Pre-session attach: rides tab.id meta until the first message lands a
    // session id (setSessionId migrates the meta; the server records membership).
    patchMeta(tab.id, { workspaceId: selected.id });
    onClose?.();
  }, [selected, isHome, addTab, onClose]);

  const openConversation = useCallback(async (conv: ConversationSummary) => {
    const openTab = tabs.find(t => t.sessionId === conv.conversation_id);
    try {
      if (openTab) switchTab(openTab.id);
      else await restoreConversation(conv.conversation_id);
    } catch (err) {
      notify.notifyError(err, 'Could not open conversation');
      return;
    }
    onClose?.();
  }, [tabs, switchTab, restoreConversation, notify, onClose]);

  const removeConversation = useCallback(async (conv: ConversationSummary) => {
    if (!selectedId) return;
    try {
      await api.unlinkConversation(selectedId, conv.conversation_id);
      patchMeta(conv.conversation_id, { workspaceId: undefined });
      void refreshConversations(selectedId);
      void refreshHistory();
    } catch (err) {
      notify.notifyError(err, 'Could not remove from project');
    }
  }, [selectedId, refreshConversations, refreshHistory, notify]);

  const uploadFiles = useCallback(async (files: FileList | File[]) => {
    if (!selectedId) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await api.uploadDocument(selectedId, file);
      }
      await refreshDocuments(selectedId);
      await refreshWorkspaces();  // refresh used_bytes / count
    } catch (err) {
      notify.notifyError(err, 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [selectedId, notify, refreshDocuments, refreshWorkspaces]);

  const createTextDocument = useCallback(async () => {
    if (!selectedId) { setNamingDoc(false); return; }
    let name = docNameDraft.trim();
    if (!name) { setNamingDoc(false); return; }
    if (!/\.(md|markdown|txt)$/i.test(name)) name = `${name}.md`;
    try {
      const { document } = await api.createTextDocument(selectedId, name, `# ${name.replace(/\.(md|markdown|txt)$/i, '')}\n`);
      setNamingDoc(false); setDocNameDraft('');
      await refreshDocuments(selectedId);
      setPreview({ doc: document, edit: true });
    } catch (err) {
      const apiErr = toApiError(err);
      notify.notifyError(apiErr, apiErr.status === 409
        ? 'A file with that name already exists'
        : 'Could not create document');
    }
  }, [selectedId, docNameDraft, notify, refreshDocuments]);

  const onPreviewUpdated = useCallback((updated: WorkspaceDocument) => {
    // Reflect the save immediately; the pending-poll takes over for re-indexing.
    setDocuments(prev => prev.map(d => (d.id === updated.id ? updated : d)));
  }, []);

  const retryIngest = useCallback(async (doc: WorkspaceDocument) => {
    if (!selectedId) return;
    try {
      await api.reingestDocument(selectedId, doc.id);
      await refreshDocuments(selectedId);  // shows pending; the poll takes over
    } catch (err) {
      notify.notifyError(err, 'Could not retry ingestion');
    }
  }, [selectedId, refreshDocuments, notify]);

  const deleteDocument = useCallback(async (doc: WorkspaceDocument) => {
    const ok = await confirm({
      title: `Remove "${doc.filename}"?`, confirmLabel: 'Remove', danger: true,
    });
    if (!ok || !selectedId) return;
    try {
      await api.deleteDocument(selectedId, doc.id);
      await refreshDocuments(selectedId);
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not remove file');
    }
  }, [confirm, notify, refreshDocuments, refreshWorkspaces, selectedId]);

  const clearGroup = useCallback(async (group: DocGroup) => {
    if (!selectedId) return;
    const noun = group.docs.length === 1 ? 'file' : 'files';
    const ok = await confirm({
      title: `Clear ${group.docs.length} ${group.label.toLowerCase()} ${noun}?`,
      body: 'This permanently removes these files from the project. This cannot be undone.',
      confirmLabel: 'Clear', danger: true,
    });
    if (!ok) return;
    try {
      for (const doc of group.docs) await api.deleteDocument(selectedId, doc.id);
      await refreshDocuments(selectedId);
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not clear files');
    }
  }, [confirm, notify, refreshDocuments, refreshWorkspaces, selectedId]);

  const docGroups = useMemo(() => groupDocuments(documents), [documents]);
  const projects = useMemo(() => workspaces.filter(w => w.id !== HOME_ID), [workspaces]);
  const homeWs = useMemo(() => workspaces.find(w => w.id === HOME_ID) ?? null, [workspaces]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-surface-base text-fg">
      <header className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="flex items-center gap-2 text-base font-semibold tracking-tight">
          <FolderKanban size={18} className="text-accent" /> Projects
        </h2>
        <div className="flex items-center gap-1.5">
          <Button size="sm" onClick={() => { setCreating(true); setDraftName(''); }}>
            <Plus size={15} /> New
          </Button>
          {onClose && (
            <IconButton aria-label="Close" onClick={onClose}>
              <X size={18} />
            </IconButton>
          )}
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Project rail — sunken so the detail column reads as the raised page */}
        <aside className="flex w-56 shrink-0 flex-col overflow-y-auto border-r border-line-subtle bg-surface-sunken/50 p-2">
          {creating && (
            <form
              onSubmit={e => { e.preventDefault(); void createWorkspace(); }}
              className="mb-2 flex items-center gap-1"
            >
              <Input
                autoFocus value={draftName} placeholder="Project name…"
                onChange={e => setDraftName(e.target.value)}
                onBlur={() => { if (!draftName.trim()) setCreating(false); }}
                onKeyDown={e => { if (e.key === 'Escape') { setCreating(false); setDraftName(''); } }}
                className="ax-field--sm"
              />
              <IconButton type="submit" aria-label="Create project" size="sm" tone="accent">
                <Check size={15} />
              </IconButton>
            </form>
          )}
          <div className="min-h-0 flex-1">
            {loading ? (
              <div className="flex items-center gap-2 p-3 text-sm text-fg-muted">
                <Loader2 size={15} className="animate-spin" /> Loading…
              </div>
            ) : projects.length === 0 && !creating ? (
              <p className="p-3 text-sm text-fg-muted">
                No projects yet. Create one to bundle files, instructions, and conversations.
              </p>
            ) : (
              projects.map(ws => (
                <div
                  key={ws.id}
                  className={`group flex items-center justify-between rounded-md px-2.5 py-2 text-sm transition-colors ${
                    ws.id === selectedId
                      ? 'bg-[var(--accent-tint-soft)]'
                      : 'hover:bg-surface-hover'
                  }`}
                >
                  <button onClick={() => setSelectedId(ws.id)} className="min-w-0 flex-1 bg-transparent text-left">
                    <span className="flex items-center gap-1.5">
                      <span className={`truncate font-medium ${ws.id === selectedId ? 'text-accent' : 'text-fg'}`}>
                        {ws.name}
                      </span>
                      {ws.id === attachedId && <Link2 size={12} className="shrink-0 text-accent" />}
                    </span>
                    <span className="block text-2xs text-fg-muted">
                      {ws.document_count} files · {formatBytes(ws.used_bytes)}
                    </span>
                  </button>
                  <IconButton
                    aria-label={`Delete ${ws.name}`}
                    size="xs"
                    tone="danger"
                    className="ml-1 opacity-0 group-hover:opacity-100"
                    onClick={() => void deleteWorkspace(ws)}
                  >
                    <Trash2 size={14} />
                  </IconButton>
                </div>
              ))
            )}
          </div>
          {/* Home — personal media space, kept apart from projects */}
          {homeWs && (
            <div className="mt-2 border-t border-line-subtle pt-2">
              <button
                onClick={() => setSelectedId(HOME_ID)}
                className={`flex w-full items-center gap-2 rounded-md bg-transparent px-2.5 py-2 text-left text-sm transition-colors ${
                  selectedId === HOME_ID ? 'bg-[var(--accent-tint-soft)]' : 'hover:bg-surface-hover'
                }`}
              >
                <Home size={14} className={selectedId === HOME_ID ? 'shrink-0 text-accent' : 'shrink-0 text-fg-muted'} />
                <span className="min-w-0 flex-1">
                  <span className={`block truncate font-medium ${selectedId === HOME_ID ? 'text-accent' : 'text-fg'}`}>
                    {homeWs.name}
                  </span>
                  <span className="block text-2xs text-fg-muted">
                    Personal media · {homeWs.document_count} files
                  </span>
                </span>
              </button>
            </div>
          )}
        </aside>

        {/* Project hub */}
        <section className="flex min-w-0 flex-1 flex-col">
          {!selected ? (
            <div className="flex flex-1 items-center justify-center text-sm text-fg-muted">
              Select or create a project.
            </div>
          ) : (
            <>
              <div className="flex items-start justify-between gap-2 border-b border-line px-3 py-2">
                <div className="min-w-0 flex-1">
                  {renaming ? (
                    <form
                      onSubmit={e => { e.preventDefault(); void renameWorkspace(); }}
                      className="flex min-w-0 items-center gap-1"
                    >
                      <Input
                        autoFocus value={draftName}
                        onChange={e => setDraftName(e.target.value)}
                        onBlur={() => setRenaming(false)}
                        onKeyDown={e => { if (e.key === 'Escape') setRenaming(false); }}
                        className="ax-field--sm"
                      />
                      <IconButton type="submit" aria-label="Save name" size="sm" tone="accent">
                        <Check size={15} />
                      </IconButton>
                    </form>
                  ) : (
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="min-w-0 truncate text-lg font-semibold tracking-tight">{selected.name}</span>
                      <IconButton
                        aria-label="Rename project"
                        size="xs"
                        onClick={() => { setRenaming(true); setDraftName(selected.name); }}
                      >
                        <Pencil size={13} />
                      </IconButton>
                    </div>
                  )}
                  {!isHome && (editingDesc ? (
                    <form
                      onSubmit={e => { e.preventDefault(); void saveDescription(); }}
                      className="mt-1"
                    >
                      <Input
                        autoFocus value={descDraft} maxLength={DESCRIPTION_MAX}
                        placeholder="What is this project about?"
                        onChange={e => setDescDraft(e.target.value)}
                        onBlur={() => void saveDescription()}
                        onKeyDown={e => { if (e.key === 'Escape') setEditingDesc(false); }}
                        className="ax-field--sm text-xs"
                      />
                    </form>
                  ) : (
                    <button
                      className="group/desc mt-0.5 flex max-w-full items-center gap-1 bg-transparent text-left text-xs text-fg-secondary hover:text-fg"
                      title="Edit description"
                      onClick={() => { setEditingDesc(true); setDescDraft(selected.description ?? ''); }}
                    >
                      <span className="truncate">{selected.description || 'Add a description…'}</span>
                      <Pencil size={11} className="shrink-0 opacity-0 transition-opacity group-hover/desc:opacity-60" />
                    </button>
                  ))}
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => toggleShell(selected)}
                    className={selected.allow_shell ? 'border-warning/40 bg-warning/15 text-warning' : undefined}
                    title={
                      selected.allow_shell
                        ? 'Agents in this project can run sandboxed shell commands. Click to disable.'
                        : 'Allow agents to run sandboxed shell commands against this project (off by default).'
                    }
                  >
                    <Terminal size={13} />
                    {selected.allow_shell ? 'Shell on' : 'Allow shell'}
                  </Button>
                  {convKey ? (
                    <Button
                      variant={attachedId === selected.id ? 'primary' : 'secondary'}
                      size="sm"
                      onClick={() => void toggleAttach(selected.id)}
                      title={isHome
                        ? 'Attach Home to the active conversation so the agent can use its files'
                        : 'Add the active conversation to this project (its files and instructions apply)'}
                    >
                      {attachedId === selected.id ? <Link2Off size={13} /> : <Link2 size={13} />}
                      {attachedId === selected.id
                        ? (isHome ? 'Detach' : 'Remove from project')
                        : (isHome ? 'Attach to conversation' : 'Add to project')}
                    </Button>
                  ) : (
                    <span className="text-xs text-fg-muted">Open a chat to add it</span>
                  )}
                </div>
              </div>

              {selected.allow_shell && (
                <div className="border-b border-line px-3 py-2">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">Shell backend</span>
                    <SegmentedControl
                      size="sm"
                      ariaLabel="Shell backend"
                      value={selected.shell_backend}
                      onChange={b => void setBackend(selected, b)}
                      options={[
                        { value: 'bubblewrap', label: 'Bubblewrap', title: 'Lightweight jail — no install, no network' },
                        { value: 'container', label: 'Container', title: 'Persistent Docker container — installs + network' },
                      ]}
                    />
                  </div>
                </div>
              )}
              {selected.allow_shell && selected.shell_backend === 'container' && (
                <WorkspaceContainerCard workspaceId={selected.id} />
              )}

              <div className="min-h-0 flex-1 overflow-y-auto">
                {/* Instructions — injected into every turn in this project */}
                {!isHome && (
                  <div className="border-b border-line-subtle px-3 py-2.5">
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">
                        Instructions
                      </span>
                      <span className="font-mono text-2xs text-fg-muted">
                        {instrStatus === 'saving' && (
                          <span className="flex items-center gap-1">
                            <Loader2 size={10} className="animate-spin" /> Saving…
                          </span>
                        )}
                        {instrStatus === 'saved' && 'Saved'}
                        {instrStatus === 'idle' && `${instrDraft.length.toLocaleString()} / ${INSTRUCTIONS_MAX.toLocaleString()}`}
                      </span>
                    </div>
                    <Textarea
                      value={instrDraft}
                      maxLength={INSTRUCTIONS_MAX}
                      rows={instrDraft ? 5 : 2}
                      placeholder="Standing guidance the agent follows in every conversation of this project — tone, goals, constraints, what the files are for…"
                      onChange={e => onInstructionsChange(selected.id, e.target.value)}
                      className="ax-field--sm resize-y"
                    />
                  </div>
                )}

                {/* Click-to-upload is the primary path — HTML5 drag-drop doesn't
                    deliver files in the Tauri webview, so drop is best-effort (web only). */}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={e => {
                    e.preventDefault(); setDragOver(false);
                    if (e.dataTransfer.files.length) void uploadFiles(e.dataTransfer.files);
                  }}
                  disabled={uploading}
                  className={`m-3 flex w-[calc(100%-1.5rem)] flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed bg-transparent p-6 text-center text-sm transition-colors ${
                    dragOver
                      ? 'border-accent bg-accent/10 text-fg'
                      : 'border-line-subtle text-fg-muted hover:border-accent hover:bg-accent/5 hover:text-fg'
                  }`}
                >
                  {uploading
                    ? <Loader2 size={20} className="animate-spin text-accent" />
                    : <Upload size={20} className="text-accent" />}
                  <span><span className="font-medium text-accent">Click to upload</span> a file, or drop it here</span>
                  <span className="text-2xs uppercase tracking-caps">PDF · text · markdown · code</span>
                  <input
                    ref={fileInputRef} type="file" multiple className="hidden"
                    onChange={e => { if (e.target.files?.length) void uploadFiles(e.target.files); e.target.value = ''; }}
                  />
                </button>

                {/* New document — create a markdown file directly in the project */}
                <div className="mx-3 -mt-1 mb-2">
                  {namingDoc ? (
                    <form
                      onSubmit={e => { e.preventDefault(); void createTextDocument(); }}
                      className="flex items-center gap-1.5"
                    >
                      <Input
                        autoFocus value={docNameDraft} placeholder="notes.md"
                        onChange={e => setDocNameDraft(e.target.value)}
                        onBlur={() => { if (!docNameDraft.trim()) setNamingDoc(false); }}
                        onKeyDown={e => { if (e.key === 'Escape') { setNamingDoc(false); setDocNameDraft(''); } }}
                        className="ax-field--sm"
                      />
                      <Button type="submit" size="sm">Create</Button>
                    </form>
                  ) : (
                    <Button
                      variant="ghost" size="sm" className="text-accent"
                      onClick={() => { setNamingDoc(true); setDocNameDraft(''); }}
                    >
                      <FilePlus2 size={13} /> New document
                    </Button>
                  )}
                </div>

                <div className="space-y-3 px-3 pb-3">
                  {documents.length === 0 ? (
                    <p className="p-3 text-sm text-fg-muted">No files yet.</p>
                  ) : (
                    docGroups.map(group => (
                      <section key={group.folder || 'root'}>
                        <div className="mb-1.5 flex items-center justify-between px-0.5">
                          <span className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">
                            {group.label}
                            <span className="ml-1.5 font-mono font-normal">{group.docs.length}</span>
                          </span>
                          {CLEARABLE_FOLDERS.has(group.folder) && (
                            <Button variant="ghost" size="sm" onClick={() => void clearGroup(group)}>
                              Clear
                            </Button>
                          )}
                        </div>
                        <ul className="space-y-1.5">
                          {group.docs.map(doc => (
                            <li
                              key={doc.id}
                              className="group flex items-start gap-2 rounded-lg border border-line-subtle bg-surface-raised p-2.5 transition-colors hover:border-line"
                            >
                              <FileText size={16} className="mt-0.5 shrink-0 text-fg-muted" />
                              <button
                                type="button"
                                onClick={() => setPreview({ doc, edit: false })}
                                title={`Preview ${doc.filename}`}
                                className="min-w-0 flex-1 bg-transparent text-left"
                              >
                                <div className="flex items-center gap-2">
                                  <span className="truncate text-sm font-medium">{leafName(doc.filename, group.folder)}</span>
                                  <Badge
                                    size="sm"
                                    variant={doc.status === 'ready' ? 'success' : doc.status === 'pending' ? 'warning' : 'danger'}
                                  >
                                    {doc.status === 'pending' && <Loader2 size={10} className="animate-spin" />}
                                    {doc.status === 'pending' ? 'ingesting' : doc.status}
                                  </Badge>
                                </div>
                                {doc.summary && <p className="mt-0.5 text-xs text-fg-muted">{doc.summary}</p>}
                                {doc.tags.length > 0 && (
                                  <div className="mt-1 flex flex-wrap gap-1">
                                    {doc.tags.map(tag => (
                                      <span key={tag} className="rounded-md bg-surface-sunken px-1.5 py-0.5 text-2xs text-fg-secondary">
                                        {tag}
                                      </span>
                                    ))}
                                  </div>
                                )}
                                {doc.error && <p className="mt-0.5 text-xs text-error">{doc.error}</p>}
                                <span className="mt-0.5 block font-mono text-2xs text-fg-muted">{formatBytes(doc.size_bytes)}</span>
                              </button>
                              {doc.status === 'failed' && (
                                <IconButton
                                  aria-label={`Retry ingesting ${doc.filename}`}
                                  title="Retry ingestion"
                                  size="xs"
                                  tone="accent"
                                  onClick={() => void retryIngest(doc)}
                                >
                                  <RotateCcw size={14} />
                                </IconButton>
                              )}
                              <IconButton
                                aria-label={`Remove ${doc.filename}`}
                                size="xs"
                                tone="danger"
                                className="opacity-0 group-hover:opacity-100"
                                onClick={() => void deleteDocument(doc)}
                              >
                                <Trash2 size={14} />
                              </IconButton>
                            </li>
                          ))}
                        </ul>
                      </section>
                    ))
                  )}
                </div>

                {/* Conversations in this project (durable server membership) */}
                {!isHome && (
                  <div className="border-t border-line-subtle px-3 py-2.5 pb-4">
                    <div className="mb-1.5 flex items-center justify-between px-0.5">
                      <span className="text-2xs font-semibold uppercase tracking-caps text-fg-muted">
                        Conversations
                        <span className="ml-1.5 font-mono font-normal">{conversations.length}</span>
                      </span>
                      <Button variant="ghost" size="sm" className="text-accent" onClick={newChatInProject}>
                        <MessageSquarePlus size={13} /> New chat in this project
                      </Button>
                    </div>
                    {conversations.length === 0 ? (
                      <p className="p-2 text-sm text-fg-muted">
                        No conversations yet — start one and it will live here.
                      </p>
                    ) : (
                      <ul className="space-y-1">
                        {conversations.map(conv => (
                          <li
                            key={conv.conversation_id}
                            className="group flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors hover:bg-surface-hover"
                          >
                            <MessageSquare size={14} className="shrink-0 text-fg-muted" />
                            <button
                              onClick={() => void openConversation(conv)}
                              className="min-w-0 flex-1 bg-transparent text-left"
                            >
                              <span className="block truncate text-sm">
                                {getDisplayTitle(conv.conversation_id, conv.title)}
                              </span>
                              <span className="block font-mono text-2xs text-fg-muted">
                                {formatDate(conv.last_message_at)} · {conv.message_count} messages
                              </span>
                            </button>
                            <IconButton
                              aria-label="Remove from project"
                              title="Remove from project (keeps the conversation)"
                              size="xs"
                              tone="danger"
                              className="opacity-0 group-hover:opacity-100"
                              onClick={() => void removeConversation(conv)}
                            >
                              <X size={13} />
                            </IconButton>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      </div>

      {preview && selectedId && (
        <DocumentPreviewModal
          workspaceId={selectedId}
          document={preview.doc}
          startInEdit={preview.edit}
          onClose={() => setPreview(null)}
          onDocumentUpdated={onPreviewUpdated}
        />
      )}
    </div>
  );
}

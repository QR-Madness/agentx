/**
 * WorkspacesPanel — manage file workspaces & their documents (Document RAG).
 *
 * Left: workspace list (create / select / rename / delete). Right: the selected
 * workspace's document manifest with ingestion status, plus drag-drop / picker
 * upload. Polls while any document is still ingesting (pending).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { FileText, FolderPlus, Link2, Link2Off, Loader2, Plus, Trash2, Upload } from 'lucide-react';
import { api, type Workspace, type WorkspaceDocument } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { useConversation } from '../../contexts/ConversationContext';
import { getMeta, patchMeta, useConversationMeta } from '../../lib/conversationMeta';
import { useConfirm } from '../ui/ConfirmDialog';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const STATUS_STYLE: Record<WorkspaceDocument['status'], string> = {
  ready: 'text-success',
  pending: 'text-warning',
  failed: 'text-error',
};

export function WorkspacesPanel() {
  const notify = useNotify();
  const confirm = useConfirm();
  const { activeTab } = useConversation();
  useConversationMeta();  // re-render when the conversation↔workspace tag changes
  const convKey = activeTab ? (activeTab.sessionId ?? activeTab.id) : null;
  const attachedId = convKey ? getMeta(convKey).workspaceId : undefined;

  const toggleAttach = useCallback((workspaceId: string) => {
    if (!convKey) return;
    patchMeta(convKey, { workspaceId: attachedId === workspaceId ? undefined : workspaceId });
  }, [convKey, attachedId]);

  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshWorkspaces = useCallback(async () => {
    try {
      const { workspaces } = await api.listWorkspaces();
      setWorkspaces(workspaces);
      setSelectedId(prev => prev ?? workspaces[0]?.id ?? null);
    } catch (err) {
      notify.notifyError(err, 'Could not load workspaces');
    } finally {
      setLoading(false);
    }
  }, [notify]);

  const refreshDocuments = useCallback(async (workspaceId: string) => {
    try {
      const { documents } = await api.listDocuments(workspaceId);
      setDocuments(documents);
    } catch (err) {
      notify.notifyError(err, 'Could not load documents');
    }
  }, [notify]);

  useEffect(() => { void refreshWorkspaces(); }, [refreshWorkspaces]);
  useEffect(() => {
    if (selectedId) void refreshDocuments(selectedId);
    else setDocuments([]);
  }, [selectedId, refreshDocuments]);

  // Poll while any document is still ingesting.
  useEffect(() => {
    if (!selectedId || !documents.some(d => d.status === 'pending')) return;
    const t = setInterval(() => { void refreshDocuments(selectedId); }, 3000);
    return () => clearInterval(t);
  }, [selectedId, documents, refreshDocuments]);

  const createWorkspace = useCallback(async () => {
    const name = window.prompt('Workspace name?')?.trim();
    if (!name) return;
    try {
      const { workspace } = await api.createWorkspace(name);
      await refreshWorkspaces();
      setSelectedId(workspace.id);
    } catch (err) {
      notify.notifyError(err, 'Could not create workspace');
    }
  }, [notify, refreshWorkspaces]);

  const deleteWorkspace = useCallback(async (ws: Workspace) => {
    const ok = await confirm({
      title: `Delete "${ws.name}"?`,
      body: 'This removes the workspace and all its documents. This cannot be undone.',
      confirmLabel: 'Delete', danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteWorkspace(ws.id);
      if (selectedId === ws.id) setSelectedId(null);
      await refreshWorkspaces();
    } catch (err) {
      notify.notifyError(err, 'Could not delete workspace');
    }
  }, [confirm, notify, refreshWorkspaces, selectedId]);

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
      notify.notifyError(err, 'Could not remove document');
    }
  }, [confirm, notify, refreshDocuments, refreshWorkspaces, selectedId]);

  const selected = workspaces.find(w => w.id === selectedId) ?? null;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-center justify-between border-b border-line px-4 py-3">
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <FolderPlus size={18} /> Workspaces
        </h2>
        <button
          onClick={createWorkspace}
          className="flex items-center gap-1 rounded-md bg-accent px-2.5 py-1.5 text-sm text-fg-inverse hover:opacity-90"
        >
          <Plus size={15} /> New
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Workspace list */}
        <aside className="w-56 shrink-0 overflow-y-auto border-r border-line p-2">
          {loading ? (
            <div className="flex items-center gap-2 p-3 text-sm text-fg-muted">
              <Loader2 size={15} className="animate-spin" /> Loading…
            </div>
          ) : workspaces.length === 0 ? (
            <p className="p-3 text-sm text-fg-muted">No workspaces yet. Create one to upload files.</p>
          ) : (
            workspaces.map(ws => (
              <button
                key={ws.id}
                onClick={() => setSelectedId(ws.id)}
                className={`group flex w-full items-center justify-between rounded-md px-2.5 py-2 text-left text-sm ${
                  ws.id === selectedId ? 'bg-accent-secondary text-fg' : 'hover:bg-surface-hover'
                }`}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{ws.name}</span>
                  <span className="block text-xs text-fg-muted">
                    {ws.document_count} files · {formatBytes(ws.used_bytes)}
                  </span>
                </span>
                <Trash2
                  size={14}
                  className="ml-1 shrink-0 text-fg-muted opacity-0 hover:text-error group-hover:opacity-100"
                  onClick={e => { e.stopPropagation(); void deleteWorkspace(ws); }}
                />
              </button>
            ))
          )}
        </aside>

        {/* Documents + upload */}
        <section className="flex min-w-0 flex-1 flex-col">
          {!selected ? (
            <div className="flex flex-1 items-center justify-center text-sm text-fg-muted">
              Select or create a workspace.
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between gap-2 border-b border-line px-3 py-2">
                <span className="min-w-0 truncate text-sm font-medium">{selected.name}</span>
                {convKey ? (
                  <button
                    onClick={() => toggleAttach(selected.id)}
                    className={`flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs ${
                      attachedId === selected.id
                        ? 'bg-accent text-fg-inverse'
                        : 'border border-line text-fg-secondary hover:bg-surface-hover'
                    }`}
                    title="Attach this workspace to the active conversation so the agent can use it"
                  >
                    {attachedId === selected.id ? <Link2Off size={13} /> : <Link2 size={13} />}
                    {attachedId === selected.id ? 'Attached — detach' : 'Attach to conversation'}
                  </button>
                ) : (
                  <span className="shrink-0 text-xs text-fg-muted">Open a chat to attach</span>
                )}
              </div>
              {/* Click-to-browse is the primary path — HTML5 drag-drop doesn't
                  deliver files in the Tauri webview, so it's best-effort (web only). */}
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
                className={`m-3 flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors ${
                  dragOver ? 'border-accent bg-accent-secondary' : 'border-line text-fg-muted hover:border-accent hover:text-fg'
                }`}
              >
                {uploading ? <Loader2 size={20} className="animate-spin" /> : <Upload size={20} />}
                <span><span className="text-accent">Click to upload</span> a file, or drop it here</span>
                <span className="text-xs">PDF, text, markdown, or code</span>
                <input
                  ref={fileInputRef} type="file" multiple className="hidden"
                  onChange={e => { if (e.target.files?.length) void uploadFiles(e.target.files); e.target.value = ''; }}
                />
              </button>

              <ul className="min-h-0 flex-1 space-y-1.5 overflow-y-auto px-3 pb-3">
                {documents.length === 0 ? (
                  <li className="p-3 text-sm text-fg-muted">No documents yet.</li>
                ) : (
                  documents.map(doc => (
                    <li
                      key={doc.id}
                      className="group flex items-start gap-2 rounded-md border border-line bg-surface-raised p-2.5"
                    >
                      <FileText size={16} className="mt-0.5 shrink-0 text-fg-muted" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="truncate text-sm font-medium">{doc.filename}</span>
                          <span className={`text-xs ${STATUS_STYLE[doc.status]}`}>
                            {doc.status === 'pending' ? 'ingesting…' : doc.status}
                          </span>
                        </div>
                        {doc.summary && <p className="mt-0.5 text-xs text-fg-muted">{doc.summary}</p>}
                        {doc.tags.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {doc.tags.map(tag => (
                              <span key={tag} className="rounded bg-surface-sunken px-1.5 py-0.5 text-[11px] text-fg-secondary">
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                        {doc.error && <p className="mt-0.5 text-xs text-error">{doc.error}</p>}
                        <span className="mt-0.5 block text-[11px] text-fg-muted">{formatBytes(doc.size_bytes)}</span>
                      </div>
                      <Trash2
                        size={14}
                        className="shrink-0 cursor-pointer text-fg-muted opacity-0 hover:text-error group-hover:opacity-100"
                        onClick={() => void deleteDocument(doc)}
                      />
                    </li>
                  ))
                )}
              </ul>
            </>
          )}
        </section>
      </div>
    </div>
  );
}

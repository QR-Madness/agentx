/**
 * DocumentPreviewModal — view/edit a project document from the Projects hub.
 *
 * Type dispatch: markdown renders through the shared chat renderer
 * (`MessageContent`), txt/code render as mono text, images and PDFs render from
 * an authed object URL (a bare src can't carry the auth header). Markdown/txt
 * documents get an Edit mode with explicit Save/Cancel (content is not a
 * setting — no autosave) using the `expected_sha256` ETag: a 409 means the
 * document changed elsewhere (e.g. the agent updated it mid-edit).
 * "Export as PDF" prints the rendered document (print stylesheet hides the app).
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Download, Loader2, Pencil, Printer } from 'lucide-react';
import { api, apiErrorMessage, toApiError, type WorkspaceDocument } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { useIsMobile } from '../../lib/hooks';
import { Badge, Button, Textarea } from '../ui';
import { ModalDialog } from '../modals/ModalDialog';
import MessageContent from '../chat/MessageContent';
import './DocumentPreviewModal.css';

const EDITABLE_EXTS = new Set(['md', 'markdown', 'txt']);

function extOf(filename: string): string {
  const leaf = filename.slice(filename.lastIndexOf('/') + 1);
  const dot = leaf.lastIndexOf('.');
  return dot === -1 ? '' : leaf.slice(dot + 1).toLowerCase();
}

type PreviewKind = 'markdown' | 'text' | 'image' | 'pdf';

function kindOf(doc: WorkspaceDocument): PreviewKind {
  const ct = doc.content_type ?? '';
  if (ct.startsWith('image/')) return 'image';
  if (ct === 'application/pdf' || extOf(doc.filename) === 'pdf') return 'pdf';
  const ext = extOf(doc.filename);
  if (ext === 'md' || ext === 'markdown') return 'markdown';
  return 'text'; // txt/code/config — everything else the server ingests is utf-8 text
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentPreviewModal({
  workspaceId,
  document: initialDoc,
  startInEdit = false,
  onClose,
  onDocumentUpdated,
}: {
  workspaceId: string;
  document: WorkspaceDocument;
  /** Open directly in the editor (the hub's "New document" flow). */
  startInEdit?: boolean;
  onClose: () => void;
  /** Fired after a successful save so the hub can refresh its manifest. */
  onDocumentUpdated?: (doc: WorkspaceDocument) => void;
}) {
  const notify = useNotify();
  const isMobile = useIsMobile();
  // Track the live row locally: saves bump sha/status without reopening.
  const [doc, setDoc] = useState(initialDoc);
  const kind = useMemo(() => kindOf(doc), [doc]);
  const editable = EDITABLE_EXTS.has(extOf(doc.filename));

  const [text, setText] = useState<string | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [editing, setEditing] = useState(startInEdit && editable);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);

  // Load content for the current sha. Object URLs are revoked on cleanup.
  useEffect(() => {
    let cancelled = false;
    let url: string | null = null;
    setLoadError(null);
    if (kind === 'markdown' || kind === 'text') {
      api.fetchDocumentText(workspaceId, doc.id, doc.sha256)
        .then(t => { if (!cancelled) { setText(t); setDraft(t); } })
        .catch(err => { if (!cancelled) setLoadError(apiErrorMessage(toApiError(err))); });
    } else {
      api.fetchDocumentBlob(workspaceId, doc.id, doc.sha256)
        .then(blob => {
          url = URL.createObjectURL(blob);
          if (cancelled) URL.revokeObjectURL(url);
          else setObjectUrl(url);
        })
        .catch(err => { if (!cancelled) setLoadError(apiErrorMessage(toApiError(err))); });
    }
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [workspaceId, doc.id, doc.sha256, kind]);

  const save = useCallback(async () => {
    setSaving(true);
    try {
      const { document: updated } = await api.updateTextDocument(
        workspaceId, doc.id, draft, doc.sha256,
      );
      setDoc(updated);
      setText(draft);
      setEditing(false);
      onDocumentUpdated?.(updated);
    } catch (err) {
      const apiErr = toApiError(err);
      notify.notifyError(
        apiErr,
        apiErr.status === 409
          ? 'Document changed elsewhere — close and reopen to load the latest version'
          : 'Could not save document',
      );
    } finally {
      setSaving(false);
    }
  }, [workspaceId, doc.id, doc.sha256, draft, notify, onDocumentUpdated]);

  const exportPdf = useCallback(() => {
    // Print stylesheet (DocumentPreviewModal.css) makes .doc-print-area the only
    // visible content; the OS print dialog offers "Save as PDF" everywhere.
    window.print();
  }, []);

  const download = useCallback(async () => {
    try {
      const blob = await api.fetchDocumentBlob(workspaceId, doc.id, doc.sha256);
      const url = URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = doc.filename.slice(doc.filename.lastIndexOf('/') + 1);
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      notify.notifyError(err, 'Could not download file');
    }
  }, [workspaceId, doc, notify]);

  return (
    <ModalDialog size="xl" onClose={onClose}>
      <div className={`flex min-h-0 flex-col ${isMobile ? 'h-full flex-1' : 'h-[80vh]'}`}>
        <header className={`flex items-center gap-2 border-b border-line py-3 pr-12 ${isMobile ? 'flex-wrap px-3' : 'px-4'}`}>
          <span className="min-w-0 flex-1 truncate text-sm font-semibold">{doc.filename}</span>
          <Badge
            size="sm"
            className="shrink-0"
            variant={doc.status === 'ready' ? 'success' : doc.status === 'pending' ? 'warning' : 'danger'}
          >
            {doc.status === 'pending' && <Loader2 size={10} className="animate-spin" />}
            {doc.status === 'pending' ? 're-indexing' : doc.status}
          </Badge>
          <span className="shrink-0 font-mono text-2xs text-fg-muted">{formatBytes(doc.size_bytes)}</span>
          <div className={`flex items-center gap-1.5 ${isMobile ? 'basis-full justify-end' : 'ml-auto'}`}>
            {editable && !editing && (
              <Button size="sm" variant="ghost" onClick={() => { setDraft(text ?? ''); setEditing(true); }}>
                <Pencil size={13} /> Edit
              </Button>
            )}
            {(kind === 'markdown' || kind === 'text') && !editing && (
              <Button size="sm" variant="ghost" onClick={exportPdf} title="Print / save as PDF">
                <Printer size={13} /> Export PDF
              </Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => void download()} title="Download the file">
              <Download size={13} />
            </Button>
          </div>
        </header>

        {doc.summary && !editing && (
          <p className="border-b border-line-subtle px-4 py-2 text-xs text-fg-muted">{doc.summary}</p>
        )}

        <div className="doc-preview-body flex flex-col">
          {loadError ? (
            <p className="p-4 text-sm text-error">{loadError}</p>
          ) : editing ? (
            <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
              <Textarea
                autoFocus
                value={draft}
                onChange={e => setDraft(e.target.value)}
                className="min-h-0 flex-1 resize-none font-mono text-sm"
                placeholder="Write markdown…"
              />
              <div className="flex items-center justify-end gap-2">
                <Button size="sm" variant="ghost" disabled={saving}
                        onClick={() => { setEditing(false); setDraft(text ?? ''); }}>
                  Cancel
                </Button>
                <Button size="sm" disabled={saving || draft === text} onClick={() => void save()}>
                  {saving ? <Loader2 size={13} className="animate-spin" /> : 'Save'}
                </Button>
              </div>
            </div>
          ) : kind === 'markdown' ? (
            text === null
              ? <Loader2 size={18} className="m-6 animate-spin text-accent" />
              : <div className={`doc-print-area ${isMobile ? 'px-3 py-3' : 'p-4'}`}><MessageContent content={text} /></div>
          ) : kind === 'text' ? (
            text === null
              ? <Loader2 size={18} className="m-6 animate-spin text-accent" />
              : <pre className="doc-preview-plain doc-print-area">{text}</pre>
          ) : kind === 'image' ? (
            objectUrl === null
              ? <Loader2 size={18} className="m-6 animate-spin text-accent" />
              : <img src={objectUrl} alt={doc.filename} className="max-h-full max-w-full self-center object-contain p-3" />
          ) : (
            objectUrl === null
              ? <Loader2 size={18} className="m-6 animate-spin text-accent" />
              : <iframe src={objectUrl} title={doc.filename} className="h-full w-full flex-1 border-0" />
          )}
        </div>
      </div>
    </ModalDialog>
  );
}

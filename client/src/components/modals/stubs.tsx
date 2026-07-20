/**
 * Modal content components — wrappers for tabs rendered in modals/drawers
 */

import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Copy, Check, MessagesSquare, Braces, BookOpen } from 'lucide-react';
import { MessageContent } from '../chat/MessageContent';
import { MemoryWorkbench } from '../memory/MemoryWorkbench';
import { PlansPanel } from '../plans/PlansPanel';
import { SourcesPanel } from '../bibliography/SourcesPanel';
import { AmbassadorPanel } from '../ambassador/AmbassadorPanel';
import { deckThreadId } from '../../lib/ambassadorDeck';
import { useAuth } from '../../contexts/AuthContext';
import { useAmbassador } from '../../contexts/AmbassadorContext';
import { LogsPanel } from '../logs/LogsPanel';
import { TranslationPanel } from '../panels/TranslationPanel';
import { UnifiedSettings } from '../unified-settings/UnifiedSettings';
import { ToolkitPage } from '../toolkit/ToolkitPage';
import { UnifiedProfileEditor } from '../unified-profile-editor/UnifiedProfileEditor';
import { ConversationList } from '../chat/ConversationList';
import { ToolOutputsPanel } from '../toolkit/ToolOutputsPanel';
import { WorkspacesPanel } from '../workspaces/WorkspacesPanel';
import { Button } from '../ui';

// Re-export ProfileEditorModal for the modal registry
export { ProfileEditorModal } from './ProfileEditorModal';

interface ModalContentProps {
  onClose: () => void;
}

export function MemoryModalContent({ onClose }: ModalContentProps) {
  // Full-screen immersive Memory Workbench (in FULLSCREEN_SURFACES + SELF_CLOSING):
  // renders bare and owns its own backdrop + close, like UnifiedSettings.
  return <MemoryWorkbench onClose={onClose} />;
}

export function PlansModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <PlansPanel />
    </div>
  );
}

export function SourcesDrawerContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <SourcesPanel />
    </div>
  );
}

export function AmbassadorDrawerContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <AmbassadorPanel />
    </div>
  );
}

export function AmbassadorDeckContent({ onClose }: ModalContentProps) {
  // The standalone command deck: the ambassador with no conversation. Holds multiple named
  // Inquiries (the home deck thread + minted ones); this owns which one is selected and the
  // registry list. A BARE full-screen surface (FULLSCREEN_SURFACES + SELF_CLOSING, like
  // Memory): owns its own backdrop + Escape + scroll lock; the panel renders the inline
  // close button (never a floating one).
  const { sessionInfo } = useAuth();
  const { inquiries, listInquiries, createInquiry } = useAmbassador();
  const homeId = deckThreadId(sessionInfo?.user_id);
  const [selectedId, setSelectedId] = useState(homeId);

  useEffect(() => { void listInquiries(); }, [listInquiries]);

  // ESC to close + body scroll lock (bare full-screen surface owns these).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); }
    };
    window.addEventListener('keydown', handler);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handler);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const onNewInquiry = async () => {
    const id = await createInquiry();
    if (id) setSelectedId(id);
  };

  return (
    <>
      {/* Desktop (>600px): sit BELOW the 56px TopBar so the Deck pill stays visible,
          lit, and clickable (toggle-close) — first-class tab behavior. Mobile keeps
          full-viewport (pills don't exist there; the palette is the entry). */}
      <div
        className="fixed inset-0 z-[1999] bg-[var(--glass-backdrop)] min-[601px]:top-14"
        onClick={onClose}
      />
      <div
        className="fixed inset-0 z-[2000] flex flex-col overflow-hidden bg-surface-base text-fg animate-in fade-in-0 zoom-in-[0.99] duration-150 min-[601px]:top-14"
        role="dialog"
        aria-modal="true"
        aria-label="Command Deck"
      >
        <AmbassadorPanel
          deckThreadId={selectedId}
          deckHomeId={homeId}
          deckInquiries={inquiries}
          onSelectInquiry={setSelectedId}
          onNewInquiry={onNewInquiry}
          onClose={onClose}
        />
      </div>
    </>
  );
}

export function LogsDrawerContent({ onClose: _onClose }: ModalContentProps) {
  // Rendered into a definite-height host so the panel fills the drawer and
  // scrolls internally (toolbar/footer fixed, stream scrolls) — like Memory.
  return (
    <div className="memory-modal-content">
      <LogsPanel />
    </div>
  );
}

export function WorkspacesDrawerContent({
  onClose,
  initialWorkspaceId,
}: ModalContentProps & { initialWorkspaceId?: string }) {
  // Definite-height host so the panel fills the drawer and scrolls internally.
  // The panel renders its own inline close button (floating ones fail cross-plat),
  // so it's in SELF_CLOSING and gets `onClose`. `initialWorkspaceId` (via
  // modal props) opens the hub focused on a specific project.
  return (
    <div className="memory-modal-content">
      <WorkspacesPanel onClose={onClose} initialWorkspaceId={initialWorkspaceId} />
    </div>
  );
}

export function ToolOutputBrowserContent({ onClose: _onClose }: ModalContentProps) {
  // Definite-height host so the master-detail panel fills the drawer and scrolls
  // internally (same pattern as Logs/Memory).
  return (
    <div className="memory-modal-content">
      <ToolOutputsPanel />
    </div>
  );
}

/**
 * ConversationsDrawerContent — mobile-first drawer for browsing/switching active
 * and past conversations. Reuses the shared ConversationList (same body as the
 * desktop Conversations sidebar). The shell-owned close button (DrawerPanel)
 * dismisses it; selecting a conversation also closes via `onActivated`.
 */
export function ConversationsDrawerContent({ onClose }: ModalContentProps) {
  // Rendered directly into .drawer-content (no .modal-content-wrapper) so the
  // flex chain resolves: .conversations-drawer fills the drawer height and the
  // inner .history-list owns the scroll. The wrapper's min-height:100vh would
  // otherwise break the fill and leave the list collapsed to its content.
  return (
    <div className="conversations-drawer">
      <div className="conversations-drawer-header">
        <MessagesSquare size={18} />
        <span>Conversations</span>
      </div>
      <ConversationList onActivated={onClose} autoFocusSearch={false} />
    </div>
  );
}

export function ToolsModalContent({ onClose }: ModalContentProps) {
  // Phase 18.2: full-screen Toolkit replaces the read-only ToolsPanel drawer.
  return <ToolkitPage isOpen={true} onClose={onClose} />;
}

export function TranslationModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <TranslationPanel />
    </div>
  );
}

/**
 * UnifiedSettingsModalContent — New unified settings interface
 */
export function UnifiedSettingsModalContent({ onClose }: ModalContentProps) {
  return <UnifiedSettings isOpen={true} onClose={onClose} />;
}

/**
 * UnifiedProfileEditorModalContent — Full-screen immersive profile editor
 */
export function UnifiedProfileEditorModalContent({
  onClose,
  initialProfileId,
  isNew,
}: ModalContentProps & { initialProfileId?: string; isNew?: boolean }) {
  return (
    <UnifiedProfileEditor
      isOpen={true}
      onClose={onClose}
      initialProfileId={initialProfileId}
      isNew={isNew}
    />
  );
}

/**
 * Fallback stub for unregistered components
 */
export function StubModal({ onClose }: ModalContentProps) {
  return (
    <div style={{ padding: 24 }}>
      <h3 style={{ color: 'var(--text-primary)', marginBottom: 12 }}>Coming Soon</h3>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
        This panel will be implemented in a future phase.
      </p>
      <Button variant="secondary" onClick={onClose}>
        Close
      </Button>
    </div>
  );
}

/**
 * ToolOutputDrawer — Displays full tool output in a side drawer
 */
interface ToolOutputDrawerProps extends ModalContentProps {
  toolName?: string;
  content?: string;
  success?: boolean;
  durationMs?: number;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function tryFormatJson(content: string): { formatted: string; isJson: boolean } {
  try {
    const parsed = JSON.parse(content);
    return { formatted: JSON.stringify(parsed, null, 2), isJson: true };
  } catch {
    return { formatted: content, isJson: false };
  }
}

/** Markdown-ish tool output (compressed summaries, web research reports) reads
 *  far better through the shared chat renderer than as a mono blob. */
function looksLikeMarkdown(content: string): boolean {
  return /(^|\n)#{1,4} |\*\*[^*\n]+\*\*|(^|\n)[-*] |(^|\n)\d+\. /.test(content);
}

export function ToolOutputDrawer({
  onClose,
  toolName = 'Tool',
  content = '',
  success = true,
  durationMs,
}: ToolOutputDrawerProps) {
  const [copied, setCopied] = useState(false);
  const { formatted, isJson } = tryFormatJson(content);
  // Markdown-ish output defaults to the rendered reading view; Raw is a click
  // away (debugging always deserves the verbatim bytes).
  const markdown = !isJson && looksLikeMarkdown(content);
  const [showRaw, setShowRaw] = useState(false);
  const rendered = markdown && !showRaw;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="tool-output-drawer">
      <div className="tool-output-header">
        <div className="tool-output-title">
          {success ? (
            <CheckCircle size={18} style={{ color: 'var(--feedback-success)' }} />
          ) : (
            <XCircle size={18} style={{ color: 'var(--feedback-error)' }} />
          )}
          <h3>{toolName}</h3>
          {durationMs !== undefined && (
            <span className="tool-output-duration">{formatDuration(durationMs)}</span>
          )}
        </div>
        <div className="tool-output-actions">
          {markdown && (
            <button
              className="copy-btn"
              onClick={() => setShowRaw(r => !r)}
              title={showRaw ? 'Rendered reading view' : 'Raw output (verbatim)'}
            >
              {showRaw ? <BookOpen size={14} /> : <Braces size={14} />}
              <span>{showRaw ? 'Pretty' : 'Raw'}</span>
            </button>
          )}
          <button className="copy-btn" onClick={handleCopy} title="Copy to clipboard">
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
      <div className="tool-output-content">
        {rendered ? (
          <div className="tool-output-md">
            <MessageContent content={content} />
          </div>
        ) : (
          <pre className={isJson ? 'json-content' : ''}>
            {formatted}
          </pre>
        )}
      </div>
      <style>{`
        .tool-output-drawer {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: var(--bg-primary);
        }
        .tool-output-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1rem 1.25rem;
          border-bottom: 1px solid var(--border-subtle);
          background: var(--bg-secondary);
        }
        .tool-output-title {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .tool-output-title h3 {
          margin: 0;
          font-size: 1rem;
          font-weight: 600;
          color: var(--text-primary);
        }
        .tool-output-duration {
          font-size: 0.75rem;
          color: var(--text-tertiary);
          margin-left: 0.5rem;
        }
        .tool-output-actions {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .copy-btn {
          display: flex;
          align-items: center;
          gap: 0.375rem;
          padding: 0.375rem 0.75rem;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 6px;
          color: var(--text-secondary);
          font-size: 0.75rem;
          cursor: pointer;
          transition: all 0.15s;
        }
        .copy-btn:hover {
          background: var(--bg-hover);
          border-color: var(--accent-primary);
          color: var(--accent-primary);
        }
        .tool-output-md {
          padding: 1rem 1.5rem 2rem;
          max-width: 72ch;
          font-size: 0.875rem;
          line-height: 1.65;
        }
        .tool-output-content {
          flex: 1;
          overflow: auto;
          padding: 1rem;
        }
        .tool-output-content pre {
          margin: 0;
          padding: 1rem;
          background: var(--bg-tertiary);
          border-radius: 8px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.8125rem;
          line-height: 1.5;
          color: var(--text-secondary);
          white-space: pre-wrap;
          word-break: break-word;
        }
        .tool-output-content pre.json-content {
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}

/**
 * Modal content components — wrappers for tabs rendered in modals/drawers
 */

import { useState } from 'react';
import { CheckCircle, XCircle, Copy, Check } from 'lucide-react';
import { SettingsPanel } from '../panels/SettingsPanel';
import { MemoryPanel } from '../panels/MemoryPanel';
import { ToolsPanel } from '../panels/ToolsPanel';
import { TranslationPanel } from '../panels/TranslationPanel';

// Re-export ProfileEditorModal for the modal registry
export { ProfileEditorModal } from './ProfileEditorModal';

interface ModalContentProps {
  onClose: () => void;
}

export function SettingsModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <SettingsPanel />
    </div>
  );
}

export function MemoryModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <MemoryPanel />
    </div>
  );
}

export function ToolsModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <ToolsPanel />
    </div>
  );
}

export function TranslationModalContent({ onClose: _onClose }: ModalContentProps) {
  return (
    <div className="modal-content-wrapper">
      <TranslationPanel />
    </div>
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
      <button className="button-secondary" onClick={onClose}>
        Close
      </button>
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

export function ToolOutputDrawer({
  onClose,
  toolName = 'Tool',
  content = '',
  success = true,
  durationMs,
}: ToolOutputDrawerProps) {
  const [copied, setCopied] = useState(false);
  const { formatted, isJson } = tryFormatJson(content);

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
            <CheckCircle size={18} style={{ color: '#22c55e' }} />
          ) : (
            <XCircle size={18} style={{ color: '#ef4444' }} />
          )}
          <h3>{toolName}</h3>
          {durationMs !== undefined && (
            <span className="tool-output-duration">{formatDuration(durationMs)}</span>
          )}
        </div>
        <div className="tool-output-actions">
          <button className="copy-btn" onClick={handleCopy} title="Copy to clipboard">
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>
          <button className="button-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
      <div className="tool-output-content">
        <pre className={isJson ? 'json-content' : ''}>
          {formatted}
        </pre>
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

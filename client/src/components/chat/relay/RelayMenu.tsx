import { useEffect, useRef, useState } from 'react';
import {
  Box,
  Database,
  Inbox,
  Mic,
  Paperclip,
  Send,
  Sparkles,
  X,
} from 'lucide-react';
import { api } from '../../../lib/api';
import type { BackgroundChatJob } from '../../../lib/api';
import './RelayMenu.css';

interface RelayMenuProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  noMemorization: boolean;
  onToggleNoMemorization: () => void;
  canToggleNoMemorization: boolean;
  canEnhance: boolean;
  onEnhance: () => void;
  isEnhancing: boolean;
  canSendBackground: boolean;
  onSendBackground: () => void;
  onJobsChanged?: (jobs: BackgroundChatJob[]) => void;
}

export function RelayMenu({
  isOpen,
  onClose,
  anchorRef,
  noMemorization,
  onToggleNoMemorization,
  canToggleNoMemorization,
  canEnhance,
  onEnhance,
  isEnhancing,
  canSendBackground,
  onSendBackground,
  onJobsChanged,
}: RelayMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [jobs, setJobs] = useState<BackgroundChatJob[]>([]);
  const [loading, setLoading] = useState(false);

  // Close on outside click / Escape
  useEffect(() => {
    if (!isOpen) return;
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        menuRef.current?.contains(target) ||
        anchorRef.current?.contains(target)
      ) {
        return;
      }
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [isOpen, onClose, anchorRef]);

  // Poll background jobs while open
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;

    const load = async () => {
      try {
        setLoading(true);
        const { jobs: next } = await api.listBackgroundChats();
        if (!cancelled) {
          setJobs(next);
          onJobsChanged?.(next);
        }
      } catch {
        // ignore — inbox is best-effort
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    const id = window.setInterval(load, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [isOpen, onJobsChanged]);

  const dismiss = async (jobId: string) => {
    await api.dismissBackgroundChat(jobId);
    setJobs(prev => prev.filter(j => j.job_id !== jobId));
  };

  if (!isOpen) return null;

  return (
    <div
      ref={menuRef}
      className="relay-menu"
      role="dialog"
      aria-label="Relay Module"
    >
      <div className="relay-menu-header">
        <Box size={14} />
        <span>Relay</span>
        <button className="relay-close" onClick={onClose} aria-label="Close">
          <X size={14} />
        </button>
      </div>

      <div className="relay-section">
        <div className="relay-section-title">Send</div>
        <button
          className="relay-item"
          onClick={() => {
            onSendBackground();
            onClose();
          }}
          disabled={!canSendBackground}
          title="Run in the background; the result lands in this inbox when done."
        >
          <Send size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">Send to background</span>
            <span className="relay-item-hint">
              Fire-and-forget; surfaces below when complete.
            </span>
          </div>
        </button>
        <button
          className={`relay-item ${isEnhancing ? 'active' : ''}`}
          onClick={() => {
            onEnhance();
          }}
          disabled={!canEnhance || isEnhancing}
        >
          <Sparkles size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">
              {isEnhancing ? 'Enhancing…' : 'Enhance prompt'}
            </span>
            <span className="relay-item-hint">
              Rewrite the current draft using the prompt enhancer.
            </span>
          </div>
        </button>
      </div>

      <div className="relay-section">
        <div className="relay-section-title">Memory</div>
        <button
          className={`relay-item toggle ${noMemorization ? 'on' : ''}`}
          onClick={onToggleNoMemorization}
          disabled={!canToggleNoMemorization}
          title={
            canToggleNoMemorization
              ? undefined
              : 'No Memorization can only be set on a new conversation.'
          }
        >
          <Database size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">No Memorization</span>
            <span className="relay-item-hint">
              {!canToggleNoMemorization
                ? 'Locked — start a new conversation to change this.'
                : noMemorization
                  ? 'Turns in this conversation will not be stored.'
                  : 'Memory is on — turns will be stored and recalled.'}
            </span>
          </div>
          <span className="relay-toggle-pill">{noMemorization ? 'ON' : 'OFF'}</span>
        </button>
      </div>

      <div className="relay-section">
        <div className="relay-section-title">Tools</div>
        <button className="relay-item" disabled title="Coming soon">
          <Mic size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">Voice input</span>
            <span className="relay-item-hint">Coming soon.</span>
          </div>
        </button>
        <button className="relay-item" disabled title="Coming soon">
          <Paperclip size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">Attach file</span>
            <span className="relay-item-hint">Coming soon.</span>
          </div>
        </button>
      </div>

      <div className="relay-section relay-inbox">
        <div className="relay-section-title">
          <Inbox size={12} />
          <span>Background runs</span>
          {loading && <span className="relay-loading-dot" />}
        </div>
        {jobs.length === 0 ? (
          <div className="relay-inbox-empty">No background runs yet.</div>
        ) : (
          <ul className="relay-inbox-list">
            {jobs.slice(0, 10).map(job => (
              <li key={job.job_id} className={`relay-job status-${job.status}`}>
                <div className="relay-job-main">
                  <span className={`relay-job-status status-${job.status}`}>
                    {job.status}
                  </span>
                  <span className="relay-job-message" title={job.message}>
                    {job.message.slice(0, 80)}
                    {job.message.length > 80 ? '…' : ''}
                  </span>
                </div>
                {job.status === 'done' && job.response && (
                  <div className="relay-job-response">
                    {job.response.slice(0, 220)}
                    {job.response.length > 220 ? '…' : ''}
                  </div>
                )}
                {job.status === 'failed' && job.error && (
                  <div className="relay-job-error">{job.error}</div>
                )}
                <button
                  className="relay-job-dismiss"
                  onClick={() => dismiss(job.job_id)}
                  aria-label="Dismiss"
                >
                  <X size={12} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

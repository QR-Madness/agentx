import { useEffect, useState } from 'react';
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
import { DropdownPortal } from '../../ui/DropdownPortal';
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
  const [jobs, setJobs] = useState<BackgroundChatJob[]>([]);
  const [loading, setLoading] = useState(false);

  // Poll background jobs while open — adaptively. A flat interval hammered the
  // backend even when nothing was in flight; instead we poll quickly only while
  // a job is queued/running and back off hard once everything has settled.
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      try {
        setLoading(true);
        const { jobs: next } = await api.listBackgroundChats();
        if (cancelled) return;
        setJobs(next);
        onJobsChanged?.(next);
        const hasActive = next.some(j => j.status === 'queued' || j.status === 'running');
        // 3s while work is in flight; 30s idle heartbeat to catch new arrivals.
        timer = setTimeout(load, hasActive ? 3000 : 30000);
      } catch {
        // ignore — inbox is best-effort; retry on the idle cadence.
        if (!cancelled) timer = setTimeout(load, 30000);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [isOpen, onJobsChanged]);

  const dismiss = async (jobId: string) => {
    await api.dismissBackgroundChat(jobId);
    setJobs(prev => prev.filter(j => j.job_id !== jobId));
  };

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="top"
      align="start"
      estimatedHeight={520}
    >
    <div
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
    </DropdownPortal>
  );
}

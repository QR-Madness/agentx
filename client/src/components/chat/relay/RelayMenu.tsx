import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Box,
  FolderOpen,
  Image as ImageIcon,
  Inbox,
  Mic,
  Paperclip,
  Play,
  Radio,
  Send,
  Sparkles,
  X,
} from 'lucide-react';
import { api } from '../../../lib/api';
import type { ActiveChatRun, BackgroundChatJob } from '../../../lib/api';
import { useConversation } from '../../../contexts/ConversationContext';
import { orphanedRuns } from '../../../contexts/conversation/orphanedRuns';
import { DropdownPortal } from '../../ui/DropdownPortal';
import { useIsMobile } from '../../../lib/hooks';
import './RelayMenu.css';

interface RelayMenuProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  canEnhance: boolean;
  onEnhance: () => void;
  isEnhancing: boolean;
  canArmBackground: boolean;
  backgroundArmed: boolean;
  onToggleBackground: () => void;
  onJobsChanged?: (jobs: BackgroundChatJob[]) => void;
  /** Open the document picker — attaches files to the conversation's workspace (RAG). */
  onAttachFile?: () => void;
  attachingFile?: boolean;
  /** Open the image picker — attaches images to the next message (vision input). */
  onAttachImage?: () => void;
  uploadingImage?: boolean;
  /** Whether the active model/profile supports vision (gates the image item). */
  visionEnabled?: boolean;
  /** Open the Projects hub for this conversation's project. */
  onOpenProject?: () => void;
  /** Name of the conversation's project, if it belongs to one. */
  projectName?: string;
  hasProject?: boolean;
}

export function RelayMenu({
  isOpen,
  onClose,
  anchorRef,
  canEnhance,
  onEnhance,
  isEnhancing,
  canArmBackground,
  backgroundArmed,
  onToggleBackground,
  onJobsChanged,
  onAttachFile,
  attachingFile,
  onAttachImage,
  uploadingImage,
  visionEnabled,
  onOpenProject,
  projectName,
  hasProject,
}: RelayMenuProps) {
  const isMobile = useIsMobile();
  const [jobs, setJobs] = useState<BackgroundChatJob[]>([]);
  const [runs, setRuns] = useState<ActiveChatRun[]>([]);
  const [loading, setLoading] = useState(false);
  const { tabs, resumeRun } = useConversation();

  // Runs still going whose owning tab is closed — the recoverable ones.
  const liveRuns = useMemo(() => orphanedRuns(runs, tabs), [runs, tabs]);

  // Keep the change callback in a ref so the poll effect depends only on
  // `isOpen`. Otherwise an inline `onJobsChanged` from the parent changes
  // identity on every render (e.g. during streaming), tearing down and
  // re-running the effect — which fires an immediate fetch each time and
  // hammers the backend.
  const onJobsChangedRef = useRef(onJobsChanged);
  onJobsChangedRef.current = onJobsChanged;

  // Poll background jobs + detached runs while open — adaptively, on a single
  // timer. A flat interval hammered the backend even when nothing was in
  // flight; instead we poll quickly only while something is running and back
  // off hard once everything has settled.
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      try {
        setLoading(true);
        const [jobsRes, runsRes] = await Promise.all([
          api.listBackgroundChats(),
          api.listChatRuns().catch(() => ({ runs: [] as ActiveChatRun[] })),
        ]);
        if (cancelled) return;
        setJobs(jobsRes.jobs);
        setRuns(runsRes.runs);
        onJobsChangedRef.current?.(jobsRes.jobs);
        const hasActive =
          jobsRes.jobs.some(j => j.status === 'queued' || j.status === 'running') ||
          runsRes.runs.some(r => r.status === 'running');
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
  }, [isOpen]);

  const handleResume = async (run: ActiveChatRun) => {
    await resumeRun(run);
    onClose();
  };

  const dismiss = async (jobId: string) => {
    await api.dismissBackgroundChat(jobId);
    setJobs(prev => prev.filter(j => j.job_id !== jobId));
  };

  // On mobile the menu renders as a bottom sheet (its own portal), so it needs
  // its own Escape-to-close — the desktop path gets this from DropdownPortal.
  useEffect(() => {
    if (!isOpen || !isMobile) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [isOpen, isMobile, onClose]);

  const body = (
    <div
      className={`relay-menu${isMobile ? ' relay-menu--sheet' : ''}`}
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
        <div className="relay-section-title">Project</div>
        <button
          className="relay-item"
          onClick={() => { onOpenProject?.(); onClose(); }}
          disabled={!onOpenProject}
          title={hasProject ? 'Open this conversation’s project' : 'Browse projects and attach one'}
        >
          <FolderOpen size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">
              {hasProject ? (projectName || 'Open project') : 'Projects'}
            </span>
            <span className="relay-item-hint">
              {hasProject
                ? 'Open this conversation’s project — files, instructions, chats.'
                : 'Browse projects, or attach files below to start one.'}
            </span>
          </div>
        </button>
        {visionEnabled && (
          <button
            className={`relay-item ${uploadingImage ? 'active' : ''}`}
            onClick={() => { onAttachImage?.(); onClose(); }}
            disabled={!onAttachImage || uploadingImage}
            title="Attach an image to the next message (vision input) — or just paste one into the composer"
          >
            <ImageIcon size={14} />
            <div className="relay-item-body">
              <span className="relay-item-label">{uploadingImage ? 'Uploading…' : 'Attach image'}</span>
              <span className="relay-item-hint">
                Add a PNG / JPEG / WebP / GIF for the model to see. Paste works too.
              </span>
            </div>
          </button>
        )}
        <button
          className={`relay-item ${attachingFile ? 'active' : ''}`}
          onClick={() => { onAttachFile?.(); onClose(); }}
          disabled={!onAttachFile || attachingFile}
          title="Attach a document to this conversation's project — the agent can search it (Document RAG)"
        >
          <Paperclip size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">{attachingFile ? 'Attaching…' : 'Attach file'}</span>
            <span className="relay-item-hint">
              Add a PDF / text / code file to this conversation's project.
            </span>
          </div>
        </button>
      </div>

      <div className="relay-section">
        <div className="relay-section-title">Send</div>
        <button
          className={`relay-item toggle ${backgroundArmed ? 'on' : ''}`}
          onClick={() => {
            onToggleBackground();
            onClose();
          }}
          disabled={!canArmBackground}
          title="Stage the next message to run in the background; the result lands in this inbox when done."
        >
          <Send size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">
              {backgroundArmed ? 'Background mode armed' : 'Send next to background'}
            </span>
            <span className="relay-item-hint">
              {backgroundArmed
                ? 'Your next message runs in the background. Click to cancel.'
                : 'Arms the next send — it surfaces below when complete.'}
            </span>
          </div>
          <span className="relay-toggle-pill">{backgroundArmed ? 'ON' : 'OFF'}</span>
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
        <div className="relay-section-title">Tools</div>
        <button className="relay-item" disabled title="Coming soon">
          <Mic size={14} />
          <div className="relay-item-body">
            <span className="relay-item-label">Voice input</span>
            <span className="relay-item-hint">Coming soon.</span>
          </div>
        </button>
      </div>

      {liveRuns.length > 0 && (
        <div className="relay-section relay-inbox">
          <div className="relay-section-title">
            <Radio size={12} />
            <span>Live runs</span>
          </div>
          <ul className="relay-inbox-list">
            {liveRuns.map(run => (
              <li key={run.run_id} className="relay-job status-running relay-live-run">
                <div className="relay-job-main">
                  <span className="relay-live-dot" aria-hidden />
                  <span className="relay-job-message" title={run.message}>
                    {run.message.slice(0, 80) || 'Running conversation'}
                    {run.message.length > 80 ? '…' : ''}
                  </span>
                </div>
                <button
                  className="relay-resume-btn"
                  onClick={() => handleResume(run)}
                  title="Reopen this run and continue streaming"
                >
                  <Play size={12} />
                  <span>Resume</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

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

  // Mobile: a thumb-reachable bottom sheet with a dimming backdrop. Desktop:
  // the anchor-positioned dropdown above the composer.
  if (isMobile) {
    if (!isOpen) return null;
    return createPortal(
      <div className="relay-sheet-backdrop" onClick={onClose}>
        <div onClick={(e) => e.stopPropagation()}>{body}</div>
      </div>,
      document.body,
    );
  }

  return (
    <DropdownPortal
      isOpen={isOpen}
      onClose={onClose}
      anchorRef={anchorRef}
      preferredSide="top"
      align="start"
      estimatedHeight={520}
    >
      {body}
    </DropdownPortal>
  );
}

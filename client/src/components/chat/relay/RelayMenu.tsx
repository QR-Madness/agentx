/**
 * RelayMenu — the conversation command center behind the composer's Orbit
 * button. Control-center layout: a status strip, a tiled action grid (wide
 * Thinking-Mode tile + toggle/opener tiles), then the Live-runs and
 * Background-runs sections. Desktop: glass popover anchored above the
 * composer. Mobile: bottom sheet — on small screens the composer chip row is
 * hidden entirely and THIS is where every conversation control lives.
 *
 * Glass discipline: the container is the hero glass surface; tiles stay
 * opaque (`--surface-raised`) — see the WebKitGTK paint-cost note in
 * ui/DropdownMenu.tsx.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { ReactNode } from 'react';
import {
  Box,
  Brain,
  Cpu,
  FolderOpen,
  Ghost,
  Gauge,
  Image as ImageIcon,
  Inbox,
  MemoryStick,
  Mic,
  Music,
  NotebookPen,
  Orbit,
  Paperclip,
  Play,
  Radio,
  Sparkles,
  Square,
  Telescope,
  User,
  Users,
  UserX,
  WandSparkles,
  X,
} from 'lucide-react';
import { api } from '../../../lib/api';
import type { ActiveChatRun, BackgroundChatJob } from '../../../lib/api';
import { useConversation } from '../../../contexts/ConversationContext';
import { useNotify } from '../../../contexts/NotificationContext';
import { orphanedRuns } from '../../../contexts/conversation/orphanedRuns';
import { DropdownPortal } from '../../ui/DropdownPortal';
import { useIsMobile } from '../../../lib/hooks';
import { RESEARCH_MODE, THINKING_MODE_LABELS, type ThinkingModeOption } from '../../../lib/thinkingModes';
import { ThinkingModeMenu } from '../ThinkingModeMenu';
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
  /** Whether the active model/profile supports vision (gates the image tile). */
  visionEnabled?: boolean;
  /** Open the audio picker — attaches clips to the next message (audio input). */
  onAttachAudio?: () => void;
  /** Start/stop a voice-note recording (undefined when the mic isn't available). */
  onRecordVoice?: () => void;
  uploadingAudio?: boolean;
  recordingVoice?: boolean;
  /** Global audio-input gate (audio.input_enabled) — hides both audio tiles. */
  audioEnabled?: boolean;
  /** Open the Projects hub for this conversation's project. */
  onOpenProject?: () => void;
  /** Name of the conversation's project, if it belongs to one. */
  projectName?: string;
  hasProject?: boolean;
  /** Auto-title the active conversation (LLM). */
  onAutoTitle?: () => void;
  canAutoTitle?: boolean;
  titling?: boolean;
  // — Command-center tiles —
  /** Unified thinking mode ('' = Auto … 'research'). */
  thinkingMode: string;
  onThinkingModeChange: (mode: string) => void;
  thinkingModeOptions: ThinkingModeOption[];
  useMemory: boolean;
  canToggleMemory: boolean;
  onToggleMemory: () => void;
  showSoloToggle: boolean;
  soloMode: boolean;
  onToggleSolo: () => void;
  modelLabel: string;
  modelOverridden: boolean;
  onOpenModelPicker: () => void;
  contextChip: { label: string; warn: boolean; title: string } | null;
  agentName?: string;
  /** Open the Conversation State drawer (goals/decisions/threads). */
  onOpenState: () => void;
  conversationId?: string;
}

/** One control-center tile: icon + label (+hint), tinted when ON. */
function RelayTile({
  icon, label, hint, on, warn, wide, disabled, title, onClick, pill,
}: {
  icon: ReactNode;
  label: string;
  hint?: string;
  on?: boolean;
  warn?: boolean;
  wide?: boolean;
  disabled?: boolean;
  title?: string;
  onClick?: () => void;
  pill?: string;
}) {
  return (
    <button
      className={
        `relay-tile${wide ? ' relay-tile--wide' : ''}${on ? ' on' : ''}${warn ? ' warn' : ''}`
      }
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      <span className="relay-tile-top">
        <span className="relay-tile-icon">{icon}</span>
        {pill && <span className="relay-tile-pill">{pill}</span>}
      </span>
      <span className="relay-tile-label">{label}</span>
      {hint && <span className="relay-tile-hint">{hint}</span>}
    </button>
  );
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
  onAttachAudio,
  onRecordVoice,
  uploadingAudio,
  recordingVoice,
  audioEnabled,
  onOpenProject,
  projectName,
  hasProject,
  onAutoTitle,
  canAutoTitle,
  titling,
  thinkingMode,
  onThinkingModeChange,
  thinkingModeOptions,
  useMemory,
  canToggleMemory,
  onToggleMemory,
  showSoloToggle,
  soloMode,
  onToggleSolo,
  modelLabel,
  modelOverridden,
  onOpenModelPicker,
  contextChip,
  agentName,
  onOpenState,
}: RelayMenuProps) {
  const isMobile = useIsMobile();
  const [jobs, setJobs] = useState<BackgroundChatJob[]>([]);
  const [runs, setRuns] = useState<ActiveChatRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [stopping, setStopping] = useState<string | null>(null);
  const { tabs, resumeRun } = useConversation();
  const { notify, notifyError } = useNotify();

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
    try {
      await resumeRun(run);
      onClose();
    } catch (err) {
      notifyError(err);
    }
  };

  const handleStop = async (run: ActiveChatRun) => {
    setStopping(run.run_id);
    try {
      const res = await api.cancelChatRun(run.run_id);
      // A live run settles cooperatively (next SSE boundary); an orphaned one
      // is settled server-side immediately. Either way it leaves "running".
      setRuns(prev => prev.map(r =>
        r.run_id === run.run_id
          ? { ...r, status: res.status && res.status !== 'running' ? res.status : 'cancelled' }
          : r));
      notify({ kind: 'success', title: 'Run stopped', message: run.message.slice(0, 80) || run.run_id });
    } catch (err) {
      notifyError(err);
    } finally {
      setStopping(null);
    }
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

  const research = thinkingMode === RESEARCH_MODE;
  const modeLabel = THINKING_MODE_LABELS[thinkingMode] ?? 'Auto';
  const modeHint =
    thinkingModeOptions.find(o => o.value === thinkingMode)?.hint
    ?? 'Pick the best pattern per message';

  const body = (
    <div
      className={`relay-menu${isMobile ? ' relay-menu--sheet' : ''}`}
      role="dialog"
      aria-label="Relay — conversation command center"
    >
      {isMobile && <div className="relay-sheet-handle" aria-hidden />}
      <div className="relay-menu-header">
        <Orbit size={14} />
        <span>Relay</span>
        <button className="relay-close" onClick={onClose} aria-label="Close">
          <X size={14} />
        </button>
      </div>

      {/* Status strip — a read-only pulse of the conversation. */}
      {(contextChip || modelLabel || agentName) && (
        <div className="relay-status">
          {agentName && (
            <span className="relay-status-item" title="Active agent">
              <User size={11} />
              <span>{agentName}</span>
            </span>
          )}
          {modelLabel && (
            <span className="relay-status-item" title="Active model">
              <Cpu size={11} />
              <span>{modelLabel}</span>
            </span>
          )}
          {contextChip && (
            <span
              className={`relay-status-item${contextChip.warn ? ' warn' : ''}`}
              title={contextChip.title}
            >
              <Gauge size={11} />
              <span>{contextChip.label}</span>
            </span>
          )}
        </div>
      )}

      <div className="relay-grid">
        {/* Thinking Mode — the wide hero tile. Research replaces patterns. */}
        <ThinkingModeMenu
          value={thinkingMode}
          onChange={onThinkingModeChange}
          options={thinkingModeOptions}
        >
          <button
            className={`relay-tile relay-tile--wide relay-tile--mode${thinkingMode ? ' on' : ''}${research ? ' research' : ''}`}
            title="Thinking mode — how the agent works this conversation"
          >
            <span className="relay-tile-top">
              <span className="relay-tile-icon">
                {research ? <Telescope size={16} /> : <Brain size={16} />}
              </span>
              <span className="relay-tile-pill">{modeLabel}</span>
            </span>
            <span className="relay-tile-label">Thinking mode</span>
            <span className="relay-tile-hint">{modeHint}</span>
          </button>
        </ThinkingModeMenu>

        <RelayTile
          icon={useMemory ? <MemoryStick size={16} /> : <Ghost size={16} />}
          label={useMemory ? 'Memory' : 'No memory'}
          hint={useMemory ? 'This chat is remembered' : 'Ephemeral chat'}
          on={useMemory}
          warn={!useMemory}
          disabled={!canToggleMemory}
          onClick={onToggleMemory}
          title={
            canToggleMemory
              ? 'Toggle memorization for this chat'
              : 'Memorization locked once the conversation has started'
          }
        />
        {showSoloToggle && (
          <RelayTile
            icon={soloMode ? <UserX size={16} /> : <Users size={16} />}
            label={soloMode ? 'Solo' : 'Team'}
            hint={soloMode ? 'No delegation' : 'May delegate to teammates'}
            on={!soloMode}
            onClick={onToggleSolo}
            title="Toggle ad-hoc delegation for this conversation"
          />
        )}
        <RelayTile
          icon={<Box size={16} />}
          label="Background"
          hint={backgroundArmed ? 'Next send runs detached' : 'Arm the next send'}
          on={backgroundArmed}
          disabled={!canArmBackground}
          onClick={onToggleBackground}
          pill={backgroundArmed ? 'ARMED' : undefined}
          title="Stage the next message to run in the background; the result lands in the inbox below."
        />
        <RelayTile
          icon={<Cpu size={16} />}
          label={modelLabel || 'Model'}
          hint={modelOverridden ? 'Overridden for this chat' : 'From the profile'}
          on={modelOverridden}
          onClick={() => { onClose(); onOpenModelPicker(); }}
          title="Choose the model for this conversation"
        />
        <RelayTile
          icon={<FolderOpen size={16} />}
          label={hasProject ? (projectName || 'Project') : 'Projects'}
          hint={hasProject ? 'Files, instructions, chats' : 'Browse or attach one'}
          on={!!hasProject}
          disabled={!onOpenProject}
          onClick={() => { onOpenProject?.(); onClose(); }}
          title={hasProject ? 'Open this conversation’s project' : 'Browse projects'}
        />
        <RelayTile
          icon={<NotebookPen size={16} />}
          label="State"
          hint="Goals · decisions · threads"
          onClick={() => { onClose(); onOpenState(); }}
          title="View and edit this conversation's structured state"
        />
        {visionEnabled && (
          <RelayTile
            icon={<ImageIcon size={16} />}
            label={uploadingImage ? 'Uploading…' : 'Attach image'}
            hint="For the model to see"
            disabled={!onAttachImage || uploadingImage}
            onClick={() => { onAttachImage?.(); onClose(); }}
            title="Attach an image to the next message — or just paste one into the composer"
          />
        )}
        {audioEnabled && (
          <RelayTile
            icon={<Music size={16} />}
            label={uploadingAudio ? 'Uploading…' : 'Attach audio'}
            hint="For the model to hear"
            disabled={!onAttachAudio || uploadingAudio}
            onClick={() => { onAttachAudio?.(); onClose(); }}
            title="Attach an audio clip — audio models hear it; others get a transcript"
          />
        )}
        {audioEnabled && onRecordVoice && (
          <RelayTile
            icon={<Mic size={16} />}
            label={recordingVoice ? 'Recording…' : 'Voice note'}
            hint={recordingVoice ? 'Stop from the composer' : 'Record for the message'}
            on={recordingVoice}
            disabled={recordingVoice}
            onClick={() => { onRecordVoice(); onClose(); }}
            title="Record a voice note to attach to the next message"
          />
        )}
        <RelayTile
          icon={<Paperclip size={16} />}
          label={attachingFile ? 'Attaching…' : 'Attach file'}
          hint="Into the project (RAG)"
          disabled={!onAttachFile || attachingFile}
          onClick={() => { onAttachFile?.(); onClose(); }}
          title="Attach a document the agent can search (Document RAG)"
        />
        <RelayTile
          icon={<Sparkles size={16} />}
          label={isEnhancing ? 'Enhancing…' : 'Enhance'}
          hint="Rewrite the draft"
          disabled={!canEnhance || isEnhancing}
          onClick={onEnhance}
          title="Rewrite the current draft using the prompt enhancer"
        />
        <RelayTile
          icon={<WandSparkles size={16} />}
          label={titling ? 'Titling…' : 'Auto-title'}
          hint="Name this chat"
          disabled={!onAutoTitle || !canAutoTitle || titling}
          onClick={() => { onAutoTitle?.(); onClose(); }}
          title="Generate a concise title for this conversation"
        />
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
                <div className="relay-run-actions">
                  <button
                    className="relay-resume-btn"
                    onClick={() => handleResume(run)}
                    title="Reopen this run and continue streaming"
                  >
                    <Play size={12} />
                    <span>Resume</span>
                  </button>
                  <button
                    className="relay-stop-btn"
                    onClick={() => handleStop(run)}
                    disabled={stopping === run.run_id}
                    title="Stop this run"
                  >
                    <Square size={12} />
                    <span>Stop</span>
                  </button>
                </div>
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
      estimatedHeight={560}
    >
      {body}
    </DropdownPortal>
  );
}

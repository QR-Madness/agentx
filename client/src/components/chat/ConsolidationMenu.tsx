/**
 * ConsolidationMenu — Dropdown menu for the lightning icon in the TopBar.
 *
 * Shows:
 *  - Trigger button with "don't ask again" checkbox
 *  - Live streaming progress when consolidation is active
 *  - Final result summary after completion
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Zap, CheckCircle, XCircle, Loader2, ChevronRight } from 'lucide-react';
import './ConsolidationMenu.css';

const CONSOLIDATION_SKIP_KEY = 'agentx:consolidation:skipPrompt';

interface ConsolidationMenuProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: React.RefObject<HTMLElement | null>;
  consolidation: {
    isActive: boolean;
    isStreaming: boolean;
    currentJob: string | null;
    currentJobIndex: number;
    totalJobs: number;
    stage: string | null;
    progress: Record<string, unknown> | null;
    result: Record<string, unknown> | null;
    error: string | null;
    trigger: (jobs?: string[]) => { abort: () => void } | undefined;
    stop: () => void;
  };
}

const JOB_LABELS: Record<string, string> = {
  consolidate: 'Extract & Store',
  patterns: 'Detect Patterns',
  promote: 'Promote to Global',
  decay: 'Apply Decay',
  cleanup: 'Cleanup Old',
  entity_linking: 'Link Entities',
};

function formatStage(stage: string | null): string {
  if (!stage) return '';
  const labels: Record<string, string> = {
    discovery: 'Discovering conversations…',
    processing: 'Processing turns…',
    storing: 'Storing to memory…',
    complete: 'Complete',
    starting: 'Starting…',
    done: 'Done',
  };
  return labels[stage] || stage;
}

function formatProgressDetail(progress: Record<string, unknown> | null): string | null {
  if (!progress) return null;
  const parts: string[] = [];

  if (progress.conversation) parts.push(`Conv ${progress.conversation}`);
  if (progress.conversations_found) parts.push(`${progress.conversations_found} conversations`);
  if (progress.entities_stored) parts.push(`${progress.entities_stored} entities`);
  if (progress.facts_stored) parts.push(`${progress.facts_stored} facts`);
  if (progress.relationships_stored) parts.push(`${progress.relationships_stored} rels`);

  return parts.length > 0 ? parts.join(' · ') : null;
}

export function ConsolidationMenu({ isOpen, onClose, anchorRef, consolidation }: ConsolidationMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, right: 0 });
  const [skipPrompt, setSkipPrompt] = useState(
    () => localStorage.getItem(CONSOLIDATION_SKIP_KEY) === 'true'
  );

  // Position the menu below the anchor button
  useEffect(() => {
    if (!isOpen || !anchorRef.current) return;

    const rect = anchorRef.current.getBoundingClientRect();
    setPosition({
      top: rect.bottom + 6,
      right: window.innerWidth - rect.right,
    });
  }, [isOpen, anchorRef]);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;

    const handleClick = (e: MouseEvent) => {
      if (
        menuRef.current && !menuRef.current.contains(e.target as Node) &&
        anchorRef.current && !anchorRef.current.contains(e.target as Node)
      ) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [isOpen, onClose, anchorRef]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;

    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  const handleSkipChange = useCallback((checked: boolean) => {
    setSkipPrompt(checked);
    if (checked) {
      localStorage.setItem(CONSOLIDATION_SKIP_KEY, 'true');
    } else {
      localStorage.removeItem(CONSOLIDATION_SKIP_KEY);
    }
  }, []);

  const handleTrigger = useCallback(() => {
    consolidation.trigger();
  }, [consolidation]);

  if (!isOpen) return null;

  const { isStreaming, currentJob, currentJobIndex, totalJobs, stage, progress, result, error } = consolidation;
  const progressDetail = formatProgressDetail(progress);

  const menu = (
    <div
      ref={menuRef}
      className="consolidation-menu"
      style={{ top: position.top, right: position.right }}
    >
      <div className="consolidation-menu-header">
        <Zap size={14} />
        <span>Memory Consolidation</span>
      </div>

      {/* Idle state — show trigger */}
      {!isStreaming && !result && !error && (
        <div className="consolidation-menu-idle">
          <p className="consolidation-menu-desc">
            Extract entities, facts, and relationships from recent conversations.
          </p>
          <label className="consolidation-menu-check">
            <input
              type="checkbox"
              checked={skipPrompt}
              onChange={e => handleSkipChange(e.target.checked)}
            />
            <span>Don't ask again</span>
          </label>
          <button className="consolidation-menu-trigger" onClick={handleTrigger}>
            <Zap size={13} />
            Run Consolidation
          </button>
        </div>
      )}

      {/* Streaming state — show live progress */}
      {isStreaming && (
        <div className="consolidation-menu-progress">
          <div className="consolidation-progress-header">
            <Loader2 size={14} className="spinning" />
            <span>
              {currentJob
                ? `${JOB_LABELS[currentJob] || currentJob} (${currentJobIndex}/${totalJobs})`
                : 'Starting…'}
            </span>
          </div>

          {stage && (
            <div className="consolidation-progress-stage">
              <ChevronRight size={12} />
              <span>{formatStage(stage)}</span>
            </div>
          )}

          {progressDetail && (
            <div className="consolidation-progress-detail">
              {progressDetail}
            </div>
          )}

          {totalJobs > 0 && (
            <div className="consolidation-progress-bar-track">
              <div
                className="consolidation-progress-bar-fill"
                style={{ width: `${Math.max(5, ((currentJobIndex - 1) / totalJobs) * 100)}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Result state */}
      {result && !isStreaming && (
        <div className="consolidation-menu-result">
          <div className={`consolidation-result-icon ${result.success ? 'success' : 'failure'}`}>
            {result.success ? <CheckCircle size={16} /> : <XCircle size={16} />}
            <span>{result.success ? 'Consolidation Complete' : 'Consolidation Failed'}</span>
          </div>
          {typeof result.duration_ms === 'number' && (
            <div className="consolidation-result-time">
              {(result.duration_ms / 1000).toFixed(1)}s
            </div>
          )}
          <button className="consolidation-menu-trigger" onClick={handleTrigger}>
            <Zap size={13} />
            Run Again
          </button>
        </div>
      )}

      {/* Error state */}
      {error && !isStreaming && !result && (
        <div className="consolidation-menu-error">
          <XCircle size={14} />
          <span>{error}</span>
          <button className="consolidation-menu-trigger" onClick={handleTrigger}>
            <Zap size={13} />
            Retry
          </button>
        </div>
      )}
    </div>
  );

  return createPortal(menu, document.body);
}

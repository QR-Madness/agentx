/**
 * PlanExecutionBlock — Renders plan execution progress with subtask status tracking
 */

import { useState } from 'react';
import {
  ListChecks,
  ChevronDown,
  ChevronRight,
  Clock,
  PlayCircle,
} from 'lucide-react';
import type { PlanSubtask } from '../../lib/messages';
import { PlanProgressBar, SubtaskList } from '../plans/PlanSubtaskList';
import './PlanExecutionBlock.css';

export interface PlanExecutionBlockProps {
  planId: string;
  task: string;
  complexity: string;
  subtaskCount: number;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  subtasks: PlanSubtask[];
  totalTimeMs?: number;
  completedCount?: number;
  /** When set, an interrupted plan offers a Resume action (continues remaining subtasks). */
  onResume?: (planId: string) => void;
  /** Whether resume is currently offered (interrupted + not mid-stream). */
  canResume?: boolean;
}

export function PlanExecutionBlock({
  planId,
  task,
  complexity,
  subtaskCount,
  status,
  subtasks,
  totalTimeMs,
  completedCount,
  onResume,
  canResume,
}: PlanExecutionBlockProps) {
  const [expanded, setExpanded] = useState(true);

  const completed = completedCount ?? subtasks.filter(s => s.status === 'completed').length;
  const showResume = !!(canResume && onResume);

  return (
    <div className="plan-execution-block">
      <div className="plan-header" onClick={() => setExpanded(!expanded)}>
        <div className="plan-icon">
          <ListChecks size={14} />
        </div>
        <div className="plan-info">
          <span className="plan-title">Plan Execution</span>
          <span className="plan-complexity">{complexity}</span>
          <span className="plan-progress-text">
            {completed}/{subtaskCount} steps
          </span>
        </div>
        <span className={`plan-status-badge ${status}`}>{status}</span>
        <button className="plan-toggle">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>

      <PlanProgressBar completed={completed} total={subtaskCount} status={status} />

      {expanded && (
        <div className="plan-content">
          {/* Task description */}
          <div className="plan-task">
            <span className="section-label">Task:</span>
            <span className="task-text">{task}</span>
          </div>

          <SubtaskList subtasks={subtasks} />

          {showResume && (
            <button
              type="button"
              className="plan-resume-btn"
              onClick={(e) => { e.stopPropagation(); onResume!(planId); }}
              title="Continue this plan's remaining steps"
            >
              <PlayCircle size={13} /> Resume plan
            </button>
          )}
        </div>
      )}

      {/* Footer with timing */}
      {status !== 'running' && totalTimeMs != null && (
        <div className="plan-footer">
          <div className="plan-timing">
            <Clock size={11} />
            <span>{(totalTimeMs / 1000).toFixed(1)}s</span>
          </div>
          <span>{completed}/{subtaskCount} completed</span>
        </div>
      )}
    </div>
  );
}

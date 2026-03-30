/**
 * PlanExecutionBlock — Renders plan execution progress with subtask status tracking
 */

import { useState } from 'react';
import {
  ListChecks,
  ChevronDown,
  ChevronRight,
  Circle,
  Loader2,
  CheckCircle,
  XCircle,
  SkipForward,
  Clock,
} from 'lucide-react';
import type { PlanSubtask } from '../../lib/messages';
import './PlanExecutionBlock.css';

export interface PlanExecutionBlockProps {
  planId: string;
  task: string;
  complexity: string;
  subtaskCount: number;
  status: 'running' | 'completed' | 'failed';
  subtasks: PlanSubtask[];
  totalTimeMs?: number;
  completedCount?: number;
}

const STATUS_ICONS: Record<PlanSubtask['status'], typeof Circle> = {
  pending: Circle,
  running: Loader2,
  completed: CheckCircle,
  failed: XCircle,
  skipped: SkipForward,
};

function SubtaskItem({ subtask }: { subtask: PlanSubtask }) {
  const Icon = STATUS_ICONS[subtask.status];

  return (
    <li className={`subtask-item ${subtask.status}`}>
      <span className={`subtask-status-icon ${subtask.status}`}>
        <Icon size={14} />
      </span>
      <div className="subtask-body">
        <div className="subtask-description">{subtask.description}</div>
        {subtask.subtaskType && (
          <div className="subtask-type">{subtask.subtaskType}</div>
        )}
        {subtask.resultPreview && subtask.status === 'completed' && (
          <div className="subtask-result">{subtask.resultPreview}</div>
        )}
        {subtask.error && subtask.status === 'failed' && (
          <div className="subtask-error">{subtask.error}</div>
        )}
      </div>
    </li>
  );
}

export function PlanExecutionBlock({
  task,
  complexity,
  subtaskCount,
  status,
  subtasks,
  totalTimeMs,
  completedCount,
}: PlanExecutionBlockProps) {
  const [expanded, setExpanded] = useState(true);

  const completed = completedCount ?? subtasks.filter(s => s.status === 'completed').length;
  const progressPct = subtaskCount > 0 ? (completed / subtaskCount) * 100 : 0;

  const progressBarClass = status === 'completed'
    ? 'complete'
    : status === 'failed'
      ? 'failed'
      : '';

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

      {/* Progress bar */}
      <div className="plan-progress-bar">
        <div
          className={`plan-progress-fill ${progressBarClass}`}
          style={{ width: `${progressPct}%` }}
        />
      </div>

      {expanded && (
        <div className="plan-content">
          {/* Task description */}
          <div className="plan-task">
            <span className="section-label">Task:</span>
            <span className="task-text">{task}</span>
          </div>

          {/* Subtask list */}
          {subtasks.length > 0 && (
            <ul className="plan-subtasks">
              {subtasks.map((subtask) => (
                <SubtaskItem key={subtask.subtaskId} subtask={subtask} />
              ))}
            </ul>
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

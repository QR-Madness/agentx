/**
 * Shared plan-subtask rendering used by both the in-chat PlanExecutionBlock
 * and the "Plans in Progress" drawer, so the two surfaces stay visually
 * identical. Relies on the class names defined in
 * `../chat/PlanExecutionBlock.css` — callers must import that stylesheet.
 */

import {
  Circle,
  Loader2,
  CheckCircle,
  XCircle,
  SkipForward,
} from 'lucide-react';
import type { PlanSubtask } from '../../lib/messages';

export const STATUS_ICONS: Record<PlanSubtask['status'], typeof Circle> = {
  pending: Circle,
  running: Loader2,
  completed: CheckCircle,
  failed: XCircle,
  skipped: SkipForward,
};

export function SubtaskItem({
  subtask,
  emphasized,
  as = 'li',
}: {
  subtask: PlanSubtask;
  emphasized?: boolean;
  /** Element to render as. Use 'div' when nesting inside another <li> (e.g. the
   *  drawer's jump-wrapper) to avoid invalid <li>-in-<li> DOM nesting. */
  as?: 'li' | 'div';
}) {
  const Icon = STATUS_ICONS[subtask.status];
  const El = as;

  return (
    <El className={`subtask-item ${subtask.status}${emphasized ? ' current' : ''}`}>
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
    </El>
  );
}

export function SubtaskList({ subtasks }: { subtasks: PlanSubtask[] }) {
  if (subtasks.length === 0) return null;
  return (
    <ul className="plan-subtasks">
      {subtasks.map((s) => (
        <SubtaskItem
          key={s.subtaskId}
          subtask={s}
          emphasized={s.status === 'running'}
        />
      ))}
    </ul>
  );
}

export type PlanVisualStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'stale' | 'interrupted';

/**
 * Freeze the live spinner on a non-streaming plan. A `running` subtask only
 * genuinely spins while the plan is actively streaming; on a cancelled/finished
 * (`terminal`) or interrupted-but-resumable (`resumable`) plan that same icon is
 * misleading — it implies work is happening when nothing is.
 *
 *  - `live`      → unchanged (a real subtask is running).
 *  - `resumable` → `running` shown as `pending` (it will re-run on resume).
 *  - `terminal`  → `running` shown as `skipped` (it never finished, won't).
 */
export function freezeSubtasks(
  subtasks: PlanSubtask[],
  mode: 'live' | 'resumable' | 'terminal',
): PlanSubtask[] {
  if (mode === 'live') return subtasks;
  const next: PlanSubtask['status'] = mode === 'resumable' ? 'pending' : 'skipped';
  return subtasks.map(s => (s.status === 'running' ? { ...s, status: next } : s));
}

/** Maps the (wider) plan status to the progress-fill modifier class. */
export function progressFillClass(status: PlanVisualStatus): string {
  if (status === 'completed') return 'complete';
  if (status === 'failed' || status === 'cancelled' || status === 'stale') return 'failed';
  return '';
}

export function PlanProgressBar({
  completed,
  total,
  status,
}: {
  completed: number;
  total: number;
  status: PlanVisualStatus;
}) {
  const pct = total > 0 ? Math.min((completed / total) * 100, 100) : 0;
  return (
    <div className="plan-progress-bar">
      <div
        className={`plan-progress-fill ${progressFillClass(status)}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

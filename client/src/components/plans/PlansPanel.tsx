/**
 * PlansPanel — the "Plans in Progress" drawer body.
 *
 * Lists every plan (live + persisted across tabs) as its own collapsible
 * block: compact status when collapsed, full step list + progress when
 * expanded. Running plans owned by any tab can be cancelled; each step links
 * back to its annotated message group in the owning conversation.
 *
 * On mount it reconciles persisted `running` plans against the server's
 * Redis-tracked state (`api.getPlanStatus`) so a reload mid-plan never leaves
 * an eternal spinner — interrupted/expired plans settle to a terminal state.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ListChecks,
  ChevronDown,
  ChevronRight,
  Clock,
  X,
  CornerDownRight,
} from 'lucide-react';
import { api } from '../../lib/api';
import { useConversation } from '../../contexts/ConversationContext';
import {
  usePlans,
  derivePersistedPlans,
  mergePlans,
  isTerminalPlanStatus,
  type PlanRecord,
  type PlanRecordStatus,
} from '../../contexts/PlansContext';
import type { ConversationMessage, PlanSubtask } from '../../lib/messages';
import { PlanProgressBar, SubtaskItem, freezeSubtasks } from './PlanSubtaskList';
import '../chat/PlanExecutionBlock.css';
import './PlansPanel.css';

/** Cross-component signal consumed by ChatPanel to focus a plan / step. */
export interface JumpToStepDetail {
  tabId: string;
  planId: string;
  subtaskId?: number;
}

function dispatchJump(detail: JumpToStepDetail) {
  window.dispatchEvent(new CustomEvent<JumpToStepDetail>('agentx:jump-to-step', { detail }));
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

function PlanBlock({
  plan,
  now,
  live,
  onCancel,
  onJump,
  defaultExpanded,
}: {
  plan: PlanRecord;
  now: number;
  live: boolean;
  onCancel: (plan: PlanRecord) => void;
  onJump: (detail: JumpToStepDetail) => void;
  defaultExpanded: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const running = plan.status === 'running';
  // Freeze the spinner unless this plan is actively streaming.
  const shownSubtasks = freezeSubtasks(
    plan.subtasks,
    live ? 'live' : running ? 'resumable' : 'terminal',
  );
  const elapsedMs = running
    ? now - new Date(plan.startedAt).getTime()
    : plan.totalTimeMs ?? 0;

  return (
    <div className={`plan-block ${plan.status}`}>
      <div className="plan-block-header" onClick={() => setExpanded(e => !e)}>
        <button className="plan-block-toggle" aria-label={expanded ? 'Collapse' : 'Expand'}>
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
        <div className="plan-block-icon">
          <ListChecks size={14} />
        </div>
        <div className="plan-block-info">
          <span className="plan-block-task" title={plan.task}>{plan.task}</span>
          <span className="plan-block-meta">
            {plan.tabTitle && <span className="plan-block-tab">{plan.tabTitle}</span>}
            <span className="plan-block-complexity">{plan.complexity}</span>
            {elapsedMs > 0 && (
              <span className="plan-block-elapsed">
                <Clock size={10} /> {formatElapsed(elapsedMs)}
              </span>
            )}
          </span>
        </div>
        <span className={`plan-status-badge ${plan.status}`}>{plan.status}</span>
      </div>

      <PlanProgressBar
        completed={plan.completedCount}
        total={plan.subtaskCount}
        status={plan.status}
      />

      {expanded && (
        <div className="plan-block-body">
          <div className="plan-block-progress-text">
            {plan.completedCount}/{plan.subtaskCount} steps complete
          </div>
          {shownSubtasks.length > 0 ? (
            <ul className="plan-subtasks">
              {shownSubtasks.map((s: PlanSubtask) => (
                <li
                  key={s.subtaskId}
                  className="plan-step-jump"
                  onClick={() => onJump({ tabId: plan.tabId, planId: plan.planId, subtaskId: s.subtaskId })}
                  title="Jump to this step in the conversation"
                >
                  <SubtaskItem subtask={s} emphasized={s.status === 'running'} as="div" />
                  <CornerDownRight className="plan-step-jump-icon" size={12} />
                </li>
              ))}
            </ul>
          ) : (
            <div className="plan-block-empty">Waiting for the first step…</div>
          )}

          <div className="plan-block-actions">
            <button
              className="plan-block-action"
              onClick={() => onJump({ tabId: plan.tabId, planId: plan.planId })}
            >
              Open in conversation
            </button>
            {running && (
              <button
                className="plan-block-action plan-block-action--danger"
                onClick={() => onCancel(plan)}
              >
                <X size={12} /> Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function PlansPanel() {
  const { livePlans, upsertPlan } = usePlans();
  const { tabs, switchTab, updateTab } = useConversation();

  // Tick once a second while any plan is running, to refresh elapsed time.
  const [now, setNow] = useState(() => Date.now());
  const plans = useMemo(
    () => mergePlans(livePlans.values(), derivePersistedPlans(tabs)),
    [livePlans, tabs],
  );
  const hasRunning = plans.some(p => p.status === 'running');
  useEffect(() => {
    if (!hasRunning) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [hasRunning]);

  // Reconcile persisted `running` plans once on mount: a reload kills the
  // live stream, so anything still "running" in localStorage is settled
  // against Redis (completed/cancelled) or marked stale.
  const reconciledRef = useRef(false);
  useEffect(() => {
    if (reconciledRef.current) return;
    reconciledRef.current = true;

    const stale = derivePersistedPlans(tabs).filter(
      p => p.status === 'running' && !livePlans.has(p.planId),
    );
    if (stale.length === 0) return;

    for (const rec of stale) {
      const tab = tabs.find(t => t.id === rec.tabId);
      const sessionId = tab?.sessionId;
      const settle = (status: PlanRecordStatus, completedCount?: number) => {
        // Patch the in-chat card (any tab) and surface the corrected record
        // in the drawer via the live registry (live wins on merge).
        if (tab) {
          updateTab(rec.tabId, {
            messages: tab.messages.map((m: ConversationMessage) =>
              m.type === 'plan_execution' && m.planId === rec.planId
                ? { ...m, status: status === 'stale' ? 'cancelled' : status, completedCount: completedCount ?? m.completedCount }
                : m,
            ),
          });
        }
        upsertPlan({ ...rec, status, completedCount: completedCount ?? rec.completedCount });
      };

      if (!sessionId) {
        settle('stale');
        continue;
      }
      api
        .getPlanStatus(rec.planId, sessionId)
        .then(res => {
          if (!res.found) return settle('stale');
          if (res.status === 'complete') return settle('completed', res.completed_count);
          if (res.status === 'cancelled') return settle('cancelled', res.completed_count);
          if (res.resumable) {
            // Interrupted but resumable — keep it 'running' so the in-conversation
            // Resume affordance stays available; just refresh progress. (Flipping
            // it to cancelled/stale here is what made Resume disappear.)
            if (tab) {
              updateTab(rec.tabId, {
                messages: tab.messages.map((m: ConversationMessage) =>
                  m.type === 'plan_execution' && m.planId === rec.planId
                    ? { ...m, completedCount: res.completed_count ?? m.completedCount }
                    : m,
                ),
              });
            }
            upsertPlan({ ...rec, status: 'running', completedCount: res.completed_count ?? rec.completedCount });
            return;
          }
          // "active" but not resumable (expired snapshot) — the run is dead.
          settle('stale', res.completed_count);
        })
        .catch(() => settle('stale'));
    }
    // Intentionally mount-only: reconcile reflects state at drawer-open time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleCancel = async (plan: PlanRecord) => {
    const tab = tabs.find(t => t.id === plan.tabId);
    // Reflect intent immediately — the cooperative plan-cancel only lands at the
    // next subtask boundary (which is slow for a long research step), so without
    // this the button looks dead.
    upsertPlan({ ...plan, status: 'cancelled' });
    if (tab) {
      updateTab(plan.tabId, {
        messages: tab.messages.map((m: ConversationMessage) =>
          m.type === 'plan_execution' && m.planId === plan.planId
            ? { ...m, status: 'cancelled' }
            : m,
        ),
      });
    }
    // Hard-stop the live detached run if we have it (aborts at the next event
    // boundary — finer-grained than the per-subtask cooperative flag); always
    // also set the cooperative plan flag so a re-attached/other client stops too.
    const runId = tab?.activeRun?.runId;
    if (runId) {
      try { await api.cancelChatRun(runId); } catch { /* best-effort */ }
    }
    if (tab?.sessionId) {
      try { await api.cancelPlan(plan.planId, tab.sessionId); } catch { /* best-effort */ }
    }
  };

  const handleJump = (detail: JumpToStepDetail) => {
    switchTab(detail.tabId);
    // Defer so the target tab's messages mount before we scroll.
    requestAnimationFrame(() => dispatchJump(detail));
  };

  const active = plans.filter(p => !isTerminalPlanStatus(p.status));
  const recent = plans.filter(p => isTerminalPlanStatus(p.status));

  return (
    <div className="plans-panel">
      <div className="plans-panel-header">
        <ListChecks size={18} />
        <h2>Plans in Progress</h2>
      </div>

      {plans.length === 0 ? (
        <div className="plans-panel-empty">
          <ListChecks size={32} />
          <p>No plans yet.</p>
          <span>
            When a task is complex enough to be decomposed into steps, its
            progress shows up here.
          </span>
        </div>
      ) : (
        <div className="plans-panel-list">
          {active.length > 0 && (
            <>
              <div className="plans-panel-section">Active</div>
              {active.map(p => (
                <PlanBlock
                  key={p.planId}
                  plan={p}
                  now={now}
                  live={livePlans.has(p.planId)}
                  onCancel={handleCancel}
                  onJump={handleJump}
                  defaultExpanded
                />
              ))}
            </>
          )}
          {recent.length > 0 && (
            <>
              <div className="plans-panel-section">Recent</div>
              {recent.map(p => (
                <PlanBlock
                  key={p.planId}
                  plan={p}
                  now={now}
                  live={false}
                  onCancel={handleCancel}
                  onJump={handleJump}
                  defaultExpanded={false}
                />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

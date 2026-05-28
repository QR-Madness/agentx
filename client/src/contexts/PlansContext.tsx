/**
 * PlansContext — global registry of plan-execution runs, independent of which
 * conversation tab is active.
 *
 * Plan state is otherwise dual-natured and tab-local: ephemeral in
 * `useChatStream` while streaming, and persisted inside each tab's `messages`
 * as a `PlanExecutionMessage`. Neither gives a cross-tab view, which is what
 * the "Plans in Progress" drawer and the animated toolbar indicator need.
 *
 * This context holds the *live* records fed by `useChatStream` (one map entry
 * per `planId`). The drawer merges these with a derived list scanned from all
 * tabs' persisted plan messages (see `derivePersistedPlans`), so finished plans
 * survive a reload even though their live record is gone. Live records win on
 * `planId` collision.
 */

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';
import type { PlanSubtask } from '../lib/messages';

export type PlanRecordStatus = 'running' | 'completed' | 'failed' | 'cancelled' | 'stale';

export interface PlanRecord {
  planId: string;
  /** Owning conversation tab id. */
  tabId: string;
  tabTitle?: string;
  task: string;
  complexity: string;
  status: PlanRecordStatus;
  subtaskCount: number;
  completedCount: number;
  subtasks: PlanSubtask[];
  /** ISO timestamp the plan started (live) or last-known (persisted). */
  startedAt: string;
  totalTimeMs?: number;
  /** True for records reconstructed from persisted tab messages, not live SSE. */
  persisted?: boolean;
}

interface PlansContextValue {
  /** Live records keyed by planId, fed by useChatStream. */
  livePlans: Map<string, PlanRecord>;
  upsertPlan: (record: PlanRecord) => void;
  patchPlan: (planId: string, patch: Partial<PlanRecord>) => void;
  removePlan: (planId: string) => void;
}

const PlansContext = createContext<PlansContextValue | null>(null);

const TERMINAL_STATUSES: ReadonlySet<PlanRecordStatus> = new Set([
  'completed',
  'failed',
  'cancelled',
  'stale',
]);

export function isTerminalPlanStatus(status: PlanRecordStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function PlansProvider({ children }: { children: ReactNode }) {
  const [livePlans, setLivePlans] = useState<Map<string, PlanRecord>>(new Map());

  const upsertPlan = useCallback((record: PlanRecord) => {
    setLivePlans(prev => {
      const next = new Map(prev);
      next.set(record.planId, record);
      return next;
    });
  }, []);

  const patchPlan = useCallback((planId: string, patch: Partial<PlanRecord>) => {
    setLivePlans(prev => {
      const existing = prev.get(planId);
      if (!existing) return prev;
      const next = new Map(prev);
      next.set(planId, { ...existing, ...patch });
      return next;
    });
  }, []);

  const removePlan = useCallback((planId: string) => {
    setLivePlans(prev => {
      if (!prev.has(planId)) return prev;
      const next = new Map(prev);
      next.delete(planId);
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ livePlans, upsertPlan, patchPlan, removePlan }),
    [livePlans, upsertPlan, patchPlan, removePlan],
  );

  return <PlansContext.Provider value={value}>{children}</PlansContext.Provider>;
}

export function usePlans(): PlansContextValue {
  const ctx = useContext(PlansContext);
  if (!ctx) throw new Error('usePlans must be used within a PlansProvider');
  return ctx;
}

/** A conversation tab as seen by the persisted-plan selector (structural subset). */
interface TabLike {
  id: string;
  title: string;
  messages: Array<
    | {
        type: 'plan_execution';
        planId: string;
        task: string;
        complexity: string;
        subtaskCount: number;
        status: PlanRecord['status'] | 'running' | 'completed' | 'failed' | 'cancelled';
        subtasks: PlanSubtask[];
        totalTimeMs?: number;
        completedCount?: number;
        timestamp: string;
      }
    | { type: string }
  >;
}

/**
 * Build PlanRecords from the `plan_execution` messages persisted across all
 * tabs. Pure so the drawer can memoize on `tabs` and it can be unit-tested.
 */
export function derivePersistedPlans(tabs: TabLike[]): PlanRecord[] {
  const out: PlanRecord[] = [];
  for (const tab of tabs) {
    for (const msg of tab.messages) {
      if (msg.type !== 'plan_execution') continue;
      const m = msg as Extract<TabLike['messages'][number], { type: 'plan_execution' }>;
      out.push({
        planId: m.planId,
        tabId: tab.id,
        tabTitle: tab.title,
        task: m.task,
        complexity: m.complexity,
        status: m.status as PlanRecordStatus,
        subtaskCount: m.subtaskCount,
        completedCount:
          m.completedCount ?? m.subtasks.filter(s => s.status === 'completed').length,
        subtasks: m.subtasks,
        startedAt: m.timestamp,
        totalTimeMs: m.totalTimeMs,
        persisted: true,
      });
    }
  }
  return out;
}

/**
 * Merge live records over persisted ones (live wins on planId), returning a
 * single list sorted newest-first. Live records that lost their tab title
 * inherit it from the persisted twin when available.
 */
export function mergePlans(
  live: Iterable<PlanRecord>,
  persisted: PlanRecord[],
): PlanRecord[] {
  const byId = new Map<string, PlanRecord>();
  for (const p of persisted) byId.set(p.planId, p);
  for (const l of live) {
    const twin = byId.get(l.planId);
    byId.set(l.planId, twin ? { ...l, tabTitle: l.tabTitle ?? twin.tabTitle } : l);
  }
  return [...byId.values()].sort((a, b) => b.startedAt.localeCompare(a.startedAt));
}

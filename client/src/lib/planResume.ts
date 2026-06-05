/**
 * Helpers for the "resume an interrupted plan" choice exhibit.
 *
 * When a conversation loads with a plan still resumable in Redis, the client
 * auto-appends a `choice` exhibit (id `exh_resume_{planId}`). Picking "Resume
 * plan" submits a model-nudge turn that hands the agent the done/remaining steps
 * so it continues naturally (no PlanExecutor re-run); "Dismiss" just resolves it.
 */

import type { PlanStatusResponse } from './api/types';

export const RESUME_EXHIBIT_PREFIX = 'exh_resume_';
export const RESUME_CONFIRM = 'Resume plan';
export const RESUME_DISMISS = 'Dismiss';

export function resumeExhibitId(planId: string): string {
  return `${RESUME_EXHIBIT_PREFIX}${planId}`;
}

export function isResumeExhibitId(id: string): boolean {
  return id.startsWith(RESUME_EXHIBIT_PREFIX);
}

export function planIdFromResumeExhibit(id: string): string {
  return id.slice(RESUME_EXHIBIT_PREFIX.length);
}

/** Absolute expiry clock from the Redis TTL — stays accurate across reloads
 *  (unlike a baked "~Xm", which would go stale once persisted). */
export function expiryLabel(ttlSeconds?: number | null): string | null {
  if (ttlSeconds == null || ttlSeconds <= 0) return null;
  const at = new Date(Date.now() + ttlSeconds * 1000);
  return at.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

/** Coarse remaining-lifetime phrase for the prompt ("~47m", "~1h 10m"). */
export function ttlPhrase(ttlSeconds?: number | null): string | null {
  if (ttlSeconds == null || ttlSeconds <= 0) return null;
  const m = Math.round(ttlSeconds / 60);
  if (m < 60) return `~${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `~${h}h ${rem}m` : `~${h}h`;
}

const _stepText = (s: { id: number; description?: string }): string =>
  `- ${s.description?.trim() || `step ${s.id + 1}`}`;

/** Build the model-nudge message sent when the user picks "Resume plan". */
export function buildResumeNudge(status: PlanStatusResponse): string {
  const subs = status.subtasks ?? [];
  const done = subs.filter(s => s.status === 'complete');
  // pending/running (and anything non-terminal) is what's left to do.
  const remaining = subs.filter(
    s => s.status === 'pending' || s.status === 'running',
  );

  const lines: string[] = ["Let's pick the plan we started earlier back up."];
  if (done.length) {
    lines.push('', 'Already done:', ...done.map(_stepText));
  }
  if (remaining.length) {
    lines.push('', 'Still to do:', ...remaining.map(_stepText));
  }
  lines.push('', 'Please continue from where we left off.');
  return lines.join('\n');
}

/**
 * Pure pre-pass that folds consecutive messages sharing the same plan subtask
 * into a step group, so the transcript can render a "Step k/n · title" header
 * and visually bundle everything a step produced. Messages without a
 * `planStep` (including the plan_execution overview card and normal chat) pass
 * through as standalone items.
 *
 * Kept pure + dependency-free so it's unit-testable and cheap to memoize on
 * the message array.
 */

import type { ConversationMessage, PlanStepRef } from '../../lib/messages';

export interface StandaloneItem {
  kind: 'message';
  message: ConversationMessage;
}

export interface StepGroupItem {
  kind: 'stepGroup';
  /** Stable key: `${planId}:${subtaskId}`. */
  key: string;
  step: PlanStepRef;
  messages: ConversationMessage[];
}

export type GroupedItem = StandaloneItem | StepGroupItem;

export function groupMessagesBySteps(messages: ConversationMessage[]): GroupedItem[] {
  const out: GroupedItem[] = [];
  let current: StepGroupItem | null = null;

  for (const message of messages) {
    const step = message.planStep;
    if (!step) {
      current = null;
      out.push({ kind: 'message', message });
      continue;
    }

    const key = `${step.planId}:${step.subtaskId}`;
    if (current && current.key === key) {
      current.messages.push(message);
    } else {
      current = { kind: 'stepGroup', key, step, messages: [message] };
      out.push(current);
    }
  }

  return out;
}

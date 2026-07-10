/**
 * Pure pre-passes over the transcript:
 *
 * 1. `groupMessagesBySteps` folds consecutive messages sharing the same plan
 *    subtask into a step group ("Step k/n · title" bundles).
 * 2. `foldToolRuns` stacks runs of ≥2 consecutive tool_call messages into one
 *    ToolRunItem, so a long tool loop reads as a single collapsible stack
 *    instead of a mile of cards. Applied to the standalone stretch between
 *    step groups here, and inside each step by StepGroup itself.
 *
 * Kept pure + dependency-free so they're unit-testable and cheap to memoize
 * on the message array.
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

export interface ToolRunItem {
  kind: 'toolRun';
  /** Stable key: the first call's message id. */
  key: string;
  messages: ConversationMessage[];
}

export type GroupedItem = StandaloneItem | StepGroupItem | ToolRunItem;

/** Stack runs of ≥2 consecutive tool_call messages; a lone call stays standalone. */
export function foldToolRuns(
  messages: ConversationMessage[],
): Array<StandaloneItem | ToolRunItem> {
  const out: Array<StandaloneItem | ToolRunItem> = [];
  let run: ToolRunItem | null = null;

  for (const message of messages) {
    if (message.type === 'tool_call') {
      if (run) {
        run.messages.push(message);
      } else {
        run = { kind: 'toolRun', key: `toolrun-${message.id}`, messages: [message] };
        out.push(run);
      }
    } else {
      run = null;
      out.push({ kind: 'message', message });
    }
  }

  return out.map(item =>
    item.kind === 'toolRun' && item.messages.length === 1
      ? { kind: 'message', message: item.messages[0] }
      : item,
  );
}

export function groupMessagesBySteps(messages: ConversationMessage[]): GroupedItem[] {
  const grouped: Array<StandaloneItem | StepGroupItem> = [];
  let current: StepGroupItem | null = null;

  for (const message of messages) {
    const step = message.planStep;
    if (!step) {
      current = null;
      grouped.push({ kind: 'message', message });
      continue;
    }

    const key = `${step.planId}:${step.subtaskId}`;
    if (current && current.key === key) {
      current.messages.push(message);
    } else {
      current = { kind: 'stepGroup', key, step, messages: [message] };
      grouped.push(current);
    }
  }

  // Second pass: stack consecutive standalone tool calls between step groups.
  const out: GroupedItem[] = [];
  let buffer: ConversationMessage[] = [];
  const flush = () => {
    if (buffer.length > 0) {
      out.push(...foldToolRuns(buffer));
      buffer = [];
    }
  };
  for (const item of grouped) {
    if (item.kind === 'message') {
      buffer.push(item.message);
    } else {
      flush();
      out.push(item);
    }
  }
  flush();
  return out;
}
